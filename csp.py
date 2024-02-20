'''
CSP is computed by mne-bids-pipeline (plots are shown in the report, and decoding
scores are saved in an excel file), but the classifiers themselves are not saved.

Here we write a script that gives the same results as the pipeline (so we can
extract the CSP output for individual subjects), then project to source space.

General steps:
0) band-pass filtering the epochs with epochs.filter,
1) mne.decoding.Scaler to deal with channel types,
2) sklearn PCA to reduce rank to that of the data,
3) mne.decoding.CSP,
4) Logistic Regression.

To start, should follow roughly:



Then, projecting to source space can follow the decoding tutorial:
https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#projecting-sensor-space-patterns-to-source-space

And the patterns tutorial:
https://mne.tools/stable/auto_examples/decoding/linear_model_patterns.html

TODO:
- Fix MNE-Python bug with get_coef
- Fix MNE-Python bug when bad channels are present (index error)
- Fix MNE-BIDS-Pipeline bug where Scalar isn't used
'''

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import mne
from mne.decoding import CSP, UnsupervisedSpatialFilter, Scaler, LinearModel, get_coef
from sklearn.decomposition import PCA


# adjust mne options to fix rendering issues (only needed in Linux / WSL)
mne.viz.set_3d_options(antialias = False, depth_peeling = False, 
                    smooth_shading = False, multi_samples = 1) 

#############################################################################

subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
            '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24', 
            '25', '26', '27'] # excluding subj 12 & 20

sub = '03'
path = Path(__file__).parents[1] / "Natural_Conversations_study" / "analysis" / 'natural-conversations-bids' / 'derivatives' / 'mne-bids-pipeline' / 'sub-' + sub / 'meg'
epochs_fname = path / 'sub-' + sub + '_task-conversation_proc-clean_epo.fif'
inv_fname = path / 'sub-' + sub + '_task-conversation_inv.fif'
subjects_dir = '/mnt/d/Work/analysis_ME206/processing/mri/' # only for plotting stc

# Read data
epochs = mne.read_epochs(epochs_fname).load_data()

# only select the conditions we are interested in
epochs = epochs[['conversation', 'repetition']].pick(["meg", "eeg"], exclude="bads")
assert epochs.info["bads"] == []  # should have picked good only
epochs.equalize_event_counts()
labels = epochs.events[:, 2] # conversation=2, repetition=4
ranks = mne.compute_rank(inst=epochs)
rank = sum(ranks.values())
print(f"Ranks={ranks} (total={rank})")

# Define the CSP parameters
contrasts = [("conversation", "repetition")]
decoding_csp_times = [-1, -0.5, 0, 0.5, 1]  #[-1, 0, 1] # before and after
decoding_csp_freqs = {
    'theta': [4, 7],
    'alpha': [8, 13],
    'beta': [14, 30],
    'gamma': [31, 49],
}
n_components = 4


# Construct the time bins
# Note: We don't support varying time ranges for different frequency
# ranges to avoid leaking of information.
time_bins = np.array(decoding_csp_times)
if time_bins.ndim == 1:
    time_bins = np.array(list(zip(time_bins[:-1], time_bins[1:])))
assert time_bins.ndim == 2

# Loop over frequency bands x time bins
for band, (fmin, fmax) in decoding_csp_freqs.items():
    for (tmin, tmax) in time_bins:
        
        # 0) band-pass filtering the epochs to get the relevant freq band
        epochs_filt = epochs.copy().filter(
            fmin, fmax, l_trans_bandwidth=1, h_trans_bandwidth=1, verbose="error",
        )

        # 1) mne.decoding.Scaler to deal with channel types
        scaler = Scaler(epochs.info)

        # 2) sklearn PCA to reduce rank to that of the data
        # Scaler should ensure that channel types are close enough in scale that using
        # the total rank should work.
        msg = f"Reducing data dimension via PCA; new rank: {rank}."
        pca = UnsupervisedSpatialFilter(PCA(rank), average=False)

        # 3) mne.decoding.CSP
        # https://github.com/mne-tools/mne-bids-pipeline/blob/d76abaa0cc1c0fbf7dc25e6ba6bcc6e0d4ed4284/mne_bids_pipeline/steps/sensor/_05_decoding_csp.py#L180
        # NB: Regularization can make a big difference in classification score!
        csp = CSP(n_components=n_components, reg=0.1, log=True, norm_trace=False)

        # 4) LDA
        lr = LinearModel(LogisticRegression(solver="liblinear"))

        # Put it in a pipeline
        steps = [("scaler", scaler), ("PCA", pca), ("CSP", csp), ("LR", lr)]
        clf = Pipeline(steps)
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        # Crop data to the time window of interest
        if tmax is not None:  # avoid warnings about outside the interval
            tmax = min(tmax, epochs_filt.times[-1])
        epochs_filt.crop(tmin, tmax)

        # Get the data for all time points
        X = epochs_filt.get_data(copy=False)
        del epochs_filt

        # Calculate the decoding scores
        scores = cross_val_score(clf, X, labels, cv=cv, verbose=True, scoring="roc_auc")
        print(f"{band} band, {tmin} to {tmax}s:")
        print(f"cv scores: {scores}")
        print(f"mean: {np.mean(scores)}")
        
        # plot CSP patterns
        clf.fit(X, labels)
        # In theory we should be able to extract the coef from the classifier:
        # coef = get_coef(clf, "patterns_", inverse_transform=True, verbose=True)
        # but there is a MNE-Python bug, so instead do it manually:
        coef = csp.patterns_[:n_components]
        assert coef.shape == (n_components, pca.estimator.n_components_), coef.shape
        coef = pca.estimator.inverse_transform(coef)
        assert coef.shape == (n_components, len(epochs.ch_names)), coef.shape
        coef = scaler.inverse_transform(coef.T[np.newaxis])[0]
        assert coef.shape == (len(epochs.ch_names), n_components), coef.shape

        evoked = mne.EvokedArray(coef, epochs.info, tmin=0)
        '''
        fig, axes = plt.subplots(2, n_components, figsize=(n_components * 2, 4), layout="constrained")
        for ci, ch_type in enumerate(("mag", "eeg")):
            fig = evoked.plot_topomap(axes=axes[ci], times=evoked.times, colorbar=False, show=False, ch_type=ch_type)
        fig.suptitle(f"{band}")
        '''

        # Project sensor-space patterns to source space
        inv = mne.minimum_norm.read_inverse_operator(inv_fname)
        stc = mne.minimum_norm.apply_inverse(evoked, inv, 1.0 / 9.0, "dSPM")
        #stc.save(f"{save_path}/sub-{sub}_{band}_{tmin}_to_{tmax}s.stc")
        '''
        brain = stc.plot(
            hemi="split", views=("lat", "med"), initial_time=0.01, subjects_dir=subjects_dir
        )
        '''
        
        # TODO: 
        # 1. In the STC, take the absolute value at each vertex, then binarize it -
        # perhaps the top quartile of weights have value 1 and the rest are zero:
        #data = (np.abs(stc.data) >= np.percentile(np.abs(stc.data), 75)).astype(float)

        # 2. The source loc for each CSP component is represented as a "time point"
        # (misnomer) in the STC. So to sum across the sources from the 4 CSP components,
        # we can just sum across the "time points". The resulting STC_sum will have 
        # discrete integer values 0, 1, 2, 3, and 4 only, indicating for a given vertex 
        # the number of components it was highly influential in.

        # 3. For group analysis, you can add up the STC_sum from each subject -  
        # this grand total STC will contain integer values between 0 and 120 
        # (30 subj x 4 components) at each vertex. In practice, no area should be 
        # used in all components for all subjects, so the max value will probably be 
        # much lower than 120. Plotting these tells us how much each source vertex 
        # contributes to the decoding, across subjects & CSP components.

        # Note: this is not a statistical test - it's more like a grand average/total


        #raise RuntimeError
        print("Done")
