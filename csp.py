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


#############################################################################
# Read data

path = Path(__file__).parents[1] / "Natural_Conversations_study" / "analysis" / 'natural-conversations-bids' / 'derivatives' / 'mne-bids-pipeline' / 'sub-01' / 'meg'
epochs_fname = path / 'sub-01_task-conversation_proc-clean_epo.fif'
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
decoding_csp_times = [-1, 0, 1]  # before and after
decoding_csp_freqs = {
    'theta': [4, 7],
    'alpha': [8, 13],
    'beta': [14, 30],
}
n_components = 4

# loop through the frequency bands
for band, (fmin, fmax) in decoding_csp_freqs.items():
    print(f"Decoding band: {band}")

    # 0) band-pass filtering the epochs with epochs.filter
    epochs_filt = epochs.copy().filter(
        fmin, fmax, l_trans_bandwidth=1, h_trans_bandwidth=1, verbose="error",
    )

    # Get the data for all time points
    X = epochs_filt.get_data(copy=False)
    del epochs_filt

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
    scores = cross_val_score(clf, X, labels, cv=cv, verbose=True, scoring="roc_auc")

    # Printing the results
    print(f"roc_auc: {np.mean(scores)}")

    # plot CSP patterns
    clf.fit(X, labels)
    # In theory we should be able to do this, but there is a MNE-Python bug:
    # coef = get_coef(clf, "patterns_", inverse_transform=True, verbose=True)
    # But it doesn't work, so instead do it manually:
    coef = csp.patterns_[:n_components]
    assert coef.shape == (n_components, pca.estimator.n_components_), coef.shape
    coef = pca.estimator.inverse_transform(coef)
    assert coef.shape == (n_components, len(epochs.ch_names)), coef.shape
    coef = scaler.inverse_transform(coef.T[np.newaxis])[0]
    assert coef.shape == (len(epochs.ch_names), n_components), coef.shape
    evoked = mne.EvokedArray(coef, epochs.info, tmin=0)
    fig, axes = plt.subplots(2, n_components, figsize=(n_components * 2, 4), layout="constrained")
    for ci, ch_type in enumerate(("mag", "eeg")):
        fig = evoked.plot_topomap(axes=axes[ci], times=evoked.times, colorbar=False, show=False, ch_type=ch_type)
    fig.suptitle(f"{band}")
    raise RuntimeError
