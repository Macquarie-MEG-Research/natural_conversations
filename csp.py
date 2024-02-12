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
4) LDA

Then, projecting to source space can follow the decoding tutorial:
https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#projecting-sensor-space-patterns-to-source-space

'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

import mne
from mne.decoding import CSP, UnsupervisedSpatialFilter
from sklearn.decomposition import PCA


#############################################################################
# Read data

path = '/mnt/d/Work/analysis_ME206/processing/bids/sub-01/meeg/'
epochs_fname = path + 'sub-01_task-conversation_proc-clean_epo.fif'
epochs = mne.read_epochs(epochs_fname)
    
# only select the conditions we are interested in
epochs = epochs[['conversation', 'repetition']]
epochs.equalize_event_counts()
labels = epochs.events[:, 2] # conversation=2, repetition=4

# Define the CSP parameters
contrasts = [("conversation", "repetition")]
decoding_csp_times = [-1, 0, 1]  # before and after
decoding_csp_freqs = {
    'theta': [4, 7],
    'alpha': [8, 13],
    'beta': [14, 30],
}

# loop through the frequency bands
for band, (fmin, fmax) in decoding_csp_freqs.items():
    print(f"Decoding band: {band}")

    # 0) band-pass filtering the epochs with epochs.filter
    epochs_filt = epochs.filter(fmin, fmax, n_jobs=1, verbose="error")

    # Get the data for all time points
    X = epochs_filt.get_data(picks="data", copy=False)  # omit bad channels

    # 1) mne.decoding.Scaler to deal with channel types
    scaler = mne.decoding.Scaler(epochs_filt.info)
    X = scaler.fit_transform(X, y=labels)

    # 2) sklearn PCA to reduce rank to that of the data 
    # Select the channel type with the smallest rank.
    # Limit it to a maximum of 100.
    ranks = mne.compute_rank(inst=epochs_filt, rank="info")
    ch_type_smallest_rank = min(ranks, key=ranks.get)
    rank = min(ranks[ch_type_smallest_rank], 100)

    msg = f"Reducing data dimension via PCA; new rank: {rank}."
    pca = UnsupervisedSpatialFilter(PCA(rank), average=False)

    # We apply PCA before running CSP:
    # - much faster CSP processing
    # - reduced risk of numerical instabilities.
    X = pca.fit_transform(X)
    
    # 3) mne.decoding.CSP
    # 4) LDA

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(X, y=labels)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, X, labels, cv=cv, n_jobs=None)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

    # plot CSP patterns
    csp.fit_transform(X, labels)
    csp.plot_patterns(epochs_filt.info, units="Patterns (AU)", size=1.5)
