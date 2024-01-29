"""MNE-BIDS-Pipeline configuration for natural-conversations-bids.

Run with:

    $ mne_bids_pipeline ./config.py

For options, see https://mne.tools/mne-bids-pipeline/stable/settings/general.html.
"""

from pathlib import Path

on_error = "debug"

study_name = "natural-conversations"
data_root = (Path(__file__).parent / ".." / "Natural_Conversations_study").resolve()
bids_root = data_root / f'{study_name}-bids'
interactive = False
sessions = "all"
task = "conversation"
subjects = ["01"]  # "all" TODO: Process all
runs = ["01", "02", "03", "04", "05", "06"]

ch_types = ["meg", "eeg"]
data_type = "meg"
eeg_reference = "average"

l_freq = None
h_freq = 40.0
epochs_decim = 5
process_rest = True

regress_artifact = dict(picks="meg", picks_artifact=["MISC 001", "MISC 002", "MISC 003"])

spatial_filter = 'ssp'
n_proj_ecg = dict(n_mag=1, n_eeg=0)
n_proj_eog = dict(n_mag=1, n_eeg=1)

reject = {'eeg': 200e-6, 'mag': 2000e-15}
conditions = ["ba", "da", "dummy"]  # dummy just to make event processing happy
epochs_tmin = -0.2
epochs_tmax = 0.5
baseline = (None, 0)

run_source_estimation = True
subjects_dir = bids_root / "derivatives" / "freesurfer" / "subjects"
use_template_mri = "fsaverage"
adjust_coreg = True  # TODO: Add option to use manual -trans.fif files
spacing = "oct6"
mindist = 1
inverse_method = "dSPM"
noise_cov = (None, 0.0)
