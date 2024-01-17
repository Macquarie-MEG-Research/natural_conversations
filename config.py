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

use_maxwell_filter = False
# find_flat_channels_meg = True  # TODO: Enable for files w/o cal + cross-talk
# find_noisy_channels_meg = True
# mf_head_origin = (0., 0., 0.04)
ch_types = ["meg", "eeg"]
data_type = "meg"

l_freq = None
h_freq = 40.0
epochs_decim = 5
process_rest = True

reject = "autoreject_global"
conditions = ["ba", "da", "dummy"]  # dummy just to make event processing happy
epochs_tmin = -0.2
epochs_tmax = 0.5
baseline = (None, 0)
spatial_filter = 'ssp'
n_proj_ecg = dict(n_mag=1, n_eeg=0)
n_proj_eog = dict(n_mag=1, n_eeg=1)

run_source_estimation = False
spacing = "oct6"
mindist = 1
source_info_path_update = {"processing": "clean", "suffix": "epo"}
inverse_method = "dSPM"
noise_cov = "rest"
