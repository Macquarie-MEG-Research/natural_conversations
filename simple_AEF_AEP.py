#!/usr/bin/python3
#
# Simple AEF & AEP analysis (for sanity check)
#
# Authors: Paul Sowman, Judy Zhu, Eric Larson

#######################################################################################

from pathlib import Path

import mne
import mne_bids
import meegkit  # noqa

bids_root = Path("../Natural_Conversations_study/natural-conversations-bids").resolve(strict=True)
bids_path = mne_bids.BIDSPath(subject="01", datatype="meg", task="conversation", run="06", root=bids_root)
raw = mne_bids.read_raw_bids(bids_path).load_data()
picks_ref = ["MISC 001", "MISC 002", "MISC 003"]

# Apply reference regression
raw, _ = mne.preprocessing.regress_artifact(raw, picks="meg", picks_artifact=picks_ref)

# Apply TSPCA
# noisy_data = raw.get_data(picks="meg").T
# noisy_ref = raw.get_data(picks=picks_ref).T
# data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
# raw._data[0:160] = data_after_tspca.T
# raw.plot()

raw.filter(0.5, 40)
events, event_id = mne.events_from_annotations(raw)
assert event_id == {"ba": 1, "da": 2}
# downsample to 200 Hz
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-0.1, tmax=0.41, preload=True, decim=5, baseline=None)
epochs.equalize_event_counts(event_id)
fig = epochs.average().plot(spatial_colors=True, gfp=True)
