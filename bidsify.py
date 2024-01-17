"""BIDSify the Natural Conversations dataset.

https://mqoutlook.sharepoint.com/sites/Natural_Conversations/

Done
----
- File sanity/existence check
- dev_head_t using _ini.mrk (no checking of others!)
- Checked for acceptable drift rate
- Checked event numbers in localiser
- Add EEG sensor positions
- Align MEG and EEG and concatenate channels
- Check jitter of triggers and auditory events on MEG for localiser
- Incorporate fixed/improved auditory localiser trigger timing

Todo
----
- Mark channels as bad somehow (automatic? PTP/flat? Could inspect...)
- Add TSPCA or reference regression to MNE-BIDS-Pipeline
- Add annotations for speaking and listening
  (https://github.com/Macquarie-MEG-Research/natural_conversations/blob/main/align_and_segment_audios.py)
  using 1-sec segments and run CSP
- Run STRF-type analysis on M/EEG using auditory
- Source localization using fsaverage using MBP
- Anonymize for eventual sharing
- Improve BIDS descriptions and authors
- Set tasks properly in BIDS formatting and update MNE-BIDS-Pipeline to handle it

Notes
-----
- For localizer data, triggers on MEG channel 181/ba and 182/da. Auditory on ch166.
- For resting state, triggers on 181 at the start of the block.
- Conversation (B1, B2, B3)
  - Trigger at beginning of each block (ch181/B1, ch182/B2, ch183/B3, etc.)
  - The interviewer speech recorded or ch166, participant speech on ch167
  - Trigger at the beginning of each block can map the timing between EEG and MEG
- Repetition (B2, B4) are 2 repetition blocks of 3 mins each, same as conversation
"""  # noqa: E501

import glob
from pathlib import Path

import mne
import mne_bids
import numpy as np
from scipy.spatial.distance import cdist

import utils  # local module

n_bas_meg = dict(G03=99, G18=99, G19=98)
n_das_meg = dict(G15=99, G18=98, G19=99)
n_bas_eeg = dict(G03=99, G18=99, G19=98)
n_das_eeg = dict(G15=99, G18=98, G19=99)
drift_tols = dict(G04=0.08, G09=0.026, G15=0.024)
bad_envelope_subjects = ("G04", "G08", "G11", "G13", "G22")  # naive envelope fails
empty_map = dict(G02="G01", G05="G06", G13="G01", G18="G17", G20="G19")

eeg_map = """
Fp1 AF3 AF7 Fz F1 F3 F5 F7 FC1 FC3
FC5 FT7 Cz C1 C3 C5 T7 CP1 CP3 CP5
TP7 TP9 Pz P1 P3 P5 P7 PO3 PO7 Oz
O1 ECG Fpz Fp2 AF4 AF8 F2 F4 F6 F8
FC2 FC4 FC6 FT8 C2 C4 C6 T8 CPz CP2
CP4 CP6 TP8 TP10 P2 P4 P6 P8 POz PO4
PO8 O2 EOG
""".strip().split()
assert len(eeg_map) == 63  # omit ref, which is at FCz
eeg_renames = {str(ci): ch for ci, ch in enumerate(eeg_map, 1)}
ch_types_map = dict(ECG="ecg", EOG="eog")
min_dur = 0.002  # for events

subjects = tuple(f"G{s:02d}" for s in range(1, 28))
blocks = dict(  # to BIDS task and run
    B1=("conversation", "01"),
    B2=("conversation", "02"),
    B3=("conversation", "03"),
    B4=("conversation", "04"),
    B5=("conversation", "05"),
    localiser=("conversation", "06"),  # ("localizer", "01"),  # TODO: Revert
    resting=("rest", None),
)
event_id = dict(ba=1, da=2, dummy=99)

# BIDS stuff
name = "natural-conversations"
datatype = suffix = "meg"
this_path = Path(__file__).parent
data_root = this_path / ".." / "Natural_Conversations_study"
bids_root = data_root / f"{name}-bids"
bids_root.mkdir(exist_ok=True)
mne_bids.make_dataset_description(
    path=bids_root,
    name=name,
    authors=["Paul Sowman", "Judy Zhu", "Eric Larson"],  # noqa: E501
    how_to_acknowledge='If you use this data, please cite the references provided in this dataset description.',  # noqa: E501
    data_license='CC-BY-SA',
    ethics_approvals=['Human Research Ethics at the Macquarie University'],
    references_and_links=[],
    overwrite=True,
)
(bids_root / 'README').write_text("""
Listening and speaking M/EEG dataset
====================================

This dataset contains M/EEG data.

The data were converted with MNE-BIDS:

- Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896). https://doi.org/10.21105/joss.01896
- Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J., Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data structure extended to magnetoencephalography. Scientific Data, 5, 180110. https://doi.org/10.1038/sdata.2018.110
""".strip())

for subject in subjects:
    print(f"Subject {subject}...")
    for block in blocks:
        if block == "empty" and subject in no_empty_subjects:
            continue  # missing for this one subject
        print(f"  Block {block}...")
        # MEG
        subject_root = data_root / f"{subject}"
        fnames = dict(
            input_fname=subject_root / "meg" / f"*_{block}.con",
            mrk=subject_root / "meg" / "*_ini.mrk",
            elp=subject_root / "meg" / "*.elp",
            hsp=subject_root / "meg" / "*.hsp",
        )
        for key, pattern in fnames.items():
            fname = glob.glob(str(pattern))
            assert len(fname) == 1, (key, pattern, fname)
            fnames[key] = fname[0]
        raw_meg = mne.io.read_raw_kit(
            **fnames,
            stim=[166, *range(176, 190)],
            slope="+",
            stim_code="channel",
            stimthresh=2,
        ).load_data()
        assert not np.allclose(raw_meg.info["dev_head_t"]["trans"], np.eye(4))
        assert raw_meg.first_time == 0
        # Empty room data (if present)
        empty_room = empty_map.get(subject, subject)
        if empty_room is not None:
            empty_room = glob.glob(str(data_root / empty_room / "meg" / "*_empty.con"))
            assert len(empty_room) == 1, empty_room
            empty_room = mne.io.read_raw_kit(empty_room[0]).load_data()
            empty_room.info["dev_head_t"] = None
        # EEG
        if block != "empty":
            eeg_fname = glob.glob(str(subject_root / "eeg" / f"*{block}_new.vhdr"))
            if len(eeg_fname) == 0:
                eeg_fname = glob.glob(str(subject_root / "eeg" / f"*{block}.vhdr"))
            assert len(eeg_fname) == 1
            eeg_fname = eeg_fname[0]
            raw_eeg = mne.io.read_raw_brainvision(eeg_fname).load_data()
            raw_eeg.rename_channels(eeg_renames)
            raw_eeg.set_channel_types(ch_types_map)
            assert raw_eeg.first_time == 0
            assert raw_eeg.info["sfreq"] == raw_meg.info["sfreq"]
            mne.add_reference_channels(raw_eeg, "FCz", copy=False)
            raw_eeg.set_eeg_reference(["FCz"])
            assert raw_eeg.ch_names[-1] == "FCz"
            raw_eeg.set_montage("standard_1020")
            # fix some accounting that MNE-Python should probably take care of for us
            with raw_eeg.info._unlock():
                for ch in raw_eeg.info["chs"]:
                    ch["loc"][3:6] = raw_eeg.info["chs"][-1]["loc"][3:6]
                raw_eeg.info["custom_ref_applied"] = 0
            raw_eeg.set_eeg_reference(projection=True)
            if block.startswith("B"):
                trig_offset = int(block[1]) - 1
            else:
                trig_offset = 0
            meg_event = mne.find_events(
                raw_meg,
                stim_channel=raw_meg.ch_names[181 + trig_offset],
                min_duration=min_dur,
            )[:1]
            eeg_event = mne.events_from_annotations(
                raw_eeg,
                event_id={f"Stimulus/S {53 + trig_offset}": 1},
            )[0][:1]
            meg_samp = meg_event[0, 0]
            eeg_samp = eeg_event[0, 0]
            # Instead of cropping MEG, let's just zero-order hold the first or last EEG
            # sample. This will make timing of events align with the original MEG
            # data.
            if eeg_samp < meg_samp:
                n_pad = meg_samp - eeg_samp
                raw_eeg_pad = raw_eeg.copy().crop(0, (n_pad - 1) / raw_eeg.info["sfreq"])
                assert len(raw_eeg_pad.times) == n_pad
                raw_eeg_pad._data[:] = raw_eeg[:, 0][0]
                raw_eeg_pad.set_annotations(None)
                raw_eeg = mne.concatenate_raws([raw_eeg_pad, raw_eeg])
                del raw_eeg_pad
            elif eeg_samp > meg_samp:
                raw_eeg.crop((eeg_samp - meg_samp) / raw_eeg.info["sfreq"], None)
            if len(raw_eeg.times) < len(raw_meg.times):
                n_pad = len(raw_meg.times) - len(raw_eeg.times)
                raw_eeg_pad = raw_eeg.copy().crop(0, (n_pad - 1) / raw_eeg.info["sfreq"])
                assert len(raw_eeg_pad.times) == n_pad
                raw_eeg_pad._data[:] = raw_eeg[:, -1][0]
                raw_eeg_pad.set_annotations(None)
                raw_eeg = mne.concatenate_raws([raw_eeg, raw_eeg_pad])
                del raw_eeg_pad
            elif len(raw_eeg.times) > len(raw_meg.times):
                raw_eeg.crop(0, (len(raw_meg.times) - 1) / raw_eeg.info["sfreq"])
            extra_idx = np.where([d["kind"] == 4 for d in raw_meg.info["dig"]])[0][0]
            for di, d in enumerate(raw_eeg.info["dig"][3:]):  # omit fiducials
                raw_meg.info["dig"].insert(extra_idx + di, d)
            raw_eeg.info["dig"][:] = raw_meg.info["dig"]
            for key in ("dev_head_t", "description"):
                raw_eeg.info[key] = raw_meg.info[key]
            with raw_eeg.info._unlock():
                for key in ("highpass", "lowpass"):
                    raw_eeg.info[key] = raw_meg.info[key]
            raw_meg.add_channels([raw_eeg])
        if block == "localiser":
            # Figure out our first-order coefficient
            mba = mne.find_events(raw_meg, stim_channel=raw_meg.ch_names[181], min_duration=min_dur)
            mba[:, 2] = 1
            mda = mne.find_events(raw_meg, stim_channel=raw_meg.ch_names[182], min_duration=min_dur)
            mda[:, 2] = 2
            m_ev = np.concatenate([mba, mda])
            m_ev = m_ev[np.argsort(m_ev[:, 0])]
            eba = mne.events_from_annotations(raw_eeg, event_id={"Stimulus/S 53": 1})[0]
            eda = mne.events_from_annotations(raw_eeg, event_id={"Stimulus/S 54": 2})[0]
            e_ev = np.concatenate([eba, eda])
            e_ev = e_ev[np.argsort(e_ev[:, 0])]
            # if subject == "G18":
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots(figsize=(6, 1.5), layout="constrained")
            #     ax.plot(raw_meg.times, raw_meg[182][0][0], zorder=4)
            #     ax.set(xlabel="Time (s)", ylabel="ch182")
            #     pe = eda[:, 0] / raw_eeg.info["sfreq"] + (mda[0, 0] / raw_meg.info["sfreq"] - eda[0, 0] / raw_eeg.info["sfreq"])
            #     for val in pe:
            #         ax.axvline(val, color='r', lw=0.5, zorder=3)
            assert len(mba) == n_bas_meg.get(subject, 100), len(mba)
            assert len(mda) == n_das_meg.get(subject, 100), len(mda)
            assert len(eba) == n_bas_eeg.get(subject, 100), len(eba)
            assert len(eda) == n_das_eeg.get(subject, 100), len(eda)
            e_ev = e_ev[:len(m_ev)]
            np.testing.assert_array_equal(m_ev[:, 2], e_ev[:, 2])
            m = m_ev[:, 0] / raw_meg.info["sfreq"] - raw_meg.first_time
            e = e_ev[:, 0] / raw_eeg.info["sfreq"] - raw_eeg.first_time
            p = np.polyfit(m, e, 1)
            first_ord = p[0]
            off = e - first_ord * m - p[1]
            np.testing.assert_allclose(off, 0., atol=2e-3)
            print(f"  Drift rate: {(1 - first_ord) * 1e6:+0.1f} PPM")
            # Correct jitter
            events, _ = utils.triggerCorrection(raw_meg, subject, plot=False)
            events[events[:, 2] == 181, 2] = 1
            events[events[:, 2] == 182, 2] = 2
            assert np.in1d(events[:, 2], (1, 2)).all()
            max_off = cdist(m_ev[:, :1], events[:, :1]).min(axis=0).max()
            assert max_off < 50, max_off
        else:
            # TODO: Need at least one dummy event for MNE-BIDS-Pipeline to work
            # but put it far enough out that it won't lead to empty epochs
            events = np.array([[raw_meg.first_samp + int(round(raw_meg.info["sfreq"])), 0, event_id["dummy"]]], int)
        task, run = blocks[block]
        int(subject[1:])  # make sure it's an int
        bids_subject = subject[1:]  # remove "G"
        bids_path = mne_bids.BIDSPath(
            subject=bids_subject,
            task=task,
            run=run,
            suffix=suffix,
            datatype=datatype,
            root=bids_root,
        )
        mne_bids.write_raw_bids(
            raw_meg,
            bids_path,
            events=events,
            event_id=event_id,
            empty_room=empty_room,
            overwrite=True,
            allow_preload=True,
            format="FIF",
            verbose="error",  # ignore stuff about events being empty for now
        )
