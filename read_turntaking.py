import mne
import os.path as op
import pandas as pd
import numpy as np


def get_participant_turns(file):
    min_event_duration = 1 # seconds
    # note: segments in repetition block are shorter (need to use 1s to catch most of them)
    df = pd.read_csv(file)
    participant_turns = []
    for _, row in df.iterrows():
        if row['person'] == 'participant' \
            and (row['end']-row['start']) > min_event_duration \
            and bool(row['is_full_turn']):
                participant_turns.append([row['start'],row['end']])
    return participant_turns


path = '/mnt/d/Work/analysis_ME206/processing/'
path_turntaking = op.join(path, 'audios_metadata_labelled')

subject_MEG = 'G02'
subject_bids = 'sub-' + subject_MEG[1:3]
path_preprocessed = op.join(path, 'bids', subject_bids, 'meeg')


# initialise the epochs
#epochs = {'conversation': [], 'repetition': []} #TODO: does this structure work?? Nope...

for block in ['B1', 'B2', 'B3', 'B4', 'B5']:
    participant_turns = get_participant_turns(op.join(path_turntaking, subject_MEG + '_' + block + '.csv'))

    print(subject_MEG + " " + block + ": Participant had " + str(len(participant_turns)) + " full turns.")
    print(participant_turns)

    # get the onset & duration of each participant turn
    participant_onsets = []
    participant_durations = []
    for turn in participant_turns:
        participant_onsets.append(turn[0])
        participant_durations.append(turn[1] - turn[0])
    #print(participant_onsets)
    #print(participant_durations)

    
    # create an "_events.tsv" file with these columns: onset, duration, trial_type
    if block == 'B1' or block == 'B3' or block == 'B5':
        ttype = 'conversation'
    elif block == 'B2' or block == 'B4':
        ttype = 'repetition'
    #duration = 1 # for 0~1s epochs
    
    output_file = op.join(path_preprocessed, subject_bids + '_task-conversation_run-0' + block[1] + '_events.tsv')
    df = pd.DataFrame({'onset': participant_onsets, 'duration': participant_durations, 'trial_type': ttype})
    df.to_csv(output_file, sep='\t', index=False, header=True)


    '''
    # read in preprocessed data for this block
    fname = op.join(path_preprocessed, subject_bids + '_task-conversation_run-0' + block[1] + '_proc-clean_raw.fif')
    raw = mne.io.read_raw_fif(fname)

    # create the events array (need integers)
    participant_onsets = participant_onsets * 1000 # convert to ms
    participant_onsets = [round(x) for x in participant_onsets] # convert to integer
    ntrials = len(participant_onsets)
    events = np.column_stack((participant_onsets, np.repeat(0, ntrials), np.repeat(1, ntrials))) # should change the event IDs to indicate conversation vs repetition?
    
    # TODO: epoching (-1~0, 0~1, -1~1 seconds)
    new_epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=1, preload=True)

    # add epochs to the correct condition
    if block == 'B1' or block == 'B3' or block == 'B5':
        epochs['conversation'].append(new_epochs)
    elif block == 'B2' or block == 'B4':
        epochs['repetition'].append(new_epochs)
    '''
