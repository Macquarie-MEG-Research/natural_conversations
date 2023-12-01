import mne
import matplotlib.pyplot as plt
import numpy as np
import copy


def triggerCorrection(raw, subject_MEG):
    """Adjust trigger timing based on MEG audio channel.

    Parameters
    ----------
    raw : instance of Raw
    subject_MEG : string
        Used for specifying special thresholds for certain subjects
        
    Returns
    -------
    events_corrected : array, shape (m, 3)
        The events with corrected timing.
    AD_delta : list of integers
        The audio delay values (i.e. offset between normal triggers and 
        detected sound onset in audio channel signal), in ms.
    """

   # Find events from normal trigger channels
    events = mne.find_events(
        raw,
        output="onset",
        consecutive=False,
        min_duration=0,
        shortest_event=1,  # 5 for adult
        mask=None,
        uint_cast=False,
        mask_type="and",
        initial_event=False,
        verbose=None,
    )

    # get rid of audio triggers for now
    events = np.delete(events, np.where(events[:, 2] == 166), 0)

    # get raw audio signal from ch166
    aud_ch_data_raw = raw.get_data(picks="MISC 007")


    #raw.load_data().apply_function(getEnvelope, picks="MISC 006")
    if subject_MEG == 'G22':
        envelope = getEnvelope(aud_ch_data_raw, 3.5)
    else:
        envelope = getEnvelope(aud_ch_data_raw)
    envelope = envelope.tolist() # convert ndarray to list
    # detect the beginning of each envelope (set the rest of the envelope to 0)
    new_stim_ch = np.clip(np.diff(envelope),0,1)
    # find all the 1s (i.e. audio triggers)
    stim_tps = np.where(new_stim_ch==1)[0]

    # compare number of events from trigger channels & from AD
    print("Number of events from trigger channels:", events.shape[0])
    print("Number of events from audio channel (166) signal:", stim_tps.shape[0])


    # apply timing correction onto the events array
    events_corrected = copy.copy(events) # work on a copy so we don't affect the original

    # Missing AD triggers can be handled:
    # if there's an AD trigger within 50ms following the normal trigger
    # (this ensures we've got the correct trial), update to AD timing;
    # if there's no AD trigger in this time range, discard the trial
    AD_delta = []
    missing = [] # keep track of the trials to discard (due to missing AD trigger)
    for i in range(events.shape[0]):
        idx = np.where((stim_tps >= events[i,0]-30) & (stim_tps < events[i,0]+50))
        if len(idx[0]): # if an AD trigger exists within the specified window
            idx = idx[0][0] # use the first AD trigger (if there are multiple)
            AD_delta.append(stim_tps[idx] - events[i,0]) # keep track of audio delay values
            events_corrected[i,0] = stim_tps[idx] # update event timing
        else:
            missing.append(i)
    # discard events which could not be corrected
    events_corrected = np.delete(events_corrected, missing, 0)
    print("Could not correct", len(missing), "events - these were discarded!")

    # histogram showing the distribution of audio delays
    n, bins, patches = plt.hist(
        x=AD_delta, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Delay (ms)")
    plt.ylabel("Frequency")
    plt.title("Audio Detector Delays")
    plt.text(
        70,
        50,
        r"$mean="
        + str(round(np.mean(AD_delta)))
        + ", std="
        + str(round(np.std(AD_delta)))
        + "$",
    )
    maxfreq = n.max()
    # set a clean upper y-axis limit
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    return events_corrected, AD_delta


def getEnvelope(inputSignal, thresh=0.2):
    """Get the envelope of an audio signal, and then binarise it at the given threshold.
    """

    # Taking the absolute value
    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append(abs(sample))
    absoluteSignal = absoluteSignal[0]

    # Peak detection
    intervalLength = 15  # Experiment with this number!
    outputSignal = []

    # Like a sample and hold filter
    for baseIndex in range(intervalLength, len(absoluteSignal)):
        maximum = 0
        for lookbackIndex in range(intervalLength):
            maximum = max(absoluteSignal[baseIndex - lookbackIndex], maximum)
        outputSignal.append(maximum)

    outputSignal = np.concatenate(
        (
            np.zeros(intervalLength),
            np.array(outputSignal)[:-intervalLength],
            np.zeros(intervalLength),
        )
    )
    # finally binarise the output at the given threshold
    return np.array([1 if np.abs(x) > thresh else 0 for x in outputSignal])
