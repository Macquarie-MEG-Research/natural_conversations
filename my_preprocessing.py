# MEG preprocessing: filtering & ICA
#
# (autoreject & Ransac are not finding much?? 
# maybe they only target super large artefact - that makes sense; 
# in most cases, ICA alone seems to suffice - it can fix bad sensors &
# some LFN as well)
#
# Authors: Paul Sowman, Judy Zhu

def reject_artefact(raw, l_freq, h_freq, do_ICA, ica_fname):
    import mne
    import copy
    import os.path as op
        
    #from mne.preprocessing import find_bad_channels_maxwell
    from autoreject import get_rejection_threshold  # noqa
    from autoreject import Ransac  # noqa
    from autoreject.utils import interpolate_bads  # noqa


    # do filtering first - so when we display waveforms for ICA comps, it will show filtered data
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    
    if do_ICA:

        # ICA takes a long time - if we've run it before, just load the components
        if op.exists(ica_fname):
            ica = mne.preprocessing.read_ica(ica_fname)
        else:
            # filter again (1Hz high-pass) before ICA
            raw_for_ICA = raw.copy()
            raw_for_ICA.filter(l_freq=1, h_freq=None)
            
            # 'autoreject' requires epoched data
            # here we create arbitrary epochs (making use of all data - useful for short recordings)
            tstep = 1.0 # make 1 second epochs
            events_ICA = mne.make_fixed_length_events(raw_for_ICA, duration=tstep)
            epochs_ICA = mne.Epochs(
                raw_for_ICA, events_ICA, tmin=0.0, tmax=tstep, baseline=None, preload=True
            )
            
            # use 'autoreject' to compute a threshold for removing large noise
            reject = get_rejection_threshold(epochs_ICA)
            reject # print the result
            # remove large noise before running ICA
            #epochs_ICA.load_data() # to avoid reading epochs from disk multiple times
            epochs_ICA.drop_bad(reject=reject)

            # could also try Ransac
            #ransac = Ransac(verbose=True, n_jobs=1)
            #epochs_ICA = ransac.fit_transform(epochs_ICA)
            
            # run ICA
            ica = mne.preprocessing.ICA(n_components=60, max_iter="auto", random_state=97)
            ica.fit(epochs_ICA, tstep=tstep) # 'reject' param here only works for continuous data, so we use drop_bad() above instead
            
            ica.save(ica_fname)
        
        # plot ICA results
        ica.plot_sources(raw) # plot IC time series
        ica.plot_components() # plot IC topography

        # manually select which components to reject
        select_comps = input("Enter the IC components to remove (e.g. 0 3 5): ")
        ica.exclude = list(map(int, select_comps.split())) # convert to array of ints
        #ica.exclude = [8, 17, 23, 58]
        
        # can also use automatic methods to select comps for rejection:
        # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#using-a-simulated-channel-to-select-ica-components
        # https://github.com/LanceAbel/MQ_MEG_Analysis (selecting channels to simulate EOG)


        # Compare raw data before & after IC rejection
        raw_orig = copy.deepcopy(raw) # need to make a copy, otherwise the 'before'
            # and 'after' plots become the same (even if you do the 'before' plot
            # first, then apply ICA, it still gets updated to look the same as the 'after' plot)
        raw_orig.plot(title='before ICA')
        ica.apply(raw) # apply component rejection onto raw (continuous) data
                       # Note: data will be modified in-place
        raw.plot(title='after ICA')

    return raw
