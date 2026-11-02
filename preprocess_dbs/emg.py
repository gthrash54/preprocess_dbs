import mne

#target=12500 and above=15000 are default arguments, used when the caller doesn't give them (the lab's values,
#preprocess.m:786-823: EMG recorded above 15 kHz goes to 12.5 kHz; DBS and ECoG are never resampled)
def downsample_emg(raw, target=12500, above=15000):
    """
    downsamples emg files to make them need less storage
    Input:
    raw: raw emg data from to_mne
    target: the downsampled fs of interest with a Nyquist (target/2) above the signal of interest. Default is 12500
    above: only resample when the current rate (Hz) is above this
    Returns: a copy of raw at target Hz, or an unchanged copy if it was already at or below above. Refuses non-EMG channels
    """
    #get_channel_types() = one type per channel ("emg", "dbs", ...); set(...) keeps each distinct value once, so a pure EMG
    #recording gives {"emg"}
    types = set(raw.get_channel_types())
    if types != {"emg"}: #e.g. an ecos recording passed by mistake: resampling would wreck the 48.8 kHz DBS/ECoG for ERNA
        #sorted(...) turns the set into an alphabetical list for a tidy message
        raise ValueError(f"downsample_emg only takes EMG channels, got {sorted(types)}; load the EMG stream on its own")
    raw = raw.copy() # resample below works in place, so work on a copy
    #raw.info["sfreq"] = the sample rate in Hz
    if raw.info["sfreq"] > above:
        #raw.resample changes the sample rate. method="polyphase" (scipy's resample_poly, like MATLAB's resample, which the
        #lab used) low-pass filters before keeping fewer samples: anything above the new Nyquist (6.25 kHz) would otherwise
        #fold back ("alias") into lower frequencies and look like real signal. Stim annotations keep their times in s
        raw.resample(target, method="polyphase")
    return raw
