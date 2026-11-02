import mne

def downsample_emg(raw, target=12500, above=15000):
    """
    downsamples emg files to make them need less storage
    Input:
    raw: raw emg data from to_mne
    target: the downsampled fs of interest with a Nyquist (target/2) above the signal of interest. Default is 12500
    above: limit that fs must meet to require downsampling
    """
    raw = raw.copy()
    if raw.info["sfreq"] > above:
        raw.resample(target, method="polyphase")
    return raw
