import mne

def downsample_emg(raw, target=12500, above=15000):
    """
    downsamples emg files to make them need less storage
    Input:
    raw: raw emg data from to_mne
    target: the downsampled fs of interest with a Nyquist (target/2) above the signal of interest. Default is 12500
    above: only resample when the current rate (Hz) is above this
    Returns: a copy of raw at target Hz, or an unchanged copy if it was already at or below above. Refuses non-EMG channels
    """
    types = set(raw.get_channel_types())
    if types != {"emg"}: #e.g. an ecos recording passed by mistake: resampling would wreck the 48.8 kHz DBS/ECoG for ERNA
        raise ValueError(f"downsample_emg only takes EMG channels, got {sorted(types)}; load the EMG stream on its own")
    raw = raw.copy()
    if raw.info["sfreq"] > above:
        raw.resample(target, method="polyphase")
    return raw
