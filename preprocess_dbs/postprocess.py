"""Sarah's postprocess.m: the cleanup after de-artifacting (trim to the artifact peak, crop and demean, detrend choice,
adaptive Savitzky-Golay smoothing, baseline)."""
import numpy as np


def trim_to_artifact(data,peaks,fs):
    """
    Cuts every channel so it starts 2 ms before its artifact peak, so all channels and conditions line up on the
    artifact. Sarah's postprocess.m:399-470.
    Each channel's peak is given as a position inside Sarah's alignment window, which starts at epoch sample
    round(0.002*fs) - 1 = 97; so the peak sits at epoch sample 97 + peak. The new start is round(that) - round(2 ms)
    (98 samples at 48.8 kHz, 2.007 ms), so the ROUNDED peak lands at sample 98 of the trimmed signal, which is time 0 on
    Sarah's time axis (it starts at -98 samples, postprocess.m:552). A fractional peak sits within half a sample
    (0.01 ms) of time 0. Channels can start at different samples, so all are cut to the shortest.

    Two deliberate differences from Sarah (see notes_for_walker.md):
    - Callers pass the SINGLE pulse's peaks (deartifact_segment's peaks["single"]) for every condition: one reference
      per channel. The template was aligned onto the single pulse and every condition onto the template, so this is
      the point they all share (the single itself barely moves, by its tiny shift to the aligned template). Sarah
      trims each condition on its own peak measured before it was moved, so each is off by its own alignment shift.
      This function doesn't enforce that; the caller passes the right peaks.
    - The rounded peak sits at time 0. Sarah's start, round(peak) counted from 1, is one sample later, so her peak sits
      about 0.02 ms before time 0.
    Python's round sends exact halves to the even number (MATLAB rounds them away from zero); a refined peak landing
    exactly on .5 is practically impossible, so this is noted, not handled.

    Input:
    data: one condition's cleaned average, channels x time (the stim at 2 ms), e.g. deartifact_segment's
        cleaned["single"].data
    peaks: one per channel, the artifact peak's position in Sarah's alignment window (samples from the window start,
        counted from 0; can be a fraction), e.g. deartifact_segment's peaks["single"]
    fs: sampling freq, usually 48.8 kHz
    Returns: a new array, channels x (the shortest channel's length), each channel starting 98 samples before its
        (rounded) artifact peak. The input isn't changed
    """
    window_start = round(0.002*fs)-1 #Sarah's alignment window starts here (97), so the peaks count from here
    two_ms_in_samples = 0.002*fs #97.66, rounded to 98 below
    trimmed_channels = []
    for ch in range(data.shape[0]):
        peak_position = window_start + peaks[ch] #the peak's position in the whole epoch
        new_start = round(peak_position)-round(two_ms_in_samples) #98 samples before the rounded peak
        #fail loudly: a negative start would silently count from the END (data[ch, -1:] keeps only the last sample),
        #and a start past the end would leave nothing, cutting every channel to length 0
        if new_start < 0 or new_start >= data.shape[1]:
            raise ValueError(f"channel {ch}: artifact peak {peaks[ch]} gives a trim start of {new_start}, outside the "
                             f"{data.shape[1]}-sample epoch; can't keep 2 ms before it")
        trimmed_channels.append(data[ch,new_start:])
    #channels can start at different samples, so their pieces differ in length: cut all to the shortest
    shortest = min(len(piece) for piece in trimmed_channels)
    output = np.zeros((data.shape[0],shortest))
    for ch in range(data.shape[0]):
        output[ch] = trimmed_channels[ch][:shortest]
    return output
