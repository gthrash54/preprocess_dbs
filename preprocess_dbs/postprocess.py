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

def crop_and_demean(data,fs):
    """
    Cuts the channels from -2 to 45 ms around the artifact peak, then subtracts each channel's level, the average of a
    pre-artifact mean (m1, -2 to -1 ms) and a post-artifact mean (m2, about +1 ms to the end; with her indexing it
    starts at +0.96 ms). Sarah's postprocess.m:570-579.
    Both windows skip the artifact itself but the artifact stays in the signal.
    The post-artifact window has ERNA in it because ERNA swings up and down, it mostly averages out to around zero.
    But its first, biggest swings don't fully cancel, so a big response nudges the level. Using only a late window
    (e.g. 25-45 ms, after the ERNA has died out) could increase rigor (open question Q6, in notes_for_walker.md).
    Sarah's comment calls this a "general/dirty demean" on purpose; the final median baseline (postprocess's last
    step) sets the level again.
    Input:
    data: the output of trim_to_artifact, channels x time, the artifact peak at sample 98 (time 0)
    fs: sampling freq in hz
    Return:
    demeaned: a new array, channels x 2295 samples (-2 to +45 ms at 48.8 kHz; all of it if the signal is shorter), each
        channel centered on its level. The input isn't changed
    Raises ValueError if the signal is too short for m2 (146 samples or fewer), which would otherwise turn every value
    into NaN.
    """
    #45ms after the artifact plus 2ms before (2295 samples)
    after = round(0.045*fs)
    before = round(0.002*fs)
    crop = data[:,:after+before]
    #fail loudly: with 145 samples or fewer, m2's window (from sample 145) is empty, its mean is NaN, and so is every
    #output value
    if crop.shape[1] <= round(0.003*fs)-1:
        raise ValueError(f"only {crop.shape[1]} samples: too short for the post-artifact window (sample "
                         f"{round(0.003*fs)-1} on)")
    m1 = np.mean(crop[:,:round(0.001*fs)], axis=1, keepdims=True) # the first 1ms slice(prestim baseline, -2->-1ms) because it is relatively quiet to demean
    m2 = np.mean(crop[:,round(0.003*fs)-1:],axis=1, keepdims=True) # 1ms after artifact peak through +45ms
    demeaned = crop - ((m1+m2)/2)
    return demeaned


def fit_drifts(signal,fs):
    """
    Fits both a line and a parabola to the slow drift that runs through each recording, mostly from the amplifier and
    electrode recovering after the pulse. The drift is slow compared to ERNA, but ERNA inside the window can still pull
    on the fit a little. Part 1 of Sarah's detrend choice (postprocess.m:583-600).
    The window is every sample after 3 ms (about 1 ms after the artifact peak) and before the last sample's time minus
    0.5 ms. That keeps the artifact and the edge out of the fit. We fit on that window (a mask and np.polyfit), then
    evaluate the curves over the entire signal, since they get subtracted from all of it.
    Input:
    signal: one channel (1-D) from crop_and_demean, the artifact peak at about 2 ms
    fs: sampling freq in Hz
    Returns:
    linear_drift: the fitted line, evaluated at every sample
    quadratic_drift: the fitted parabola, evaluated at every sample. Outside the window it's extrapolated and can bend
        away sharply. The longest stretch it's extrapolated over is the first 3 ms, where the artifact is anyway
        (Sarah's method; see Q2)
    mask: True for the samples the fits used
    Raises ValueError if fewer than 3 samples are in the window, since a parabola needs at least 3 points.
    """
    time_axis = ((np.arange(len(signal)))/fs*1000) #array the length of the signal in ms
    mask = (time_axis > 3) & (time_axis < (time_axis[-1]-0.5)) #window to fit the drift on: after 3 ms to avoid the artifact, until 0.5 ms before the end to avoid edge effects
    #fail loudly: a parabola needs at least 3 points (a short signal from crop_and_demean can leave fewer)
    if mask.sum() < 3:
        raise ValueError(f"only {mask.sum()} samples in the drift window: too few to fit a parabola")
    line_fit = np.polyfit(time_axis[mask], signal[mask], deg=1)
    parabola_fit = np.polyfit(time_axis[mask],signal[mask],deg=2) #deg 2 is quadratic fit
    linear_drift = np.polyval(line_fit,time_axis)
    quadratic_drift = np.polyval(parabola_fit,time_axis) #when extrapolating a parabola, it can bend away sharply from the data but this is most prominent where artifact is anyway.
    return linear_drift, quadratic_drift, mask


def score_drifts(signal, linear_drift, quadratic_drift, mask):
    """
    Scores the drift window for three versions of one channel: as it is, minus the linear drift, and minus the
    quadratic drift. A smaller score means the window is smaller compared to that version's own largest value, which
    usually means more drift was removed (it can't tell drift from response, though). Part 2 of Sarah's detrend choice
    (postprocess.m:601-610).
    Each version is scaled by its own largest absolute value, then scored as the mean absolute value inside the window.
    In real data the largest value is usually the artifact, so the three end up on about the same scale, but that's an
    assumption, not a guarantee. Sarah calls these scores "rms", but they're mean absolute values (open question Q2 in
    notes_for_walker.md). A version that's all zeros gives 0/0, so its score is NaN.
    Input:
    signal: one channel (1-D) from crop_and_demean
    linear_drift, quadratic_drift, mask: from fit_drifts on that same signal
    Returns: a list of three scores, in the order none, linear, quadratic
    """
    signals = [signal,(signal-linear_drift),(signal-quadratic_drift)] #the three versions to compare
    scores = []
    for x in signals:
        scaled = x/(np.max(np.abs(x))) #its biggest point becomes +-1
        score = np.mean(np.abs(scaled[mask])) #how far from flat it is, inside the drift window only
        scores.append(score)
    return scores
