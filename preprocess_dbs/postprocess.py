"""Sarah's postprocess.m: the cleanup after de-artifacting (trim to the artifact peak, crop and demean, detrend choice,
adaptive Savitzky-Golay smoothing, baseline)."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


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


def choose_detrend(scores,condition):
    """
    Picks which detrend to use for one DBS channel (none, linear or quadratic) from its three scores, using Sarah's
    rule. Part 3 of her detrend choice (postprocess.m:612-655).
    Branch A, bursts 1-9: their epochs are too short for a quadratic, so only none and linear are compared. The two
    scores are normalized by the larger one, and linear wins if it at least halves the score.
    Branch B, everything else (single, pairs, probes, burst_10): all three scores are normalized by the largest, then
    checked in order, and the first rule that matches wins:
    1. quadratic, if it cuts the score by at least 35% (q <= 0.65 n) and is at least 0.10 lower than linear (an
       absolute difference in normalized scores, not 10% of linear's)
    2. linear, if the quadratic cuts it by 35% but isn't 0.10 lower than linear (not worth the extra bend)
    3. linear, if the quadratic doesn't cut it by 35% but linear's normalized score is at most 0.85
    4. none otherwise
    Rule 3's comment in her code says "linear provides 25% reduction", but the code only requires linear's score to be
    at most 85% of the LARGEST score. That needn't be an improvement over no detrending at all: scores 0.2, 0.4, 1.0
    pick linear even though linear doubles the score (open question Q2 in notes_for_walker.md). Copied as her code runs.
    Input:
    scores: the three scores from score_drifts, in the order none, linear, quadratic
    condition: the condition's name, e.g. "single", "burst_3", "paired_0.96_1"
    Returns: "none", "linear" or "quadratic"
    """
    scores = scores.copy()
    #branch A: Sarah's nine short-burst names exactly (an allow-list), so burst_10, burstprobes_ and anything new go to B
    short_bursts = [f"burst_{x}" for x in range(1,10)]
    if condition in short_bursts:
        burst_scores = np.array(scores[0:2]) #none and linear only
        none_burst_normalized, linear_burst_normalized = burst_scores/np.max(burst_scores)
        if linear_burst_normalized <= none_burst_normalized/2: #linear at least halves the score
            return "linear"
        else:
            return "none"
    #branch B: all three, normalized by the largest, checked in order (the first match wins)
    else:
        n, l, q = np.array(scores)/np.max(scores)
        if q <= 0.65*n and q - l <= -0.10: #rule 1: quadratic cuts 35% and beats linear by 0.10
            return "quadratic"
        elif q <= 0.65*n and q-l > -0.10: #rule 2: quadratic helps, but not enough beyond linear
            return "linear"
        elif q > 0.65*n and l <= 0.85: #rule 3: quadratic doesn't help much, linear does a little
            return "linear"
        else: #rule 4: neither helps enough
            return "none"


def detrend_channel(signal,fs,condition,channel_type):
    """
    Detrends one channel the way Sarah does (postprocess.m:583-655): DBS channels get the drift picked by her rule taken
    off; ECoG channels are left as they are (her code skips them). Any other type (EMG, misc) raises an error: her code
    leaves EMG out of this step (postprocess.m:548), so getting one here means the caller made a mistake.
    For DBS this chains the three parts: fit_drifts (a line and a parabola), score_drifts (scores the drift window for
    each option), choose_detrend (her rule), then subtracts the chosen drift from the whole signal ("none" subtracts
    nothing).
    Input:
    signal: one channel (1-D) from crop_and_demean
    fs: sampling freq in Hz
    condition: the condition's name, e.g. "single", "burst_3" (bursts 1-9 use her simpler rule)
    channel_type: the channel's MNE type as to_mne sets it, "dbs" or "ecog" (her code checks whether the type contains
        "ecog"; to_mne's types are exact)
    Returns: the detrended signal (a new array, the input isn't changed) and the choice: "none", "linear" or "quadratic"
    Raises ValueError for any channel_type other than "dbs" or "ecog".
    """
    #fail loudly: only DBS and ECoG belong here (an allow-list, like the short bursts in choose_detrend)
    if channel_type not in ("dbs", "ecog"):
        raise ValueError(f"detrend_channel takes 'dbs' or 'ecog' channels, got {channel_type!r}")
    if channel_type == "ecog":
        return signal.copy(), "none" #ECoG skips detrending (a copy, so changing the result can't change the input)
    else:
        linear_drift, quadratic_drift, mask = fit_drifts(signal,fs)
        scores = score_drifts(signal,linear_drift,quadratic_drift,mask)
        choice = choose_detrend(scores,condition)
        #the choice word picks the drift to take off
        chosen_drift = {"none":np.zeros(len(signal)), "linear":linear_drift, "quadratic":quadratic_drift}[choice]
        detrended_signal = signal - chosen_drift
        return detrended_signal, choice



def moving_mean(x,window):
    """
    Averages each sample with its neighbors, the way MATLAB's movmean does (Sarah uses it for the smoothing features,
    postprocess.m:682-690). The window is centered on each sample, with (window-1)/2 samples on each side.
    Near the edges there aren't enough neighbors, so the window shrinks and averages only the samples that exist, like
    movmean. With a 5-sample window, the first sample is the mean of samples 0-2.
    Input:
    x: a 1-D signal with no NaN or Inf
    window: how many samples to average, a positive odd number
    Returns: the smoothed signal, the same length as x
    Raises ValueError for an even or non-positive window. An even window has no middle sample, and MATLAB's rule for
    placing it (one extra sample before) isn't copied here.
    Raises ValueError if x has NaN or Inf. pandas quietly leaves them out of the average, while movmean spreads a NaN to
    every window it touches, so the two would differ without any warning.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"Window must be a positive odd number, got {window}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x has NaN or Inf values")
    smoothed = pd.Series(x).rolling(window, center=True, min_periods=1).mean().to_numpy()
    return smoothed


def _odd_window(seconds,fs):
    """
    Turns a window length in seconds into a whole, odd number of samples, so the window has a middle sample. Same as
    Sarah's round(seconds*fs), plus 1 if even (postprocess.m:678-681, 688-689).
    Python rounds exact halves to the even number, and MATLAB rounds them away from zero. That never changes the result
    here: of the two candidates one is even, and adding 1 to it lands on the same odd number. (Neither of her windows
    lands on a half at 48828.125 Hz anyway.)
    """
    window = round(seconds*fs)
    if window % 2 == 0:
        window += 1
    return window


def smoothing_features(signal,fs):
    """
    Computes the two features Sarah's adaptive Savitzky-Golay uses to pick a smoothing state at each sample
    (postprocess.m:674-691). Both are built from moving RMS values (square, average over a window, square root).
    - activity: how fast the signal is changing. The moving RMS of the slope plus half the moving RMS of the bend
      (second derivative), over a 0.1 ms window (5 samples at 48.8 kHz). It spikes at the artifact. Sarah's
      activity_fast.
    - envelope: how big the signal is. The moving RMS of the signal itself over a 1.5 ms window (73 samples at
      48.8 kHz), so it changes slowly. It stays high through the response and falls as the response fades. Sarah's
      energy_slow.
    Each is divided by its own max, so both stay between 0 and 1 with a max of exactly 1, and her state thresholds
    (0.4, 0.2, 0.25) apply.
    The slope is per second and the bend per second squared, so they aren't on the same scale. For a wave at frequency
    f, the bend is about 2*pi*f times the slope (about 1,900 times for a 300 Hz ERNA, far more at the artifact), so
    activity is really just the scaled moving RMS of the bend; the slope only matters for content slower than about
    0.3 Hz. Copied as her code runs; see notes_for_walker.md.
    Input:
    signal: one channel (1-D) from detrend_channel
    fs: sampling freq in Hz
    Returns: activity, envelope (each the same length as signal, between 0 and 1)
    Raises ValueError for a flat signal. Its max is 0, so scaling would divide by 0 and quietly give NaN (Sarah's code
    doesn't check).
    """
    #slope and bend, per second like Sarah's (diff is one sample shorter, so a 0 in front keeps the length)
    dx = np.concat(([0.0],np.diff(signal)))*fs
    ddx = np.concat(([0.0],np.diff(dx)))*fs
    feature_window = _odd_window(0.0001,fs) #0.1 ms: 5 samples at 48.8 kHz
    envelope_window = _odd_window(0.0015,fs) #1.5 ms: 73 samples
    dx_rms = np.sqrt(moving_mean(dx**2, feature_window))
    ddx_rms = np.sqrt(moving_mean(ddx**2, feature_window))
    combined_rms = dx_rms + ddx_rms/2 #her weights: the bend counts half as much as the slope (but see the docstring)
    signal_rms = np.sqrt(moving_mean(signal**2, envelope_window))
    #fail loudly: a flat signal has nothing to scale (a constant has no slope; all zeros has no size either)
    if np.max(combined_rms) == 0 or np.max(signal_rms) == 0:
        raise ValueError("signal is flat, so its features can't be scaled to 0-1")
    activity = combined_rms/np.max(combined_rms) #0 to 1
    envelope = signal_rms/np.max(signal_rms) #0 to 1
    return activity,envelope

    