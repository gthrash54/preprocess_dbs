import numpy as np
import warnings
from scipy.signal import butter, sosfiltfilt


def refine_peak(signal, idx, half_width=3):
    """
    Finds a peak's position to a fraction of a sample: fits a parabola through the samples around it and returns the
    parabola's top. Sarah's refinePeak (alignsignals_fractional.m). A pulse fires anywhere between two samples, so its
    artifact peak usually falls between them too; the stim artifact is steep enough that a fraction of a sample matters
    Input:
    signal: one channel's samples, 1-D
    idx: the position of the peak's biggest sample (from the peak finder)
    half_width: how many samples on each side to fit through (3, as Sarah: 7 points)
    Returns: the peak's position as a decimal, on the same 0-based count as idx (e.g. 41.8 = between samples 41 and 42).
        Kept inside the fitted samples: Sarah's code doesn't do this, so a nearly flat stretch could send its peak hundreds of
        samples away and wreck the alignment (open question: confirm with Harrison). idx unchanged when fewer than 3
        samples are left near an edge or the samples lie on a straight line
    """
    #the positions around the peak, kept inside the signal: max/min stop them running off either end
    first_position = max(0, idx - half_width)
    last_position = min(len(signal), idx + half_width + 1) #+1 because np.arange stops before its end
    range_of_positions = np.arange(first_position, last_position)
    values_of_positions = signal[range_of_positions] #the heights at those positions
    if len(range_of_positions) < 3: #a parabola needs at least 3 points
        return idx
    a, b, c = np.polyfit(range_of_positions, values_of_positions, 2) #fits a parabola onto the peak: y = a*x^2 + b*x + c
    #a straight line has no top. The fit never gives exactly 0 for a, only something tiny from rounding (Sarah's code checks
    #a == 0, so it misses this), and dividing by it puts the "peak" absurdly far away: treat tiny as 0
    if abs(a) <= 1e-12 * np.max(np.abs(values_of_positions)):
        return idx
    vertex = -b / (2 * a) #the parabola's top
    return min(max(vertex, first_position), last_position - 1) #kept inside the fitted samples


def find_first_big_deflection(signal, fs, desired_sign=None):
    """
    Finds the stim artifact's reference peak, used to line the template up with a signal (Sarah's
    findFirstBigDeflection in alignsignals_fractional.m). The artifact goes up and down more than once, and which of
    those peaks is biggest can differ between the template and the signal. So instead of the biggest peak, this takes
    the first big one: the biggest peak, unless there's another peak at least half its size in the 0.4 ms before it
    Input:
    signal: one channel's samples, 1-D, lightly smoothed
    fs: sampling rate in Hz
    desired_sign: None for the first call; +1 or -1 for the second call, to only look at peaks pointing the same way
        as the first call's (pass in the sign the first call returned)
    Returns: (position of the reference peak, its sign: +1 for a peak pointing up, -1 for one pointing down)
    """
    #starting guess: the position of the biggest value, up or down
    idx_largest = np.argmax(np.abs(signal))

    if desired_sign == None:
        #first call. In short paired-pulse signals, the first pulse's leftover sits in the first 0.2 ms and can be the
        #biggest value, so ignore that stretch: 0.2 ms = 10 samples in Sarah's 1-based count, the first 9 here
        if idx_largest < round(0.2/1000*fs)-1:
            idx_largest = np.argmax(abs(signal[(round(0.2/1000*fs)-1):]))
            idx_largest = idx_largest + (round(0.2/1000*fs)-1) #back to a position in the whole signal
        sign_value = np.sign(signal[idx_largest]) #+1 if the peak points up, -1 if down
    else:
        #second call: the biggest value pointing the given way (True/False mask -> biggest in the shorter list ->
        #its position in the whole signal). 
        mask = np.sign(signal) == desired_sign
        if mask.any()==True:
            rel_idx = np.argmax(np.abs(signal[mask]))
            idx_candidates = np.flatnonzero(mask)
            idx_largest = idx_candidates[rel_idx]
            sign_value =desired_sign
        else:
            #nothing points the given way: use the biggest value overall but say so, because the two
            #signals' reference peaks would then point opposite ways and the alignment would likely be wrong
            idx_largest = np.argmax(np.abs(signal))
            sign_value = np.sign(signal[idx_largest])
            warnings.warn(f"no sample points in the desired direction ({int(desired_sign):+d}); using the biggest peak "
                          f"({int(sign_value):+d}) instead, so the alignment may line up peaks pointing opposite ways")

    #look back: the 0.4 ms before the peak, up to and including it, where an earlier big peak would be
    search_region_start = max(0,idx_largest - round(0.0004*fs))
    search_region = signal[search_region_start:(idx_largest+1)]
    #candidates: at least half the peak's size (and, in the second call, pointing the same way). The last 4 samples
    #are the peak itself and the samples rising into it, so they can't count as an earlier peak
    big = np.abs(search_region) >= (np.abs(signal[idx_largest])/2)
    if desired_sign == None:
        candidates = big
    else:
        candidates = (big) & ((np.sign(search_region) == sign_value))
    candidates[-4:] = False
    #keep only candidates that are the top of a peak (or the bottom of a dip): at least as far from zero as the 2
    #samples on each side, so a point partway up a slope doesn't count. Needs 2 samples on each side to compare with
    local_peaks = []
    for i in np.flatnonzero(candidates):
        if i < 2 or i > len(search_region) - 3:
            continue
        if search_region[i] > 0:
            if (search_region[i] >= search_region[i+1] and search_region[i] >= search_region[i+2] and search_region[i] 
             >= search_region[i-1] and search_region[i] >= search_region[i-2]):
                local_peaks.append(i)
        elif search_region[i] < 0:
            if (search_region[i] <= search_region[i+1] and search_region[i] <= search_region[i+2] and search_region[i] 
             <= search_region[i-1] and search_region[i] <= search_region[i-2]):
                local_peaks.append(i)

    #if there's an earlier peak, the biggest one becomes the reference peak (position back in the whole signal)
    if local_peaks:
        peak_height = abs(search_region[local_peaks])
        peak_max_height = np.argmax(peak_height)
        max_peak = local_peaks[peak_max_height]
        idx_largest = search_region_start + max_peak
        sign_value = np.sign(signal[idx_largest])
    return idx_largest, sign_value


def smooth_signal(signal, fs):
    """
    Smooths and filters the artifact using a 4th order Butterworth and 8khz low pass
    to preserve artifact and remove sample to sample noise. It is zero phase by running forwards and backwards (filtfilt)
    Input:
    signal: signal of interest that requires smoothing, likely has already been preprocessed.
    fs: Sampling frequency, usually 48.8khz
    Returns: smoothed_signal, the filtered signal of interest
    """
    fourth_order_butter = butter(4, 8000,btype="low",fs=fs,output="sos") #4th order Butterworth with a 8khz cutoff with low pass filtering. sos is just the numerically safe output
    smoothed_signal = sosfiltfilt(sos=fourth_order_butter,x=signal) #zero-phase using filtfilt
    return smoothed_signal
def align_signals(signal_x,signal_y,fs):
    """
    Measures how far apart the signals are to a fraction of a sample.
    We first smooth the signals, then find their reference peaks and refine them. Then we subtract them to find the difference. Adapted from alignsignals_fractional.m from Sarah
    Input:
    signal_x: the signal we measure against, one channel. Sarah uses it twice:
        building the template: the single-pulse average
        removing the artifact: one condition's average (single, burst, probes, or pair minus single)
    signal_y: the signal compared with x, one channel, same length as signal_x. Its peak is found pointing the same
        way as x's (both up or both down)
        building the template: the raw_template output (pp1 - single)
        removing the artifact: the finished template
    fs: sampling freq, usually 48.8 kHz
    Returns:
    shift: the difference between the two peaks in units of samples (measured to the fraction), with a positive value indicating y following x temporally.
        e.g. shift = 2.4 means y's peak comes 2.4 samples (about 0.05 ms) after x's
    refined_x_peak,refined_y_peak: where each signal's reference peak sits, in samples from the start of that signal
        (a fraction, from the parabola fit on the smoothed signal). shift = refined_y_peak - refined_x_peak
    """
    if len(signal_x) != len(signal_y):
        raise ValueError(f'Length of signal_x ({len(signal_x)}) and signal_y ({len(signal_y)}) are not equal')
    smoothed_x,smoothed_y = smooth_signal(signal_x,fs), smooth_signal(signal_y,fs)
    x_reference_peak,x_reference_sign = find_first_big_deflection(smoothed_x,fs)
    y_reference_peak,y_reference_sign = find_first_big_deflection(smoothed_y, fs, desired_sign = x_reference_sign)
    refined_x_peak = refine_peak(smoothed_x, x_reference_peak, half_width = 3)
    refined_y_peak = refine_peak(smoothed_y, y_reference_peak, half_width = 3)
    shift = refined_y_peak - refined_x_peak #positive means y comes after
    return shift, refined_x_peak, refined_y_peak



    