import numpy as np


def refine_peak(signal, idx, half_width=3):
    """
    Finds a peak's position to a fraction of a sample: fits a parabola through the samples around it and returns the
    parabola's top. The lab's refinePeak (alignsignals_fractional.m). A pulse fires anywhere between two samples, so its
    artifact peak usually falls between them too; the stim artifact is steep enough that a fraction of a sample matters
    Input:
    signal: one channel's samples, 1-D
    idx: the position of the peak's biggest sample (from the peak finder)
    half_width: how many samples on each side to fit through (3, as the lab: 7 points)
    Returns: the peak's position as a decimal, on the same 0-based count as idx (e.g. 41.8 = between samples 41 and 42).
        Kept inside the fitted samples: the lab doesn't do this, so a nearly flat stretch could send its peak hundreds of
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
    #a straight line has no top. The fit never gives exactly 0 for a, only something tiny from rounding (the lab checks
    #a == 0, so it misses this), and dividing by it puts the "peak" absurdly far away: treat tiny as 0
    if abs(a) <= 1e-12 * np.max(np.abs(values_of_positions)):
        return idx
    vertex = -b / (2 * a) #the parabola's top
    return min(max(vertex, first_position), last_position - 1) #kept inside the fitted samples


def find_first_big_deflection(signal, fs, desired_sign=None):
    """
    """
    idx_largest = np.argmax(np.abs(signal)) # find the position of the largest signal ( both positive and negative)
    
    if desired_sign == None: #without a desired sign, we continue doing what we wrote above
        if idx_largest < round(0.2/1000*fs)-1: # if the biggest signal is in the first 9 samples, dont trust it bc its the leftover of first pulse. multiplying by fs gives you sample #
            idx_largest = np.argmax(abs(signal[(round(0.2/1000*fs)-1):]))
            idx_largest = idx_largest + (round(0.2/1000*fs)-1)
        sign_value = np.sign(signal[idx_largest]) #sign of the largest value
    else:
        mask = np.sign(signal) == desired_sign # if there is a desired sign we look at it
        if mask.any()==True:
            rel_idx = np.argmax(np.abs(signal[mask]))
            idx_candidates = np.flatnonzero(mask)
            idx_largest = idx_candidates[rel_idx]
            sign_value =desired_sign
        else: # if the biggest signal is in the first 9 samples, dont trust it bc its the leftover of first pulse. multiplying by fs gives you sample #
            idx_largest = np.argmax(np.abs(signal))
            sign_value = np.sign(signal[idx_largest]) 
    stretch = signal[max(0,idx_largest - round(0.0004*fs)):(idx_largest+1)]
    big = np.abs(stretch) >= (np.abs(signal[idx_largest])/2)
    if desired_sign == None:
        candidates = big
    else:
        candidates = (big) & ((np.sign(stretch) == sign_value))
    candidates[-4:] = False

    return idx_largest, sign_value

        