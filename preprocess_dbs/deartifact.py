import numpy as np
from preprocess_dbs.align import align_signals
from preprocess_dbs.align import shift_signal


def raw_template(single, pp1, gap_s, fs):
    """
    The stim artifact on its own, from the lab's refractory-pair method (Nakhmani et al. 2024; deartifact.m:250-262)
    single = A(t) + B(t), the artifact plus the brain response. In the shortest pair (pp1, about 0.3-0.45 ms apart) the
    second pulse comes while the neurons are still refractory, so it adds only its artifact: pp1 = A(t) + B(t) + A(t - gap).
    pp1 - single leaves A(t - gap), the second pulse's artifact; shifting it back by the gap gives A(t), the template
    Input:
    single: the single-pulse average, channels x time (e.g. average_segment(...)["single"].data)
    pp1: the shortest pair's average, channels x time, same channels in the same order
    gap_s: time between the pair's two pulses in s (onset to onset, e.g. 0.000451)
    fs: sampling rate in Hz
    Returns: the template A(t), channels x time, gap_s*fs samples shorter than the shorter input. Not yet fine-aligned
        to the single pulse or blanked (the lab's next steps)
    """
    n = min(single.shape[1], pp1.shape[1]) #the two averages can differ by a few samples: use the shorter length
    single = single[:, :n]
    pp1 = pp1[:, :n]
    second_pulse_artifact = pp1 - single #A(t - gap): the second pulse's artifact, since it adds no brain response
    gap_samples = int(round(gap_s *fs)) #the pair's gap in samples (22 for 0.451 ms at 48.8 kHz)
    #shift back by the true gap: keep everything from gap_samples on, so the artifact starts where the first pulse's did.
    #The lab shifts by round(isi*fs) - 1 with isi = gap - 0.1 ms, about 5 samples short (its alignment step absorbs most of it)
    first_pulse_artifact = second_pulse_artifact[:, gap_samples:]
    return first_pulse_artifact


def align_template(single, template, fs):
    """
    Slides the artifact template in time, channel by channel, until its artifact sits directly on top of the single
    pulse's artifact, so it can be subtracted without leaving a leftover spike. Sarah's deartifact.m (about lines 263-299).
    Per channel, in Sarah's order (deartifact.m:265-272): take both signals from one sample before the stim to the end
    of the epoch, demean each over that whole stretch, then keep the first 4.5 ms; measure the shift on those windows
    with align_signals, then move the whole template channel by -shift (the template is `shift` samples later than the
    single, so it has to move back by that much).

    Input:
    single: the single-pulse average, channels x time (the stim at 2 ms)
    template: the artifact template from raw_template, channels x time, same channels in the same order
    fs: sampling freq, usually 48.8 kHz
    Returns:
    aligned: the template moved onto the single pulse, same shape as template. Not yet blanked (Sarah's next step)
    shifts: one per channel, in samples (can be a fraction). Positive means the template's artifact came later than
        the single's and was moved earlier
    """
    aligned = template.copy() #a copy, so filling it in below doesn't change the caller's template
    #start of the window, one sample before the stim. Epochs begin 2 ms before the stim, so the stim is at
    #round(0.002*fs) = 98; Sarah starts at MATLAB sample 98, which is Python 97 (MATLAB counts from 1). It matters
    #because find_first_big_deflection ignores the first 0.2 ms counted from this start
    start = round(0.002*fs) - 1
    n_window = round(0.0045*fs) #window length: 4.5 ms (220 samples), where the artifact is
    shifts = [] #one shift per channel, filled in by the loop
    for ch in range(template.shape[0]):
        #this channel from the window start to the end of the epoch
        single_tail = single[ch, start:]
        template_tail = template[ch, start:]
        #Sarah's order: demean over that whole stretch FIRST, then keep only the 4.5 ms window. The mean decides which
        #peak looks biggest, so demeaning after cutting can pick a different peak
        single_window = (single_tail - np.mean(single_tail))[:n_window]
        template_window = (template_tail - np.mean(template_tail))[:n_window]
        #how far the template's artifact is from the single's (positive = the template is later)
        shift, aligned_single_peak, aligned_template_peak = align_signals(single_window,template_window,fs)
        aligned[ch] = shift_signal(template[ch],-shift) #move the whole channel, not just the window, back by the shift
        shifts.append(shift)
    return aligned, np.array(shifts)


def blank_template(template,gap_s,fs,blank=True):
    """
    Flattens the start of the aligned template: from one sample before the stim through the pair's isi + 0.35 ms, each
    channel is replaced by its own mean over the 2 ms before the stim. Sarah's deartifact.m:309-314.
    For pp1 that is about 0.5 ms (gap 0.246 ms, samples 97-121) or 0.7 ms (gap 0.451 ms, samples 97-131).

    Optional (blank=False skips it) because it's open question Q1. Sarah's comment says it blanks the "conditioning
    pulse region" (the pair's first pulse), but the template has already been shifted back by the gap, so that region
    now sits before the stim; what gets flattened is the start of the aligned artifact itself, where its largest swing
    usually is. That can change which deflection the later alignment of each condition locks onto (to check on real
    data). Her code also averages the epoch's 5-9 ms (3-7 ms after the stim) with the pre-stim mean ("interpolated
    mean"), then fills with the pre-stim mean alone, so that average has no effect; it may be unfinished.
    Copied as her code runs; ask Walker, and compare both ways on real data.

    Input:
    template: the aligned template from align_template, channels x time (the stim at 2 ms)
    gap_s: time between the pair's two pulses in s (onset to onset, as in raw_template, e.g. 0.000451)
    fs: sampling freq, usually 48.8 kHz
    blank: True (default) flattens as Sarah does; False returns the template unchanged
    Returns: a copy of the template, same shape, with the stretch flattened (or untouched if blank=False)
    """
    if not blank:
        return template.copy() #still a copy, so the caller can change the result without changing their template
    start = round(0.002*fs)-1 #one sample before the stim (97), the same start as align_template (Sarah's sample 98)
    isi = gap_s - 0.0001 #Sarah's isi is the gap minus 0.1 ms (parse.m's msafterstim)
    #how many samples to flatten: the isi, plus 0.35 ms, plus 1 because her start:end range includes both ends
    #(gap 0.451 ms: 17 + 17 + 1 = 35)
    length = round(isi*fs) + round(0.00035*fs) +1
    template = template.copy() #work on a copy so the caller's template isn't changed
    #fill value: each channel's own mean over samples 0 to start, including start (her 1:98), the 2 ms before the stim.
    #keepdims keeps it a column (channels x 1), so each row gets its own mean in the next line
    mean_per_channel = np.mean(template[:,:start+1],axis=1,keepdims=True)
    template[:,start:start+length] = mean_per_channel #every sample in the stretch becomes its row's mean
    return template