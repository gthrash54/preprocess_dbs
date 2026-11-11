import numpy as np
from preprocess_dbs.align import align_signals
from preprocess_dbs.align import shift_signal
import mne


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


def _measure_shift(x, y, fs):
    """
    Measures how far y's artifact sits from x's, one channel at a time, on Sarah's window (deartifact.m:265-276 and
    420-431, the same recipe both times): from one sample before the stim to the end of the epoch, demean each signal
    over that whole stretch, keep the first 4.5 ms, then align_signals. Shared by align_template and the template
    subtraction so the window is written once and can't drift apart between the two.

    Input:
    x: one channel, 1-D, the signal measured against (the stim at 2 ms)
    y: one channel, 1-D, the signal compared with x (the stim at 2 ms). x and y can differ in length; both windows are
        4.5 ms long
    fs: sampling freq, usually 48.8 kHz
    Returns: what align_signals returns, on the windows: shift (samples, can be a fraction; positive = y is later),
        x's peak and y's peak (positions within the window, counted from the window start)
    """
    #start of the window, one sample before the stim. Epochs begin 2 ms before the stim, so the stim is at
    #round(0.002*fs) = 98; Sarah starts at MATLAB sample 98, which is Python 97 (MATLAB counts from 1). It matters
    #because find_first_big_deflection ignores the first 0.2 ms counted from this start
    start = round(0.002*fs) - 1
    n_window = round(0.0045*fs) #window length: 4.5 ms (220 samples), where the artifact is
    #each signal from the window start to the end of the epoch
    x_tail = x[start:]
    y_tail = y[start:]
    #Sarah's order: demean over that whole stretch FIRST, then keep only the 4.5 ms window. The mean decides which
    #peak looks biggest, so demeaning after cutting can pick a different peak
    x_window = (x_tail - np.mean(x_tail))[:n_window]
    y_window = (y_tail - np.mean(y_tail))[:n_window]
    return align_signals(x_window, y_window, fs)


def align_template(single, template, fs):
    """
    Slides the artifact template in time, channel by channel, until its artifact sits directly on top of the single
    pulse's artifact, so it can be subtracted without leaving a leftover spike. Sarah's deartifact.m (about lines 263-299).
    Per channel: measure the shift on Sarah's window with _measure_shift (from one sample before the stim, demeaned to
    the end of the epoch, first 4.5 ms kept), then move the whole template channel by -shift (the template is `shift`
    samples later than the single, so it has to move back by that much).

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
    shifts = [] #one shift per channel, filled in by the loop
    for ch in range(template.shape[0]):
        #how far the template's artifact is from the single's (positive = the template is later)
        shift, single_peak, template_peak = _measure_shift(single[ch], template[ch], fs)
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


def subtract_template(data,template, fs, single=None, gap_s=None):
    """
    Removes the stim artifact from one condition's average by subtracting the finished template, leaving the brain
    response. Sarah's deartifact.m:350-480.
    1. Pairs only (single and gap_s given): subtract the single pulse and shift back by the gap (raw_template), so what
       is left is the second pulse's artifact + response, starting at the stim like everything else. Singles, bursts
       and probes are used as they are.
    2. Per channel: measure the shift on Sarah's window (_measure_shift, data as x, template as y) and move the DATA
       onto the template by +shift (the opposite of align_template, which moves the template). The brain response
       moves with it, by a fraction of a sample.
    3. Subtract the template over the samples both have; any extra data samples past the template's end are kept as
       they are (Sarah's lines 471-480).

    Input:
    data: one condition's average, channels x time (the stim at 2 ms), e.g. average_segment(...)["single"].data
    template: the finished template (align_template, then blank_template), same channels in the same order
    fs: sampling freq, usually 48.8 kHz
    single: pairs only, the single-pulse average (channels x time); leave as None for everything else
    gap_s: pairs only, time between the pair's two pulses in s (onset to onset, e.g. 0.000451)
    Returns:
    deartifacted: the condition with the artifact subtracted, a new array (the input isn't changed). As long as the
        data, or for pairs as long as raw_template's output
    shifts: one per channel, in samples (can be a fraction). Positive means the template's artifact was later than the
        data's, so the data was moved later
    peaks: one per channel, where the data's artifact peak sat in Sarah's window, in samples from the window start
        (before the data was moved). Sarah's artifact_peak_frame (hers counts from 1), which postprocess uses to trim
    """
    if single is not None:
        #pair: minus the single, shifted back by the gap, exactly what raw_template does (a new array)
        data = raw_template(single, data, gap_s, fs)
    else:
        data = data.copy() #not a pair: a copy, so filling it in below doesn't change the caller's data
    #line the data up with the template, one channel at a time
    shifts = []
    peaks = []
    for ch in range(data.shape[0]):
        #how far the template's artifact is from the data's (positive = the template is later)
        shift, data_peak, template_peak = _measure_shift(data[ch], template[ch], fs)
        data[ch] = shift_signal(data[ch], shift) #move the whole data channel onto the template (+shift)
        shifts.append(shift)
        peaks.append(data_peak)
    #subtract over the overlap: the two can differ in length, so only the first n samples have a template to subtract
    n = min(data.shape[1],template.shape[1])
    data[:,:n] = data[:,:n] - template[:,:n]
    return data, np.array(shifts), np.array(peaks)

def deartifact_segment(averages, fs, blank=True):
    """
    """
    # find the refractory pair
    refractory_pair = []
    for name in averages: #each conditions name ie single, paired_0.45_1
        if name.startswith("paired_"): # only pull pairs
            gap = float(name.split("_")[1]) #"paired_0.45_1" into ['paired','0.45','1'] indexed to pull the gap 
            if gap - 0.1 < 0.4: #sarah's isi used gap-0.1 as the isi, refractory if <0.4ms
                refractory_pair.append(name) #keep its name
    if len(refractory_pair) != 1: # need exactly one to build the template
        raise ValueError(f"needs exactly one refractory pair (isi<0.4), found {refractory_pair}")
    # build the template from the refractory pair
    template_name = refractory_pair[0]
    template_gap_s = (float(template_name.split("_")[1])/1000) #template gap in seconds 
    picks = mne.pick_types(averages["single"].info, dbs=True, ecog=True, exclude=[])#pick the DBS and Ecog rows to clean
    single = averages["single"].data[picks] #the single-pulse average, DBS and ECoG rows only (channels x time)
    pair = averages[template_name].data[picks] #the refractory pair's average, same rows
    template = raw_template(single,pair,template_gap_s,fs=fs) #pair - single, shifted back by the gap: the artifact alone
    template, template_shifts = align_template(single, template, fs) #slid onto the single pulse's artifact
    template = blank_template(template,template_gap_s,fs,blank=blank) #start flattened (open question Q1), unless blank=False

    #clean every condition in dbs and ecog rows
    #channel info for just the picked rows, so the cleaned Evokeds have the right channel names. EMG isn't in them: a
    #cleaned pair is shifted back by its gap (time 0 = its second pulse), an unshifted EMG row couldn't share that time
    #axis, so EMG stays as it was in averages (Sarah copies it unchanged too)
    info = mne.pick_info(averages["single"].info,picks)
    cleaned = {} #{condition: cleaned Evoked}
    peaks = {} #{condition: each channel's artifact peak in Sarah's window}, for postprocess's trim later
    for name, evoked in averages.items(): #each condition's name and its Evoked together
        data = evoked.data[picks] #this condition's DBS and ECoG rows
        if name.startswith("paired_"):
            #a pair: subtract_template also removes the single first and shifts back by this pair's own gap
            gap_s = float(name.split("_")[1])/1000
            clean, shifts, peak = subtract_template(data,template,fs=fs,single=single,gap_s=gap_s)
        else:
            #single, bursts, probes: just line up with the template and subtract it
            clean, shifts, peak = subtract_template(data,template,fs=fs)
        #wrap the cleaned array back into an Evoked: channel names, start time (-2 ms), pulses averaged, condition name
        cleaned[name] = mne.EvokedArray(clean,info,tmin=evoked.tmin,nave=evoked.nave, comment=name)
        peaks[name] = peak
    return cleaned, peaks
