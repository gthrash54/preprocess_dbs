import warnings # python's built-in way to print a warning without stopping the program
import numpy as np # numpy = arrays and math on whole arrays at once; "np" is the usual short name
import scipy.signal # scipy.signal = signal processing tools (filters, detrend, resample)


#"gap_s = 3.0" is a default argument: if the caller doesn't pass gap_s, it is 3.0
#why a pause: the operator moves the stim cable to the next contact by hand between runs, which stops the pulses for a
#few seconds; 3 s is the lab's cutoff for calling a pause a move (diff(PeA_.onset) > 3, preprocess.m:248)
def split_by_stim_contact(times, gap_s = 3.0):
    """
    Input:
    times: pulse onsets in s from the start of the recording (from stim_times)
    gap_s: a pause longer than this means the operator moved the stim cable to a new contact
    Returns:
    {"stim2": (t1, t2, idx), "stim3": ..., "stim4": ...}
        stim2/3/4 = the contact the cable was on; t1, t2 = the segment's data window in s; idx = that contact's pulse rows
    """
    #np.asarray turns a list (or anything list-like) into a numpy array, so the array math below works
    times = np.asarray(times)
    #np.diff gives each value minus the one before it: [1, 3, 10] -> [2, 7]. One shorter than times;
    #gaps[i] = times[i+1] - times[i]
    gaps = np.diff(times) # the pause after each pulse
    #gaps > gap_s is a True/False array; np.flatnonzero returns the positions (indices) where it is True
    moves = np.flatnonzero(gaps > gap_s) # pulses followed by a cable move based on gap_s
    #indexing an array with a True/False array keeps only the entries where it is True
    moves = moves[moves > 0] # ignore a pause right after the first pulse (the lab's idx(idx > 1); numpy counts from 0)
    #3 contacts (2, 3, 4) means 2 moves between them; with fewer we can't split
    if len(moves) < 2:
        #raise stops the function with an error. f"..." is an f-string: {expression} inside it is replaced by its value
        #two strings side by side ("..." "...") are joined into one
        raise ValueError(f"found {len(moves)} cable moves (pauses > {gap_s} s), expected 2 (stim2 to stim3 and stim3 to stim4); "
                         "can't tell which pulses belong to which contact")
    if len(moves) > 2:
        #warnings.warn prints the message but lets the code keep going (unlike raise)
        warnings.warn(f"found {len(moves)} cable moves (pauses > {gap_s} s), expected 2; using the first 2 and treating "
                      "everything after the second as stim4, as the lab does (preprocess.m:361)")
    #np.arange(n) = 0, 1, ..., n-1
    rows = np.arange(len(times)) # pulse numbers 0, 1, 2 ...
    #the lab keeps a lone first pulse in stim2 (start_pts(1) = first pulse, preprocess.m:361); we drop it, because
    #a pulse seconds away from the rest isn't part of the stim2 run (a test pulse or TDT's t=0 pulse?)
    #"a if cond else b" is a one-line if: start is 1 when the first gap is long, otherwise 0
    start = 1 if gaps[0] > gap_s else 0
    #{key: value, ...} is a dict (a lookup table from names to values). rows[a:b] is a slice: entries a up to b-1 (the end is
    #excluded), so +1 makes the pulse before each move the last one in its run. rows[b:] = from b to the end
    groups = {"stim2": rows[start:moves[0]+1], "stim3": rows[moves[0]+1:moves[1]+1], "stim4": rows[moves[1]+1:]}
    #each contact gets the same protocol, so the runs should be about the same size. A much smaller run usually means
    #a cable move was missed (under gap_s) and another long pause was taken for it
    #a dict comprehension: builds a dict in one line, one entry per (name, idx) in groups. .items() gives (key, value) pairs
    sizes = {name: len(idx) for name, idx in groups.items()}
    #.values() = just the dict's values (here the 3 run sizes)
    if min(sizes.values()) < 0.5 * max(sizes.values()):
        warnings.warn(f"runs are uneven ({sizes} pulses); a cable move may have been missed or a pause inside a run "
                      f"taken for one, check the split")
    out = {} # an empty dict, filled in the loop
    #"for name, idx in ..." unpacks each (key, value) pair into two variables
    for name, idx in groups.items():
        #idx[0] = first entry, idx[-1] = last entry (negative indices count from the end)
        #2 ms before the first pulse leaves a little pre-stim baseline; 50 ms after the last lets its response finish
        t1 = times[idx[0]] - 0.002 # from sarah: first pulse - 2ms
        t2 = times[idx[-1]] + 0.050 # last pulse + 50 ms
        out[name] = (t1, t2, idx) # (a, b, c) is a tuple: a fixed group of values stored together
    return out


#impedance_times=() defaults to an empty tuple, meaning "no impedance checks"
def rest_window(times, impedance_times=()):
    """
    Input:
    times: pulse onsets in s (from stim_times)
    impedance_times: when impedance checks happened in this block, in s (empty if none)
    Returns: (t1, t2) of the rest window in s, or None if there isn't enough time before the first pulse
    """
    #1 where rest starts: 1s in or 9s after the last impedance check
    #lab: preprocess.m:364-390, only the else branch ever runs (its comments describe the dead branch)
    #(an impedance check passes a small test current through the contacts, so the signal right after it isn't clean rest;
    #the lab's code doesn't say why it chose 1 s and 9 s)
    if len(impedance_times) == 0:
        start = 1 # 1 second in
    else:
        start = max(impedance_times) + 9 # 9 seconds after the last impedance check

    #2 too little time before first pulse? no rest
    #TDT logs a fake pulse at exactly t=0 that the lab drops (preprocess.m:217) and clean_dbs keeps, so skip it
    times = np.asarray(times)
    real = times[times > 0] # True/False indexing again: keep only pulses after t=0
    if len(real) == 0:
        raise ValueError("no pulses after t=0, so there's no first pulse to end rest at")
    first = real[0]
    #the first pulse must come more than 3 s after start (the lab's start_pts(1) > 3 + remove)
    if first <= start + 3:
        #an early return: the function stops here. None is python's "no value"; the caller checks for it
        return None

    #3 rest ends half a second before the first pulse
    #(start, first - 0.5) is a tuple; the caller can unpack it with t1, t2 = rest_window(...)
    return(start, first - 0.5)


def stim_contact(segment, stim_lead):
    """
    Input:
    segment: "stim2", "stim3", or "stim4" (from split_by_stim_contact)
    stim_lead: the lead being stimulated, 1 or 2 (from the block's stim contacts, not from experiment name)
    Returns:
    the stimulated contact's channel name (e.g. lead1_c2)
    """
    #slicing works on text too: "stim2"[4:] = everything from character 4 (counting from 0) on = "2"
    contact = segment[4:]
    name = f"lead{stim_lead}_c{contact}"
    return name


def mark_bad_stim_contact(raw, segment, stim_lead):
    """
    Marks the stimulated contact bad in raw. Sarah dropped it by row position (preprocess.m:228-243): if a lower contact
    was do-not-use, the rows shift and she drops the wrong one. This function finds it by name instead
    """
    #why drop it at all: on these TDT noSwitch rigs the stim contact is unplugged from the amplifier while it stimulates,
    #so it reads nearly flat, and any pair or average using it would be wrong
    #raw.copy() makes an independent copy; changing the copy leaves the caller's raw as it was ("in place" would change it)
    raw = raw.copy() #like the other steps, don't change the caller's raw (marking stim2 then stim3 would leave both bad)
    name = stim_contact(segment, stim_lead) # get the name from stim_contact, e.g. lead1_c2
    #raw.ch_names = the list of channel names in this Raw; "x not in list" is True when x isn't one of them
    if name not in raw.ch_names: # channel not found in the recording at all
        raise ValueError(f"no channel {name} in the recording; is stim_lead={stim_lead} correct and was lead {stim_lead} working/loaded?")
    #raw.info["bads"] = MNE's list of bad channel names. The data stays, but MNE's tools (and ours) skip these channels
    if name not in raw.info["bads"]:
        raw.info["bads"].append(name) # .append adds one item to the end of a list
    return raw


def detrend_segment(raw):
    """
    Fits and removes a straight line from each channel over one whole segment, to remove electrode drift (voltage creep
    from electrode polarization and amplifier offset). The lab's detrend(x, 1), preprocess.m:828-846
    Input:
    raw: one segment's MNE Raw (e.g. stim2), after rereferencing
    Returns:
    a copy of raw with each channel's best-fit straight line removed
    """
    raw = raw.copy()
    #lambda x: ... is a tiny unnamed function: takes x, returns the detrended x
    #raw.apply_function runs that function on the data and stores the result back into raw (in place, which is why we copied
    #first). By default it's channel_wise=True: called once per channel, x = one channel's samples, so each channel gets its
    #own line
    raw.apply_function(lambda x: scipy.signal.detrend(x, type="linear"))
    return raw
