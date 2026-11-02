import warnings
import numpy as np


def split_by_stim_contact(times, gap_s = 3.0):
    """
    Input: 
    times: pulse onsets in s from the start of the recording (from stim_times)
    gap_s: a pause longer than this means the operator moved the stim cable to a new contact
    Returns:
    {"stim2":(t1,t2,idx), "stim3":..., "stim4":...; where stim2/3/4 is contact the cable was on.}
    """
    times = np.asarray(times)
    gaps = np.diff(times) # the pause after each pulse
    moves = np.flatnonzero(gaps > gap_s) # pulses followed by a cable move based on gap_s
    moves = moves[moves > 0]
    if len(moves) < 2:
        raise ValueError(f"found {len(moves)} cable moves (pauses > {gap_s} s), expected 2 (stim2 to stim3 and stim3 to stim4); "
                         "can't tell which pulses belong to which contact")
    if len(moves) > 2:
        warnings.warn(f"found {len(moves)} cable moves (pauses > {gap_s} s), expected 2; using the first 2 and treating "
                      "everything after the second as stim4, as the lab does (preprocess.m:361)")
    rows = np.arange(len(times)) # pulse numbers 0, 1, 2 ...
    #the lab keeps a lone first pulse in stim2 (start_pts(1) = first pulse, preprocess.m:361); we drop it, because
    #a pulse seconds away from the rest isn't part of the stim2 run (a test pulse or TDT's t=0 pulse?)
    start = 1 if gaps[0] > gap_s else 0
    groups = {"stim2": rows[start:moves[0]+1], "stim3": rows[moves[0]+1:moves[1]+1], "stim4": rows[moves[1]+1:]}
    #each contact gets the same protocol, so the runs should be about the same size. A much smaller run usually means
    #a cable move was missed (under gap_s) and another long pause was taken for it
    sizes = {name: len(idx) for name, idx in groups.items()}
    if min(sizes.values()) < 0.5 * max(sizes.values()):
        warnings.warn(f"runs are uneven ({sizes} pulses); a cable move may have been missed or a pause inside a run "
                      f"taken for one, check the split")
    out = {}
    for name, idx in groups.items():
        t1 = times[idx[0]] - 0.002 # from sarah: first pulse - 2ms
        t2 = times[idx[-1]] + 0.050 # last pulse + 50 ms
        out[name] = (t1, t2, idx)
    return out


def rest_window(times, impedance_times=()):
    """
    Input:
    times: pulse onsets in s (from stim_times)
    impedance_times: when impedance checks happened in this block, in s(empty if none)
    Returns: (t1,t2) of the rest window in s, or None if there isnt enough time before the first pulse
    """
    #1 where rest starts: 1s in or 9s after the last impedance check
    #lab: preprocess.m:364-390, only the else branch ever runs (its comments describe the dead branch)
    if len(impedance_times) ==0:
        start = 1 # 1 second in
    else:
        start = max(impedance_times) + 9 # 9 seconds after the last impedance check

    #2 too little time before first pulse? no rest
    first = times[0]
    if first <= start + 3:
        return None

    #3 rest ends half a second before the first pulse
    return(start, first - 0.5)


def stim_contact(segment, stim_lead):
    """
    Input:
    segment: "stim2", "stim3", or "stim4" (from split_by_stim_contact)
    stim_lead: the lead being stimulated, 1 or 2(from the block's stim contacts, not from experiment name)
    Returns:
    the stimulated contacts channel name (lead1_c2)
     """
    contact = segment[4:]
    name = f"lead{stim_lead}_c{contact}"
    return name


def mark_bad_stim_contact(raw, segment, stim_lead):
    """
    Marks the stimulated contact bad in raw. Sarah dropped it by row position (preprocess.m 228-243), so a do not use
    contact below it shifts which one goes. this fx finds it by name instead
    """
    name = stim_contact(segment, stim_lead) #asking stim contact for the name ie lead1_c2
    if name not in raw.ch_names: # channel not found in the recording at all
        raise ValueError(f"no channel {name} in the recording; is stim_lead= {stim_lead} correct and was lead {stim_lead} working/loaded")
    if name not in raw.info["bads"]:
        raw.info["bads"].append(name)
    return raw

    