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