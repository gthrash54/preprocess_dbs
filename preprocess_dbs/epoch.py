import h5py
import warnings
import numpy as np
from pathlib import Path
import mne 

def stim_times(h5_path, voice=None, confirmed_only=True):
    """
    Input:
    h5_path: a cleaned h5 file from clean_dbs
    voice: which stimulator output to keep ("A", "B", ...), None keeps every voice. TDT only: BrainVision files have no voices
    confirmed_only: True keeps only the pulses clean_dbs is sure are real stim (confirmed = 1). TDT pulses always are;
        BrainVision pulses only when a stimulator file backs them up, so some BrainVision blocks need False
    Returns: the pulse times as a numpy array, in seconds from the start of the recording (empty if the block had no stim)
    """
    if not Path(h5_path).is_file():
        raise FileNotFoundError(f"no cleaned file at {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if "stim" not in f:
            raise KeyError(f"{h5_path} has no stim table (cleaned with an old clean_dbs version?)")
        stim = f["stim"]
        times = stim["time"][:]
        keep = np.ones(len(times), dtype=bool) #start with every pulse kept
        if confirmed_only:
            keep &= stim["confirmed"][:] == 1 #&= means "and also": drop any pulse that isn't confirmed
            if len(times) and not keep.any():
                warnings.warn(f"all {len(times)} pulses in {Path(h5_path).name} are unconfirmed, none kept; "
                              "use confirmed_only=False if you trust them")
            elif not keep.all():
                warnings.warn(f"{(~keep).sum()} of {len(times)} pulses in {Path(h5_path).name} are unconfirmed and were dropped")
        if voice is not None:
            keep &= stim["voice"][:] == voice.encode() #the file stores voices as bytes (b"A")
    return times[keep]

def make_epochs(raw, tmin=-0.002, tmax= 0.025, event= "stim", baseline=None):
    """
    Cuts a short window of signal around every event, e.g. every stim pulse
    Input:
    raw: MNE Raw from to_mne (its pulses are annotations called "stim")
    tmin, tmax: window around each event, in seconds (default -2 ms to +25 ms). Not the lab's window: its epochs run to the next
        pulse (about 100 ms for singles), and its z-score and baselines use the last 5-7 ms, so set tmax to match when you need those
    event: which annotation to cut around (default "stim")
    baseline: MNE baseline window to subtract, e.g. (-0.2, 0); None (default) subtracts nothing, which ERNA needs
        since only 2 ms come before the stim artifact
    Returns: MNE Epochs, one slice per event (epochs x channels x samples)
    """
    n_events = sum(d == event for d in raw.annotations.description)
    if n_events == 0: #MNE's own error here is hard to read
        raise ValueError(f'no "{event}" annotations in this Raw, nothing to epoch (no pulses in the window?)')
    events, event_id = mne.events_from_annotations(raw, event_id={event:1})
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    if len(epochs) < n_events: #MNE drops events whose window runs past the start or end of the Raw
        warnings.warn(f"{n_events - len(epochs)} of {n_events} {event} events were too close to the edge of the Raw and were dropped")
    return epochs

def select_pulses(raw, gap_before=0.05, gap_after=0.0, event="stim"):
    """
    Keeps only the pulses with enough quiet time around them, so other pulses' artifacts don't land in their epochs
    Input:
    raw: MNE Raw with pulses as annotations (from to_mne)
    gap_before: seconds with no other pulse before a pulse (default 50 ms: keeps the first pulse of each pair, burst or train)
    gap_after: seconds with no other pulse after it (default 0: no rule). Set it to the epoch length (e.g. 0.025) for single
        pulses whose whole epoch is clean
    event: which annotation the pulses are (default "stim")
    Returns: a copy of the Raw with only the kept pulses; other annotations are left as they are
    """
    ann = raw.annotations
    is_event = ann.description == event
    times = ann.onset[is_event]
    before = np.diff(times, prepend=-np.inf) #gap to the previous pulse (the first pulse has none: infinity)
    after = np.diff(times, append=np.inf) #gap to the next pulse
    keep = (before >= gap_before) & (after >= gap_after)
    if len(times) and not keep.any():
        warnings.warn(f"none of the {len(times)} {event} pulses have {gap_before * 1000:g} ms quiet before and "
                      f"{gap_after * 1000:g} ms after")
    drop = np.flatnonzero(is_event)[~keep] #positions in the annotation list of the pulses that didn't pass
    out = raw.copy()
    out.set_annotations(ann.copy())
    out.annotations.delete(drop)
    return out

def classify_pulses(counts,periods):
    """
    Gives each pulse its condition label from the stim table: single, paired with its gap, or its position in a burst
    Input:
    counts: the stim table's "count" column, how many pulses are in the group each pulse belongs to
        (1 = single, 2 = paired, 10 = burst). Both pulses of a pair say 2, and the 10 probes after a burst also say 10
    periods: the stim table's "period_ms" column, the gap in ms the stimulator puts between pulses of a group.
        For a pair this is its interpulse interval
    Returns: a list of labels, one per pulse in the same order as the stim table: "single", "paired_<gap in ms>_<1 or 2>"
        (e.g. "paired_2.89_1" and "paired_2.89_2", first and second pulse of a pair), "burst_1" to "burst_10", "burstprobes_1" to "burstprobes_20" (a burst, then 10 slower
        probes), or "" for a count it doesn't know
    Not the same as the lab's MATLAB code (parse.m), so its labels won't line up one to one with the lab's output:
        - pairs are named by the stimulator's own gap ("paired_0.45"), not by the lab's lookup names (pp1, pp5 ... pp23),
          which come from the measured gap minus 0.1 ms. Any gap works, with no lookup table and no "CHECK"
        - every pulse gets a label. The lab's loop stops one short, so a group that starts on the last pulse of a segment
          (e.g. a single) goes unlabeled. A last pulse that ends an earlier pair or burst still gets labeled there
        - unknown counts stay "" and a burst cut off by the end of the table is labeled as far as it goes,
          where the lab's loop hangs
        - burst + probes also needs the next 10 pulses at a different rate, not just count 10, so two bursts in a row
          aren't mistaken for one burst + probes
    """
    labels = [""] *len(counts)
    m = 0
    while m < len(counts):
        if counts[m] == 1:
            labels[m] = "single" 
            m += 1 
        elif counts[m] == 2:
            for p in range(min(2,len(counts)-m)):
                labels[m+p] = f"paired_{periods[m]:.2f}_{p+1}"
            m += 2
        elif counts[m] == 10:
            if m+10 < len(counts) and counts[m+10] == 10 and periods[m+10] != periods[m]:
                for k in range(min(20,len(counts) - m)):
                    labels[m+k] = f"burstprobes_{k+1}"
                m += 20
            else:
                for k in range(min(10, len(counts) - m)): #a burst cut off by the end of the table: label only the pulses that are there
                    labels[m+k] = f"burst_{k+1}"
                m += 10
        else:
            m += 1
    return labels