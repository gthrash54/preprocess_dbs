import warnings
import numpy as np
import mne
from preprocess_dbs.pulses import _pair_position


def make_epochs(raw, tmin=-0.002, tmax= 0.025, event= "stim", baseline=None):
    """
    Cuts a short window of signal around every event, e.g. every stim pulse
    Input:
    raw: MNE Raw from to_mne (its pulses are annotations called "stim")
    tmin, tmax: window around each event, in seconds (default -2 ms to +25 ms). Not the lab's window: its epochs run to the next
        pulse (about 100 ms for singles; see condition_windows). The lab's later steps also need the end of the epoch (the ERNA
        z-score uses its last 5 ms, the coastline burst index its last 7 ms), so set tmax to match when you need those
    event: which annotation to cut around (default "stim")
    baseline: MNE baseline window to subtract, e.g. (-0.2, 0); None (default) subtracts nothing. Only 2 ms come before the stim
        artifact, so a pre-stim baseline doesn't fit ERNA. The lab's parse.m instead subtracts each epoch's own mean (whole window)
        and, after averaging, the mean of 4-6 ms from the epoch start (about 2-4 ms after the pulse); neither is done here
    Returns: MNE Epochs, one slice per event (epochs x channels x samples)
    """
    n_events = sum(d == event for d in raw.annotations.description)
    if n_events == 0: #MNE's own error here is hard to read
        raise ValueError(f'no "{event}" annotations in this Raw, nothing to epoch (no pulses in the window?)')
    events, event_id = mne.events_from_annotations(raw, event_id={event:1})
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, verbose=False)
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


def condition_windows(times, labels, max_gap=2.0, pair_tail=None):
    """
    Measures how long each condition's epochs can be: from its pulse to just before the next pulse
    Input:
    times: the stim table's "time" column in seconds, the full table, so times[k] and labels[k] are the same pulse.
        Not stim_times(), which can drop pulses (unconfirmed, other voices) whose artifacts are still in the signal
    labels: the labels from classify_pulses or label_by_timing, one per row of the same stim table
    max_gap: a gap longer than this (seconds, default 2) is a pause, not a next pulse (14 s and 21 s between stim contacts
        in uh3021), so the pulse before it is skipped like the last pulse of the table
    pair_tail: None (default), or the fixed time in seconds to keep after a pair's second pulse instead of running to the
        next pulse: conditions.doubled_pair_tail for uh3001/uh3006, as the lab does (parse.m:261-263). Never runs past
        0.2 ms before the next pulse
    Returns: a dictionary {label: window end in seconds after the pulse}, to use as make_epochs' tmax for that condition.
        Each window ends 0.1 ms before the next pulse in the table. A pair's window starts at its first pulse and ends
        0.2 ms before the pulse after the pair (the lab takes 0.1 ms off each of the pair's two gaps), so it holds both
        pulses, and second pulses ("paired_<gap>_2") get no window of their own. Each condition uses its shortest gap,
        so no epoch runs into the next pulse. Unlabeled pulses ("") are skipped. With tmin = -0.002, the lengths for uh3021
        (41 conditions, e.g. single 4977 samples, 0.45 ms pair 4984, burst_1 469) equal those from the lab's parse.m rule
        recomputed by the uh3021 test; they haven't been compared with any of the lab's saved outputs
    Warns when there's nothing to window (pulses but no labels; an empty table quietly gives {}), when a condition gets
    no window (none of its pulses has a next pulse within max_gap), and when a condition is dropped because one of its
    pulses has a next pulse less than 0.1 ms away (no window would be safe for it)
    Raises instead of guessing when times are missing or out of order, when two pulses happen at the same moment, or
    when another pulse falls between a pair's two pulses. Crosstalk blocks do both: they fire two voices together
    (sometimes one at 0 uA, logged with no current) and interleave them. Which of those pulses should end a window
    isn't decided yet
    Not the same as the lab's MATLAB code (parse.m):
        - the lab works out each epoch's length separately (and MATLAB errors if a single's differ). MNE epochs of one
          condition share one length, so each condition gets its shortest. Same result whenever a condition's gaps
          are equal, as they are in uh3021
        - the lab gives a burst or pair that ends a segment a fixed length (500.08 ms, 49.89 ms), never labels a
          segment's last pulse when it starts a group (its loop stops one short), and errors if a plain burst starts in a
          segment's last 19 pulses. Here a pulse followed by a pause (over max_gap) or by nothing is skipped
    """
    times = np.asarray(times, dtype=float)
    if len(times) != len(labels): #rows must line up: label k belongs to time k
        raise ValueError(f"{len(times)} times but {len(labels)} labels: use the full stim table's times, "
                         "the same rows classify_pulses labeled")
    if not (np.isfinite(max_gap) and max_gap > 0):
        raise ValueError(f"max_gap has to be a positive number of seconds, not {max_gap}")
    if not np.all(np.isfinite(times)):
        raise ValueError("times has missing values (NaN), so the gaps can't be measured")
    steps = np.diff(times)
    if np.any(steps < 0):
        raise ValueError("times are out of order: use the stim table's own order, earliest pulse first")
    if np.any(steps == 0):
        raise ValueError(f"{(steps == 0).sum()} pulses happen at the same moment as the one before (e.g. two voices "
                         "firing together, as in crosstalk blocks): which pulses should end a window isn't decided yet")
    if len(labels) == 0: #a block with no stim: nothing to window, nothing to warn about
        return {}
    if not any(labels):
        warnings.warn(f"none of the {len(labels)} pulses has a label, so there are no windows")
        return {}
    windows = {}
    too_close = set()
    for k in range(len(times)):
        label = labels[k]
        if (label == ""):
            continue
        if _pair_position(label) == 2:
            continue
        first_of_pair = _pair_position(label) == 1
        if first_of_pair and k+1 < len(times) and labels[k+1] != label[:-1] + "2":
            raise ValueError(f"another pulse falls between the two pulses of {label} at row {k} (e.g. a second voice): "
                             "which pulses should end a window isn't decided yet")
        if first_of_pair:
            next_row = k+2
        else:
            next_row = k+1
        if first_of_pair and pair_tail is not None and k+1 < len(times):
            #doubled-period patients: a fixed time after the second pulse, capped before the next pulse if that comes first
            end = (times[k+1] - times[k]) - 0.0001 + pair_tail
            if next_row < len(times) and times[next_row] - times[k] <= max_gap:
                end = min(end, times[next_row] - times[k] - 0.0002)
            if label not in windows or end < windows[label]:
                windows[label] = end
            continue
        if next_row >= len(times):
            continue
        gap = times[next_row] - times[k]
        if gap > max_gap: #a pause between runs, not a next pulse
            continue
        if first_of_pair:
            end = gap - 0.0002
        else:
            end = gap -0.0001
        if end <= 0: #a next pulse less than 0.1 ms away: no window is safe for this condition
            too_close.add(label)
            continue
        if label not in windows or end < windows[label]:
            windows[label] = end
    for label in too_close:
        windows.pop(label, None)
    if too_close:
        warnings.warn(f"no window for {sorted(too_close)}: one of their pulses has a next pulse less than 0.1 ms away")
    wanted = {label for label in labels if label and _pair_position(label) != 2}
    missing = sorted(wanted - set(windows) - too_close)
    if missing:
        warnings.warn(f"no window for {missing}: none of their pulses has a next pulse within max_gap ({max_gap} s)")
    return windows

def label_pulses(raw, labels, event="stim"):
    """
    Renames each stim pulse to its condition, so make_epochs can cut one condition at a time
    Input:
    raw: one segment's MNE Raw from to_mne, its pulses annotations called event ("stim"), in time order
    labels: one label per pulse in this segment, same order (classify_pulses labels for the segment's rows, e.g.
        labels[idx] from split_by_stim_contact); "" = unlabelled, and that pulse is left out. classify_pulses and
        split_by_stim_contact must have run on the same pulse table as to_mne's annotations (same voice and
        confirmed_only): the count check catches a different number of pulses, not the same number shifted
    event: which annotations are the pulses (default "stim"); any other annotations are kept as they are
    Returns: a copy of raw whose pulses are named by condition ("single", "burst_3", ...), at the same times
    """
    raw = raw.copy()
    is_pulse = np.asarray(raw.annotations.description) == event #only the pulses get relabelled
    if is_pulse.sum() != len(labels):
        raise ValueError(f"The number of {event} annotations ({is_pulse.sum()}) does not match the number of labels "
                         f"({len(labels)}). The labels could land on the wrong pulses")
    keep = np.asarray(labels) != ""
    labels = np.asarray(labels)[keep]
    onsets = np.asarray(raw.annotations.onset)[is_pulse][keep]
    #the other annotations, unchanged, plus the relabelled pulses, back in time order
    other = raw.annotations[~is_pulse]
    all_onsets = np.concatenate([other.onset, onsets])
    all_names = np.concatenate([other.description, labels])
    all_durations = np.concatenate([other.duration, np.zeros(len(onsets))]) #e.g. a BAD_ span keeps its length
    all_ch_names = list(other.ch_names) + [()] * len(onsets)
    order = np.argsort(all_onsets, kind="stable")
    #orig_time kept, so onsets stay on the same clock (else a Raw that doesn't start at sample 0, e.g. after crop, shifts)
    new_annotations = mne.Annotations(onset=all_onsets[order], duration=all_durations[order], description=all_names[order],
                                      orig_time=raw.annotations.orig_time, ch_names=[all_ch_names[i] for i in order])
    raw.set_annotations(new_annotations)
    return raw

def average_condition(epochs):
    """
    Averages one condition's epochs into its evoked response, the lab's way (parse.m). Each pulse's response is small and
    buried in noise, but it's the same every time while the noise isn't, so averaging many pulses keeps the response and
    cancels the noise
    Input:
    epochs: MNE Epochs of ONE condition (e.g. all "single" pulses), from label_pulses + make_epochs with tmin=-0.002,
        so every epoch starts 2 ms before its pulse. Its data is epochs x channels x time
    Steps:
    1. subtract each epoch's own mean (over time), so pulses that landed at different voltage levels line up
    2. average across epochs: epochs x channels x time -> channels x time
    3. subtract, per channel, the mean of samples round(0.004*fs) to round(0.006*fs) counted from the epoch start
       (the lab's 1-based inclusive MATLAB range, so [start-1:end] here). That's about 2-4 ms after the pulse, which may
       overlap the response itself (open question Q6); copied from the lab for now
    Returns: the evoked response as a numpy array, channels x time (same channel order as epochs)
    """
    data = epochs.get_data()
    demeaned_time = data - data.mean(axis=2, keepdims=True)
    mean_over_epochs = demeaned_time.mean(axis=0)
    fs = epochs.info["sfreq"]
    start = round(0.004*fs)-1
    end = round(0.006*fs)
    window = slice(start,end)
    baseline = mean_over_epochs[:,window].mean(axis=1, keepdims=True)
    evoked = mean_over_epochs - baseline
    return evoked


def average_segment(raw, times, labels, pair_tail=None):
    """
    Averages every condition in one segment into its evoked response: the lab's parse.m for one stim contact
    Input:
    raw: one segment's MNE Raw (e.g. stim2), after rereferencing and detrending, its pulses annotated "stim"
    times: that segment's pulse times in s, in order (times[idx] from split_by_stim_contact)
    labels: the matching condition labels (classify_pulses labels[idx]); "" = unlabelled, left out
    pair_tail: passed to condition_windows; conditions.doubled_pair_tail for uh3001/uh3006, None for everyone else
    Steps:
    1. condition_windows: how long each condition's epochs are (to just before the next pulse). Second pulses of pairs
       get no window of their own, since they sit inside the first pulse's epoch
    2. label_pulses: renames the pulses by condition, once, so each condition can be cut on its own
    3. for each condition: make_epochs from 2 ms before each pulse to its window end, then average_condition
    Returns: {condition: MNE Evoked}, e.g. {"single": ..., "burst_1": ..., "paired_0.45_1": ...}. Each Evoked has the
        channel names, the time axis (0 = the pulse), nave = how many pulses were averaged, and comment = the condition
    """
    windows = condition_windows(times, labels, pair_tail=pair_tail) #{condition: tmax} eg. {"single": 0.0999, "burst_1": 0.0096, ...}
    labelled = label_pulses(raw, labels) # a copy of raw and its "stim" markers renamed "single", "burst_1", etc

    out = {}
    for condition, tmax in windows.items():
        epochs = make_epochs(labelled, tmin=-0.002, tmax=tmax, event=condition)
        data = average_condition(epochs)
        out[condition] = mne.EvokedArray(data, epochs.info, tmin=-0.002, nave=len(epochs), comment=condition)
    return out