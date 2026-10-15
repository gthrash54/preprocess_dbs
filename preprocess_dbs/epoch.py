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

def classify_pulses(counts, periods, times, voices=None, tol_ms=0.05):
    """
    Gives each pulse its condition label from the stim table: single, paired with its gap, or its position in a burst
    Input:
    counts: the stim table's "count" column, how many pulses are in the group each pulse belongs to
        (1 = single, 2 = paired, 10 = burst). Both pulses of a pair say 2, and the 10 probes after a burst also say 10.
        -1 means unknown, which is every BrainVision file so far (its pulses come from markers)
    periods: the stim table's "period_ms" column, the gap in ms the stimulator puts between pulses of a group.
        For a pair this is its interpulse interval
    times: the stim table's "time" column in seconds. Every group is checked against it (see below). Raises if a time
        is missing or the times are out of order. Two pulses at the same moment are allowed (two voices firing
        together), but on one voice they aren't checked here: condition_windows refuses them
    voices: the stim table's "voice" column, or None (default) for one voice. Each voice is labeled on its own: a table
        with several voices interleaves their pulses (crosstalk blocks fire A and B at the same moments)
    tol_ms: how far apart two periods, or a period and a measured gap, can be and still match (default 0.05 ms, about
        one sample at TDT's 24 kHz)
    Returns: a list of labels, one per pulse in the same order as the stim table: "single", "paired_<gap in ms>_<1 or 2>"
        (e.g. "paired_2.89_1" and "paired_2.89_2", first and second pulse of a pair), "burst_1" to "burst_10",
        "burstprobes_1" to "burstprobes_20" (a burst, then 10 slower probes right after it), or "" for a pulse it can't
        label. Warns with how many pulses were left unlabeled and why
    A group only counts when it checks out against the times as well as the counts: its pulses all say that count and
    the same period, they really are one period apart, and nothing (on any voice) comes one period after its last
    pulse or one period before its first, which would mean the train has more pulses than its count or another voice
    pairs with it. A pulse whose group doesn't check out is left unlabeled and the walk moves on one row. Rows past the
    end of the table don't count against a group, so one cut off at the end is labeled as far as it goes, as long as at
    least two of its pulses are there to check. Burst + probes needs the next rows to be a group of 10 at a slower
    rate, starting within 1.5 probe periods of the burst's last pulse. A single is checked the same way, against its
    logged period, which for a single is a leftover setting (49.99 ms in uh3021). So singles delivered exactly one such
    period apart look like a train and are left unlabeled (256 in uh3005's crosstalk assays), and so are groups that
    follow each other with a gap equal to their own period: the timing can't tell those apart from one longer train.
    Pair names round the gap to 2 decimals, so gaps that round to the same value share a name
    Works on TDT stim tables whose count and period describe each voice's groups (the uh increment blocks). With more
    than one voice it warns, because the labels are per voice: a crosstalk condition depends on both voices (which
    carry current, whether one follows the other), so crosstalk needs its own step, not built yet. Left unlabeled,
    with a warning:
        - BrainVision tables (count -1): their groups have to come from somewhere else first (a sidecar condition
          column, a TTL code or the pulse timing), not built yet
        - the first pulse of a pair split across voices (crosstalk: an A pulse, then a B pulse one gap later). The
          second one usually logs an unrelated period and still comes out "single"
        - tables whose count and period don't match the pulse times (uh3001's crosstalk blocks)
    Not the same as the lab's MATLAB code (parse.m), so its labels won't line up one to one with the lab's output:
        - pairs are named by the stimulator's own gap ("paired_0.45"), not by the lab's lookup names (pp1, pp5 ... pp23),
          which come from the measured gap minus 0.1 ms. Any gap works, with no lookup table and no "CHECK"
        - the lab labels each stim-contact segment on its own; this labels the whole table. Same labels as long as no
          group runs from one segment into the next, as in uh3021
        - the lab's preprocess.m stops at 2601 pulses, so in uh3021 depth1 (3060 pulses) it never labels the last 459.
          This labels them all
        - the lab's loop stops one short, so a group that starts on the last pulse of a segment (e.g. a single) goes
          unlabeled there. This labels it
        - the lab takes a group's count on trust and never looks at the times, so one odd row shifts every label after
          it. Unknown counts are skipped where the lab's loop hangs, and a burst cut off by the end of the table is
          labeled as far as it goes where the lab's code errors
        - the lab calls any 10 count-10 rows after a burst probes, so two bursts in a row become burst + probes there
    """
    counts = np.asarray(counts, dtype=float)
    periods = np.asarray(periods, dtype=float)
    times = np.asarray(times, dtype=float)
    if not (len(counts) == len(periods) == len(times)):
        raise ValueError(f"{len(counts)} counts, {len(periods)} periods and {len(times)} times: all three have to be "
                         "columns of the same stim table")
    if not (np.isfinite(tol_ms) and tol_ms > 0):
        raise ValueError(f"tol_ms has to be a positive number of ms, not {tol_ms}")
    if not np.all(np.isfinite(times)):
        raise ValueError("times has missing or infinite values, so the groups can't be checked")
    if np.any(np.diff(times) < 0):
        raise ValueError("times are out of order: use the stim table's own order, earliest pulse first")
    if voices is None:
        voices = [""] * len(counts) #every pulse is the same voice
    voices = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in voices]) #the file stores voices as bytes (b"A")
    if len(voices) != len(counts):
        raise ValueError(f"{len(voices)} voices but {len(counts)} counts: both have to be columns of the same stim table")
    every_pulse = np.sort(times[np.isfinite(times)]) #all voices, for "nothing comes one period after the group"
    labels = [""] * len(counts)
    for voice in dict.fromkeys(voices.tolist()): #each voice on its own, in the order they first appear
        rows = np.flatnonzero(voices == voice) #this voice's rows, in table order
        for row, label in zip(rows, _label_one_voice(counts[rows], periods[rows], times[rows], every_pulse, tol_ms)):
            labels[row] = label #back to the row it came from
    if len(set(voices.tolist())) > 1:
        warnings.warn(f"{len(set(voices.tolist()))} voices, labeled one at a time: conditions that involve more than one "
                      "voice (pulses on both at once, one voice's pulse followed by the other's) aren't in these labels")
    unlabeled = np.array([label == "" for label in labels], dtype=bool)
    if unlabeled.any(): #say how many and why, so an empty result is never silent
        no_count = unlabeled & ((counts == -1) | ~np.isfinite(counts))
        no_rule = unlabeled & ~no_count & ~np.isin(counts, [1, 2, 10])
        odd = unlabeled & ~no_count & ~no_rule
        why = []
        if no_count.any():
            why.append(f"{no_count.sum()} have no count (-1, e.g. BrainVision markers): their groups have to come from elsewhere")
        if no_rule.any():
            why.append(f"{no_rule.sum()} have a count with no rule yet ({', '.join(f'{c:g}' for c in sorted(set(counts[no_rule])))})")
        if odd.any():
            why.append(f"{odd.sum()} belong to a group that doesn't check out (its count or period disagrees with its "
                       "neighbors or with the pulse times, e.g. a pair split across voices)")
        warnings.warn(f"{unlabeled.sum()} of {len(labels)} pulses left unlabeled: " + "; ".join(why))
    return labels

def _label_one_voice(counts, periods, times, every_pulse, tol_ms):
    #the walk through one voice's pulses: m is always the first row of the next group
    labels = [""] *len(counts)
    m = 0
    while m < len(counts):
        if counts[m] == 1 and _is_group(counts, periods, times, every_pulse, m, 1, tol_ms):
            labels[m] = "single" 
            m += 1 
        elif counts[m] == 2 and _is_group(counts, periods, times, every_pulse, m, 2, tol_ms):
            for p in range(min(2,len(counts)-m)):
                labels[m+p] = f"paired_{periods[m]:.2f}_{p+1}"
            m += 2
        elif counts[m] == 10 and _is_group(counts, periods, times, every_pulse, m, 10, tol_ms):
            if _probes_follow(counts, periods, times, every_pulse, m, tol_ms):
                for k in range(min(20,len(counts) - m)):
                    labels[m+k] = f"burstprobes_{k+1}"
                m += 20
            else:
                for k in range(min(10, len(counts) - m)): #a burst cut off by the end of the table: label only the pulses that are there
                    labels[m+k] = f"burst_{k+1}"
                m += 10
        else:
            m += 1 #an unknown count, or a group that doesn't check out: left unlabeled, and the next row gets its own chance
    return labels

def _is_group(counts, periods, times, every_pulse, m, n, tol_ms, check_before=True):
    #True when rows m to m+n-1 (the ones inside the table) are one real group of n: they all say count n and one known
    #period, they sit one period apart, and nothing on any voice comes one period after the last of them or one period
    #before the first (a train with more pulses than its count, or another voice pairing with it)
    if m >= len(counts) or not (np.isfinite(periods[m]) and periods[m] > 0):
        return False
    rows = slice(m, min(m + n, len(counts)))
    if n > 1 and rows.stop - m < 2: #a group cut off after its first pulse: nothing to check it against
        return False
    period = periods[m]
    if not (np.all(counts[rows] == n) and np.all(np.abs(periods[rows] - period) <= tol_ms)):
        return False
    if not np.all(np.abs(np.diff(times[rows]) * 1000 - period) <= tol_ms): #the group's own gaps, in ms
        return False
    places = [times[rows][-1] + period / 1000] #where the next pulse would be if the train went on
    if check_before:
        places.append(times[m] - period / 1000) #and where the one before would be
    for at in places:
        first, last = np.searchsorted(every_pulse, [at - tol_ms / 1000, at + tol_ms / 1000])
        if first != last: #a pulse there
            return False
    return True

def _probes_follow(counts, periods, times, every_pulse, m, tol_ms):
    #True when the 10 rows after the burst starting at row m are its probes: a real group of 10 at a slower rate that
    #starts right after the burst (within 1.5 probe periods), not a second burst after a pause
    p = m + 10
    if not _is_group(counts, periods, times, every_pulse, p, 10, tol_ms, check_before=False) or periods[p] <= periods[m] + tol_ms:
        return False
    return bool((times[p] - times[p - 1]) * 1000 <= 1.5 * periods[p])

def condition_windows(times, labels, max_gap=2.0):
    """
    Measures how long each condition's epochs can be: from its pulse to just before the next pulse
    Input:
    times: the stim table's "time" column in seconds, the full table, so times[k] and labels[k] are the same pulse.
        Not stim_times(), which can drop pulses (unconfirmed, other voices) whose artifacts are still in the signal
    labels: the labels from classify_pulses, one per row of the same stim table
    max_gap: a gap longer than this (seconds, default 2) is a pause, not a next pulse (14 s and 21 s between stim contacts
        in uh3021), so the pulse before it is skipped like the last pulse of the table
    Returns: a dictionary {label: window end in seconds after the pulse}, to use as make_epochs' tmax for that condition.
        Each window ends 0.1 ms before the next pulse in the table. A pair's window starts at its first pulse and ends
        0.2 ms before the pulse after the pair (the lab takes 0.1 ms off each of the pair's two gaps), so it holds both
        pulses, and second pulses ("paired_<gap>_2") get no window of their own. Each condition uses its shortest gap,
        so no epoch runs into the next pulse. Unlabeled pulses ("") are skipped. With tmin = -0.002 this gives the lab's
        epoch lengths in uh3021 (all 41 conditions, e.g. single 4977 samples, 0.45 ms pair 4984, burst_1 469)
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
        - the lab gives the last pulse of a segment a fixed length and skips the very last pulse of the last plain
          burst. Here a pulse followed by a pause (over max_gap) or by nothing is skipped
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
        if label.startswith("paired") and label.endswith("_2"):
            continue
        first_of_pair = label.startswith("paired") and label.endswith("_1")
        if first_of_pair and k+1 < len(times) and labels[k+1] != label[:-1] + "2":
            raise ValueError(f"another pulse falls between the two pulses of {label} at row {k} (e.g. a second voice): "
                             "which pulses should end a window isn't decided yet")
        if first_of_pair:
            next_row = k+2
        else:
            next_row = k+1
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
    wanted = {label for label in labels if label and not (label.startswith("paired") and label.endswith("_2"))}
    missing = sorted(wanted - set(windows) - too_close)
    if missing:
        warnings.warn(f"no window for {missing}: none of their pulses has a next pulse within max_gap ({max_gap} s)")
    return windows
