import h5py
import warnings
import collections
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
        2.4 samples at TDT's 48.8 kHz)
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

def _pair_position(label):
    #1 or 2 for the first or second pulse of a pair label ("paired_2.89_1", "sham_paired_0.3_2"), otherwise None
    parts = label.split("_")
    if len(parts) >= 3 and parts[-3] == "paired" and parts[-1] in ("1", "2") and parts[:-3] in ([], ["sham"]):
        return int(parts[-1])
    return None

def condition_windows(times, labels, max_gap=2.0):
    """
    Measures how long each condition's epochs can be: from its pulse to just before the next pulse
    Input:
    times: the stim table's "time" column in seconds, the full table, so times[k] and labels[k] are the same pulse.
        Not stim_times(), which can drop pulses (unconfirmed, other voices) whose artifacts are still in the signal
    labels: the labels from classify_pulses or label_by_timing, one per row of the same stim table
    max_gap: a gap longer than this (seconds, default 2) is a pause, not a next pulse (14 s and 21 s between stim contacts
        in uh3021), so the pulse before it is skipped like the last pulse of the table
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
def group_by_gaps(times, max_gap_ms=40.0):
    """
    Splits pulses into groups by the gaps between them: a long gap starts a new group
    Input:
    times: pulse times in seconds, in table order. For BrainVision use the stim table's "time_clock" column, not "time":
        BrainVision glues the data together after a recording dropout, so on "time" a gap across a dropout looks too short
    max_gap_ms: a gap longer than this starts a new group (default 40 ms, which suits UG: its longest common gap inside a
        group is 33.5 ms and its trial clock is 50 ms; only garbled stretches near dropouts have gaps in between)
    Returns: a list of group numbers, one per pulse, counting up from 0
    """
    if not (np.isfinite(max_gap_ms) and max_gap_ms > 0):
        raise ValueError(f"max_gap_ms has to be a positive number of ms, not {max_gap_ms}")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("times have to be known and in order, earliest pulse first")
    groups = [0] * len(times)
    group = 0
    for k in range(1, len(times)):
        gap_ms = (times[k]-times[k-1])*1000
        if gap_ms > max_gap_ms:
            group += 1
        groups[k] = group
    return groups

def label_by_timing(times, clock_times=None, gap_times=(), amplitudes=None, max_gap_ms=40.0, near_gap_s=1.0, burst_tol_ms=0.15,
                    min_groups=5):
    """
    Labels each pulse's condition from the timing alone, for stim tables with no stimulator count (BrainVision, e.g. UG)
    Input:
    times: the stim table's "time" column in seconds (the recording's own time axis, the one epochs are cut on)
    clock_times: the stim table's "time_clock" column (BrainVision). Groups are found on it, because on "time" a gap
        across a recording dropout looks too short (see group_by_gaps). None (default): use times
    gap_times: when the real recording dropouts happen, on the "time" axis: flags/new_segments/time where is_gap is 1
    amplitudes: each pulse's current (uA, or mV where that's all a file has), NaN where unknown, or None. A group whose
        pulses all carry zero current is a sham trial
    max_gap_ms: a longer gap starts a new group (default 40 ms, see group_by_gaps)
    near_gap_s: a group with a pulse this close to a recording dropout is left unlabeled (default 1 s): around a dropout
        the markers come out garbled, as merged or cut-off groups
    burst_tol_ms: how far a burst's gaps can stray from their middle value and still count as even (default 0.15 ms,
        about one sample at UG's 10 kHz)
    min_groups: a condition with fewer groups than this, counted after the other rules, is left unlabeled as a likely
        glitch (default 5: UG's designed conditions repeat 41 times or more, while a few one-off "pairs" at 7.2, 29.7 and
        38.8 ms sit in garbled stretches)
    Returns: a list of labels, one per pulse in the same order: "single", "paired_<gap in ms, to 0.1 ms>_<1 or 2>" (e.g.
        "paired_0.3_1"), "burst_1" to "burst_10", any of these with "sham_" in front when the whole group carries zero
        current, or "" for a pulse it can't label. Warns with how many pulses were left unlabeled and why, and when there
        are no amplitudes (then sham trials can't be told apart from real ones)
    A burst is any 10 evenly spaced pulses inside one group: there's no check of the rate itself (UG's are 7.7 ms).
    Left unlabeled: groups of any other size, 10-pulse groups whose gaps aren't even, groups with two pulses at the same
    moment, groups near a recording dropout,
    the first and last group of the table (the recording may have cut them off), groups that mix zero and nonzero
    current (or zero and unknown), and conditions seen in fewer than min_groups groups
    Built for UG's ERP blocks. Check max_gap_ms on any other block type first (UG's sdcurve and lfp blocks are built
    differently), and filter out marker codes that aren't stim pulses before calling it
    """
    times = np.asarray(times, dtype=float)
    clock = times if clock_times is None else np.asarray(clock_times, dtype=float)
    if len(clock) != len(times) or (amplitudes is not None and len(amplitudes) != len(times)):
        raise ValueError("times, clock_times and amplitudes have to be columns of the same stim table (same length)")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(clock))):
        raise ValueError("times or clock_times has missing or infinite values, so the gaps can't be measured")
    if np.any(np.diff(clock) < 0) or np.any(np.diff(times) < 0):
        raise ValueError("times are out of order: use the stim table's own order, earliest pulse first")
    for name, value in (("near_gap_s", near_gap_s), ("burst_tol_ms", burst_tol_ms)):
        if not (np.isfinite(value) and value >= 0):
            raise ValueError(f"{name} has to be a number of 0 or more, not {value}")
    if not (isinstance(min_groups, (int, np.integer)) and min_groups >= 1):
        raise ValueError(f"min_groups has to be a whole number of 1 or more, not {min_groups}")
    if not np.all(np.isfinite(np.asarray(gap_times, dtype=float))):
        raise ValueError("gap_times has missing values (NaN)")
    if len(times) == 0:
        return []
    current = np.full(len(times), np.nan) if amplitudes is None else np.asarray(amplitudes, dtype=float)
    dropouts = np.sort(np.asarray(gap_times, dtype=float))
    groups = np.asarray(group_by_gaps(clock, max_gap_ms))
    near_split = np.abs(np.diff(clock) * 1000 - max_gap_ms) <= 0.125 * max_gap_ms #gaps within 12.5% of the split (35-45 ms)
    if near_split.any():
        warnings.warn(f"{near_split.sum()} gaps fall close to max_gap_ms ({max_gap_ms:g} ms), so the split between groups may "
                      "not be clean in this file: check them (in UG they sit in garbled stretches near dropouts)")
    starts = np.flatnonzero(np.r_[True, np.diff(groups) != 0]) #first row of each group
    ends = np.r_[starts[1:], len(times)] #one past its last row
    labels = [""] * len(times)
    why = collections.Counter()
    for i, (a, b) in enumerate(zip(starts, ends)):
        size = b - a
        if i == 0 or i == len(starts) - 1:
            why["are in the first or last group of the table (may be cut off)"] += size
            continue
        lo, hi = np.searchsorted(dropouts, [times[a] - near_gap_s, times[b - 1] + near_gap_s]) #dropouts near the group
        if hi > lo:
            why[f"are within {near_gap_s:g} s of a recording dropout"] += size
            continue
        gaps = np.diff(clock[a:b]) * 1000 #the group's own gaps, in ms
        if np.any(gaps < 0.05): #two pulses at the same moment (a marker logged twice?), not a pair
            why["are in a group with two pulses at the same moment"] += size
            continue
        if size == 1:
            names = ["single"]
        elif size == 2:
            names = [f"paired_{gaps[0]:.1f}_1", f"paired_{gaps[0]:.1f}_2"]
        elif size == 10 and np.all(np.abs(gaps - np.median(gaps)) <= burst_tol_ms):
            names = [f"burst_{k}" for k in range(1, 11)]
        else:
            why["are in a group that isn't a single, a pair or an even 10-pulse burst"] += size
            continue
        if np.all(current[a:b] == 0): #no current in the whole group: a sham trial
            names = ["sham_" + name for name in names]
        elif np.any(current[a:b] == 0):
            why["are in a group that mixes zero and nonzero current"] += size
            continue
        labels[a:b] = names
    groups_seen = collections.Counter() #how many groups each condition has
    for label in labels:
        if label in ("single", "sham_single"):
            groups_seen[label] += 1
        elif label.endswith("_1"): #a pair or burst, counted once by its first pulse ("burst_10" ends in "_10")
            groups_seen[label.rsplit("_", 1)[0]] += 1
    for k, label in enumerate(labels):
        condition = label if label in ("single", "sham_single") else label.rsplit("_", 1)[0]
        if label and groups_seen[condition] < min_groups:
            labels[k] = ""
            why[f"are in a condition seen in fewer than {min_groups} groups (likely a glitch)"] += 1
    if why:
        warnings.warn(f"{sum(why.values())} of {len(labels)} pulses left unlabeled: " +
                      "; ".join(f"{n} {reason}" for reason, n in why.most_common()))
    if np.all(np.isnan(current)):
        warnings.warn("no amplitudes, so sham (zero-current) trials can't be told apart from real ones")
    elif np.any(np.isnan(current)):
        unknown = sum(1 for k, label in enumerate(labels) if label and np.isnan(current[k]))
        if unknown:
            warnings.warn(f"{unknown} labeled pulses have no amplitude, so they could be sham trials labeled as real ones")
    return labels
