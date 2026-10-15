import h5py
import warnings
import numpy as np
import pytest
from pathlib import Path
from preprocess_dbs.epoch import stim_times


def _fake_stim(path, with_table=True):
    #a small cleaned file with 4 pulses: voices A, A, B, A and one unconfirmed
    with h5py.File(path, "w") as f:
        if with_table:
            s = f.create_group("stim")
            s.create_dataset("time", data=np.array([1.0, 1.5, 2.0, 2.5]))
            s.create_dataset("voice", data=np.array(["A", "A", "B", "A"], dtype="S"))
            s.create_dataset("confirmed", data=np.array([1, 1, 1, 0]))


def test_filters(tmp_path):
    path = tmp_path / "b.h5"
    _fake_stim(path)
    assert list(stim_times(path)) == [1.0, 1.5, 2.0] #unconfirmed pulse dropped
    assert list(stim_times(path, confirmed_only=False)) == [1.0, 1.5, 2.0, 2.5]
    assert list(stim_times(path, voice="A")) == [1.0, 1.5]
    assert list(stim_times(path, voice="C")) == []


def test_no_stim_table_says_so(tmp_path):
    path = tmp_path / "b.h5"
    _fake_stim(path, with_table=False)
    with pytest.raises(KeyError, match="no stim table"):
        stim_times(path)


def test_dropping_unconfirmed_pulses_warns(tmp_path):
    path = tmp_path / "b.h5"
    _fake_stim(path)
    with pytest.warns(UserWarning, match="1 of 4"):
        stim_times(path)


def _raw_with_events(onsets, desc="stim", seconds=1.0):
    import mne
    raw = mne.io.RawArray(np.zeros((2, int(1000 * seconds))), mne.create_info(["lead1_c2", "lead1_c3"], 1000.0, "dbs"), verbose=False)
    raw.set_annotations(mne.Annotations(onsets, 0, desc))
    return raw


def test_epochs_window_and_event_choice():
    from preprocess_dbs.epoch import make_epochs
    raw = _raw_with_events([0.2, 0.5], ["stim", "cue"])
    ep = make_epochs(raw) #only the "stim" one
    assert len(ep) == 1 and ep.get_data().shape == (1, 2, 28) #-2 to +25 ms at 1000 Hz = 28 samples
    assert len(make_epochs(raw, event="cue", tmin=-0.1, tmax=0.1)) == 1


def test_no_events_and_edge_drops():
    from preprocess_dbs.epoch import make_epochs
    with pytest.raises(ValueError, match="no \"stim\""):
        make_epochs(_raw_with_events([0.5], "cue"))
    with pytest.warns(UserWarning, match="1 of 2"):
        ep = make_epochs(_raw_with_events([0.5, 0.99])) #0.99 s + 25 ms runs past the end
    assert len(ep) == 1


def test_select_pulses_by_gaps():
    from preprocess_dbs.epoch import select_pulses
    #a pair (0.100, 0.103), a single pulse (0.300) and a cue that must be left alone
    raw = _raw_with_events([0.1, 0.103, 0.3, 0.5], ["stim", "stim", "stim", "cue"])
    first = select_pulses(raw) #first of each group
    assert list(first.annotations.onset) == [0.1, 0.3, 0.5]
    single = select_pulses(raw, gap_after=0.025) #whole epoch clean: only the lone pulse
    assert [o for o, d in zip(single.annotations.onset, single.annotations.description) if d == "stim"] == [0.3]
    assert len(raw.annotations) == 4 #original untouched


def _groups(*groups):
    #stim table columns (counts, periods, times) for groups back to back. Each group is (count, period_ms, pulses) or
    #(count, period_ms, pulses, ms since the previous group's last pulse), default 100 ms of quiet in between. A group's
    #pulses are one period apart, so singles need one group each
    counts, periods, times, t = [], [], [], -0.1
    for group in groups:
        count, period, n = group[:3]
        t += (group[3] if len(group) > 3 else 100.0) / 1000
        for i in range(n):
            counts.append(count); periods.append(period); times.append(t + i * period / 1000)
        t = times[-1]
    return np.array(counts), np.array(periods), np.array(times)


def test_classify_pulses():
    from preprocess_dbs.epoch import classify_pulses
    #the first 43 rows of uh3021 increment1_depth1: pairs, a single, a burst, more pairs, then a burst + probes
    times = np.array([0.0, 0.00288768, 0.10287104, 0.1086464, 0.20865024, 0.30865408, 0.31633408, 0.41633792, 0.42307584,
                      0.5230592, 0.5307392, 0.5384192, 0.5460992, 0.5537792, 0.5614592, 0.5691392, 0.5768192, 0.5844992,
                      0.5921792, 0.84226048, 0.84322304, 0.94322688, 0.95860736, 1.0586112, 1.0662912, 1.0739712,
                      1.0816512, 1.0893312, 1.0970112, 1.1046912, 1.1123712, 1.1200512, 1.1277312, 1.15089408,
                      1.17395456, 1.19701504, 1.22007552, 1.243136, 1.26619648, 1.28925696, 1.31231744, 1.33537792,
                      1.3584384])
    counts = np.array([2, 2, 2, 2, 1, 2, 2, 2, 2] + [10] * 10 + [2, 2, 2, 2] + [10] * 20)
    periods = np.array([2.88767982, 2.88767982, 5.77535963, 5.77535963, 49.99167633, 7.67999935, 7.67999935, 6.73791981,
                        6.73791981] + [7.67999935] * 10 + [0.96255994, 0.96255994, 15.38047886, 15.38047886] +
                       [7.67999935] * 10 + [23.06047821] * 10)
    assert classify_pulses(counts, periods, times) == (
        ["paired_2.89_1", "paired_2.89_2", "paired_5.78_1", "paired_5.78_2", "single", "paired_7.68_1", "paired_7.68_2",
         "paired_6.74_1", "paired_6.74_2"] + [f"burst_{k}" for k in range(1, 11)] +
        ["paired_0.96_1", "paired_0.96_2", "paired_15.38_1", "paired_15.38_2"] + [f"burstprobes_{k}" for k in range(1, 21)])


def test_classify_pulses_edges():
    from preprocess_dbs.epoch import classify_pulses
    assert classify_pulses([], [], []) == [] #no pulses
    assert classify_pulses(*_groups((1, 49.99, 1), (1, 49.99, 1))) == ["single", "single"] #back-to-back singles both labeled
    with pytest.warns(UserWarning, match=r"no rule yet \(5\)"): #unknown counts skipped, no hang
        assert classify_pulses(*_groups((5, 4.0, 2), (1, 50.0, 1))) == ["", "", "single"]
    #a burst cut off by the end of the table: labeled as far as it goes, no crash
    assert classify_pulses(*_groups((1, 50.0, 1), (10, 7.68, 3))) == ["single", "burst_1", "burst_2", "burst_3"]


def test_classify_bursts_and_probes():
    from preprocess_dbs.epoch import classify_pulses
    burst, probes = (10, 7.67999935, 10), (10, 23.06047821, 10, 23.16288) #probes start 23.16 ms after the burst, as in uh3021
    assert classify_pulses(*_groups(burst, probes)) == [f"burstprobes_{k}" for k in range(1, 21)]
    #two bursts at the same rate stay two bursts (the lab's count-only rule would call them burst + probes)
    assert classify_pulses(*_groups(burst, burst)) == [f"burst_{k}" for k in range(1, 11)] * 2
    #a slower burst after a 250 ms pause is a second burst, not probes
    assert classify_pulses(*_groups(burst, (10, 15.36, 10, 250.0))) == [f"burst_{k}" for k in range(1, 11)] * 2
    #burst + probes running straight into a plain burst (3 times in uh3021, found by Muse): 30 count-10 rows in a row
    assert classify_pulses(*_groups(burst, probes, burst)) == \
        [f"burstprobes_{k}" for k in range(1, 21)] + [f"burst_{k}" for k in range(1, 11)]
    #burst + probes cut off by the end of the table: labeled as far as it goes
    assert classify_pulses(*_groups(burst, probes[:2] + (5,) + probes[3:])) == [f"burstprobes_{k}" for k in range(1, 16)]


def test_classify_pair_positions():
    from preprocess_dbs.epoch import classify_pulses
    #two pairs with the same gap back to back (3 times in uh3021): positions show where one pair ends
    assert classify_pulses(*_groups((2, 1.925, 2), (2, 1.925, 2))) == ["paired_1.93_1", "paired_1.93_2"] * 2
    #a pair cut off after its first pulse can't be checked: left unlabeled, with a warning
    with pytest.warns(UserWarning, match="doesn't check out"):
        assert classify_pulses(*_groups((1, 50.0, 1), (2, 2.888, 1))) == ["single", ""]


def test_classify_groups_must_check_out():
    from preprocess_dbs.epoch import classify_pulses
    #a group only counts when its counts, periods and times agree; otherwise that pulse is left unlabeled and the walk
    #moves on one row (cases from Muse's reviews: before, each one shifted, invented or contradicted labels)
    with pytest.warns(UserWarning, match="doesn't check out"):
        assert classify_pulses([2, 1, 1], [2.0, 50, 50], [0.0, 0.002, 0.1]) == ["", "single", "single"]
    with pytest.warns(UserWarning, match="5 of 11"): #5 burst pulses where 10 were promised
        assert classify_pulses(*_groups((10, 7.68, 5), *[(1, 50.0, 1)] * 6)) == [""] * 5 + ["single"] * 6
    #a burst, then one lone pulse that says count 10, then singles: not burst + probes
    with pytest.warns(UserWarning, match="1 of 20"):
        labels = classify_pulses(*_groups((10, 7.68, 10), (10, 23.06, 1, 23.16), *[(1, 50.0, 1)] * 9))
    assert labels == [f"burst_{k}" for k in range(1, 11)] + [""] + ["single"] * 9
    #after an odd row the walk can't stitch a pair together from two different pairs: their pulses aren't 4 ms apart
    with pytest.warns(UserWarning, match="2 of 4"):
        assert classify_pulses([2, 2, 2, 2], [3.0, 4, 4, 4], [0.0, 0.004, 0.1, 0.104]) == ["", "", "paired_4.00_1", "paired_4.00_2"]
    #count and period that don't match the times (as in uh3001's crosstalk blocks): nothing is labeled, not wrong labels
    with pytest.warns(UserWarning, match="4 of 4"):
        assert classify_pulses([2, 2, 1, 1], [49.99, 4.81, 4.81, 4.81], [0.0, 0.0548, 0.1096, 0.11441]) == [""] * 4
    #rates that differ by float noise are the same rate: two bursts, not burst + probes
    assert classify_pulses(*_groups((10, 7.68, 10), (10, 7.68 + 1e-7, 10))) == [f"burst_{k}" for k in range(1, 11)] * 2
    with pytest.warns(UserWarning, match="doesn't check out"): #no period, so no "paired_nan" labels
        assert classify_pulses([2, 2], [np.nan, np.nan], [0.0, 0.003]) == ["", ""]


def test_classify_says_why_pulses_are_unlabeled():
    from preprocess_dbs.epoch import classify_pulses
    with pytest.warns(UserWarning, match="3 have no count"): #BrainVision markers: count -1
        assert classify_pulses([-1] * 3, [np.nan] * 3, [0.0, 0.1, 0.2]) == [""] * 3
    with pytest.warns(UserWarning, match=r"no rule yet \(3\)"): #e.g. iTBS triplets
        assert classify_pulses(*_groups((3, 20.0, 3))) == [""] * 3
    with pytest.raises(ValueError, match="3 counts, 2 periods and 3 times"):
        classify_pulses([1, 1, 1], [50.0, 50.0], [0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="tol_ms"):
        classify_pulses([1], [50.0], [0.0], tol_ms=-1)
    with pytest.raises(ValueError, match="missing or infinite"): #times are checked here too, not only in condition_windows
        classify_pulses([1], [50.0], [np.nan])
    with pytest.raises(ValueError, match="out of order"):
        classify_pulses([1, 1], [50.0, 50.0], [0.2, 0.0])
    with pytest.warns(UserWarning, match=r"no rule yet \(1.5\)"):
        classify_pulses([1.5], [50.0], [0.0])


def test_classify_singles_one_logged_period_apart():
    from preprocess_dbs.epoch import classify_pulses
    #a single's logged period is a leftover setting (49.99 ms in uh3021): singles exactly that far apart look like a
    #train, so they're left unlabeled (uh3005's crosstalk assays have 256), never mislabeled. The docstring says so
    with pytest.warns(UserWarning, match="3 of 3"):
        assert classify_pulses([1, 1, 1], [50.0] * 3, [0.0, 0.05, 0.1]) == [""] * 3
    assert classify_pulses([1, 1, 1], [50.0] * 3, [0.0, 0.1, 0.2]) == ["single"] * 3 #100 ms apart, as in uh3021


def test_classify_each_voice_on_its_own():
    from preprocess_dbs.epoch import classify_pulses
    #a voice-A pair and a voice-B pair at the same moments (crosstalk blocks fire A and B together), then a B single;
    #voices come out of the file as bytes
    counts, periods = [2, 2, 2, 2, 1], [3.85, 3.85, 3.85, 3.85, 50.0]
    times = [0.0, 0.0, 0.00385, 0.00385, 0.1]
    with pytest.warns(UserWarning, match="2 voices"): #the labels are per voice, and it says so
        labels = classify_pulses(counts, periods, times, voices=[b"A", b"B", b"A", b"B", b"B"])
    assert labels == ["paired_3.85_1", "paired_3.85_1", "paired_3.85_2", "paired_3.85_2", "single"]
    #a pair split across voices (A, then B 4.79 ms later, each logged as count 1): A's pulse is caught and left
    #unlabeled; B's logs an unrelated period and still comes out "single", which the docstring says
    with pytest.warns(UserWarning):
        labels = classify_pulses([1, 1], [4.79, 45.02], [0.0, 0.00479], voices=["A", "B"])
    assert labels == ["", "single"]


def test_condition_windows():
    from preprocess_dbs.epoch import condition_windows
    #a pair (0.45 ms), a single, a 3-pulse burst start, an unknown count, then the last pulse with nothing after it
    times = [0.0, 0.00045, 0.1, 0.2, 0.20768, 0.21536, 0.3, 0.4]
    labels = ["paired_0.45_1", "paired_0.45_2", "single", "burst_1", "burst_2", "burst_3", "", "single"]
    w = condition_windows(times, labels)
    assert set(w) == {"paired_0.45_1", "single", "burst_1", "burst_2", "burst_3"} #no second pulses, no "", no window for the last pulse
    assert w["paired_0.45_1"] == pytest.approx(0.1 - 0.0002) #to the pulse after the pair, 0.2 ms short
    assert w["single"] == pytest.approx(0.1 - 0.0001) #the last single has no next pulse, so only the first one counts
    assert w["burst_1"] == pytest.approx(0.00768 - 0.0001)
    assert w["burst_3"] == pytest.approx(0.08464 - 0.0001) #to the unknown-count pulse: it's still a pulse with an artifact


def test_condition_windows_shortest_wins():
    from preprocess_dbs.epoch import condition_windows
    #the shortest gap wins whichever order the pulses come in (a long gap after one single must not stretch them all)
    assert condition_windows([0.0, 0.1, 1.5], ["single"] * 3)["single"] == pytest.approx(0.0999)
    assert condition_windows([0.0, 1.4, 1.5], ["single"] * 3)["single"] == pytest.approx(0.0999)
    with pytest.raises(ValueError, match="2 times but 3 labels"):
        condition_windows([0.0, 0.1], ["single"] * 3)


def test_condition_windows_needs_good_times():
    from preprocess_dbs.epoch import condition_windows
    with pytest.raises(ValueError, match="out of order"):
        condition_windows([1.0, 0.0, 0.5], ["single"] * 3)
    with pytest.raises(ValueError, match="NaN"):
        condition_windows([0.0, np.nan], ["single"] * 2)
    with pytest.raises(ValueError, match="same moment"): #two voices at once: not decided yet, so no window of 0
        condition_windows([0.0, 0.0, 0.1], ["single"] * 3)
    with pytest.raises(ValueError, match="between the two pulses"): #another voice inside a pair: not decided yet either
        condition_windows([0.0, 0.005, 0.010, 0.1], ["paired_10.00_1", "single", "paired_10.00_2", "single"])
    with pytest.raises(ValueError, match="max_gap"):
        condition_windows([0.0, 0.1], ["single"] * 2, max_gap=np.nan)


def test_condition_windows_warns_instead_of_going_quiet():
    from preprocess_dbs.epoch import condition_windows
    #a lone single before a 14 s pause between stim contacts: no 14 s window, and a warning says so
    with pytest.warns(UserWarning, match=r"no window for \['single'\]"):
        assert condition_windows([0.0, 14.0], ["single"] * 2) == {}
    assert condition_windows([0.0, 14.0], ["single"] * 2, max_gap=20)["single"] == pytest.approx(13.9999)
    with pytest.warns(UserWarning, match="none of the 3 pulses has a label"): #e.g. every BrainVision file so far
        assert condition_windows([0.0, 0.1, 0.2], [""] * 3) == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error") #a block with no stim: quietly nothing, any warning fails the test
        assert condition_windows([], []) == {}
    #a pulse 0.05 ms before the next: no window is safe for that condition, so it gets none instead of a wrong one
    with pytest.warns(UserWarning, match="less than 0.1 ms"):
        assert condition_windows([0.0, 0.1, 0.10005], ["single"] * 3) == {}


UH3021 = Path("/home/gwt/clean_dbs/clean_dbs_final/out_v3/uh3021/stage1/increment1_depth1.h5")


@pytest.mark.skipif(not UH3021.is_file(), reason="uh3021 increment1_depth1 cleaned file not on this machine")
def test_uh3021_matches_the_lab():
    import mne
    from preprocess_dbs.epoch import classify_pulses, condition_windows
    with h5py.File(UH3021, "r") as f:
        s = f["stim"]
        times, counts, periods, voices = s["time"][:], s["count"][:], s["period_ms"][:], s["voice"][:]
    labels = classify_pulses(counts, periods, times, voices=voices)
    totals = {}
    for label in labels:
        kind = "pair" if label.startswith("paired") else label.rsplit("_", 1)[0]
        totals[kind] = totals.get(kind, 0) + 1
    assert totals == {"single": 60, "pair": 1200, "burst": 600, "burstprobes": 1200} #every pulse labeled
    windows = condition_windows(times, labels)
    assert len(windows) == 41
    #every condition's epoch length in samples, the lab's way (spec 2.2: fs rounded to 48828, each epoch runs to 0.1 ms
    #before the next pulse, a pair's to 0.2 ms before the pulse after the pair, length = 98 + duration + 1), against
    #MNE's length for tmin = -0.002 and the window at the data's real rate
    lab = {}
    for k, label in enumerate(labels):
        if label.startswith("paired") and label.endswith("_2"):
            continue
        pair = label.startswith("paired")
        nxt = k + 2 if pair else k + 1
        if nxt >= len(times) or times[nxt] - times[k] > 2:
            continue
        duration = round(((times[nxt] - times[k]) * 1000 - (0.2 if pair else 0.1)) / 1000 * 48828)
        lab[label] = min(lab.get(label, 10**9), 98 + duration + 1)
    fs = 48828.125
    raw = mne.io.RawArray(np.zeros((1, int(fs))), mne.create_info(1, fs, "misc"), verbose=False)
    for label, n in lab.items():
        epochs = mne.Epochs(raw, np.array([[int(0.3 * fs), 0, 1]]), tmin=-0.002, tmax=windows[label], baseline=None, verbose=False)
        assert len(epochs.times) == n, label
    assert lab["single"] == 4977 and lab["paired_0.45_1"] == 4984 and lab["burst_1"] == 469 #the spec's own examples


def test_group_by_gaps():
    from preprocess_dbs.epoch import group_by_gaps
    #a 0.3 ms pair, a single 50 ms later, then a burst starting 50 ms after that (the lesson's toy)
    assert group_by_gaps([0.0, 0.0003, 0.0503, 0.1003, 0.1080, 0.1157]) == [0, 0, 1, 2, 2, 2]
    assert group_by_gaps([]) == []
    assert group_by_gaps([5.0]) == [0]
    assert group_by_gaps([0.0, 0.040, 0.0801]) == [0, 0, 1] #exactly 40 ms stays in the group; 40.1 ms starts a new one
    assert group_by_gaps([0.0, 0.040, 0.0801], max_gap_ms=30) == [0, 1, 2]


def _timing_table(groups, quiet_ms=50.0):
    #pulse times (s) for groups back to back: each group is a list of gaps in ms between its pulses ([] = a single),
    #with quiet_ms between groups; plus a leading and a trailing single, since the first and last group are never labeled
    times, t = [0.0], 0.0
    for gaps in [[]] + groups + [[]]:
        t += quiet_ms / 1000
        times.append(t)
        for g in gaps:
            t += g / 1000
            times.append(t)
    return np.array(times[1:])


def test_label_by_timing():
    from preprocess_dbs.epoch import label_by_timing
    groups = [[]] * 5 + [[0.3]] * 5 + [[7.7] * 9] * 5 #5 singles, 5 pairs 0.3 ms apart, 5 bursts of 10 at 7.7 ms
    times = _timing_table(groups)
    labels = label_by_timing(times)[1:-1] #the first and last group are left unlabeled (may be cut off)
    assert labels == ["single"] * 5 + ["paired_0.3_1", "paired_0.3_2"] * 5 + [f"burst_{k}" for k in range(1, 11)] * 5


def test_label_by_timing_sham_and_mixed_current():
    from preprocess_dbs.epoch import label_by_timing
    times = _timing_table([[]] * 10)
    current = np.array([3000.0] * 6 + [0.0] * 6) #first 6 singles live, last 6 zero current
    labels = label_by_timing(times, amplitudes=current)
    assert labels[1:5] == ["single"] * 4 and labels[6:11] == ["sham_single"] * 5
    #a pair whose two pulses differ, one live and one at zero, doesn't add up
    times = _timing_table([[0.8]] * 6)
    current = np.full(len(times), 3000.0)
    current[1 + 2 * 2] = 0.0 #second pulse of the third pair
    with pytest.warns(UserWarning, match="mixes zero and nonzero"):
        labels = label_by_timing(times, amplitudes=current)
    assert labels[5:7] == ["", ""]


def test_label_by_timing_leaves_out_what_it_cant_trust():
    from preprocess_dbs.epoch import label_by_timing
    #an uneven 10-pulse group, a 4-pulse group, and a one-off 38.8 ms "pair" among 5 real pairs
    times = _timing_table([[7.7] * 8 + [9.1]] + [[2.0, 2.0, 2.0]] + [[38.8]] + [[2.0]] * 5)
    with pytest.warns(UserWarning) as caught:
        labels = label_by_timing(times)
    assert labels[1:17] == [""] * 16 and labels[17:-1] == ["paired_2.0_1", "paired_2.0_2"] * 5
    message = " ".join(str(w.message) for w in caught)
    assert "isn't a single, a pair or an even 10-pulse burst" in message and "fewer than 5 groups" in message
    assert "close to max_gap_ms" in message #the 38.8 ms gap sits near the 40 ms split
    #a group within 1 s of a recording dropout is left out, however clean it looks
    times = _timing_table([[]] * 30, quiet_ms=300) #singles 300 ms apart, so only the ones near the dropout are close to it
    with pytest.warns(UserWarning, match="recording dropout"):
        labels = label_by_timing(times, gap_times=[times[15]])
    assert labels[15] == "" and labels[2] == "single"


def test_label_by_timing_groups_on_the_clock():
    from preprocess_dbs.epoch import label_by_timing
    #after a dropout BrainVision's own time axis loses the missing time, so two singles 50 ms apart on the clock can sit
    #10 ms apart on "time"; grouping on the clock keeps them apart
    clock = _timing_table([[]] * 8)
    times = clock.copy()
    times[5:] -= 0.040 #40 ms went missing before pulse 5
    with pytest.warns(UserWarning):
        assert label_by_timing(times, clock_times=clock, near_gap_s=0)[4:6] == ["single", "single"]
        assert label_by_timing(times, near_gap_s=0)[4:6] == ["", ""] #on "time" they'd merge into a 10 ms "pair" seen once
    with pytest.raises(ValueError, match="same length"):
        label_by_timing([0.0, 1.0], clock_times=[0.0])


def test_label_by_timing_checks_its_inputs():
    from preprocess_dbs.epoch import label_by_timing, group_by_gaps, condition_windows, _pair_position
    clock = _timing_table([[]] * 8)
    with pytest.raises(ValueError, match="out of order"): #a shuffled time column next to a sorted clock
        label_by_timing(clock[::-1].copy(), clock_times=clock)
    for bad in ({"near_gap_s": np.nan}, {"burst_tol_ms": -1}, {"min_groups": 0}, {"gap_times": [np.nan]}):
        with pytest.raises(ValueError):
            label_by_timing(clock, **bad)
    with pytest.raises(ValueError, match="max_gap_ms"):
        group_by_gaps(clock, max_gap_ms=-1)
    with pytest.raises(ValueError, match="in order"):
        group_by_gaps([0.2, 0.1])
    #two pulses at the same moment are a marker logged twice, not a 0.0 ms pair
    times = _timing_table([[0.0]] * 6)
    with pytest.warns(UserWarning, match="same moment"):
        assert set(label_by_timing(times)) == {""}
    #some amplitudes unknown: the labels stand, with a warning that some could be shams
    times = _timing_table([[]] * 10)
    current = np.full(len(times), 3000.0)
    current[4] = np.nan
    with pytest.warns(UserWarning, match="could be sham trials"):
        label_by_timing(times, amplitudes=current)
    #gaps close to the 40 ms split get a warning
    with pytest.warns(UserWarning, match="close to max_gap_ms"):
        label_by_timing(_timing_table([[38.0]] * 6))
    #pair labels are recognised exactly
    assert (_pair_position("paired_2.89_1"), _pair_position("sham_paired_0.3_2"), _pair_position("unpaired_9_1"),
            _pair_position("burst_1")) == (1, 2, None, None)
    assert condition_windows([0.0, 0.1, 0.2], ["unpaired_9_1"] * 3)["unpaired_9_1"] == pytest.approx(0.0999)
