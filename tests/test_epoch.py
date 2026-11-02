import h5py
import warnings
import numpy as np
import pytest
import mne
from pathlib import Path


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


UH3021 = Path.home() / "Code/clean_dbs/clean_dbs_final/out_v7/uh3021/stage1/increment1_depth1.h5"


@pytest.mark.skipif(not UH3021.is_file(), reason="uh3021 increment1_depth1 cleaned file not on this machine")


def test_uh3021_matches_the_lab():
    import mne
    from preprocess_dbs.pulses import classify_pulses
    from preprocess_dbs.epoch import condition_windows
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


def _three_pulses(extra=False):
    #1 s of zeros at 1 kHz with stim pulses at 0.1, 0.2 and 0.3 s (and optionally a non-stim marker at 0.25 s)
    raw = mne.io.RawArray(np.zeros((1, 1000)), mne.create_info(["lead1_c2"], 1000.0, "dbs"), verbose=False)
    onsets, names = [0.1, 0.2, 0.3], ["stim"] * 3
    if extra:
        onsets, names = onsets + [0.25], names + ["impedance"]
    raw.set_annotations(mne.Annotations(onset=onsets, duration=0, description=names))
    return raw


def test_label_pulses_names_by_condition_and_drops_unlabelled():
    from preprocess_dbs.epoch import label_pulses
    raw = _three_pulses()
    out = label_pulses(raw, ["single", "", "burst_1"])
    assert list(out.annotations.description) == ["single", "burst_1"]
    assert np.allclose(out.annotations.onset, [0.1, 0.3])
    assert list(raw.annotations.description) == ["stim"] * 3 #the original isn't changed


def test_label_pulses_keeps_other_annotations():
    #only the stim pulses are relabelled; a different marker stays where it was, in time order
    from preprocess_dbs.epoch import label_pulses
    out = label_pulses(_three_pulses(extra=True), ["single", "single", "burst_1"])
    assert list(out.annotations.description) == ["single", "single", "impedance", "burst_1"]
    assert np.allclose(out.annotations.onset, [0.1, 0.2, 0.25, 0.3])


def test_label_pulses_count_must_match():
    from preprocess_dbs.epoch import label_pulses
    with pytest.raises(ValueError, match=r"\(3\) does not match the number of labels \(2\)"):
        label_pulses(_three_pulses(extra=True), ["single", "single"]) #3 stim pulses (the extra marker isn't counted)


def test_label_pulses_then_epoch_one_condition():
    #end to end: after labelling, make_epochs cuts only the condition asked for
    from preprocess_dbs.epoch import label_pulses, make_epochs
    out = label_pulses(_three_pulses(), ["single", "burst_1", "single"])
    epochs = make_epochs(out, tmin=-0.002, tmax=0.05, event="single")
    assert len(epochs) == 2
    assert np.allclose(epochs.events[:, 0] / 1000.0, [0.1, 0.3]) #the two singles, not the burst


def test_label_pulses_after_crop_keeps_times_and_spans():
    #a Raw that doesn't start at sample 0 (cropped): pulses must stay on their samples, and a BAD_ span keeps its length
    from preprocess_dbs.epoch import label_pulses
    raw = mne.io.RawArray(np.zeros((1, 5000)), mne.create_info(["lead1_c2"], 1000.0, "dbs"), verbose=False)
    raw.set_meas_date(0)
    raw.set_annotations(mne.Annotations(onset=[1.0, 2.0, 2.5, 3.0], duration=[0, 0, 1.0, 0],
                                        description=["stim", "stim", "BAD_seg", "stim"], orig_time=raw.info["meas_date"]))
    raw.crop(tmin=0.5)
    before = mne.events_from_annotations(raw, event_id={"stim": 1}, verbose=False)[0][:, 0]
    out = label_pulses(raw, ["single", "single", "single"])
    after = mne.events_from_annotations(out, event_id={"single": 1}, verbose=False)[0][:, 0]
    assert list(after) == list(before)
    assert out.annotations.duration[list(out.annotations.description).index("BAD_seg")] == 1.0
