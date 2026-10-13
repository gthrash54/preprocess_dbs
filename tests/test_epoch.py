import h5py
import numpy as np
import pytest
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
