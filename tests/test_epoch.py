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
