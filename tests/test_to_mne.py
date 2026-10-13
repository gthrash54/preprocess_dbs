import h5py
import numpy as np
import pytest
from preprocess_dbs.to_mne import to_mne


def _fake_cleaned(path, fs=1000.0, seconds=3):
    #a tiny file laid out like a clean_dbs output: streams/<stream>/data, channel_names, attrs fs
    names = ["ecog_a1", "ecog_p1", "lead1_c1", "lead1_c2"]
    data = np.arange(len(names) * int(fs * seconds), dtype=np.float32).reshape(len(names), -1)
    with h5py.File(path, "w") as f:
        g = f.create_group("streams").create_group("ecos")
        g.create_dataset("data", data=data)
        g.create_dataset("channel_names", data=np.array(names, dtype="S"))
        g.attrs["fs"] = fs
    return data


def test_window_names_and_types(tmp_path):
    path = tmp_path / "block.h5"
    data = _fake_cleaned(path)
    raw = to_mne(path, "ecos", 1, 2)
    assert raw.info["sfreq"] == 1000.0
    assert raw.ch_names == ["ecog_a1", "ecog_p1", "lead1_c1", "lead1_c2"]
    assert raw.get_channel_types() == ["ecog", "ecog", "dbs", "dbs"]
    assert np.array_equal(raw.get_data(), data[:, 1000:2000]) #exactly seconds 1 to 2


def test_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        to_mne(tmp_path / "nope.h5", "ecos", 0, 1)


def test_brainvision_names_become_standard(tmp_path):
    #a BrainVision file calls the contacts DBS1..; the cleaned file's group/contact say they're lead 1, so they come out lead1_c1..
    path = tmp_path / "bv.h5"
    names = ["DBS1", "DBS2", "FDI_contra"]
    with h5py.File(path, "w") as f:
        g = f.create_group("streams").create_group("lead")
        g.create_dataset("data", data=np.zeros((3, 100), dtype=np.float32))
        g.create_dataset("channel_names", data=np.array(names, dtype="S"))
        g.create_dataset("channel_group", data=np.array(["lead1", "lead1", "emg"], dtype="S"))
        g.create_dataset("channel_contact", data=np.array(["1", "2", ""], dtype="S"))
        g.attrs["fs"] = 1000.0
    raw = to_mne(path, "lead", 0, 0.1)
    assert raw.ch_names == ["lead1_c1", "lead1_c2", "FDI_contra"]
    assert raw.get_channel_types() == ["dbs", "dbs", "emg"]
