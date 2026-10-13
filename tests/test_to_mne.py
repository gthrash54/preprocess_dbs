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
