import h5py
from pathlib import Path
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


def test_window_outside_the_recording(tmp_path):
    path = tmp_path / "block.h5"
    _fake_cleaned(path) #3 s long
    with pytest.raises(ValueError, match="outside"):
        to_mne(path, "ecos", 2, 5)


def test_pulses_become_annotations(tmp_path):
    path = tmp_path / "block.h5"
    _fake_cleaned(path) #3 s at 1000 Hz
    with h5py.File(path, "a") as f:
        s = f.create_group("stim")
        s.create_dataset("time", data=np.array([0.5, 1.25, 1.5, 2.5]))
        s.create_dataset("voice", data=np.array(["A"] * 4, dtype="S"))
        s.create_dataset("confirmed", data=np.ones(4, dtype=np.int8))
    raw = to_mne(path, "ecos", 1, 2)
    assert np.allclose(raw.annotations.onset, [0.25, 0.5]) #only the pulses inside 1-2 s, measured from the window start
    assert set(raw.annotations.description) == {"stim"}


def test_no_stim_table_still_loads(tmp_path):
    path = tmp_path / "block.h5"
    _fake_cleaned(path) #has no stim table
    assert len(to_mne(path, "ecos", 0, 1).annotations) == 0


def test_stream_without_channel_names(tmp_path):
    #TDT EMG and mic streams have input numbers but no names
    path = tmp_path / "block.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("streams").create_group("emgg")
        g.create_dataset("data", data=np.zeros((2, 100), dtype=np.float32))
        g.create_dataset("channels", data=np.array([1, 2]))
        g.attrs["fs"] = 1000.0
    assert to_mne(path, "emgg", 0, 0.1).ch_names == ["emgg_1", "emgg_2"]


def test_do_not_use_channels_become_bads(tmp_path):
    path = tmp_path / "block.h5"
    _fake_cleaned(path)
    with h5py.File(path, "a") as f:
        g = f["streams"]["ecos"]
        g.create_dataset("channel_dnu", data=np.array([0, 1, 0, 1], dtype=np.int8))
        g.attrs["dnu_list"] = "lab sheet: x.xlsx"
    raw = to_mne(path, "ecos", 0, 1)
    assert raw.info["bads"] == ["ecog_p1", "lead1_c2"] and raw.ch_names == ["ecog_a1", "ecog_p1", "lead1_c1", "lead1_c2"] #marked, not dropped
    with h5py.File(path, "a") as f:
        f["streams"]["ecos"].attrs["dnu_list"] = "none"
        f["streams"]["ecos"]["channel_dnu"][...] = 0
    with pytest.warns(UserWarning, match="isn't on any do-not-use list"): #not listed: nothing known, not "none bad"
        assert to_mne(path, "ecos", 0, 1).info["bads"] == []


def test_old_file_without_flags_says_so(tmp_path):
    path = tmp_path / "block.h5"
    _fake_cleaned(path)
    with pytest.warns(UserWarning, match="before clean_dbs version 7"):
        assert to_mne(path, "ecos", 0, 1).info["bads"] == []


PRACTICE_V7 = Path.home() / "Code/clean_dbs/clean_dbs_final/out_v7/uh3021/stage1/increment1_depth1.h5"


@pytest.mark.skipif(not PRACTICE_V7.is_file(), reason="version 7 practice file not on this machine")
def test_uh3021_bads_from_the_lab_sheet():
    #the lab's sheet marks ECoG a4 and EMG input 3 do not use in this recording
    assert to_mne(PRACTICE_V7, "ecos", 10, 11).info["bads"] == ["ecog_a4"]
    assert to_mne(PRACTICE_V7, "emgg", 10, 11).info["bads"] == ["emgg_3"]
