import mne
import numpy as np
import pytest
from preprocess_dbs.rereference import bipolar_names, rereference


def test_pairs_stay_on_one_level():
    #2-4 are the lower segments and 5-7 the upper ones; a pair never crosses levels
    anode, cathode = bipolar_names("lead1")
    assert list(zip(anode, cathode)) == [("lead1_c3", "lead1_c2"), ("lead1_c4", "lead1_c3"), ("lead1_c2", "lead1_c4"),
                                         ("lead1_c6", "lead1_c5"), ("lead1_c7", "lead1_c6"), ("lead1_c5", "lead1_c7")]


def test_pair_is_anode_minus_cathode():
    names = ["ecog_a1"] + [f"lead1_c{k}" for k in range(1, 9)]
    data = np.random.default_rng(0).normal(size=(len(names), 500))
    raw = mne.io.RawArray(data, mne.create_info(names, 1000.0, ["ecog"] + ["dbs"] * 8), verbose=False)
    bip = rereference(raw, ["lead1"])
    assert np.allclose(bip.get_data(picks="lead1_c3-lead1_c2")[0], data[3] - data[2]) #the lab's dbs_3_2 = c3 - c2 (row 3 is c3, row 2 is c2)
    assert {"ecog_a1", "lead1_c1", "lead1_c8"} <= set(bip.ch_names) #rings and ECoG are left alone
    assert raw.ch_names == names #the original Raw isn't changed


def test_missing_lead_is_skipped_with_a_warning():
    names = [f"lead1_c{k}" for k in range(1, 9)] #single-lead patient: no lead2 channels
    raw = mne.io.RawArray(np.zeros((8, 100)), mne.create_info(names, 1000.0, "dbs"), verbose=False)
    with pytest.warns(UserWarning, match="lead2"):
        bip = rereference(raw, ["lead1", "lead2"])
    assert sum("-" in n for n in bip.ch_names) == 6
    with pytest.raises(ValueError):
        rereference(raw, ["lead2"])


def test_other_pairs_for_a_ring_lead():
    names = [f"lead1_c{k}" for k in range(1, 5)] #4-ring lead
    raw = mne.io.RawArray(np.zeros((4, 100)), mne.create_info(names, 1000.0, "dbs"), verbose=False)
    bip = rereference(raw, ["lead1"], pairs=[(1, 2), (2, 3), (3, 4)])
    assert bip.ch_names == ["lead1_c1-lead1_c2", "lead1_c2-lead1_c3", "lead1_c3-lead1_c4"]
