import mne
import numpy as np
import pytest
from preprocess_dbs.rereference import (bipolar_names, dbs_rereference, horizontal_pairs, vertical_pairs, ring_pairs, c1_pairs,
                                        all_pairs, erna_pairs, add_virtual_rings, common_average,
                                        ecog_rereference)


def test_pairs_stay_on_one_level():
    #2-4 are the lower segments and 5-7 the upper ones; a pair never crosses levels
    anode, cathode = bipolar_names("lead1_c", horizontal_pairs)
    assert list(zip(anode, cathode)) == [("lead1_c3", "lead1_c2"), ("lead1_c4", "lead1_c3"), ("lead1_c2", "lead1_c4"),
                                         ("lead1_c6", "lead1_c5"), ("lead1_c7", "lead1_c6"), ("lead1_c5", "lead1_c7")]


def test_pair_is_anode_minus_cathode():
    names = ["ecog_a1"] + [f"lead1_c{k}" for k in range(1, 9)]
    data = np.random.default_rng(0).normal(size=(len(names), 500))
    raw = mne.io.RawArray(data, mne.create_info(names, 1000.0, ["ecog"] + ["dbs"] * 8), verbose=False)
    bip = dbs_rereference(raw, ["lead1"])
    assert np.allclose(bip.get_data(picks="lead1_c3-lead1_c2")[0], data[3] - data[2]) #the lab's dbs_3_2 = c3 - c2 (row 3 is c3, row 2 is c2)
    assert "ecog_a1" in bip.ch_names #ECoG is left alone
    #MNE removes the single contacts a pair uses (drop_refs), and the 21 pairs use every contact, so only pairs are left,
    #as in the lab's bipolar output. The common average has to be made from the original raw
    assert not any(n.startswith("lead1_c") and "-" not in n for n in bip.ch_names)
    assert raw.ch_names == names #the original Raw isn't changed


def test_missing_lead_is_skipped_with_a_warning():
    names = [f"lead1_c{k}" for k in range(1, 9)] #single-lead patient: no lead2 channels
    raw = mne.io.RawArray(np.zeros((8, 100)), mne.create_info(names, 1000.0, "dbs"), verbose=False)
    with pytest.warns(UserWarning, match="lead2"):
        bip = dbs_rereference(raw, ["lead1", "lead2"], pairs=horizontal_pairs)
    assert sum("-" in n for n in bip.ch_names) == 6
    with pytest.raises(ValueError):
        dbs_rereference(raw, ["lead2"], pairs=horizontal_pairs)


def test_other_pairs_for_a_ring_lead():
    names = [f"lead1_c{k}" for k in range(1, 5)] #4-ring lead
    raw = mne.io.RawArray(np.zeros((4, 100)), mne.create_info(names, 1000.0, "dbs"), verbose=False)
    bip = dbs_rereference(raw, ["lead1"], pairs=[(1, 2), (2, 3), (3, 4)])
    assert bip.ch_names == ["lead1_c1-lead1_c2", "lead1_c2-lead1_c3", "lead1_c3-lead1_c4"]


def test_pairs_with_a_bad_contact_are_skipped():
    names = ["lead1_c" + str(c) for c in range(1, 9)]
    raw = mne.io.RawArray(np.random.default_rng(1).normal(size=(8, 200)), mne.create_info(names, 1000.0, "dbs"), verbose=False)
    raw.info["bads"] = ["lead1_c3"] #e.g. the lab's do-not-use list
    with pytest.warns(UserWarning, match="lead1_c3 marked bad"):
        bip = dbs_rereference(raw, ["lead1"], pairs=horizontal_pairs)
    made = [ch for ch in bip.ch_names if "-" in ch]
    assert made == ["lead1_c2-lead1_c4", "lead1_c6-lead1_c5", "lead1_c7-lead1_c6", "lead1_c5-lead1_c7"] #3-2 and 4-3 left out


def test_all_pairs_are_the_lab_21_in_its_order():
    #get_rereference.m:37-38, column k = (c2, c1) and the pair is c2 - c1, written out by hand from her table
    lab = [(3, 2), (4, 3), (2, 4), (6, 5), (7, 6), (5, 7), (5, 2), (6, 3), (7, 4),
           (234, 1), (567, 234), (8, 567), (567, 1), (8, 234),
           (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]
    assert all_pairs == lab
    assert erna_pairs == [(3, 2), (4, 3), (2, 4), (6, 5), (7, 6), (5, 7), (8, 567)] #what her ERNA script keeps


def test_pair_families_dont_overlap():
    families = [horizontal_pairs, vertical_pairs, ring_pairs, c1_pairs]
    assert [len(f) for f in families] == [6, 3, 5, 7]
    assert len(set(all_pairs)) == 21 #no pair in two families


def _lead(bads=()):
    #a fake lead 1 with random data, so a wrong mean would show
    names = [f"lead1_c{k}" for k in range(1, 9)]
    data = np.random.default_rng(2).normal(size=(8, 300))
    raw = mne.io.RawArray(data, mne.create_info(names, 1000.0, "dbs"), verbose=False)
    raw.info["bads"] = list(bads)
    return raw, data


def test_ring_is_the_mean_of_its_three_segments():
    raw, data = _lead()
    rings = add_virtual_rings(raw, ["lead1"])
    assert np.allclose(rings.get_data(picks="lead1_c234")[0], data[1:4].mean(axis=0)) #rows 1-3 = c2, c3, c4
    assert np.allclose(rings.get_data(picks="lead1_c567")[0], data[4:7].mean(axis=0)) #rows 4-6 = c5, c6, c7
    assert "lead1_c234" not in raw.ch_names #the original raw isn't changed


def test_no_ring_when_a_segment_is_bad():
    #stim2: c2 is the stim contact, so no c234 (the lab's GR:41 needs all three); c567 is still made
    raw, _ = _lead(bads=["lead1_c2"])
    rings = add_virtual_rings(raw, ["lead1"])
    assert "lead1_c234" not in rings.ch_names
    assert "lead1_c567" in rings.ch_names


def test_ring_pair_through_rereference():
    #the lab's dbs_8_567 = c8 - mean(c5, c6, c7), the one ring pair its ERNA script keeps
    raw, data = _lead()
    bip = dbs_rereference(add_virtual_rings(raw, ["lead1"]), ["lead1"], pairs=[(8, 567)])
    assert np.allclose(bip.get_data(picks="lead1_c8-lead1_c567")[0], data[7] - data[4:7].mean(axis=0))


def test_all_21_pairs_with_rings():
    #with every contact usable and the rings added, all 21 of the lab's pairs are made, none skipped
    raw, _ = _lead()
    bip = dbs_rereference(add_virtual_rings(raw, ["lead1"]), ["lead1"])
    assert sum("-" in n for n in bip.ch_names) == 21


def test_common_average_is_contact_minus_lead_mean():
    #the lab's dbs_3_avg = c3 - mean(c1 ... c8) at every time point (get_rereference.m:10-11)
    raw, data = _lead()
    averaged = common_average(raw, "lead1")
    assert averaged.ch_names == [f"lead1_c{k}-avg" for k in range(1, 9)]
    assert np.allclose(averaged.get_data(picks="lead1_c3-avg")[0], data[2] - data.mean(axis=0)) #row 2 = c3


def test_common_average_leaves_out_bad_contacts():
    #c2 bad (stim2, or do-not-use): averaged over the other 7, and c2 gets no channel of its own
    raw, data = _lead(bads=["lead1_c2"])
    averaged = common_average(raw, "lead1")
    assert "lead1_c2-avg" not in averaged.ch_names
    others = np.delete(data, 1, axis=0) #every row except row 1 (c2)
    assert np.allclose(averaged.get_data(picks="lead1_c3-avg")[0], data[2] - others.mean(axis=0))


def test_common_average_leaves_out_virtual_rings():
    #the rings only exist inside the lab's bipolar step, so adding them first mustn't change the average
    raw, data = _lead()
    averaged = common_average(add_virtual_rings(raw, ["lead1"]), "lead1")
    assert len(averaged.ch_names) == 8
    assert np.allclose(averaged.get_data(picks="lead1_c3-avg")[0], data[2] - data.mean(axis=0))


def test_common_average_keeps_stim_annotations():
    #epoching later needs the stim pulse markers that to_mne puts on raw
    raw, _ = _lead()
    raw.set_annotations(mne.Annotations(onset=[0.05, 0.15], duration=[0, 0], description=["stim", "stim"]))
    averaged = common_average(raw, "lead1")
    assert list(averaged.annotations.description) == ["stim", "stim"]
    assert np.allclose(averaged.annotations.onset, [0.05, 0.15])


def test_common_average_needs_two_contacts():
    #one contact minus its own average is all zeros, so stop instead of saving zeros
    raw, _ = _lead(bads=[f"lead1_c{k}" for k in range(2, 9)]) #only c1 usable
    with pytest.raises(ValueError, match="at least 2"):
        common_average(raw, "lead1")


def _ecog(bads=()):
    #fake ECoG: anterior strip ecog_a1-a6 (rows 0-5) and posterior ecog_p1-p6 (rows 6-11), random data
    names = [f"ecog_{strip}{k}" for strip in ("a", "p") for k in range(1, 7)]
    data = np.random.default_rng(3).normal(size=(12, 300))
    raw = mne.io.RawArray(data, mne.create_info(names, 1000.0, "ecog"), verbose=False)
    raw.info["bads"] = list(bads)
    return raw, data


def test_ecog_pair_is_upper_minus_lower():
    #the lab's ecog_2_1 = a2 - a1 (get_rereference.m:30, Y(k+1) - Y(k)); 5 pairs per strip
    raw, data = _ecog()
    bip = ecog_rereference(raw)
    assert len(bip.ch_names) == 10
    assert np.allclose(bip.get_data(picks="ecog_a2-ecog_a1")[0], data[1] - data[0])
    assert np.allclose(bip.get_data(picks="ecog_p2-ecog_p1")[0], data[7] - data[6])


def test_ecog_pairs_never_cross_strips():
    #a6 and p1 sit on different strips, so they're never paired
    raw, _ = _ecog()
    bip = ecog_rereference(raw)
    assert not any("ecog_a" in n and "ecog_p" in n for n in bip.ch_names)


def test_bad_middle_contact_is_not_bridged():
    #deliberate difference: with a2 bad the lab makes ecog_3_1 (a3 - a1, double spacing); we skip a2's pairs and warn
    raw, _ = _ecog(bads=["ecog_a2"])
    with pytest.warns(UserWarning, match="lab would pair ecog_a3 with ecog_a1"):
        bip = ecog_rereference(raw)
    assert "ecog_a2-ecog_a1" not in bip.ch_names
    assert "ecog_a3-ecog_a2" not in bip.ch_names
    assert "ecog_a3-ecog_a1" not in bip.ch_names #no bridged pair
    assert len(bip.ch_names) == 8 #only pairs: the unused a1 and the bad a2 aren't left as single channels, as in the lab
