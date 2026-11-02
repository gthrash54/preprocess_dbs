import numpy as np
import pytest
from preprocess_dbs.conditions import lab_pair_name, is_doubled, doubled_pair_tail
from preprocess_dbs.epoch import condition_windows


def test_pair_names_for_standard_patients():
    #classify_pulses names pairs by gap (2 decimals); the lab's names and ERNA labels come from parse.m and the ERNA script
    assert lab_pair_name("paired_0.45_1") == ("pp1", "pp0.3") #uh3005 on
    assert lab_pair_name("paired_0.25_2") == ("pp1", "pp0.3") #older patients' pp1, second pulse works too
    assert lab_pair_name("paired_2.89_1") == ("pp14", "pp2.8")
    assert lab_pair_name("paired_15.38_1") == ("pp23", "pp15.3")


def test_pair_names_for_doubled_patients():
    #the same 1.93 ms gap is pp10 normally but pp5_2 in uh3001/uh3006; both are the ERNA script's pp1.8
    assert lab_pair_name("paired_1.93_1") == ("pp10", "pp1.8")
    assert lab_pair_name("paired_1.93_1", doubled=True) == ("pp5_2", "pp1.8")
    assert lab_pair_name("paired_5.78_1", doubled=True) == ("pp14_2", None) #the ERNA script skips this one


def test_unknown_gap_and_bad_label():
    assert lab_pair_name("paired_3.10_1") == (None, None) #the lab's "CHECK"
    with pytest.raises(ValueError, match="isn't a pair label"):
        lab_pair_name("single")


def test_doubled_patients_from_the_table():
    assert is_doubled("uh3001") and is_doubled("uh3006")
    assert not is_doubled("uh3021")


def test_doubled_pair_window_is_fixed_after_the_second_pulse():
    #pair at 1.000/1.00193 s, next pulse 100 ms later (doubled periods): window = (gap - 0.1 ms) + 49.89168 ms - 0.1 ms
    times = np.array([1.0, 1.00193, 1.1, 1.2])
    labels = ["paired_1.93_1", "paired_1.93_2", "single", ""]
    normal = condition_windows(times, labels)["paired_1.93_1"]
    fixed = condition_windows(times, labels, pair_tail=doubled_pair_tail)["paired_1.93_1"]
    assert normal == pytest.approx(0.1 - 0.0002) #runs to the next pulse
    assert fixed == pytest.approx(0.00193 - 0.0001 + 0.04989168 - 0.0001) #about 51.6 ms, like a normal patient's pair


def test_fixed_pair_window_never_runs_into_the_next_pulse():
    times = np.array([1.0, 1.00193, 1.03, 1.2]) #next pulse only 30 ms after the pair
    labels = ["paired_1.93_1", "paired_1.93_2", "single", ""]
    assert condition_windows(times, labels, pair_tail=doubled_pair_tail)["paired_1.93_1"] == pytest.approx(0.03 - 0.0002)
