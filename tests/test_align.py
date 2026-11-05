import numpy as np
import pytest
from preprocess_dbs.align import refine_peak
from preprocess_dbs.align import find_first_big_deflection


def test_peak_between_samples():
    #a parabola built with its top at 41.8: the biggest sample is 42, the true peak 41.8
    t = np.arange(100.0)
    signal = -(t - 41.8) ** 2
    assert refine_peak(signal, 42) == pytest.approx(41.8)


def test_trough_works_too():
    #artifacts can point down: an upside-down peak's bottom is found the same way
    t = np.arange(100.0)
    assert refine_peak((t - 17.25) ** 2, 17) == pytest.approx(17.25)


def test_worked_example():
    #samples 3.0, 7.0, 8.0, 6.0, 2.0 around position 3: the left side is higher, so the top is a bit left of 3
    signal = np.array([0.4, 3.0, 7.0, 8.0, 6.0, 2.0, 0.4])
    assert 2.5 < refine_peak(signal, 3) < 3.0


def test_edge_and_flat_cases_return_idx():
    assert refine_peak(np.array([1.0, 5.0, 2.0]), 0, half_width=0) == 0 #fewer than 3 points to fit
    assert refine_peak(np.arange(10.0), 5) == 5 #a straight line has no top


def test_nearly_flat_stretch_stays_inside_the_window():
    #a tiny curve on a slope: the parabola's top is far away, but the answer stays inside the 7 fitted samples
    #(deliberate difference: the lab doesn't clamp)
    t = np.arange(100.0)
    signal = t + 1e-6 * (t - 500.0) ** 2 * -1 #almost a straight line, top near 500
    assert 39 <= refine_peak(signal, 42) <= 45


def  test_spikes_for_finding_first_big_deflection():
    #the biggest peak is the -10 at 40, but the 7 at 30 comes earlier and is at least half its size (the -6 at 35 is
    #smaller), so the function picks the first big peak: position 30, pointing up
    signal = np.zeros(60)
    signal[[30,35,40]] = [7,-6,-10]
    fs = 48828.125
    position,sign = find_first_big_deflection(signal=signal,fs=fs,desired_sign=None)
    assert position == 30
    assert sign == 1

def test_first_big_deflection_with_a_desired_sign():
    #second call, told to look for downward peaks: the 7 points up, so it's ignored; the -6 at 35 is an earlier
    #downward peak at least half the size of the -10, so it wins
    signal = np.zeros(60)
    signal[[30, 35, 40]] = [7, -6, -10]
    position, sign = find_first_big_deflection(signal, 48828.125, desired_sign=-1)
    assert position == 35
    assert sign == -1


def test_first_big_deflection_ignores_points_on_a_slope():
    #6, 7, 8 rising into 34: all three are big, but only the 8 is the top of a peak (the 6 and 7 are partway up)
    signal = np.zeros(60)
    signal[[32, 33, 34, 40]] = [6, 7, 8, -10]
    position, sign = find_first_big_deflection(signal, 48828.125)
    assert position == 34
    assert sign == 1


def test_fallback_warning_for_first_big_deflection():
    #asked for a downward peak, but every peak points up: it falls back to the biggest peak (the 10 at 13) and warns,
    #then the look-back finds the 7 at 8, an earlier peak at least half its size, so 8 pointing up
    signal = np.zeros(60)
    signal[[4,8,13]] = [3,7,10]
    with pytest.warns(UserWarning, match="no sample points in the desired direction"):
        position,sign = find_first_big_deflection(signal, 48828.125, desired_sign=-1)
    assert position == 8
    assert sign == 1
