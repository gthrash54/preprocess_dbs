import numpy as np
import pytest
from preprocess_dbs.align import refine_peak


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
