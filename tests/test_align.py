import numpy as np
import pytest
from preprocess_dbs.align import refine_peak
from preprocess_dbs.align import find_first_big_deflection
from preprocess_dbs.align import align_signals
from preprocess_dbs.align import shift_signal


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


def test_align_signals_finds_known_shifts():
    #a smooth artifact-like bump peaking 2 ms in; y is the same bump moved by a known number of samples
    #np.roll(x, 5) moves everything 5 samples later, so y comes after x and the shift is +5; rolling by -3 gives -3
    fs = 48828.125
    t = np.arange(400) / fs
    x = np.exp(-((t - 0.002) / 0.0001) ** 2)
    shift, peak_x, peak_y = align_signals(x, np.roll(x, 5), fs)
    assert shift == pytest.approx(5, abs=0.01)
    assert peak_y - peak_x == pytest.approx(shift) #the shift is just the gap between the two refined peaks
    shift, peak_x, peak_y = align_signals(x, np.roll(x, -3), fs)
    assert shift == pytest.approx(-3, abs=0.01)


def test_align_signals_needs_equal_lengths():
    #x and y must be the same length; one sample short should stop with an error, not give a wrong shift
    x = np.zeros(400)
    with pytest.raises(ValueError, match="not equal"):
        align_signals(x, x[:-1], 48828.125)


def test_shift_signal_positive_shift():
    #tests a time shift to a later time by using a positive shift in shift_signal
    t = np.arange(200)
    x = np.exp(-((t-100)/4)**2) # test data that peaks at 100
    shift,a,b = align_signals((shift_signal(x,2.5)),x, 48000) #returns 3 values, a is the shifted signals location and b is the original location
    assert a == pytest.approx(102.5, abs=0.1)

def test_shift_signal_with_zeros():
    #tests a shift that causes the edge to display zeros
    signal = np.arange(1,11)
    shifted_signal = shift_signal(signal,3) 
    assert len(signal) == len(shifted_signal)
    assert np.all(shifted_signal[:3] ==0)
    assert signal[0] == shifted_signal[3]


def test_shift_signal_shifted_zero():
    # tests a shift of zero
    t = np.arange(200)
    x = np.exp(-((t-100)/4)**2) # test data that peaks at 100
    shifted_signal = shift_signal(x,0)
    assert x == pytest.approx(shifted_signal)