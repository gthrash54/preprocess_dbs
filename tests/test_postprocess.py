import numpy as np
import pytest
from preprocess_dbs.postprocess import trim_to_artifact
from preprocess_dbs.postprocess import crop_and_demean

FS = 48828.125


def _peaked(t, at):
    #a smooth peak at sample `at`, so it's easy to see where the peak ends up
    return np.exp(-((t - at) / 3) ** 2)


def test_trim_puts_each_peak_at_time_zero():
    #two channels with their artifact peaks at epoch samples 118 and 119 (window positions 21 and 22, since the
    #window starts at 97). After trimming, both peaks sit at sample 98, which is time 0 on a -98 ... axis
    t = np.arange(600.0)
    data = np.vstack([_peaked(t, 118), _peaked(t, 119)])
    trimmed = trim_to_artifact(data, np.array([21.0, 22.0]), FS)
    assert list(trimmed.argmax(axis=1)) == [98, 98]
    #channel 1 starts at 20 and channel 2 at 21, so channel 2 is one sample shorter and both are cut to its length
    assert trimmed.shape == (2, 600 - 21)
    assert np.array_equal(trimmed[0], data[0, 20:20 + 579])
    assert np.array_equal(trimmed[1], data[1, 21:21 + 579])


def test_trim_rounds_a_fractional_peak():
    #a peak between samples: window position 21.4 is epoch sample 118.4, rounded to 118, so the start is 118 - 98 = 20.
    #At 21.6 (118.6, rounded to 119) the start is 21. This is the "exactly 2 ms" fix: Sarah's start, round(peak),
    #would be 21 and 22, one sample later, leaving the peak about 0.02 ms before time 0
    t = np.arange(600.0)
    data = np.vstack([t, t]) #each sample's value is its own sample number, so the first value is the start
    trimmed = trim_to_artifact(data, np.array([21.4, 21.6]), FS)
    assert trimmed[0, 0] == 20
    assert trimmed[1, 0] == 21


def test_trim_leaves_the_input_alone():
    t = np.arange(600.0)
    data = np.vstack([_peaked(t, 118), _peaked(t, 125)])
    before = data.copy()
    trim_to_artifact(data, np.array([21.0, 28.0]), FS)
    assert np.array_equal(data, before)


def test_trim_refuses_a_start_outside_the_epoch():
    #a peak too early to keep 2 ms before it would give a negative start, which Python would silently read from the END
    #of the signal; a start past the end would leave nothing. Both must stop with an error instead
    t = np.arange(600.0)
    data = np.vstack([t, t])
    with pytest.raises(ValueError, match="outside"):
        trim_to_artifact(data, np.array([21.0, 0.0]), FS) #channel 2: 97 + 0 - 98 = -1
    with pytest.raises(ValueError, match="outside"):
        trim_to_artifact(data, np.array([21.0, 601.0]), FS) #channel 2: 97 + 601 - 98 = 600, past the last sample (599)
    #the last valid start, 599, still works: one sample per channel
    assert trim_to_artifact(data, np.array([598.0, 600.0]), FS).shape == (2, 1)


def test_crop_keeps_2_ms_before_to_45_ms_after():
    #after trim_to_artifact the peak is at sample 98, so -2 to +45 ms is the first 98 + 2197 = 2295 samples. A shorter
    #signal is kept whole (slicing past the end just stops at the end)
    data = np.ones((3, 3000))
    assert crop_and_demean(data, FS).shape == (3, 2295)
    assert crop_and_demean(np.ones((3, 1000)), FS).shape == (3, 1000)


def test_crop_and_demean_leaves_the_input_alone():
    #the crop is a view of the input; the demean must build a new array, not subtract inside the caller's data
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 3000)) + 5.0
    before = data.copy()
    crop_and_demean(data, FS)
    assert np.array_equal(data, before)


def test_crop_and_demean_off_by_one():
    #a signal whose value at each sample is its own index (sample 145 holds 145), so moving either window by one sample
    #changes its mean; a flat signal would pass even with the bug. The mean of consecutive whole numbers is
    #(first + last) / 2, so the right answer can be written out by hand, not computed the way the function does it
    x = np.arange(0,2295)
    #channel 2 is the index squared: a different level per channel (so averaging over the wrong axis, or over the
    #whole array, fails), and not a straight line (so two edge mistakes can't cancel each other out)
    data = np.vstack([x, x**2])
    demeaned = crop_and_demean(data,FS)
    #channel 1, by hand
    m1 = (0 + 48) / 2 #samples 0-48: the first 1 ms (Sarah's 1:49)
    m2 = (145 + 2294) / 2 #samples 145-2294: from about 1 ms after the artifact to the end (Sarah's 146:end)
    level = (m1 + m2) / 2
    #if m2 started one sample late (146) it would be half a sample higher, and if m1 took one sample too many (0-49)
    #it would be half a sample higher too: either way the level rises by 0.25 and every value is off. Either fails here
    assert np.allclose(demeaned[0], x - level)
    #channel 2: the same windows, written out as explicit sample ranges
    level_squared = (np.mean(np.arange(0, 49)**2) + np.mean(np.arange(145, 2295)**2)) / 2
    assert np.allclose(demeaned[1], x**2 - level_squared)



def test_crop_and_demean_refuses_a_too_short_signal():
    #m2 starts at sample 145; with 145 samples or fewer it would be empty and every output value NaN, so stop instead.
    #146 samples is the shortest that works (m2 is then just sample 145)
    with pytest.raises(ValueError, match="too short"):
        crop_and_demean(np.ones((2, 145)), FS)
    assert crop_and_demean(np.ones((2, 146)), FS).shape == (2, 146)
