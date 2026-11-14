import numpy as np
import pytest
from preprocess_dbs.postprocess import trim_to_artifact

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
