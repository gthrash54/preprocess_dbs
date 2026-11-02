import mne
import numpy as np
import pytest
from preprocess_dbs.emg import downsample_emg

FS = 24414.0625 #TDT's EMG rate (25e6 / 1024)


def _sine(freq, fs=FS, seconds=2.0):
    #one channel of a sine wave at freq Hz, with two stim markers
    t = np.arange(int(fs * seconds)) / fs
    raw = mne.io.RawArray(np.sin(2 * np.pi * freq * t)[None], mne.create_info(["emg_1"], fs, "emg"), verbose=False)
    raw.set_annotations(mne.Annotations(onset=[0.5, 1.5], duration=[0, 0], description=["stim", "stim"]))
    return raw


def test_fast_emg_goes_to_12_5_khz():
    raw = _sine(100)
    out = downsample_emg(raw)
    assert out.info["sfreq"] == 12500
    assert abs(out.n_times - raw.n_times * 64 / 125) <= 1 #true fs: 12500 / 24414.0625 is exactly 64/125
    assert raw.info["sfreq"] == FS #the original isn't changed


def test_slow_emg_is_left_alone():
    #the lab only resamples above 15 kHz (preprocess.m:796)
    raw = _sine(100, fs=10000.0)
    out = downsample_emg(raw)
    assert out.info["sfreq"] == 10000.0
    assert np.array_equal(out.get_data(), raw.get_data())


def test_emg_range_signal_survives():
    #100 Hz is well inside EMG's band and below the new Nyquist (6.25 kHz): it should come through unchanged
    out = downsample_emg(_sine(100))
    t = np.arange(out.n_times) / out.info["sfreq"]
    middle = slice(1000, -1000) #skip the filter's edge effects
    assert np.allclose(out.get_data()[0][middle], np.sin(2 * np.pi * 100 * t)[middle], atol=0.01)


def test_noise_above_new_nyquist_is_removed_not_aliased():
    #10 kHz is above 6.25 kHz: without the anti-alias filter it would fold down to a fake 2.5 kHz wave (12.5 - 10 kHz)
    out = downsample_emg(_sine(10000))
    middle = out.get_data()[0][1000:-1000]
    assert np.sqrt(np.mean(middle ** 2)) < 0.01 #a sine of amplitude 1 has RMS 0.71; almost nothing is left


def test_stim_markers_keep_their_times():
    out = downsample_emg(_sine(100))
    assert np.allclose(out.annotations.onset, [0.5, 1.5])


def test_non_emg_channels_are_refused():
    #DBS at TDT's 48.8 kHz must never be downsampled by mistake: ERNA needs the full rate
    raw = mne.io.RawArray(np.zeros((2, 1000)), mne.create_info(["lead1_c2", "emg_1"], 48828.125, ["dbs", "emg"]), verbose=False)
    with pytest.raises(ValueError, match="only takes EMG"):
        downsample_emg(raw)
