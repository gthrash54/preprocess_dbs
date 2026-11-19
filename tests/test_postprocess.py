import numpy as np
import pytest
from preprocess_dbs.postprocess import moving_mean
from preprocess_dbs.postprocess import smoothing_features
from preprocess_dbs.postprocess import trim_to_artifact
from preprocess_dbs.postprocess import crop_and_demean
from preprocess_dbs.postprocess import fit_drifts
from preprocess_dbs.postprocess import score_drifts
from preprocess_dbs.postprocess import choose_detrend
from preprocess_dbs.postprocess import detrend_channel
from preprocess_dbs.postprocess import smoothing_states
from preprocess_dbs.postprocess import guard_artifact_onset
from preprocess_dbs.postprocess import state_window
from preprocess_dbs.postprocess import find_stretches
from preprocess_dbs.postprocess import smooth_by_state
from preprocess_dbs.postprocess import subtract_median_baseline
from preprocess_dbs.postprocess import adaptive_smooth
from preprocess_dbs.postprocess import LONG, ARTIFACT, SHORT

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


def _drift_test_signal(drift):
    #a known drift plus two traps the fit window must leave out: a big artifact spike at 2 ms (in the first 3 ms) and a
    #jump in the last 0.3 ms (in the last 0.5 ms). If either got into the fit, the fitted curves would bend toward it
    t = np.arange(2295) / FS * 1000 #ms from the trimmed start
    spike = np.where(t < 3, 5 * np.exp(-((t - 2) / 0.1) ** 2), 0.0)
    end_jump = np.where(t > t[-1] - 0.3, 3.0, 0.0)
    return t, drift(t) + spike + end_jump


def test_fit_drifts_recovers_a_straight_drift_despite_the_artifact():
    #a straight drift: both fits should find exactly it, everywhere (the spike is outside the fitted window)
    t, signal = _drift_test_signal(lambda t: 0.2 + 0.01 * t)
    linear, quadratic, mask = fit_drifts(signal, FS)
    assert np.allclose(linear, 0.2 + 0.01 * t)
    assert np.allclose(quadratic, 0.2 + 0.01 * t) #a parabola can be a straight line (its squared term goes to 0)


def test_fit_drifts_quadratic_catches_a_bend():
    #a bending drift: only the parabola can follow it; the line misses it by a clear margin
    bend = lambda t: 0.1 - 0.02 * t + 0.0008 * t ** 2
    t, signal = _drift_test_signal(bend)
    linear, quadratic, mask = fit_drifts(signal, FS)
    assert np.allclose(quadratic, bend(t))
    assert np.abs(linear - bend(t))[mask].max() > 0.1


def test_fit_drifts_window_edges():
    #the fit window is t > 3 ms and t < (last time - 0.5 ms): samples 147 to 2269 of a 2295-sample signal at 48.8 kHz
    _, signal = _drift_test_signal(lambda t: 0 * t)
    _, _, mask = fit_drifts(signal, FS)
    kept = np.flatnonzero(mask)
    assert kept[0] == 147 #146 is 2.990 ms, 147 is 3.011 ms
    assert kept[-1] == 2269 #the last sample is 46.981 ms, so the window ends below 46.481 ms (2269 is 46.469 ms)
    assert len(kept) == 2269 - 147 + 1 #one unbroken stretch



def test_fit_drifts_shapes_and_input():
    #both curves and the mask are as long as the signal, the mask is True/False, and the input isn't changed
    _, signal = _drift_test_signal(lambda t: 0.2 + 0.01 * t)
    before = signal.copy()
    linear, quadratic, mask = fit_drifts(signal, FS)
    assert linear.shape == quadratic.shape == mask.shape == signal.shape
    assert mask.dtype == bool
    assert np.array_equal(signal, before)


def test_fit_drifts_refuses_a_too_short_window():
    #crop_and_demean accepts as few as 146 samples, but then no sample is after 3 ms, so there's nothing to fit
    with pytest.raises(ValueError, match="too few"):
        fit_drifts(np.ones(146), FS)


def test_score_drifts_straight_drift_favors_linear():
    #a realistic signal: an artifact spike at 2 ms, a straight drift, and a small response after 3 ms. The artifact has to
    #be there: scoring scales each version by its largest value, and in real data that's the artifact. Removing the line
    #should flatten the window a lot, and the parabola should do about the same as the line (it can fit a line too; it
    #comes out a hair lower here because its extra bend soaks up a little of the response)
    t = np.arange(2295)/FS*1000
    artifact = np.where(t < 3, 5 * np.exp(-((t - 2) / 0.1) ** 2), 0.0)
    drift = t*0.05 + 0.0010 #straight
    response = np.where(t > 3, np.sin(2*np.pi*0.3*t) * 0.05, 0)
    signal = artifact + drift + response
    linear_drift, quadratic_drift, mask = fit_drifts(signal,FS)
    none, linear, quadratic = score_drifts(signal,linear_drift,quadratic_drift,mask)
    assert linear < none/2
    assert linear == pytest.approx(quadratic, abs=0.001)


def test_score_drifts_ignore_samples_outside_the_mask():
    #invariance test: put a trap where score_drifts should not look (the last 0.3 ms, outside the window) and check the
    #scores don't change. If the mask were forgotten, the trap would leak into the scores. The trap stays smaller than
    #the artifact, so it doesn't become the largest value and change the scaling the function rightly uses
    t = np.arange(2295)/FS*1000
    artifact = np.where(t < 3, 5 * np.exp(-((t - 2) / 0.1) ** 2), 0.0)
    drift = t*0.05 + 0.0010 #straight
    response = np.where(t > 3, np.sin(2*np.pi*0.3*t) * 0.05, 0)
    signal = artifact + drift + response
    trap = np.where(t > t[-1]-0.3, 1, 0)
    #fit once on the clean signal and reuse the drifts and mask, so this tests score_drifts only
    linear_drift, quadratic_drift, mask = fit_drifts(signal,FS)
    clean_scores = score_drifts(signal,linear_drift,quadratic_drift,mask)
    trap_scores = score_drifts(signal+trap,linear_drift,quadratic_drift,mask)
    assert np.allclose(clean_scores,trap_scores)


#every rule of Sarah's detrend choice, plus the exact thresholds, plus which branch each condition name goes to.
#Each row: (scores as none, linear, quadratic; condition; expected choice). pytest runs the test once per row.
#Careful with exact decimal boundaries: 0.4 - 0.5 is -0.09999999999999998 on a computer, so an "exact" -0.10 can land
#on either side from rounding alone. 0.1 - 0.2 does come out as exactly -0.1, so that pair tests rule 1's <= exactly
@pytest.mark.parametrize("scores, condition, expected", [
    #branch A (bursts 1-9): none vs linear only (the quadratic score isn't used)
    ([0.2, 0.05, 0.9], "burst_3", "linear"), #linear cuts the score by 75%
    ([0.2, 0.15, 0.9], "burst_3", "none"), #only a 25% cut
    ([0.2, 0.10, 0.9], "burst_3", "linear"), #boundary: exactly half counts
    ([0.2, 0.1001, 0.9], "burst_3", "none"), #just over half
    #branch B: one row per rule
    ([1.0, 0.5, 0.2], "single", "quadratic"), #rule 1
    ([1.0, 0.5, 0.45], "single", "linear"), #rule 2: quadratic helps, but only 0.05 beyond linear
    ([1.0, 0.8, 0.9], "single", "linear"), #rule 3
    ([1.0, 0.9, 0.95], "single", "none"), #rule 4
    #branch B boundaries
    ([1.0, 0.9, 0.65], "single", "quadratic"), #exactly 0.65 x none still counts
    ([1.0, 0.9, 0.651], "single", "none"), #just over: rule 1 fails and linear (0.9) is too big for rule 3
    ([1.0, 0.85, 0.9], "single", "linear"), #rule 3 at exactly 0.85
    ([1.0, 0.86, 0.9], "single", "none"), #just over
    #normalizing and routing
    ([0.4, 0.24, 0.16], "single", "quadratic"), #normalized: 1, 0.6, 0.4 -> rule 1. Without normalizing, q - l would
                                                #be only -0.08 and it would wrongly say linear
    ([1.0, 0.5, 0.2], "burst_10", "quadratic"), #branch B (branch A would say linear)
    ([1.0, 0.5, 0.2], "burstprobes_4", "quadratic"), #branch B too
    ([1.0, 0.5, 0.2], "burst_3", "linear"), #the same scores go to branch A: proves the routing
    #the largest score isn't always none's
    ([0.1, 0.2, 0.9], "burst_3", "none"), #branch A: linear is the larger, so it normalizes to 1 and none to 0.5
    ([0.8, 0.5, 1.0], "single", "linear"), #branch B: quadratic is the largest; rule 3 (linear 0.5 <= 0.85)
    ([0.2, 0.4, 1.0], "single", "linear"), #rule 3 even though linear doubles none's score (her Q2 quirk). Catches
                                           #normalizing by none instead of the largest, or comparing l to 0.85 x n
    #more edges
    ([1.0, 0.2, 0.1], "single", "quadratic"), #rule 1 at exactly q - l = -0.10 (0.1 - 0.2 is exact): <= counts
    ([1.0, 0.7, 0.65], "single", "linear"), #rule 2 qualifies on the QUADRATIC's 35% cut, not linear's
    ([0.2, 0.05, 0.9], "burst_1", "linear"), #both ends of the nine-name allow-list are branch A
    ([0.2, 0.05, 0.9], "burst_9", "linear"),
    ([1.0, 0.5, 0.2], "paired_0.45_1", "quadratic"), #pairs go to branch B
])
def test_choose_detrend_rules(scores, condition, expected):
    assert choose_detrend(scores, condition) == expected




def test_score_drifts_known_answer():
    #score_drifts on its own, with a tiny hand-made signal, drifts and mask, so every score can be worked out by hand.
    #The behavior tests above can't see the arithmetic; this one catches skipping the scaling, a true RMS instead of a
    #mean absolute value, max without abs, the original signal's max for every version, and swapped outputs
    signal = np.array([4.0, 1.0, -1.0, 2.0, -2.0, 0.0])
    mask = np.array([False, True, True, True, True, False]) #the 4 middle samples; the 4.0 is outside but sets the scale
    linear_drift = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.0]) #removes everything inside the window
    quadratic_drift = np.array([7.0, 0.0, 0.0, 0.0, 0.0, 0.0]) #only changes the first sample, to -3
    #none: largest |value| 4, so the window becomes 0.25, 0.25, 0.5, 0.5 -> mean 0.375
    #linear: signal - drift = 4, 0, 0, 0, 0, 0 -> the window is all 0 -> 0.0
    #quadratic: signal - drift = -3, 1, -1, 2, -2, 0; largest |value| 3 (a negative one, so max without abs would
    #give 2) -> the window becomes 1/3, 1/3, 2/3, 2/3 -> mean 0.5
    assert score_drifts(signal, linear_drift, quadratic_drift, mask) == pytest.approx([0.375, 0.0, 0.5])


def _detrend_test_signal(drift):
    #an artifact spike at 2 ms, a drift, and a small response after 3 ms, like the score_drifts tests
    t = np.arange(2295) / FS * 1000
    artifact = np.where(t < 3, 5 * np.exp(-((t - 2) / 0.1) ** 2), 0.0)
    response = np.where(t > 3, np.sin(2 * np.pi * 0.3 * t) * 0.05, 0.0)
    return t, artifact + drift(t) + response, artifact + response


def test_detrend_channel_takes_off_a_straight_drift():
    #a straight drift on a DBS channel: her rule picks linear, and what's left is the signal without the drift
    t, signal, without_drift = _detrend_test_signal(lambda t: 0.05 * t + 0.001)
    before = signal.copy()
    detrended, choice = detrend_channel(signal, FS, "single", "dbs")
    assert choice == "linear"
    assert np.array_equal(signal, before) #the input isn't changed
    #the drift comes off the WHOLE signal, the artifact and the end included, not just the fit window
    assert np.allclose(detrended, signal - fit_drifts(signal, FS)[0])
    window = (t > 3) & (t < t[-1] - 0.5)
    #the fitted line also soaks up a little of the response, so compare with a small tolerance, inside the window
    assert np.allclose(detrended[window], without_drift[window], atol=0.005)


def test_detrend_channel_takes_off_a_bend():
    #a bending drift: her rule picks quadratic, and the bend comes off
    t, signal, without_drift = _detrend_test_signal(lambda t: 0.0004 * (t - 20) ** 2)
    detrended, choice = detrend_channel(signal, FS, "single", "dbs")
    assert choice == "quadratic"
    window = (t > 3) & (t < t[-1] - 0.5)
    assert np.allclose(detrended[window], without_drift[window], atol=0.005)


def test_detrend_channel_leaves_a_signal_without_drift_alone():
    #artifact and response but no drift: nothing helps enough, so the choice is none and the signal comes back as it
    #was, as a new array
    t, signal, _ = _detrend_test_signal(lambda t: 0.0 * t)
    detrended, choice = detrend_channel(signal, FS, "single", "dbs")
    assert choice == "none"
    assert np.array_equal(detrended, signal)
    assert not np.shares_memory(detrended, signal)


def test_detrend_channel_skips_ecog_and_returns_a_copy():
    #Sarah's code skips ECoG: same values back, choice none, but a new array, so changing it can't change the input
    t, signal, _ = _detrend_test_signal(lambda t: 0.05 * t)
    detrended, choice = detrend_channel(signal, FS, "single", "ecog")
    assert choice == "none"
    assert np.array_equal(detrended, signal)
    assert not np.shares_memory(detrended, signal)


def test_detrend_channel_passes_the_condition_on(monkeypatch):
    #the condition must reach choose_detrend (bursts 1-9 use a different rule), not be dropped or replaced
    from preprocess_dbs import postprocess
    seen = []
    def record(scores, condition):
        seen.append(condition)
        return "none"
    monkeypatch.setattr(postprocess, "choose_detrend", record)
    _, signal, _ = _detrend_test_signal(lambda t: 0.05 * t)
    detrend_channel(signal, FS, "burst_3", "dbs")
    assert seen == ["burst_3"]



def test_detrend_channel_refuses_other_channel_types():
    #EMG (or anything else) should never reach this step; if it does, stop with a clear error instead of detrending it
    _, signal, _ = _detrend_test_signal(lambda t: 0.05 * t)
    for channel_type in ("emg", "misc", "ecog_anterior"):
        with pytest.raises(ValueError, match="'dbs' or 'ecog'"):
            detrend_channel(signal, FS, "single", channel_type)


def test_moving_mean_shrinks_at_the_edges():
    #MATLAB's movmean: the first sample averages only 1 and 2, the last only 4 and 5 (no padding)
    assert np.allclose(moving_mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3), [1.5, 2, 3, 4, 4.5])


def test_moving_mean_matches_a_plain_loop():
    #against the definition written out as a loop, with Sarah's two window sizes, on random data of a real length
    rng = np.random.default_rng(0)
    x = rng.normal(size=2295)
    for window in (5, 73):
        half = window // 2
        by_hand = np.array([x[max(0, i - half):i + half + 1].mean() for i in range(len(x))])
        assert np.allclose(moving_mean(x, window), by_hand)


def test_moving_mean_refuses_an_even_window():
    with pytest.raises(ValueError, match="odd"):
        moving_mean(np.ones(10), 4)



def test_smoothing_features_run_0_to_1():
    #regression test for a bug caught while writing it: building activity from the raw slope and bend (dx, ddx)
    #instead of their moving RMS. Raw slopes swing negative, so activity went down to about -2 on this signal.
    #A moving RMS can't be negative, so both features must stay at 0 or above, with a max of exactly 1.
    #The signal: an artifact at 2 ms plus a 300 Hz ringing that starts at 0 and dies out (swings both ways, like ERNA)
    t = np.arange(2295) / FS
    signal = 5e-3 * np.exp(-((t - 0.002) / 0.00005) ** 2) + 2e-4 * np.sin(2 * np.pi * 300 * t) * np.exp(-t / 0.01)
    activity, envelope = smoothing_features(signal, FS)
    for feature in (activity, envelope):
        assert len(feature) == len(signal)
        assert feature.min() >= 0
        assert feature.max() == pytest.approx(1)


def test_smoothing_features_windows_on_a_single_spike():
    #one nonzero sample shows exactly where each feature spreads, which pins down the windows and the 0 in front of diff
    #activity: the slope is nonzero at samples k and k+1, the bend at k to k+2; the 5-sample window spreads each by 2
    #on both sides, so activity is nonzero from k-2 to k+4. (Putting the 0 at the end of diff instead gives k-4 to k+2.)
    #envelope: the squared signal is nonzero only at k; the 73-sample window spreads it 36 each way, and every one of
    #those samples averages the same single value, so the envelope is exactly 1 on k-36 to k+36 and 0 everywhere else.
    signal = np.zeros(301)
    k = 150
    signal[k] = 1.0
    activity, envelope = smoothing_features(signal, FS)
    assert np.array_equal(np.flatnonzero(activity > 1e-6), np.arange(k - 2, k + 5))
    assert np.array_equal(np.flatnonzero(envelope > 1e-6), np.arange(k - 36, k + 37))
    assert np.allclose(envelope[k - 36:k + 37], 1)


def test_smoothing_features_match_her_formulas_written_out():
    #Sarah's lines 674-691 written out one sample at a time, with the window sizes typed in by hand.
    #The bend is so much bigger than the slope that dropping the slope, changing the 0.5 or skipping a square root only
    #moves activity a tiny bit, so the comparison has to be tight (rtol 1e-9) to see it.
    #Two sampling rates: at half our rate the windows are 3 and 37 samples, so hardcoding 5 and 73 would fail.
    rng = np.random.default_rng(1)
    for fs, feature_window, envelope_window in ((FS, 5, 73), (FS / 2, 3, 37)):
        x = rng.normal(size=500)
        n = len(x)
        dx = [0.0] + [(x[i] - x[i - 1]) * fs for i in range(1, n)]
        ddx = [0.0] + [(dx[i] - dx[i - 1]) * fs for i in range(1, n)]

        def moving_rms(values, window):
            half = window // 2
            return np.array([np.sqrt(np.mean(np.square(values[max(0, i - half):i + half + 1]))) for i in range(n)])

        combined = moving_rms(dx, feature_window) + 0.5 * moving_rms(ddx, feature_window)
        signal_rms = moving_rms(x, envelope_window)
        activity, envelope = smoothing_features(x, fs)
        assert np.allclose(activity, combined / combined.max(), rtol=1e-9, atol=0)
        assert np.allclose(envelope, signal_rms / signal_rms.max(), rtol=1e-9, atol=0)


def test_moving_mean_refuses_a_window_below_1():
    for window in (0, -1):
        with pytest.raises(ValueError, match="positive odd"):
            moving_mean(np.ones(10), window)


def test_moving_mean_refuses_nan_and_inf():
    #pandas would quietly skip these (giving [1, 2, 3]), while MATLAB's movmean spreads a NaN, so stop instead
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="NaN or Inf"):
            moving_mean(np.array([1.0, bad, 3.0]), 3)


def test_smoothing_features_refuses_a_flat_signal():
    #all zeros has no size and no slope; a constant has a size but no slope. Either would divide by 0 and give NaN.
    for flat in (np.zeros(200), np.full(200, 2.0)):
        with pytest.raises(ValueError, match="flat"):
            smoothing_features(flat, FS)

def test_smoothing_states_basic_path():
    # it checks the normal path of LONG->ARTIFACT->SHORT -> LONG
    activity = [0,0.5,0.5,0.1,0.1,0.1]
    envelope = [1,1,1,1,0.2,0.2]
    states = smoothing_states(activity,envelope)
    assert np.array_equal(states,[1,2,2,3,1,1])

def test_guard_artifact_onset_one_sample_artifact():
    states = np.array([1,1,1,1,2,3,3])
    fs = 10000
    expected = [1,1,3,3,3,3,3]
    assert np.array_equal(guard_artifact_onset(states,fs),expected)


@pytest.mark.parametrize("activity, envelope, expected", [
    #one step per sample: the envelope is already low at index 2, but ARTIFACT has to pass through SHORT first
    ([0, 0.5, 0.1, 0.1], [1, 0.1, 0.1, 0.1], [1, 2, 3, 1]),
    #hysteresis: 0.3 is too low to enter ARTIFACT (needs > 0.4)...
    ([0, 0.3, 0.3], [1, 1, 1], [1, 1, 1]),
    #...but too high to leave it (needs < 0.2)
    ([0, 0.5, 0.3, 0.3, 0.1], [1, 1, 1, 1, 1], [1, 2, 2, 2, 3]),
    #landing exactly on a threshold doesn't switch (strict > and <, like Sarah's)
    ([0, 0.4], [1, 1], [1, 1]),
    ([0, 0.5, 0.2], [1, 1, 1], [1, 2, 2]),
    ([0, 0.5, 0.1, 0.1], [1, 1, 1, 0.25], [1, 2, 3, 3]),
    #SHORT can't jump straight back to ARTIFACT, even with high activity (only LONG can enter ARTIFACT)
    ([0, 0.5, 0.1, 0.9], [1, 1, 1, 1], [1, 2, 3, 3]),
    #a second full cycle works the same as the first
    ([0, 0.5, 0.1, 0.1, 0.5, 0.1], [1, 1, 1, 0.1, 1, 1], [1, 2, 3, 1, 2, 3]),
    #the first sample is LONG whatever its features say (Sarah's state(1) = LONG)
    ([0.9, 0.1], [1, 1], [1, 1]),
])
def test_smoothing_states_rules(activity, envelope, expected):
    states = smoothing_states(np.array(activity), np.array(envelope))
    assert np.array_equal(states, expected)


@pytest.mark.parametrize("states, expected", [
    #basic: the 2 samples before the first ARTIFACT sample, and that sample itself, become SHORT
    ([1, 1, 1, 1, 2, 2, 3], [1, 1, 3, 3, 3, 2, 3]),
    #near the start: the guard would run off the front, so it stops at sample 0. Without max(0, ...), start = -1 and
    #the slice [-1:2] is empty (a negative index counts from the end), so nothing would change and nothing would crash
    ([1, 2, 2, 3], [3, 3, 2, 3]),
    #no ARTIFACT anywhere: nothing changes
    ([1, 1, 3, 1], [1, 1, 3, 1]),
    #two artifact stretches: only the first one gets a guard
    ([1, 1, 1, 2, 3, 1, 1, 2, 3], [1, 3, 3, 3, 3, 1, 1, 2, 3]),
])
def test_guard_artifact_onset_cases(states, expected):
    #fs = 10000 makes the guard round(0.2 ms * 10000) = 2 samples
    assert np.array_equal(guard_artifact_onset(np.array(states), 10000), expected)


def test_guard_artifact_onset_is_0_2_ms_at_our_rate():
    #at 48828.125 Hz the guard is round(9.766) = 10 samples: with the first ARTIFACT sample at 20, samples 10-20 become
    #SHORT and sample 9 stays LONG
    states = np.full(40, LONG)
    states[20:25] = ARTIFACT
    states[25:] = SHORT
    expected = states.copy()
    expected[10:21] = SHORT
    assert np.array_equal(guard_artifact_onset(states, FS), expected)


def test_guard_artifact_onset_leaves_the_input_alone():
    states = np.array([1, 1, 1, 1, 2, 2, 3])
    before = states.copy()
    guard_artifact_onset(states, 10000)
    assert np.array_equal(states, before)
    #and a new array even when there's nothing to guard, so changing the result can't change the caller's states
    no_artifact = np.array([1, 1, 3, 1])
    assert not np.shares_memory(guard_artifact_onset(no_artifact, 10000), no_artifact)


def test_guard_artifact_onset_rounds_halves_up_like_matlab():
    #at 12.5 kHz the guard is 2.5 samples: MATLAB's round gives 3, Python's round would give 2 (halves go to even).
    #First ARTIFACT at index 5, so indices 2-5 become SHORT (with Python's round it would be 3-5)
    states = np.array([1, 1, 1, 1, 1, 2, 3])
    assert np.array_equal(guard_artifact_onset(states, 12500), [1, 1, 3, 3, 3, 3, 3])


@pytest.mark.parametrize("state, activity_value, deartifacted, expected", [
    #Sarah's windows at our rate (postprocess.m:728-736), after rounding, making odd and at least order + 2
    (LONG, 0, True, (29, 3)),       #round(0.6 ms) = 29, already odd
    (LONG, 1, True, (21, 3)),       #29 x 0.7 = 20.3 -> 20 -> odd 21: the up-to-30% shrink when the signal is sharp
    (LONG, 0.5, True, (25, 3)),     #29 x 0.85 = 24.65 -> 25
    (SHORT, 0, True, (15, 2)),      #round(0.3 ms) = 15
    (SHORT, 1, True, (15, 2)),      #only LONG uses the activity
    (ARTIFACT, 0, True, (11, 2)),   #round(0.2 ms) = 10 -> odd 11 (de-artifacted data)
    (ARTIFACT, 0, False, (5, 2)),   #round(0.1 ms) = 5 (the not-de-artifacted branch)
])
def test_state_window_at_our_rate(state, activity_value, deartifacted, expected):
    assert state_window(state, activity_value, FS, deartifacted) == expected


def test_state_window_is_at_least_order_plus_2():
    #at a low rate the windows would be tiny: at 5 kHz, 0.2 ms is 1 sample (odd already), but Sarah requires at least
    #order + 2 = 4 points, so the window becomes 4. (Sarah's order of steps allows an even window here; it can't happen at
    #our rate, where the smallest window is 5.)
    assert state_window(ARTIFACT, 0, 5000) == (4, 2)


def test_state_window_refuses_an_unknown_state():
    with pytest.raises(ValueError, match="LONG, SHORT or ARTIFACT"):
        state_window(0, 0, FS)


@pytest.mark.parametrize("states, expected", [
    ([1, 1, 2, 2, 2, 3, 1], [0, 2, 5, 6, 7]),   #the docstring example: four stretches, five fenceposts
    ([1, 1, 1], [0, 3]),                         #no change: one stretch covering everything
    ([1], [0, 1]),                               #a single sample
    ([2, 1], [0, 1, 2]),                         #a change right after the first sample
    ([1, 1, 3], [0, 2, 3]),                      #a change right before the last sample
])
def test_find_stretches(states, expected):
    assert np.array_equal(find_stretches(np.array(states)), expected)


def test_find_stretches_each_stretch_is_one_state():
    #on random states: every stretch holds a single state, and neighboring stretches differ
    states = np.random.default_rng(2).integers(1, 4, size=300)
    fenceposts = find_stretches(states)
    for k in range(len(fenceposts) - 1):
        stretch = states[fenceposts[k]:fenceposts[k + 1]]
        assert np.all(stretch == stretch[0])
        if k > 0:
            assert states[fenceposts[k]] != states[fenceposts[k] - 1]


def _smoothing_test_case():
    #50 LONG, 12 ARTIFACT, 5 SHORT samples of random data, activity 0 everywhere
    states = np.array([LONG] * 50 + [ARTIFACT] * 12 + [SHORT] * 5)
    signal = np.random.default_rng(3).normal(size=len(states))
    return signal, states, np.zeros(len(states))


def test_smooth_by_state_uses_each_stretchs_own_settings():
    #each stretch must equal savgol_filter on that stretch alone with its state's window and order:
    #LONG 29/3, ARTIFACT 11/2; the SHORT stretch (5 samples) is shorter than its window (15), so it stays raw
    from scipy.signal import savgol_filter
    signal, states, activity = _smoothing_test_case()
    smoothed = smooth_by_state(signal, states, activity, FS)
    assert np.allclose(smoothed[:50], savgol_filter(signal[:50], 29, 3, mode="interp"))
    assert np.allclose(smoothed[50:62], savgol_filter(signal[50:62], 11, 2, mode="interp"))
    assert np.array_equal(smoothed[62:], signal[62:])


def test_smooth_by_state_passes_deartifacted_on():
    #the not-de-artifacted branch uses the 5-sample ARTIFACT window (a hardcoded deartifacted=True would give 11)
    from scipy.signal import savgol_filter
    signal, states, activity = _smoothing_test_case()
    smoothed = smooth_by_state(signal, states, activity, FS, deartifacted=False)
    assert np.allclose(smoothed[50:62], savgol_filter(signal[50:62], 5, 2, mode="interp"))


def test_smooth_by_state_never_mixes_stretches():
    #invariance: a huge trap inside the ARTIFACT stretch must not change the smoothed LONG or SHORT stretches,
    #because each stretch is filtered on its own (no blending)
    signal, states, activity = _smoothing_test_case()
    trapped = signal.copy()
    trapped[50:62] += 1000
    before = smooth_by_state(signal, states, activity, FS)
    after = smooth_by_state(trapped, states, activity, FS)
    assert np.array_equal(before[:50], after[:50])
    assert np.array_equal(before[62:], after[62:])


def test_smooth_by_state_sizes_long_from_the_first_sample():
    #LONG's window shrinks with the activity at the stretch's FIRST sample only: activity 1 there gives 21 instead of
    #29, and activity anywhere else in the stretch changes nothing
    from scipy.signal import savgol_filter
    signal, states, activity = _smoothing_test_case()
    first_high = activity.copy()
    first_high[0] = 1
    assert np.allclose(smooth_by_state(signal, states, first_high, FS)[:50],
                       savgol_filter(signal[:50], 21, 3, mode="interp"))
    rest_high = activity.copy()
    rest_high[1:50] = 1
    assert np.array_equal(smooth_by_state(signal, states, rest_high, FS), smooth_by_state(signal, states, activity, FS))


def test_smooth_by_state_leaves_the_input_alone():
    signal, states, activity = _smoothing_test_case()
    before = signal.copy()
    smooth_by_state(signal, states, activity, FS)
    assert np.array_equal(signal, before)


def test_smooth_by_state_filters_a_stretch_exactly_as_long_as_its_window():
    #Sarah skips a stretch only when it's SHORTER than its window (numel(idx) < w): an 11-sample ARTIFACT stretch
    #(window 11) still gets smoothed. Catches "<=" in place of "<"
    from scipy.signal import savgol_filter
    states = np.array([LONG] * 50 + [ARTIFACT] * 11 + [SHORT] * 5)
    signal = np.random.default_rng(4).normal(size=len(states))
    smoothed = smooth_by_state(signal, states, np.zeros(len(states)), FS)
    assert np.allclose(smoothed[50:61], savgol_filter(signal[50:61], 11, 2, mode="interp"))


def test_smooth_by_state_full_cycle():
    #a whole cycle, LONG -> ARTIFACT -> SHORT -> LONG, with a SHORT stretch long enough to be filtered (20 >= 15) and a
    #second LONG stretch sized from ITS first sample (activity 1 there: window 21), not from sample 0 (activity 0: 29)
    from scipy.signal import savgol_filter
    states = np.array([LONG] * 40 + [ARTIFACT] * 12 + [SHORT] * 20 + [LONG] * 40)
    signal = np.random.default_rng(5).normal(size=len(states))
    activity = np.zeros(len(states))
    activity[72] = 1
    smoothed = smooth_by_state(signal, states, activity, FS)
    assert np.allclose(smoothed[:40], savgol_filter(signal[:40], 29, 3, mode="interp"))
    assert np.allclose(smoothed[40:52], savgol_filter(signal[40:52], 11, 2, mode="interp"))
    assert np.allclose(smoothed[52:72], savgol_filter(signal[52:72], 15, 2, mode="interp"))
    assert np.allclose(smoothed[72:], savgol_filter(signal[72:], 21, 3, mode="interp"))


def test_state_window_rounds_halves_up_like_matlab():
    #at 27.5 kHz LONG's base window is 16.5 samples. MATLAB rounds it to 17 (then 17 x 0.925 = 15.73 -> 16 -> odd 17);
    #Python's round would give 16 (-> 14.8 -> 15). Also catches multiplying before the first rounding (15.26 -> 15)
    assert state_window(LONG, 0.25, 27500) == (17, 3)


def test_subtract_median_baseline_starts_at_sample_145():
    #index-valued data: each sample's value is its own index, so the median reveals the start. With 300 samples the
    #median of samples 145-299 is (145 + 299) / 2 = 222; starting at 146 would give 222.5, at 144 221.5
    signal = np.arange(300.0)
    result = subtract_median_baseline(signal, FS)
    assert result[222] == 0
    assert np.allclose(signal - result, 222)


def test_subtract_median_baseline_zeroes_the_late_median():
    #on random data the result's median from sample 145 on is 0, and only a constant is subtracted
    signal = np.random.default_rng(6).normal(size=2295) + 5
    result = subtract_median_baseline(signal, FS)
    assert np.median(result[145:]) == pytest.approx(0, abs=1e-12)
    assert np.allclose(np.diff(signal - result), 0)


def test_subtract_median_baseline_leaves_the_input_alone():
    signal = np.random.default_rng(7).normal(size=500)
    before = signal.copy()
    result = subtract_median_baseline(signal, FS)
    assert np.array_equal(signal, before)
    assert not np.shares_memory(result, signal)


def _erna_like_signal():
    #an artifact at 2 ms, a 300 Hz ringing that dies out, and a little noise
    t = np.arange(2295) / FS
    ring = 2e-4 * np.sin(2 * np.pi * 300 * t) * np.exp(-np.clip(t - 0.002, 0, None) / 0.01) * (t > 0.0025)
    noise = 1e-5 * np.random.default_rng(8).normal(size=t.size)
    return 5e-3 * np.exp(-((t - 0.002) / 0.00005) ** 2) + ring + noise


def test_adaptive_smooth_chains_the_pieces_in_order():
    #the wrapper must equal the four pieces run by hand, in Sarah's order, for both branches
    signal = _erna_like_signal()
    activity, envelope = smoothing_features(signal, FS)
    states = guard_artifact_onset(smoothing_states(activity, envelope), FS)
    for deartifacted in (True, False):
        by_hand = smooth_by_state(signal, states, activity, FS, deartifacted=deartifacted)
        assert np.array_equal(adaptive_smooth(signal, FS, deartifacted=deartifacted), by_hand)


def test_adaptive_smooth_removes_noise_and_leaves_the_input_alone():
    #the late stretch is LONG (window 29), so sample-to-sample noise there drops a lot; the input isn't changed
    signal = _erna_like_signal()
    before = signal.copy()
    smoothed = adaptive_smooth(signal, FS)
    assert np.std(np.diff(smoothed[1500:])) < 0.2 * np.std(np.diff(signal[1500:]))
    assert np.array_equal(signal, before)
    assert len(smoothed) == len(signal)


def test_subtract_median_baseline_at_another_rate():
    #at 1.5 kHz, 3 ms is 4.5 samples: MATLAB rounds it to 5 (counted from 1), so the window starts at index 4. Index-valued
    #data: the median of 4..20 is 12. Python's round (4 -> index 3) would give 11.5, and a hardcoded 145 would fail
    signal = np.arange(21.0)
    assert np.allclose(signal - subtract_median_baseline(signal, 1500), 12)


def test_adaptive_smooth_passes_the_right_arrays_on(monkeypatch):
    #spy on the last step: it must get the original signal, the GUARDED states and the ACTIVITY (not the envelope), and
    #the deartifacted flag, which defaults to True. (Swapping in the envelope gave identical output on the ERNA-like
    #signal, so only a spy catches it.)
    from preprocess_dbs import postprocess
    seen = {}
    def record(signal, states, activity, fs, deartifacted=True):
        seen.update(signal=signal, states=states, activity=activity, fs=fs, deartifacted=deartifacted)
        return signal.copy()
    monkeypatch.setattr(postprocess, "smooth_by_state", record)
    signal = _erna_like_signal()
    activity, envelope = smoothing_features(signal, FS)
    guarded = guard_artifact_onset(smoothing_states(activity, envelope), FS)
    postprocess.adaptive_smooth(signal, FS)
    assert seen["signal"] is signal
    assert np.array_equal(seen["states"], guarded)
    assert np.array_equal(seen["activity"], activity) and not np.array_equal(seen["activity"], envelope)
    assert seen["fs"] == FS and seen["deartifacted"] is True
