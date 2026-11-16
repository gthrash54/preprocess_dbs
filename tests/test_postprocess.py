import numpy as np
import pytest
from preprocess_dbs.postprocess import trim_to_artifact
from preprocess_dbs.postprocess import crop_and_demean
from preprocess_dbs.postprocess import fit_drifts
from preprocess_dbs.postprocess import score_drifts
from preprocess_dbs.postprocess import choose_detrend
from preprocess_dbs.postprocess import detrend_channel

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

