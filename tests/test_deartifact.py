import numpy as np
from preprocess_dbs.deartifact import raw_template
from preprocess_dbs.deartifact import align_template
from preprocess_dbs.deartifact import blank_template
from preprocess_dbs.deartifact import subtract_template
from preprocess_dbs.align import shift_signal
from preprocess_dbs import deartifact
import pytest
FS = 48828.125


def _fake_pair(n=600, gap=22, onset=98, decay=10.0, scale=1.0):
    #a made-up artifact A (starts at the pulse, decays) and brain response B (starts later); single = A + B, and the
    #refractory pair adds only a second copy of A, gap samples later (the method's assumption)
    t = np.arange(n)
    A = scale * np.where(t >= onset, np.exp(-(t - onset) / decay), 0.0)
    B = np.where(t >= onset + 50, 0.1 * np.sin((t - onset - 50) / 5.0), 0.0)
    single = A + B
    pp1 = A + B + np.concatenate([np.zeros(gap), A[:-gap]])
    return single, pp1, A


def test_template_is_the_artifact_alone():
    single, pp1, A = _fake_pair()
    template = raw_template(single[None], pp1[None], 22 / FS, FS)
    assert template.shape == (1, 600 - 22) #shorter by the gap
    assert np.allclose(template[0], A[:600 - 22]) #the brain response B is gone


def test_template_trims_unequal_lengths():
    #parse.m's pair epoch can be a few samples longer than the single's: use the shorter length
    single, pp1, A = _fake_pair()
    template = raw_template(single[None, :590], pp1[None], 22 / FS, FS)
    assert template.shape == (1, 590 - 22)
    assert np.allclose(template[0], A[:590 - 22])


def test_template_per_channel():
    #each channel has its own artifact size and shape; each is recovered on its own
    pairs = [_fake_pair(decay=d, scale=s) for d, s in ((10.0, 1.0), (4.0, -3.0), (20.0, 0.5))]
    single = np.vstack([p[0] for p in pairs])
    pp1 = np.vstack([p[1] for p in pairs])
    template = raw_template(single, pp1, 22 / FS, FS)
    for ch, (_, _, A) in enumerate(pairs):
        assert np.allclose(template[ch], A[:600 - 22])


def test_align_template_known_lags_as_shift():
    #fake data: 3 channels, each just one smooth peak standing in for the artifact
    t = np.arange(600.0) #sample numbers 0 to 599
    lags = [0.0, 1.7, -2.3] #how far each channel's template peak sits from the single pulse's peak, in samples
    #single pulse: all 3 channels peak at sample 118 (the stim is at sample 98)
    single = np.array([np.exp(-((t - 118) / 4) ** 2) + t / 6000 for lag in lags])
    #template: same peak, moved by each channel's lag, so the peaks are at 118, 119.7 and 115.7
    template = np.array([np.exp(-((t - 118 - lag) / 4) ** 2) + t / 6000 for lag in lags])
    #(+ t/6000 tilts the signal up slightly; without it pchip warns about dividing by numbers too close to zero)
    aligned,shifts = align_template(single,template,FS)
    assert shifts == pytest.approx(lags,abs=0.1)
    assert aligned.shape == template.shape
    #the moved template should now sit on top of the single pulse (all peaks back at 118); if the shift went the
    #wrong way, channels 2 and 3 would end up further off instead
    assert np.allclose(aligned[:, 100:300], single[:, 100:300], atol=0.05)


def test_align_template_leaves_the_input_alone():
    #align_template fills in a copy; the template passed in should come back exactly as it went in
    t = np.arange(600.0)
    single = np.array([np.exp(-((t - 118) / 4) ** 2) + t / 6000])
    template = np.array([np.exp(-((t - 120) / 4) ** 2) + t / 6000]) #one channel, peak 2 samples late
    before = template.copy()
    align_template(single, template, FS)
    assert np.array_equal(template, before)


def test_align_template_measures_on_sarahs_windows(monkeypatch):
    #checks exactly which windows get measured, with Sarah's recipe (deartifact.m:265-272): start one sample before
    #the stim (Python 97), demean from there to the END of the epoch, then keep 4.5 ms (220 samples).
    #monkeypatch swaps align_signals for a stand-in that just writes down what it was given, so we can look at
    #the windows directly. A window starting at the stim (98), or demeaned after cutting, fails here
    seen = []
    def record_windows(x, y, fs):
        seen.append((x, y))
        return 0.0, 0.0, 0.0 #a shift of 0, so nothing gets moved
    monkeypatch.setattr(deartifact, "align_signals", record_windows)
    rng = np.random.default_rng(0)
    #2 channels of noise on a slope: on a slope, the mean of the whole tail and the mean of the first 220 samples differ
    single = rng.normal(size=(2, 600)) + np.linspace(0, 5, 600)
    template = rng.normal(size=(2, 578)) + np.linspace(0, 5, 578)
    align_template(single, template, FS)
    assert len(seen) == 2 #one measurement per channel
    for ch, (x, y) in enumerate(seen):
        assert np.allclose(x, (single[ch, 97:] - single[ch, 97:].mean())[:220])
        assert np.allclose(y, (template[ch, 97:] - template[ch, 97:].mean())[:220])


def test_blank_template_flattens_sarahs_stretch():
    #Sarah's two pp1 gaps: 0.451 ms flattens samples 97-131, 0.246 ms flattens 97-121 (deartifact.m:309-314).
    #inside the stretch each channel becomes its own mean of samples 0-97; everything outside stays as it was
    rng = np.random.default_rng(0)
    template = rng.normal(size=(2, 400)) + np.array([[5.0], [-3.0]]) #2 channels of noise around different levels
    for gap_s, last in ((0.000451, 131), (0.000246, 121)):
        blanked = blank_template(template, gap_s, FS)
        changed = np.flatnonzero(np.any(blanked != template, axis=0)) #which samples (columns) were changed
        assert changed[0] == 97
        assert changed[-1] == last
        assert len(changed) == last - 97 + 1 #one unbroken stretch, no gaps
        for ch in range(2):
            assert np.allclose(blanked[ch, 97:last + 1], template[ch, :98].mean())


def test_blank_template_can_be_switched_off():
    #blank=False gives the template back unchanged; in both modes the caller's template is never modified
    rng = np.random.default_rng(1)
    template = rng.normal(size=(2, 400))
    before = template.copy()
    unblanked = blank_template(template, 0.000451, FS, blank=False)
    assert np.array_equal(unblanked, template)
    assert not np.shares_memory(unblanked, template) #a separate copy, not the caller's array handed back
    blank_template(template, 0.000451, FS)
    assert np.array_equal(template, before)


def _bump(t, at):
    #a smooth stand-in artifact peaking at sample `at`, on a slight tilt (keeps pchip from warning about tiny numbers)
    return np.exp(-((t - at) / 4) ** 2) + t / 6000


def test_subtract_template_removes_the_artifact_and_keeps_the_response():
    #a single-pulse condition: artifact peaking at 119.6 plus a "brain response" starting at sample 250. The
    #template's artifact peaks at 118, 1.6 samples EARLIER, so the shift is about -1.6 and the data moves back onto it
    t = np.arange(600.0)
    response = np.where(t > 250, 0.05 * np.sin((t - 250) / 8), 0.0)
    data = np.array([_bump(t, 119.6) + response])
    template = np.array([_bump(t, 118)])
    before = data.copy()
    clean, shifts, peaks = subtract_template(data, template, FS)
    assert shifts == pytest.approx([-1.6], abs=0.1)
    assert np.abs(clean[0, 100:200]).max() < 0.03 #the artifact (peak 1) is gone, about 1% left
    #the response is still there, moved by the same shift as the data (it's the data that moves, not the template;
    #if the template were moved instead, the response would sit 1.6 samples off and fail this)
    assert np.allclose(clean[0, 300:550], shift_signal(response, shifts[0])[300:550], atol=0.002)
    #the data's artifact peak in Sarah's window, before moving: 119.6 - 97 (the window start) = 22.6
    assert peaks == pytest.approx([22.6], abs=0.1)
    assert np.array_equal(data, before) #the caller's data isn't changed


def test_subtract_template_pair_subtracts_the_single_first():
    #for a pair, subtract_template should do exactly raw_template (pair minus single, shifted back by the gap) and then
    #the same alignment and subtraction as for any other condition. The result is the gap (22 samples) shorter
    t = np.arange(600.0)
    single = np.array([_bump(t, 118)])
    #the pair's second pulse: its own artifact 22 samples (0.451 ms) after the first, plus its own response
    second_response = np.where(t > 272, 0.05 * np.sin((t - 272) / 8), 0.0)
    pair = single + np.array([_bump(t, 140) + second_response])
    template = np.array([_bump(t, 118)])
    clean_pair, shifts, peaks = subtract_template(pair, template, FS, single=single, gap_s=22 / FS)
    by_hand, _, _ = subtract_template(raw_template(single, pair, 22 / FS, FS), template, FS)
    assert clean_pair.shape == (1, 600 - 22)
    assert np.allclose(clean_pair, by_hand)
    #checked independently too: what's left is the second pulse's response, moved back 22 samples to start at 250
    #(plus a constant 22/6000, because the second pulse's tilt sits 22 samples further along than the template's)
    t_left = t[:600 - 22]
    expected = np.where(t_left > 250, 0.05 * np.sin((t_left - 250) / 8), 0.0) + 22 / 6000
    assert np.allclose(clean_pair[0], expected)


def test_subtract_template_with_different_lengths():
    #Sarah's lines 471-480: only the samples both have get the template subtracted; extra data samples are kept, and a
    #longer template is cut to the data's length. The output is always as long as the data
    t = np.arange(600.0)
    response = np.where(t > 250, 0.05 * np.sin((t - 250) / 8), 0.0)
    data = np.array([_bump(t, 118) + response])
    short_template = np.array([_bump(t, 118)[:500]]) #artifacts line up exactly, so nothing gets moved
    clean, _, _ = subtract_template(data, short_template, FS)
    assert clean.shape == (1, 600)
    assert np.allclose(clean[0, :500], response[:500]) #template subtracted where it exists
    assert np.allclose(clean[0, 500:], data[0, 500:]) #past its end, the data is kept as it was
    clean_short, _, _ = subtract_template(data[:, :450], np.array([_bump(t, 118)]), FS)
    assert clean_short.shape == (1, 450) #a longer template is cut to the data's length
    assert np.allclose(clean_short[0], response[:450]) #and still subtracted all the way to the data's last sample

