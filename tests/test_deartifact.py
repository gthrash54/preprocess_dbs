import numpy as np
from preprocess_dbs.deartifact import raw_template
from preprocess_dbs.deartifact import align_template
from preprocess_dbs.deartifact import blank_template
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
