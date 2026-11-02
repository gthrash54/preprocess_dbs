import numpy as np
from preprocess_dbs.deartifact import raw_template

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
