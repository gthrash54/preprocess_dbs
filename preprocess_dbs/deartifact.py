import numpy as np


def raw_template(single, pp1, gap_s, fs):
    """
    The stim artifact on its own, from the lab's refractory-pair method (Nakhmani et al. 2024; deartifact.m:250-262)
    single = A(t) + B(t), the artifact plus the brain response. In the shortest pair (pp1, about 0.3-0.45 ms apart) the
    second pulse comes while the neurons are still refractory, so it adds only its artifact: pp1 = A(t) + B(t) + A(t - gap).
    pp1 - single leaves A(t - gap), the second pulse's artifact; shifting it back by the gap gives A(t), the template
    Input:
    single: the single-pulse average, channels x time (e.g. average_segment(...)["single"].data)
    pp1: the shortest pair's average, channels x time, same channels in the same order
    gap_s: time between the pair's two pulses in s (onset to onset, e.g. 0.000451)
    fs: sampling rate in Hz
    Returns: the template A(t), channels x time, gap_s*fs samples shorter than the shorter input. Not yet fine-aligned
        to the single pulse or blanked (the lab's next steps)
    """
    n = min(single.shape[1], pp1.shape[1]) #the two averages can differ by a few samples: use the shorter length
    single = single[:, :n]
    pp1 = pp1[:, :n]
    second_pulse_artifact = pp1 - single #A(t - gap): the second pulse's artifact, since it adds no brain response
    gap_samples = int(round(gap_s *fs)) #the pair's gap in samples (22 for 0.451 ms at 48.8 kHz)
    #shift back by the true gap: keep everything from gap_samples on, so the artifact starts where the first pulse's did.
    #The lab shifts by round(isi*fs) - 1 with isi = gap - 0.1 ms, about 5 samples short (its alignment step absorbs most of it)
    first_pulse_artifact = second_pulse_artifact[:, gap_samples:]
    return first_pulse_artifact
