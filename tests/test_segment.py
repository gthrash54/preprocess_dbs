import warnings
import numpy as np
import pytest
from preprocess_dbs.segment import split_by_stim_contact

def _pulses(n=5,start=10.0, gaps=(5.0,5.0), spacing = 0.05):
    #fake block of n pulses per contact with long pause for cable move simulation in between pulses. This is to test segment.py
    run = np.arange(n) *spacing # one run starting at 0s: 0, 0.05, 0.10... with a gap of 5 showing a move
    run1 = run + start
    run2 = run + (run1[-1] + gaps[0])
    run3 = run + (run2[-1] + gaps[1])
    return np.concatenate([run1,run2,run3])


def test_three_runs_give_three_segments():
    #3 runs of 5 pulses: 10.00-10.20 (pulses 0-4), 15.20-15.40 (5-9), 20.40-20.60 (10-14)
    seg = split_by_stim_contact(_pulses())
    assert list(seg) == ["stim2", "stim3", "stim4"]

    t1, t2, idx = seg["stim2"]
    assert t1 == pytest.approx(10.00 - 0.002)
    assert t2 == pytest.approx(10.20 + 0.05)
    assert list(idx) == [0, 1, 2, 3, 4]

    t1, t2, idx = seg["stim3"]
    assert t1 == pytest.approx(15.20 - 0.002)
    assert t2 == pytest.approx(15.40 + 0.05)
    assert list(idx) == [5, 6, 7, 8, 9]

    t1, t2, idx = seg["stim4"]
    assert t1 == pytest.approx(20.40 - 0.002)
    assert t2 == pytest.approx(20.60 + 0.05)
    assert list(idx) == [10, 11, 12, 13, 14]


def test_lone_first_pulse_is_dropped():
    #a lone pulse at 5 s, then the 3 runs: long pauses after pulses 0, 5 and 10. The lab ignores the pause after the
    #first pulse (preprocess.m:249, idx(idx > 1)) but keeps that pulse in stim2 (start_pts(1) = first pulse).
    #Deliberate difference: we drop the pulse too, so it's in no segment and stim2 starts at run 1
    times = np.concatenate([[5.0], _pulses()])
    with warnings.catch_warnings():
        warnings.simplefilter("error") #any warning fails the test: 2 moves are left, so there should be none
        seg = split_by_stim_contact(times)

    t1, t2, idx = seg["stim2"]
    assert t1 == pytest.approx(10.00 - 0.002) #run 1's first pulse, not the lone one (the lab's would be 5.00 - 0.002)
    assert t2 == pytest.approx(10.20 + 0.05)
    assert list(idx) == [1, 2, 3, 4, 5]
    assert list(seg["stim3"][2]) == [6, 7, 8, 9, 10]
    assert list(seg["stim4"][2]) == [11, 12, 13, 14, 15]
    assert all(0 not in seg[name][2] for name in seg) #the lone pulse is in no segment


def test_missed_move_warns_uneven_runs():
    #6 pulses per contact, but the 2nd cable move only takes 2 s (under gap_s), and contact 3's run has a 5 s pause
    #in the middle. The code takes that pause for the 2nd move, so the runs come out 6 / 3 / 9 pulses: should warn
    run = np.arange(6) * 0.05
    half = np.arange(3) * 0.05
    run1 = run + 10.0
    run2a = half + (run1[-1] + 5.0)  #real move 1 (5 s), then the first half of contact 3's run
    run2b = half + (run2a[-1] + 5.0) #the pause inside contact 3's run (5 s)
    run3 = run + (run2b[-1] + 2.0)   #real move 2, too fast to be seen (2 s)
    times = np.concatenate([run1, run2a, run2b, run3])

    with pytest.warns(UserWarning, match="uneven"):
        seg = split_by_stim_contact(times)
    assert [len(seg[name][2]) for name in seg] == [6, 3, 9] #the wrong split it warned about

    

def test_pause_of_exactly_gap_s_is_not_a_move():
    #the 2nd pause is exactly 3.0 s. The lab's rule is strictly greater (diff > 3, preprocess.m:248), so it isn't a
    #cable move: only 1 move is found, and the code can't tell the contacts apart
    with pytest.raises(ValueError, match="found 1"):
        split_by_stim_contact(_pulses(gaps=(5.0, 3.0)))
