# preprocess_dbs

Preprocessing for the Walker lab's intraoperative DBS recordings, built on MNE. It starts from the cleaned `.h5` files that [clean_dbs](https://github.com/gthrash54/clean_dbs) makes, so it never touches raw TDT or BrainVision files.

Work in progress. It follows the lab's ERNA pipeline (the lab's MATLAB code), with the lab's known bugs fixed. The functions that work differently from the lab say how in their docstrings.

## Install

```
git clone https://github.com/gthrash54/preprocess_dbs
cd preprocess_dbs
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Two tests run on uh3021's cleaned increment file and are skipped on a machine that doesn't have it.

## Quick start

This runs without any recordings. It writes a small synthetic file, `demo.h5`, in the current folder, laid out like a clean_dbs output: one 8-contact lead, ten single pulses and ten pairs, and a made-up response on contact 3. Then it loads the file, re-references it, labels the pulses and averages the single pulses. preprocess_dbs only reads clean_dbs's `.h5` files, so clean_dbs doesn't have to be installed.

To use the package without cloning it (the editable install above is for changing the code or running the tests):

```
pip install git+https://github.com/gthrash54/preprocess_dbs
```

Save this as `quickstart.py` and run `python quickstart.py`:

```python
import h5py
import mne
import numpy as np
from preprocess_dbs.to_mne import to_mne
from preprocess_dbs.pulses import stim_times, classify_pulses
from preprocess_dbs.rereference import add_virtual_rings, dbs_rereference, erna_pairs
from preprocess_dbs.epoch import average_segment

mne.set_log_level("WARNING")  # hide MNE's progress messages

# 3 s of noise on one 8-contact lead at 10 kHz, in volts
fs = 10_000.0
rng = np.random.default_rng(0)
data = 2e-6 * rng.standard_normal((8, 30_000))

# stim: 10 single pulses, then 10 pairs 2.89 ms apart, 100 ms between groups
singles = 0.5 + 0.1 * np.arange(10)
firsts = 1.5 + 0.1 * np.arange(10)
times = np.sort(np.r_[singles, firsts, firsts + 0.00289])
counts = np.where(times < 1.45, 1, 2)            # pulses in each pulse's group
periods = np.where(times < 1.45, 49.99, 2.89)    # stimulator period, ms

# a made-up 300 Hz response after every pulse, on contact 3 only
t = np.arange(0, 0.02, 1 / fs)
response = 20e-6 * np.sin(2 * np.pi * 300 * t) * np.exp(-t / 0.005)
for p in times:
    data[2, int(p * fs):int(p * fs) + len(t)] += response

# save it the way clean_dbs lays out a cleaned file
with h5py.File("demo.h5", "w") as f:
    g = f.create_group("streams/ecos")
    g["data"] = data.astype(np.float32)
    g["channel_names"] = np.array([f"lead1_c{k}" for k in range(1, 9)], dtype="S")
    g["channel_group"] = np.array(["lead1"] * 8, dtype="S")
    g["channel_contact"] = np.array([str(k) for k in range(1, 9)], dtype="S")
    g["channel_dnu"] = np.zeros(8, dtype=np.int8)  # 1 = do not use
    g.attrs["fs"] = fs
    g.attrs["dnu_list"] = "demo"
    s = f.create_group("stim")
    s["time"], s["count"], s["period_ms"] = times, counts, periods
    s["voice"] = np.array(["A"] * len(times), dtype="S")
    s["confirmed"] = np.ones(len(times), dtype=np.int8)

# load, re-reference, label, average
raw = to_mne("demo.h5", "ecos", 0, 3)  # MNE Raw; pulses are "stim" annotations
bipolar = dbs_rereference(add_virtual_rings(raw, ["lead1"]), ["lead1"], erna_pairs)
with h5py.File("demo.h5", "r") as f:
    s = f["stim"]
    labels = classify_pulses(s["count"][:], s["period_ms"][:], s["time"][:], voices=s["voice"][:])
print(sorted(set(labels)))

evoked = average_segment(bipolar, stim_times("demo.h5"), labels)
single = evoked["single"]
print(f"single pulse, average of {single.nave}, peak to peak:")
for name, x in zip(single.ch_names, single.data):
    print(f"  {name:20s} {np.ptp(x) * 1e6:5.1f} uV")
```

It prints:

```
['paired_2.89_1', 'paired_2.89_2', 'single']
single pulse, average of 10, peak to peak:
  lead1_c3-lead1_c2     30.9 uV
  lead1_c4-lead1_c3     31.9 uV
  lead1_c2-lead1_c4      6.2 uV
  lead1_c6-lead1_c5      6.2 uV
  lead1_c7-lead1_c6      5.9 uV
  lead1_c5-lead1_c7      6.1 uV
  lead1_c8-lead1_c567    4.8 uV
```

The made-up response is only on contact 3, so only the two pairs that use contact 3 carry it; the others are averaged noise. Each pair `A-B` is A minus B. `erna_pairs` is the lab's 7-pair set (see the table below), and `lead1_c567` is the virtual ring that `add_virtual_rings` makes from contacts 5, 6 and 7.

## What's here so far

Import each function from its module, for example `from preprocess_dbs.pulses import classify_pulses`.

| function | what it does |
|---|---|
| `to_mne.to_mne(h5_path, stream, t1, t2, voice=None, confirmed_only=True)` | Loads one stream of a cleaned file between `t1` and `t2` seconds as an MNE Raw. Lead contacts are named `lead<N>_c<contact>`, and the stim pulses inside the window become `"stim"` annotations (only one voice's, or only confirmed ones, if you ask). A file with no stim table loads with no annotations. |
| `pulses.stim_times(h5_path, voice=None, confirmed_only=True)` | The pulse times from the cleaned stim table, optionally only one voice or only confirmed pulses. |
| `segment.split_by_stim_contact(times, gap_s=3.0)` | Splits an increment block into `stim2`, `stim3` and `stim4` at the cable moves (pauses over `gap_s`), with each segment's data window (first pulse - 2 ms to last pulse + 50 ms) and its pulse rows. Drops a lone first pulse, and warns when the runs come out uneven (a cable move may have been missed). |
| `segment.rest_window(times, impedance_times=())` | The rest window before stimulation: from 1 s (or 9 s after the last impedance check) to 0.5 s before the first pulse, or `None` if there's less than 3 s of room. |
| `segment.mark_bad_stim_contact(raw, segment, stim_lead)` | Marks the stimulated contact (e.g. `lead1_c2` in stim2) bad by name, so later steps skip it. `stim_lead` comes from the block's stim contacts, not from the experiment name. |
| `rereference.add_virtual_rings(raw, leads)` | Adds `lead<N>_c234` and `lead<N>_c567`, the mean of each segmented level, when all three of its segments are usable. |
| `rereference.dbs_rereference(raw, leads, pairs=all_pairs)` | The lab's 21 bipolar pairs per lead, in its order and sign (`(A, B)` = A - B). Also `horizontal_pairs`, `vertical_pairs`, `ring_pairs`, `c1_pairs`, and `erna_pairs` (the 7 the lab uses for ERNA). A pair with a missing or bad contact is skipped with a warning, and the output holds only pairs. |
| `rereference.common_average(raw, lead)` | Each usable contact c1-c8 minus the mean of the lead's usable contacts. |
| `rereference.ecog_rereference(raw, pairs=ecog_pairs)` | Neighbouring-contact pairs on each ECoG strip, never across strips. Unlike the lab, a bad middle contact isn't bridged (the pair is skipped, with a warning). |
| `rereference.ecog_common_average(raw)` | Each usable ECoG contact minus its own strip's mean. The lab means to make this but never does (preprocess.m:693). |
| `pulses.classify_pulses(counts, periods, times, voices=None)` | Labels the pulses of a TDT stim table: `single`, `paired_<gap>_1/2`, `burst_1` to `burst_10`, `burstprobes_1` to `burstprobes_20`. Each voice is labeled on its own, and every group has to check out against the pulse times. A pulse it can't label gets `""`, with a warning that says why. |
| `pulses.group_by_gaps(times, max_gap_ms=40.0)` | Splits pulses into groups wherever the gap between them is longer than `max_gap_ms`. |
| `pulses.label_by_timing(times, clock_times, gap_times, amplitudes)` | Labels pulses from their timing alone, for BrainVision stim tables that have no stimulator count. Built for UG's ERP blocks: check other block types first. Zero-current groups get `sham_` in front, which needs the amplitudes. |
| `conditions.lab_pair_name(label, doubled=False)` | The lab's name and ERNA label for a pair (`paired_0.45_1` → `pp1`, `pp0.3`), from `tables/pair_names.csv`. `is_doubled(study_id)` says whether a patient's stimulator periods were doubled (uh3001, uh3006), from `tables/doubled_period_patients.csv`. |
| `epoch.condition_windows(times, labels, max_gap=2.0, pair_tail=None)` | How long each condition's epochs can be, the lab's way: up to 0.1 ms before the next pulse, and for a pair up to 0.2 ms before the pulse after it. A gap over `max_gap` seconds counts as a pause. |
| `epoch.make_epochs(raw, tmin=-0.002, tmax=0.025, event="stim", baseline=None)` | Cuts one window around every event of one name. The default window isn't the lab's (the lab runs each epoch to the next pulse, see `condition_windows`). |
| `epoch.select_pulses(raw, gap_before=0.05, gap_after=0.0)` | Keeps the pulses with at least `gap_before` seconds of quiet before them (by default the first pulse of each pair or burst), and optionally `gap_after` after. |

The labeling and window functions warn or stop with a clear message when they can't do their job, instead of handing back a guessed result.

## Where it stands

Done:
- loading cleaned files into MNE, with do-not-use channels and unplaced ECoG strips as bads
- splitting increment blocks by stim contact, the rest window, and marking the stim contact bad
- re-referencing: the lab's 21 DBS bipolar pairs with virtual rings, the DBS common average per lead, ECoG bipolar pairs and the ECoG common average per strip
- pulse labels for uh's TDT recordings and UG's BrainVision recordings
- the lab's epoch lengths for each condition (equal to the lab's parse.m rule recomputed for all 41 conditions in uh3021; not yet compared with the lab's saved outputs)

Not built yet, roughly in the lab's order:
- resampling EMG to 12.5 kHz
- linear detrending
- cutting each condition's epochs with its own window (`make_epochs` cuts one window around one event name)
- removing the stim artifact
- post-processing and the ERNA measures
- checking each stage against the lab's saved outputs

## Which recordings it works on

- **uh, TDT:** pulse labels and epoch lengths work. The stimulator logs each group's count and period.
- **UG, BrainVision:** pulse labels come from the timing, since the markers don't carry a count.
- **Crosstalk (two voices at once):** labels come out per voice with a warning, because a condition that uses both leads isn't in them. Epoch lengths refuse these tables for now, since which pulses should end a window isn't decided.
- **The other BrainVision cohorts:** not yet. Their labels need their own source (a condition column, TTL codes or timing), and some need marker fixes in clean_dbs first.
