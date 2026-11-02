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
| `epoch.condition_windows(times, labels, max_gap=2.0)` | How long each condition's epochs can be, the lab's way: up to 0.1 ms before the next pulse, and for a pair up to 0.2 ms before the pulse after it. A gap over `max_gap` seconds counts as a pause. |
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
