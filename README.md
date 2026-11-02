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

One test runs on uh3021's cleaned increment file and is skipped on a machine that doesn't have it.

## What's here so far

Import each function from its module, for example `from preprocess_dbs.epoch import classify_pulses`.

| function | what it does |
|---|---|
| `to_mne.to_mne(h5_path, stream, t1, t2, voice=None, confirmed_only=True)` | Loads one stream of a cleaned file between `t1` and `t2` seconds as an MNE Raw. Lead contacts are named `lead<N>_c<contact>`, and the stim pulses inside the window become `"stim"` annotations (only one voice's, or only confirmed ones, if you ask). A file with no stim table loads with no annotations. |
| `epoch.stim_times(h5_path, voice=None, confirmed_only=True)` | The pulse times from the cleaned stim table, optionally only one voice or only confirmed pulses. |
| `rereference.rereference(raw, leads, pairs=PAIRS)` | Bipolar pairs within each ring level of a directional lead, by default in the lab's order and sign (c3 - c2, c4 - c3, c2 - c4, c6 - c5, c7 - c6, c5 - c7). Contacts that aren't in the recording are skipped with a warning. |
| `epoch.classify_pulses(counts, periods, times, voices=None)` | Labels the pulses of a TDT stim table: `single`, `paired_<gap>_1/2`, `burst_1` to `burst_10`, `burstprobes_1` to `burstprobes_20`. Each voice is labeled on its own, and every group has to check out against the pulse times. A pulse it can't label gets `""`, with a warning that says why. |
| `epoch.group_by_gaps(times, max_gap_ms=40.0)` | Splits pulses into groups wherever the gap between them is longer than `max_gap_ms`. |
| `epoch.label_by_timing(times, clock_times, gap_times, amplitudes)` | Labels pulses from their timing alone, for BrainVision stim tables that have no stimulator count. Built for UG's ERP blocks: check other block types first. Zero-current groups get `sham_` in front, which needs the amplitudes. |
| `epoch.condition_windows(times, labels, max_gap=2.0)` | How long each condition's epochs can be, the lab's way: up to 0.1 ms before the next pulse, and for a pair up to 0.2 ms before the pulse after it. A gap over `max_gap` seconds counts as a pause. |
| `epoch.make_epochs(raw, tmin=-0.002, tmax=0.025, event="stim", baseline=None)` | Cuts one window around every event of one name. The default window isn't the lab's (the lab runs each epoch to the next pulse, see `condition_windows`). |
| `epoch.select_pulses(raw, gap_before=0.05, gap_after=0.0)` | Keeps the pulses with at least `gap_before` seconds of quiet before them (by default the first pulse of each pair or burst), and optionally `gap_after` after. |

The labeling and window functions warn or stop with a clear message when they can't do their job, instead of handing back a guessed result.

## Where it stands

Done:
- loading cleaned files into MNE
- bipolar re-referencing
- pulse labels for uh's TDT recordings and UG's BrainVision recordings
- the lab's epoch lengths for each condition (equal to the lab's parse.m rule recomputed for all 41 conditions in uh3021; not yet compared with the lab's saved outputs)

Not built yet, roughly in the lab's order:
- dropping the channels the lab marked do not use (clean_dbs version 7 stores them as `channel_dnu`)
- splitting each recording by stim contact and dropping the stimulated contact
- the rest segment
- the lab's other bipolar pairs: it makes 21 in all (the 6 here, plus vertical 5-2, 6-3, 7-4; ring pairs with the averages of 2-4 and 5-7, among them 8 against the average of 5-7; and every contact against 1), and its common average per lead. Its code means to make a common average per ECoG strip but never does (preprocess.m:693)
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
