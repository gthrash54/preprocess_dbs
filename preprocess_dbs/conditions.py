import csv
from pathlib import Path

#the lab's names for paired pulses (parse.m pair lookup) and the ERNA script's labels for them
#(incrementingExperiment_pull_wave_parameters.m:47-57), and the patients whose stimulator periods were doubled.
#Kept as tables (tables/*.csv), not in code, so a new patient or name is a row, not a code change
tables_dir = Path(__file__).parent / "tables"

#uh3001/uh3006: the lab forces the time after a pair's second pulse to 49.89168 ms minus 0.1 ms, so their pair epochs are
#as long as everyone else's. The lab actually subtracts 0.0001 ms (a units slip, bug B8, parse.m:261-263); we take 0.1 ms
#off like every other gap, so these epochs are 0.0999 ms (about 5 samples) shorter than the lab's
doubled_pair_tail = 0.04989168 - 0.0001


def _rows(name):
    with open(tables_dir / name, newline="") as f:
        return list(csv.DictReader(f))


def is_doubled(study_id):
    """True for the patients whose stimulator periods were doubled (tables/doubled_period_patients.csv), e.g. uh3001"""
    return study_id in {r["study_id"] for r in _rows("doubled_period_patients.csv")}


def lab_pair_name(label, doubled=False, tol_ms=0.02):
    """
    Input:
    label: a pair label from classify_pulses, e.g. "paired_0.45_1" (either pulse of the pair)
    doubled: True for uh3001/uh3006 (see is_doubled), whose gaps map to a different table
    tol_ms: how close the label's gap has to be to a table gap (classify_pulses rounds gaps to 0.01 ms)
    Returns: (lab_name, erna_label), e.g. ("pp1", "pp0.3"). erna_label is None where the lab's ERNA script skips the
        condition, and both are None when the gap isn't in the table (the lab's "CHECK")
    """
    parts = label.split("_")
    if len(parts) != 3 or parts[0] != "paired":
        raise ValueError(f"{label!r} isn't a pair label like paired_0.45_1")
    gap = float(parts[1])
    for r in _rows("pair_names.csv"):
        if int(r["doubled"]) == int(doubled) and abs(float(r["gap_ms"]) - gap) <= tol_ms:
            return r["lab_name"], (r["erna_label"] or None)
    return None, None
