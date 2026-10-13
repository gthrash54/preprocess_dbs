import h5py
import warnings
import numpy as np
from pathlib import Path

def stim_times(h5_path, voice=None, confirmed_only=True):
    """
    Input:
    h5_path: a cleaned h5 file from clean_dbs
    voice: which stimulator output to keep ("A", "B", ...), None keeps every voice. TDT only: BrainVision files have no voices
    confirmed_only: True keeps only the pulses clean_dbs is sure are real stim (confirmed = 1). TDT pulses always are;
        BrainVision pulses only when a stimulator file backs them up, so some BrainVision blocks need False
    Returns: the pulse times as a numpy array, in seconds from the start of the recording (empty if the block had no stim)
    """
    if not Path(h5_path).is_file():
        raise FileNotFoundError(f"no cleaned file at {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if "stim" not in f:
            raise KeyError(f"{h5_path} has no stim table (cleaned with an old clean_dbs version?)")
        stim = f["stim"]
        times = stim["time"][:]
        keep = np.ones(len(times), dtype=bool) #start with every pulse kept
        if confirmed_only:
            keep &= stim["confirmed"][:] == 1 #&= means "and also": drop any pulse that isn't confirmed
            if len(times) and not keep.any():
                warnings.warn(f"all {len(times)} pulses in {Path(h5_path).name} are unconfirmed, none kept; "
                              "use confirmed_only=False if you trust them")
            elif not keep.all():
                warnings.warn(f"{(~keep).sum()} of {len(times)} pulses in {Path(h5_path).name} are unconfirmed and were dropped")
        if voice is not None:
            keep &= stim["voice"][:] == voice.encode() #the file stores voices as bytes (b"A")
    return times[keep]
