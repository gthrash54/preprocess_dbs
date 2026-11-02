import h5py
import warnings
import numpy as np
import mne 
from pathlib import Path
from preprocess_dbs.pulses import stim_times
def to_mne(h5_path, stream, t1, t2, voice=None, confirmed_only=True):
    """
    Input:
    h5_path: a cleaned raw h5 file, ie from clean_dbs
    stream: which signal group to load eg. ecos (ecog and DBS lead contacts)
    t1, t2: start and end of the window, in seconds from the start of the recording
    voice, confirmed_only: which stim pulses to mark, passed on to stim_times (see epoch.py)
    Returns: MNE RAW array of specified window (volts). Lead contacts are renamed lead<N>_c<contact> (e.g. lead1_c2) whatever
    the rig called them, and channels are typed ecog, dbs, emg or misc. Every stim pulse inside the window is an
    annotation called "stim" (no annotations if the file has no stim table). Channels the lab marked do not use
    (channel_dnu in files from clean_dbs version 7 on) and ECoG strips that weren't placed (group *_not_placed) are in raw.info["bads"], so MNE's own tools skip them; the data is
    kept. Warns when the recording isn't on any do-not-use list ("none": nothing is known, which isn't the same as
    nothing bad) or the file is older than version 7
    """

    if not Path(h5_path).is_file():
        raise FileNotFoundError(f"no cleaned file at {h5_path}")
    with h5py.File(h5_path,"r") as f:
        g = f["streams"][stream]
        fs = g.attrs["fs"]
        if "channel_names" in g:
            names = [n.decode() for n in g["channel_names"][:]]
        else: #some streams (TDT EMG, mic) have no names, only input numbers
            names = [f"{stream}_{k}" for k in (g["channels"][:] if "channels" in g else range(1, g["data"].shape[0] + 1))]
        #what each channel is (lead1/lead2/ecog_anterior/emg... and contact number), filled in per rig by clean_dbs
        groups = [x.decode() for x in g["channel_group"][:]] if "channel_group" in g else [""] * len(names)
        contacts = [x.decode() for x in g["channel_contact"][:]] if "channel_contact" in g else [""] * len(names)
        #1 = the lab marked this channel do not use; dnu_list says where the list came from ("none" = recording not listed)
        dnu = g["channel_dnu"][:] if "channel_dnu" in g else None
        listed = g.attrs.get("dnu_list", "")
        start = g.attrs.get("start_time", 0.0) #seconds into the recording of this stream's first sample (usually 0)
        n = g["data"].shape[1]
        i0 = int((t1 - start) * fs) # time x fs = samples
        i1 = int((t2 - start) * fs)
        if not 0 <= i0 < i1 <= n:
            raise ValueError(f"window {t1}-{t2} s is outside this stream ({start:.3f}-{start + n / fs:.3f} s)")
        data = g["data"][:,i0:i1] # all data between set times
    names = [standard_name(n, grp, c) for n, grp, c in zip(names, groups, contacts)]
    types = [channel_type(n, grp) for n, grp in zip(names, groups)]
    info = mne.create_info(names, fs, types)
    raw = mne.io.RawArray(data, info)
    if dnu is not None:
        raw.info["bads"] = [name for name, flag in zip(names, dnu) if flag == 1]
    #clean_dbs puts a strip that wasn't placed on the brain in group ecog_<side>_not_placed; its channels are still typed
    #ecog, so mark them bad too, or ECoG pairs and averages would be made from contacts recording nothing
    unplaced = [name for name, grp in zip(names, groups) if grp.endswith("_not_placed") and name not in raw.info["bads"]]
    raw.info["bads"] = raw.info["bads"] + unplaced
    if dnu is None:
        warnings.warn(f"{Path(h5_path).name} {stream}: no do-not-use flags (cleaned before clean_dbs version 7), so bad channels aren't marked")
    elif listed in ("", "none"):
        warnings.warn(f"{Path(h5_path).name} {stream}: this recording isn't on any do-not-use list, so bad channels aren't known "
                      "(not the same as none being bad)")
    with h5py.File(h5_path, "r") as f:
        has_stim = "stim" in f
    times = stim_times(h5_path, voice, confirmed_only) if has_stim else np.array([]) #no stim table (old file): the signal is still fine, no pulses to mark
    if has_stim and voice is not None and len(times) == 0:
        warnings.warn(f'no pulses with voice "{voice}" in {Path(h5_path).name} (BrainVision files have no voices)')
    t0 = start + i0 / fs #exact time of the Raw's first sample (t1 rounded down to a whole sample)
    inside = (times >= t0) & (times < t0 + data.shape[1] / fs)
    onsets = times[inside] - t0
    raw.set_annotations(mne.Annotations(onset = onsets, duration = 0, description = "stim"))
    return raw


def standard_name(name, group, contact):
    #every rig names lead contacts its own way (TDT "lead1_c2", BrainVision "DBS2"); the cleaned file's group and contact say
    #which lead and contact it really is, so the rest of the pipeline only ever sees one name per contact
    if group.startswith("lead") and contact.isdigit():
        return f"{group}_c{contact}"
    return name


def channel_type(name, group):
    #MNE channel type from the cleaned file's channel group (falls back to the name for files without groups)
    if group.startswith("ecog") or name.startswith("ecog"):
        return "ecog"
    if group.startswith("lead") or group == "dbs" or name.startswith(("lead", "dbs")):
        return "dbs"
    if group == "emg":
        return "emg"
    return "misc"
