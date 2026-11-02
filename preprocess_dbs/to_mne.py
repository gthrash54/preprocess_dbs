import h5py # reads HDF5 files (.h5), the format clean_dbs saves in
import warnings
import numpy as np
import mne
from pathlib import Path # Path objects handle file paths (check they exist, get the file name, ...)
from preprocess_dbs.pulses import stim_times # "from X import Y" brings in just Y from this toolbox's pulses.py
#voice=None, confirmed_only=True are default arguments; None is python's "no value" (here: don't filter by voice)
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

    #fail early with a clear message rather than a confusing h5py error
    if not Path(h5_path).is_file():
        raise FileNotFoundError(f"no cleaned file at {h5_path}")
    #"with ... as f" opens the file and closes it automatically when the indented block ends (even after an error).
    #"r" = read only
    with h5py.File(h5_path,"r") as f:
        #an h5 file is like folders of arrays: f["streams"]["ecos"] is the ecos stream's group
        g = f["streams"][stream]
        #.attrs holds small labelled values stored alongside the arrays; "fs" = sample rate in Hz
        fs = g.attrs["fs"]
        #"name" in g is True if the group has an entry with that name
        if "channel_names" in g:
            #[:] reads the whole array from the file; h5 stores text as bytes, .decode() turns each into a normal string
            names = [n.decode() for n in g["channel_names"][:]]
        else: #some streams (TDT EMG, mic) have no names, only input numbers
            #g["data"].shape = (channels, samples), so .shape[0] = the number of channels; range(1, n+1) = 1 to n
            names = [f"{stream}_{k}" for k in (g["channels"][:] if "channels" in g else range(1, g["data"].shape[0] + 1))]
        #what each channel is (lead1/lead2/ecog_anterior/emg... and contact number), filled in per rig by clean_dbs
        #[""] * len(names) = a list of empty strings, one per channel, for files without this info
        groups = [x.decode() for x in g["channel_group"][:]] if "channel_group" in g else [""] * len(names)
        contacts = [x.decode() for x in g["channel_contact"][:]] if "channel_contact" in g else [""] * len(names)
        #1 = the lab marked this channel do not use; dnu_list says where the list came from ("none" = recording not listed)
        dnu = g["channel_dnu"][:] if "channel_dnu" in g else None
        #.get(key, default) returns default instead of an error when the key isn't there
        listed = g.attrs.get("dnu_list", "")
        stream_type = str(g.attrs.get("type", "")) #lfp, emg, ...: the only clue for TDT EMG, which has no channel groups
        start = g.attrs.get("start_time", 0.0) #seconds into the recording of this stream's first sample (usually 0)
        n = g["data"].shape[1] # number of samples in the stream
        #int(...) drops the fraction, so the window starts and ends on whole samples
        i0 = int((t1 - start) * fs) # time x fs = samples
        i1 = int((t2 - start) * fs)
        #a chained comparison: True only if 0 <= i0 and i0 < i1 and i1 <= n
        if not 0 <= i0 < i1 <= n:
            #{start:.3f} formats the number with 3 decimals
            raise ValueError(f"window {t1}-{t2} s is outside this stream ({start:.3f}-{start + n / fs:.3f} s)")
        #[:, i0:i1] = every row (channel), columns i0 to i1-1; only this part is read from disk
        data = g["data"][:,i0:i1] # all data between set times
    #zip walks the three lists together, one channel at a time
    names = [standard_name(n, grp, c) for n, grp, c in zip(names, groups, contacts)]
    types = [channel_type(n, grp, stream_type) for n, grp in zip(names, groups)]
    #mne.create_info = the channel description (names, sample rate, types) that every MNE object carries as raw.info
    info = mne.create_info(names, fs, types)
    #mne.io.RawArray = a continuous MNE Raw made from a numpy array (channels x samples, in volts) plus its info
    raw = mne.io.RawArray(data, info)
    #"is not None" checks the file had do-not-use flags
    if dnu is not None:
        #raw.info["bads"] = MNE's list of bad channel names: the data stays, but MNE's tools skip them
        raw.info["bads"] = [name for name, flag in zip(names, dnu) if flag == 1]
    #clean_dbs puts a strip that wasn't placed on the brain in group ecog_<side>_not_placed, and a lead that was never
    #implanted or never plugged in in lead<N>_not_implanted / _not_connected; their channels are still typed ecog or dbs,
    #so mark them bad too, or pairs and averages would be made from contacts recording nothing
    unused = ("_not_placed", "_not_implanted", "_not_connected")
    #grp.endswith(unused): True if grp ends with any of the strings in the tuple
    unplaced = [name for name, grp in zip(names, groups) if grp.endswith(unused) and name not in raw.info["bads"]]
    #+ joins the two lists
    raw.info["bads"] = raw.info["bads"] + unplaced
    if dnu is None:
        warnings.warn(f"{Path(h5_path).name} {stream}: no do-not-use flags (cleaned before clean_dbs version 7), so bad channels aren't marked")
    elif listed in ("", "none"):
        warnings.warn(f"{Path(h5_path).name} {stream}: this recording isn't on any do-not-use list, so bad channels aren't known "
                      "(not the same as none being bad)")
    with h5py.File(h5_path, "r") as f:
        has_stim = "stim" in f # True if the file has a stim table
    #np.array([]) = an empty array, so the lines below still work with no pulses
    times = stim_times(h5_path, voice, confirmed_only) if has_stim else np.array([]) #no stim table (old file): the signal is still fine, no pulses to mark
    if has_stim and voice is not None and len(times) == 0:
        warnings.warn(f'no pulses with voice "{voice}" in {Path(h5_path).name} (BrainVision files have no voices)')
    t0 = start + i0 / fs #exact time of the Raw's first sample (t1 rounded down to a whole sample)
    #a True/False array: & means "and" for arrays, element by element (python's "and" doesn't work on arrays)
    inside = (times >= t0) & (times < t0 + data.shape[1] / fs)
    #keep only the pulses inside the window, as seconds from the Raw's first sample (MNE annotation times count from there)
    onsets = times[inside] - t0
    #mne.Annotations = labelled time points/spans stored with the Raw; duration = 0 marks an instant. Later steps find the
    #pulses by the "stim" label (e.g. with mne.events_from_annotations)
    raw.set_annotations(mne.Annotations(onset = onsets, duration = 0, description = "stim"))
    return raw


def standard_name(name, group, contact):
    #every rig names lead contacts its own way (TDT "lead1_c2", BrainVision "DBS2"); the cleaned file's group and contact say
    #which lead and contact it really is, so the rest of the pipeline only ever sees one name per contact
    #contact.isdigit(): True if the text is only digits ("3" yes, "" no)
    if group.startswith("lead") and contact.isdigit():
        #group.split('_') cuts the text at each _: "lead2_not_implanted" -> ["lead2", "not", "implanted"]; [0] takes "lead2"
        return f"{group.split('_')[0]}_c{contact}" #lead2_not_implanted is still lead2_c3 (and marked bad)
    return name # anything else keeps its original name


def channel_type(name, group, stream_type=""):
    #MNE channel type from the cleaned file's channel group (falls back to the stream's type, then the name, for files
    #without groups: TDT's emgg stream has none, so its channels would otherwise be misc)
    #the first test that matches returns; later ones aren't checked
    if group.startswith("ecog") or name.startswith("ecog"):
        return "ecog"
    #name.startswith(("lead", "dbs")): True if it starts with either
    if group.startswith("lead") or group == "dbs" or name.startswith(("lead", "dbs")):
        return "dbs"
    #"not group" is True for an empty string (no group stored)
    if group == "emg" or (not group and stream_type == "emg"):
        return "emg"
    return "misc"
