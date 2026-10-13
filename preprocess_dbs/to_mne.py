import h5py
import mne 
from pathlib import Path
def to_mne(h5_path, stream, t1, t2):
    """
    Input:
    h5_path: a cleaned raw h5 file, ie from clean_dbs
    stream: which signal group to load eg. ecos (ecog and DBS lead contacts)
    t1, t2: start and end of the window, in seconds from the start of the recording
    Returns: MNE RAW array of specified window (volts). Lead contacts are renamed lead<N>_c<contact> (e.g. lead1_c2) whatever
    the rig called them, and channels are typed ecog, dbs, emg or misc
    """

    if not Path(h5_path).is_file():
        raise FileNotFoundError(f"no cleaned file at {h5_path}")
    with h5py.File(h5_path,"r") as f:
        g = f["streams"][stream]
        fs = g.attrs["fs"]
        names = [n.decode() for n in g["channel_names"][:]]
        #what each channel is (lead1/lead2/ecog_anterior/emg... and contact number), filled in per rig by clean_dbs
        groups = [x.decode() for x in g["channel_group"][:]] if "channel_group" in g else [""] * len(names)
        contacts = [x.decode() for x in g["channel_contact"][:]] if "channel_contact" in g else [""] * len(names)
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
