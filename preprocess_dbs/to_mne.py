import h5py
import mne 
from pathlib import Path
def to_mne(h5_path, stream, t1, t2):
    """
    Input:
    h5_path: a cleaned raw h5 file, ie from clean_dbs
    stream: which signal group to load eg. ecos (ecog and DBS lead contacts)
    t1, t2: start and end of the window, in seconds from the start of the recording
    Returns: MNE RAW array of specified window (volts), channel typed ecog or dbs
    """

    if not Path(h5_path).is_file():
        raise FileNotFoundError(f"no cleaned file at {h5_path}")
    with h5py.File(h5_path,"r") as f:
        g = f["streams"][stream]
        fs = g.attrs["fs"]
        names = [n.decode() for n in g["channel_names"][:]]
        i0 = int(t1 * fs) # time x fs = samples
        i1 = int(t2 *fs)
        data = g["data"][:,i0:i1] # all data between set times
    types = ["ecog" if n.startswith("ecog") else "dbs" for n in names]
    info = mne.create_info(names, fs, types)
    raw = mne.io.RawArray(data, info)
    return raw