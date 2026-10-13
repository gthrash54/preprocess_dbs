import h5py
import mne 

def to_mne(h5_path, stream, t1, t2):
    with h5py.File(h5_path,"r") as f: