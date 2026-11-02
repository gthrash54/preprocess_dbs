import mne
import warnings

#all 21 pairs from the lab's get_rereference.m:37-38, in its order. (A, B) = A - B, the lab's label dbs_A_B
#(newdata = y2 - y1, labelled c2_c1; not yet compared with any of its saved outputs). Contacts 2-4 are the lower
#segments, 5-7 the upper ones; 234 and 567 are virtual rings (mean of the 3 segments at that level, made only when
#all 3 are usable)
#each family on its own, in case you want only one kind of pair
horizontal_pairs = [(3, 2), (4, 3), (2, 4), (6, 5), (7, 6), (5, 7)] #horizontal: neighbouring segments at one level, direction
vertical_pairs = [(5, 2), (6, 3), (7, 4)] #each upper segment minus the one below it: depth on one side
ring_pairs = [(234, 1), (567, 234), (8, 567), (567, 1), (8, 234)] #levels as rings, like a non-directional lead
c1_pairs = [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)] #every contact minus the bottom ring
all_pairs = horizontal_pairs + vertical_pairs + ring_pairs + c1_pairs #the lab's 21, in its order

#the 7 pairs the lab's ERNA script keeps (incrementingExperiment_pull_wave_parameters.m): horizontal + 8 vs 567
#(8, 567) = the top ring (c8) minus the upper level as a virtual ring (mean of c5, c6, c7): a non-directional pair at
#the top of the lead. Sarah's code doesn't say why this ring pair and not the others
erna_pairs = horizontal_pairs + [(8, 567)]

def bipolar_names(lead, pairs=all_pairs):
    """
    defines the name for the leads we want to bipolar re-reference 
    input: lead we want to reference in text ie "lead1"
    pairs: contact pairs (A, B) = A - B; all_pairs unless you give others (e.g. erna_pairs, or [(1, 2), (2, 3), (3, 4)] for a
        4-ring lead)
    returns(anode,cathode) ie which leads we are bipolar referencing
    """
    anode = [f"{lead}_c{a}" for a,b in pairs]
    cathode = [f"{lead}_c{b}" for a,b in pairs]
    return anode, cathode


def rereference(raw, leads, pairs=all_pairs):
    """
    This will rereference bipolar leads of your choice
    Input:
    raw = MNE raw array file from to_mne, THIS FIRST STEP MUST BE DONE
    leads = list of leads we want to rereference in text e.g. ["lead1", "lead2"]
    pairs: contact pairs (A, B) = A - B; all_pairs unless you give others (erna_pairs for only the 7 the lab uses for ERNA)
    Returns: a new Raw file with the bipolar pairs for each lead. Pairs whose contacts aren't in the recording
    (lead not implanted, channel missing) or are marked bad (raw.info["bads"], e.g. the lab's do-not-use list) are skipped
    with a warning
    """
    anode = []
    cathode = []
    for lead in leads:
        a, c = bipolar_names(lead, pairs)
        for an, ca in zip(a, c):
            bad = [ch for ch in (an, ca) if ch in raw.info["bads"]]
            if bad: #a bad contact spoils every pair it's in
                warnings.warn(f"skipping {an}-{ca}: {' and '.join(bad)} marked bad (do not use)")
            elif an in raw.ch_names and ca in raw.ch_names:
                anode.append(an)
                cathode.append(ca)
            else:
                warnings.warn(f"skipping {an}-{ca}: not in this recording")
    if not anode:
        raise ValueError(f"no usable pairs for {leads}: each pair is missing from this recording or has a contact marked bad")
    return mne.set_bipolar_reference(raw, anode, cathode, copy = True)



#the two virtual rings: a ring's name and the three segments it averages
virtual_rings = {234: (2, 3, 4), 567: (5, 6, 7)}

def add_virtual_rings(raw, leads):
    """
    Input:
    raw: MNE Raw from to_mne, with do-not-use and stim contacts already in raw.info["bads"]
    leads: leads to add rings to, e.g. ["lead1", "lead2"]
    Returns: a copy of raw with lead<N>_c234 and lead<N>_c567 added, each the mean of its three segments at every time
    point (the lab's get_rereference.m:41-46, 62-78). A ring is only made when all three segments are in the recording and
    none is bad, as in the lab, so in a stim segment c234 is never made (one of c2-c4 is the stim contact). The ring pairs
    in all_pairs then work through rereference like any other pair
    """
    raw = raw.copy()
    for lead in leads:
        for ring, segments in virtual_rings.items():
            names = [f"{lead}_c{c}" for c in segments]
            if all(n in raw.ch_names and n not in raw.info["bads"] for n in names):
                mean = raw.get_data(picks=names).mean(axis=0, keepdims=True) #the mean over the 3 segments, one row
                info = mne.create_info([f"{lead}_c{ring}"], raw.info["sfreq"], "dbs")
                raw.add_channels([mne.io.RawArray(mean, info, verbose=False)], force_update_info=True)
    return raw