import mne
import warnings

#the lab's ERNA pairs for an 8-contact directional lead: contacts 2-4 are the lower segments, 5-7 the upper ones
PAIRS = [(2, 3), (3, 4), (4, 2), (5, 6), (6, 7), (7, 5)]

def bipolar_names(lead, pairs=PAIRS):
    """
    defines the name for the leads we want to bipolar re-reference 
    input: lead we want to reference in text ie "lead1"
    pairs: contact pairs, PAIRS unless you give others (e.g. [(1, 2), (2, 3), (3, 4)] for a 4-ring lead)
    returns(anode,cathode) ie which leads we are bipolar referencing
    """
    anode = [f"{lead}_c{a}" for a,b in pairs]
    cathode = [f"{lead}_c{b}" for a,b in pairs]
    return anode, cathode


def rereference(raw, leads, pairs=PAIRS):
    """
    This will rereference bipolar leads of your choice
    Input:
    raw = MNE raw array file from to_mne, THIS FIRST STEP MUST BE DONE
    leads = list of leads we want to rereference in text e.g. ["lead1", "lead2"]
    pairs: contact pairs, PAIRS unless you give others
    Returns: a new Raw file with the bipolar pairs for each lead. Pairs whose contacts aren't in the recording
    (lead not implanted, channel missing) are skipped with a warning
    """
    anode = []
    cathode = []
    for lead in leads:
        a, c = bipolar_names(lead, pairs)
        for an, ca in zip(a, c):
            if an in raw.ch_names and ca in raw.ch_names:
                anode.append(an)
                cathode.append(ca)
            else:
                warnings.warn(f"skipping {an}-{ca}: not in this recording")
    if not anode:
        raise ValueError(f"none of the pairs for {leads} are in this recording")
    return mne.set_bipolar_reference(raw, anode, cathode, copy = True)
    