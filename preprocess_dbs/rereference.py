import mne # MNE-Python: the EEG/ECoG/DBS toolbox; Raw objects hold channels x samples plus their info
import warnings # prints a warning without stopping the program

#why bipolar: subtracting two neighbouring contacts cancels what both pick up (far-away activity, the reference
#electrode, shared noise) and keeps what differs between them, so the pair mostly reflects tissue near those two contacts
#all 21 pairs from the lab's get_rereference.m:37-38, in its order. (A, B) = A - B, the lab's label dbs_A_B
#(newdata = y2 - y1, labelled c2_c1; not yet compared with any of its saved outputs). Contacts 2-4 are the lower
#segments, 5-7 the upper ones; 234 and 567 are virtual rings (mean of the 3 segments at that level, made only when
#all 3 are usable)
#each family on its own, in case you want only one kind of pair
#each list below is a list of tuples: [(3, 2), ...] = pair 1 is contact 3 and contact 2, and so on
horizontal_pairs = [(3, 2), (4, 3), (2, 4), (6, 5), (7, 6), (5, 7)] #horizontal: neighbouring segments at one level, direction
vertical_pairs = [(5, 2), (6, 3), (7, 4)] #each upper segment minus the one below it: depth on one side
ring_pairs = [(234, 1), (567, 234), (8, 567), (567, 1), (8, 234)] #levels as rings, like a non-directional lead
c1_pairs = [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)] #every contact minus the bottom ring
all_pairs = horizontal_pairs + vertical_pairs + ring_pairs + c1_pairs #the lab's 21, in its order (+ joins lists end to end)

#the 7 pairs the lab's ERNA script keeps (incrementingExperiment_pull_wave_parameters.m): horizontal + 8 vs 567
#(8, 567) = the top ring (c8) minus the upper level as a virtual ring (mean of c5, c6, c7): a non-directional pair at
#the top of the lead. Sarah's code doesn't say why this ring pair and not the others
erna_pairs = horizontal_pairs + [(8, 567)]
## ecog pairs for anterior and posterior strip
#each ECoG contact minus its neighbour on the 6-contact strip (2-1, 3-2, ... 6-5)
ecog_pairs = [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)]


#pairs=all_pairs is a default argument: used when the caller doesn't pass pairs
def bipolar_names(prefix, pairs=all_pairs):
    """
    defines the name for the leads we want to bipolar re-reference 
    prefix: the start of each channel name, e.g. "lead1_c" (lead1_c3) or "ecog_a" (ecog_a3)
    pairs: contact pairs (A, B) = A - B; all_pairs unless you give others (e.g. erna_pairs, or [(1, 2), (2, 3), (3, 4)] for a
        4-ring lead)
    returns(anode,cathode) ie which leads we are bipolar referencing
    """
    #a list comprehension: builds a list in one line. For each (a, b) tuple in pairs (unpacked into a and b), make the
    #f-string name; f"{prefix}{a}" puts the values in, e.g. "lead1_c" + 3 -> "lead1_c3"
    anode = [f"{prefix}{a}" for a, b in pairs]
    cathode = [f"{prefix}{b}" for a, b in pairs]
    #returning two values returns a tuple; the caller unpacks it with a, c = bipolar_names(...)
    return anode, cathode


#the leading _ in _bipolar is a python convention for "internal helper, not meant to be called from outside this file"
def _bipolar(raw, prefixes, pairs):
    #shared by dbs_rereference and ecog_rereference: makes each pair for each name prefix (lead1_c, ecog_a, ...),
    #skipping with a warning any pair with a contact that's missing or marked bad
    anode = []
    cathode = []
    for prefix in prefixes:
        a, c = bipolar_names(prefix, pairs)
        #zip walks two lists side by side: (a[0], c[0]), (a[1], c[1]), ...
        for an, ca in zip(a, c):
            #the anode and/or cathode name if it's in MNE's bad list (an empty list if neither is)
            bad = [ch for ch in (an, ca) if ch in raw.info["bads"]]
            #an empty list counts as False in an if, a non-empty one as True
            if bad: #a bad contact spoils every pair it's in
                #' and '.join(bad) glues the list's strings together with " and " between them
                warnings.warn(f"skipping {an}-{ca}: {' and '.join(bad)} marked bad (do not use)")
            #raw.ch_names = the channel names in this Raw
            elif an in raw.ch_names and ca in raw.ch_names:
                anode.append(an)
                cathode.append(ca)
            else:
                warnings.warn(f"skipping {an}-{ca}: not in this recording")
    #"not anode" is True when the list is empty
    if not anode:
        raise ValueError(f"no usable pairs for {prefixes}: each pair is missing from this recording or has a contact marked bad")
    #mne.set_bipolar_reference makes one new channel per pair, anode - cathode, named "anode-cathode" (e.g.
    #"lead1_c3-lead1_c2"), and by default removes the contacts it used. copy=True works on a copy, leaving raw unchanged
    bip = mne.set_bipolar_reference(raw, anode, cathode, copy=True)
    #MNE only removes the contacts a pair used, so a contact no pair used (e.g. a bad one) would stay as a single channel;
    #the lab's bipolar output holds only pairs, so drop those. Other channels (EMG, the other modality) stay
    #n.startswith(tuple(prefixes)): True if n starts with any of the prefixes (startswith takes a tuple, not a list)
    #'"-" not in n' keeps out the new pair channels, whose names have a "-"
    leftover = [n for n in bip.ch_names if n.startswith(tuple(prefixes)) and "-" not in n]
    #drop_channels removes those channels (in place) and also returns the Raw
    return bip.drop_channels(leftover)


def dbs_rereference(raw, leads, pairs=all_pairs):
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
    #["lead1", "lead2"] -> ["lead1_c", "lead2_c"], the start of every contact name on those leads
    return _bipolar(raw, [f"{lead}_c" for lead in leads], pairs)


def ecog_rereference(raw, pairs=ecog_pairs):
    """
    Input:
    raw: MNE Raw from to_mne, with do-not-use ECoG contacts in raw.info["bads"]
    pairs: contact pairs (A, B) = A - B within a strip; ecog_pairs (each contact minus the one below it) unless you give others
    Returns: a new Raw with the bipolar pairs on the anterior (ecog_a1-6) and posterior (ecog_p1-6) strips; pairs never
        cross strips. A pair with a bad or missing contact is skipped with a warning
    """
    #only a middle contact (2-5) can be bridged over; the lab pairs the surviving neighbours across a gap
    #(get_rereference.m:24-33), so warn where our output will differ from hers
    for strip in ("a", "p"):
        #range(2, 6) = 2, 3, 4, 5 (the end is excluded, like slicing)
        for k in range(2, 6):
            name = f"ecog_{strip}{k}"
            if name in raw.info["bads"]:
                warnings.warn(f"{name} is bad: the lab would pair ecog_{strip}{k+1} with ecog_{strip}{k-1} across it (double spacing); "
                              "we skip that pair")
    return _bipolar(raw, ["ecog_a", "ecog_p"], pairs)



#the two virtual rings: a ring's name and the three segments it averages
#a dict: 234 -> (2, 3, 4) and 567 -> (5, 6, 7)
#(averaging the 3 segments at one level gives roughly what a ring contact at that level would record)
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
    #copy so the caller's raw isn't changed (add_channels below works in place)
    raw = raw.copy()
    for lead in leads:
        #.items() gives each (key, value) pair: (234, (2, 3, 4)), then (567, (5, 6, 7))
        for ring, segments in virtual_rings.items():
            names = [f"{lead}_c{c}" for c in segments]
            #all(...) is True only if the test is True for every name (any(...) would need just one)
            if all(n in raw.ch_names and n not in raw.info["bads"] for n in names):
                #get_data(picks=names) returns a numpy array, channels x samples (3 rows here). .mean(axis=0) averages down
                #the rows (across channels) at each sample; keepdims=True keeps the result 2D (1 x samples), the shape
                #RawArray needs
                mean = raw.get_data(picks=names).mean(axis=0, keepdims=True) #the mean over the 3 segments, one row
                #mne.create_info describes channels: their names, the sample rate (raw.info["sfreq"], Hz) and type ("dbs")
                info = mne.create_info([f"{lead}_c{ring}"], raw.info["sfreq"], "dbs")
                #mne.io.RawArray builds a Raw from a numpy array + info. add_channels appends its channel to raw (in place);
                #force_update_info=True makes the new Raw's info (e.g. its measurement date) match raw's, which add_channels
                #would otherwise refuse
                raw.add_channels([mne.io.RawArray(mean, info, verbose=False)], force_update_info=True)
    return raw


def common_average(raw, lead):
    """
    Input:
    raw: MNE Raw from to_mne (NOT the bipolar output, which no longer has single contacts), with do-not-use and stim
        contacts in raw.info["bads"]
    lead: e.g. "lead1"

    Returns: a new Raw with one channel per usable contact, lead<N>_c<k>-avg = that contact minus the mean of all usable
        contacts c1-c8 on the lead at each time point (the lab's get_rereference.m:8-16, labelled dbs_<k>_avg). The virtual
        rings aren't included, as in the lab. Stim annotations are kept
    """
    #why the average excludes bad contacts: a noisy or flat contact (do-not-use, or the unplugged stim contact) would be
    #mixed into the average and so leak into every other contact's signal
    #1 the usable contacts
    #range(1, 9) = 1 to 8
    names = [f"{lead}_c{k}" for k in range(1, 9)] # lead1_c1 ... lead1_c8
    #a list comprehension with a filter: keep n only if the "if" part is True
    usable = [n for n in names if n in raw.ch_names and n not in raw.info["bads"]]

    # need at least 2 or every contact minus average is just zero
    if len(usable) < 2:
        raise ValueError(f"{lead} has {len(usable)} usable contact(s) ({usable}); a common average needs at least 2")

    # copy only usable channels and subtract the mean across them at each time point
    #.pick(usable) keeps only those channels (in place, so it's done on the copy)
    averaged = raw.copy().pick(usable)
    #channel_wise=False: the function gets all channels at once, x = channels x samples. x.mean(axis=0) is one row (the mean
    #over channels at each sample); "x - row" subtracts that row from every channel (numpy "broadcasting" repeats the row)
    averaged.apply_function(lambda x: x - x.mean(axis=0), channel_wise=False)

    #rename the channels (lead1_c2 to lead1_c2-avg)
    #rename_channels takes a dict old name -> new name (built here with a dict comprehension); works in place
    averaged.rename_channels({name: f"{name}-avg" for name in usable})
    return averaged


def ecog_common_average(raw):
    """
    Input:
    raw: MNE Raw from to_mne, with do-not-use ECoG contacts in raw.info["bads"]
    Returns: a new Raw with one channel per usable ECoG contact, ecog_<strip><k>-avg = that contact minus the mean of the
        usable contacts on its own strip (anterior a1-a6, posterior p1-p6) at each time point. The lab meant to make this
        (preprocess.m:690-705, ecog_<k>_avg) but never does: its check at P:693 is always false (bug B6), so its saved data
        has no ECoG common average. A strip with fewer than 2 usable contacts is skipped with a warning. Stim annotations
        are kept
    """
    strips = [] # one Raw per strip that worked
    for strip in ("a", "p"):
        names = [f"ecog_{strip}{k}" for k in range(1, 7)] # ecog_a1 ... ecog_a6
        usable = [n for n in names if n in raw.ch_names and n not in raw.info["bads"]]
        if len(usable) < 2:
            warnings.warn(f"ecog_{strip} strip has {len(usable)} usable contact(s); no common average for it")
            continue # skip the rest of this loop pass and go on to the next strip
        averaged = raw.copy().pick(usable)
        averaged.apply_function(lambda x: x - x.mean(axis=0), channel_wise=False) #per strip, never across strips
        averaged.rename_channels({name: f"{name}-avg" for name in usable})
        strips.append(averaged)
    if not strips:
        raise ValueError("no ECoG strip has 2 usable contacts for a common average")
    #if both strips worked, add the second strip's channels to the first (strips[1:] = every item after the first) and return
    #it; if only one did, return that one
    return strips[0].add_channels(strips[1:]) if len(strips) > 1 else strips[0]
