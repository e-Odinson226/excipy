import numpy as np
import random
import concurrent.futures
from functools import partial
from excipy.processes import hopping_rate
import sys



# Exciton class  ( image counters added)

class Exciton:
    """
    Simple container for an exciton, which just knows which site it occupies.
    """
    _counter = 0  # we also need t set uniqe id for excitons to know which exciton dyes
    def __init__(self, site, image=(0,0,0)):              ### PBC ADD >>>
        self.site = site
        self.imx, self.imy, self.imz = image              ### PBC ADD <<<
        self.uid  = Exciton._counter   # persistent ID
        Exciton._counter += 1


# ------------------------------------------------------------------
# main KMC
# ------------------------------------------------------------------
def run_kmc_PBC(
    system=None,
    num_excitons=1,
    exc_list = None,
    coupling_matrix=None,        # give matrix or single hopping rate
    overlap = None,
    hop_rate = 1e9,
    decay_rate=1e9,      # s^-1
    annih_rate=1e12,     # s^-1, if 2 excitons in same site
    neighbors=None,
    max_time=1e-6,       # 1 microsecond
    max_steps=10000,
    verbose=False,
    box=None                            ### PBC ADD >>>  3‑tuple (Lx,Ly,Lz) or None
):
    """
    ...
    box : (Lx, Ly, Lz) in same units as system coordinates, or None for open boundaries
    """

    Exciton._counter = 0 # reset exciton ids
    
    if system is None:
        sys.exit("Error: Please specify your system!")
    else:    
        num_sites=len(system)

    # coordinates only if PBC enabled
    coords = None                                           ### PBC ADD >>>
    if box is not None:
        coords = np.array([mol.get_center_of_mass() for mol in system])
        box    = np.asarray(box, dtype=float)               ### PBC ADD <<<


    

    # Set up default neighbors for a 1D chain, if none provided
    if neighbors is None:
        neighbors = {}
        for i in range(num_sites):
            nb = []
            if i > 0:
                nb.append(i-1)
            if i < num_sites - 1:
                nb.append(i+1)
            neighbors[i] = nb

    # Initialize excitons
    excitons = []
    exc_ids  = []
    if exc_list is None:
        for _ in range(num_excitons):
            site_idx = random.randint(0, num_sites - 1)
            excitons.append(Exciton(site_idx))       # image = (0,0,0)
            exc_ids.append(site_idx)
    else:
        for i in exc_list:
            excitons.append(Exciton(i))
            exc_ids.append(i)

    times = [0.0]
    exciton_records = [[(ex.uid, ex.site, ex.imx, ex.imy, ex.imz)
                        for ex in excitons]]                ### PBC ADD

    current_time = 0.0
    step_count   = 0

    # 2) Main KMC loop
    while True:
        step_count += 1
        if step_count > max_steps:
            if verbose: print('Maximum number of steps reached!!')
            break

        # Build list of all possible events and their rates
        events = []
        # Decay events
        for ex_idx, ex in enumerate(excitons):
            events.append(('decay', ex_idx, decay_rate))

        # Hopping events
        for ex_idx, ex in enumerate(excitons):
            site_i = ex.site
            for site_j in neighbors[site_i]:
                if coupling_matrix is None:
                    hop_rate_ij = hop_rate
                else:
                    hop_rate_ij = hopping_rate(site_i, site_j,
                                               coupling_matrix,
                                               J=overlap)
                if hop_rate_ij > 0:
                    events.append(('hop', ex_idx, site_j, hop_rate_ij))

        # Annihilation events
        site_occupancy = {}
        for ex_idx, ex in enumerate(excitons):
            key = (ex.site, ex.imx, ex.imy, ex.imz)         ### PBC ADD
            site_occupancy.setdefault(key, []).append(ex_idx)

        for site, ex_list in site_occupancy.items():
            if len(ex_list) > 1:
                for i in range(len(ex_list)):
                    for j in range(i+1, len(ex_list)):
                        events.append(('annih', ex_list[i], ex_list[j], annih_rate))

        if not events:
            if verbose: print("No events left."); break

        total_rate = sum(ev[-1] for ev in events)
        dt = -np.log(random.random()) / total_rate
        current_time += dt
        if current_time > max_time:
            if verbose: print("Max time exceeded."); break

        # choose event
        threshold = random.random() * total_rate
        running   = 0.0
        chosen_event = None
        for ev in events:
            running += ev[-1]
            if running >= threshold:
                chosen_event = ev
                break

        etype = chosen_event[0]

        if etype == 'decay':
            ex_idx = chosen_event[1]
            excitons.pop(ex_idx)

        elif etype == 'hop':
            ex_idx, nb_site = chosen_event[1], chosen_event[2]
            ex = excitons[ex_idx]
        
            if coords is not None:
                dr = coords[nb_site] - coords[ex.site]          # raw Δ in Å
                half = box * 0.5
        
                # component-wise minimum-image shift  (−1, 0, or +1)
                shift = np.zeros(3, dtype=int)
                shift[dr >  half] = -1
                shift[dr < -half] =  1
        
                # update image counters
                ex.imx += shift[0]
                ex.imy += shift[1]
                ex.imz += shift[2]
        
            ex.site = nb_site

            
        elif etype == 'annih':
            ex_i, ex_j = chosen_event[1], chosen_event[2]
            victim = random.choice([ex_i, ex_j])
            excitons.pop(victim)

        # 2f) Record
        times.append(current_time/1e-9)
        exciton_records.append([(ex.uid, ex.site, ex.imx, ex.imy, ex.imz)
                                for ex in excitons])        ### PBC ADD

        if len(excitons) == 0:
            if verbose: print("All excitons decayed or annihilated.")
            break

    return times, exciton_records


# ------------------------------------------------------------------
# Parallel wrapper (unchanged)
# ------------------------------------------------------------------
def run_kmc_PBC_parallel(num_trajectories=10, processors=2, **kmc_kwargs):
    run_kmc_partial = partial(run_kmc_PBC, **kmc_kwargs)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=processors) as executor:
        futures_dict = {}
        for i in range(num_trajectories):
            futures_dict[executor.submit(run_kmc_partial)] = i

        for future in concurrent.futures.as_completed(futures_dict):
            traj_id = futures_dict[future]
            times, excitons = future.result()
            print(f"Trajectory {traj_id+1} done!")
            results.append((times, excitons))

    return results

