import numpy as np
import random
import concurrent.futures
from functools import partial
from excipy.processes import hopping_rate
import sys



class Exciton: 
    """
    Simple container for an exciton, which just knows which site it occupies.
    """
    _counter = 0 # we also need t set uniqe id for excitons to know which exciton dyes
    def __init__(self, site):
        self.site = site
        self.uid  = Exciton._counter   # persistent ID
        Exciton._counter += 1

def run_kmc(
    system=None,
    num_excitons=5,
    exc_list = None,
    coupling_matrix=None,        # give matrix or single hopping rate
    overlap = None,
    hop_rate = 1e9,
    decay_rate=1e9,      # s^-1
    annih_rate=1e12,     # s^-1, if 2 excitons in same site
    neighbors=None,
    max_time=1e-6,       # 1 microsecond
    max_steps=10000,
    verbose=False
):
    """

    Parameters:
    -----------
    num_sites : int
        How many lattice sites (1D for simplicity).
    num_excitons : int
        How many excitons to place initially.
    hop_rate : float
        Rate constant for exciton hopping to a neighbor (s^-1).
    decay_rate : float
        Rate constant for exciton decay (s^-1).
    annih_rate : float
        Rate constant for exciton-annihilation if 2 excitons share a site (s^-1).
    neighbors : dict or None
        A dictionary mapping site_index -> list_of_neighbor_sites.
    max_time : float
        It stop simulation if simulated time exceeds max_time (seconds).
    max_steps : int
        It also stops if we reach max_steps KMC events, to avoid infinite loops.
    verbose : bool
        If True, print debug info.

    Returns:
    --------
    times, exciton_records
        times: list of times after each event
        exciton_records: list of exciton states (which site each exciton is on)
    """

    Exciton._counter = 0 # reset exciton ids
    
    if system is None:
        sys.exit("Error: Please specify your system!")
    else:    
        num_sites=len(system)
    # 0) Set up default neighbors for a 1D chain, if none provided
    if neighbors is None:
        # Each site i has neighbors i-1 and i+1, if in bounds
        neighbors = {}
        for i in range(num_sites):
            nb = []
            if i > 0:
                nb.append(i-1)
            if i < num_sites - 1:
                nb.append(i+1)
            neighbors[i] = nb
    #print(neighbors)

    # 1) Initialize excitons
    if exc_list is None:
        excitons = []
        exc_ids = []
        for _ in range(num_excitons):
            site_idx = random.randint(0, num_sites - 1)
            excitons.append(Exciton(site_idx))
            exc_ids.append(site_idx)
    else:
        excitons = []
        exc_ids = []
        for i in exc_list:
            site_idx = i
            excitons.append(Exciton(site_idx))
            exc_ids.append(site_idx)
            
   # print("Excitons id:", exc_ids)
    
    # Lists to store info as we go
    times = [0.0]
    exciton_records = [ [(ex.uid, ex.site) for ex in excitons] ]

    current_time = 0.0
    step_count = 0

    # 2) Main KMC loop
    while True:
        step_count += 1
        if step_count > max_steps:
            if verbose:
                print('Maximum number of steps reached!!')
            break

        # 2a) Build list of all possible events and their rates
        events = []
        # - Decay events: each exciton can decay
        for ex_idx, ex in enumerate(excitons):
            events.append(('decay', ex_idx, decay_rate))

        # - Hop events: each exciton can hop to neighbor
        for ex_idx, ex in enumerate(excitons):
            site_i = ex.site
            for site_j in neighbors[site_i]:
                # 1) Calculate site-specific hopping rate
                if coupling_matrix is None:
                    hop_rate_ij =  hop_rate
                else:
                    hop_rate_ij = hopping_rate(site_i, site_j, coupling_matrix, system, J=overlap)
               
                if hop_rate_ij > 0:
                    events.append(('hop', ex_idx, site_j, hop_rate_ij))
                    
        # - Annihilation events: pairs of excitons in the same site
        #   We can find all pairs in the same site:
        site_occupancy = {}
        for ex_idx, ex in enumerate(excitons):
            site = ex.site
            site_occupancy.setdefault(site, []).append(ex_idx)
        
        for site, ex_list in site_occupancy.items():
            if len(ex_list) > 1:
                # for each distinct pair in ex_list
                for i in range(len(ex_list)):
                    for j in range(i+1, len(ex_list)):
                        ex_i = ex_list[i]
                        ex_j = ex_list[j]
                        events.append(('annih', ex_i, ex_j, annih_rate))
        #print(ex_list)
        #print(events) # use for debugging purposes only
        # 2b) Sum rates
        if not events:
            # No possible events => all excitons gone or no processes -> end
            if verbose:
                print("No events left.")
            break

        total_rate = 0.0
        for ev in events:
            # ev might be ('decay', ex_idx, rate) or ('hop', ex_idx, nb_site, rate) ...
            rate = ev[-1]  # last element
            total_rate += rate

        # 2c) Time step
        rnd = random.random()
        dt = -np.log(rnd) / total_rate
        current_time += dt
        if current_time > max_time:
            if verbose:
                print("Max time exceeded.")
            break

        # 2d) Which event occurs?
        threshold = random.random() * total_rate
        running = 0.0
        chosen_event = None
        for ev in events:
            rate = ev[-1]
            running += rate
            if running >= threshold:
                chosen_event = ev
                break

        # 2e) Apply the chosen event
        if chosen_event is None:
            # numerical edge case if random choice fails
            if verbose:
                print("No chosen event (weird!). End.")
            break

        etype = chosen_event[0]

        if etype == 'decay':
            # ('decay', ex_idx, rate)
            ex_idx = chosen_event[1]
            # remove exciton
            excitons.pop(ex_idx)
            #print("decay") #for debug

        elif etype == 'hop':
            # ('hop', ex_idx, nb_site, rate)
            ex_idx, nb_site = chosen_event[1], chosen_event[2]
            excitons[ex_idx].site = nb_site
            #print("hop", chosen_event[3]/1e9) #for debug
            
        elif etype == 'annih':
            ex_i, ex_j = chosen_event[1], chosen_event[2]
            # choose randomly which exciton to remove
            victim = random.choice([ex_i, ex_j])
            excitons.pop(victim)

            #print("annih") #for debug

        # 2f) Record system state
        times.append(current_time/1e-9)
        # exciton_records.append([ex.site for ex in excitons]) #old version
        exciton_records.append([(ex.uid, ex.site) for ex in excitons]) #new version with exciton id


        # If no excitons left => done
        if len(excitons) == 0:
            if verbose:
                print("All excitons decayed or annihilated.")
            break

    return times, exciton_records



def run_kmc_parallel(num_trajectories=10, processors=2, **kmc_kwargs):
    """
    Run multiple KMC trajectories in parallel using your *existing* run_kmc(...).

    We do NOT define a separate single-trajectory runner; we just wrap run_kmc
    via functools.partial.  No changes to run_kmc's internals are needed.
    """

    # 1) Create a partial function that calls run_kmc with the given kwargs
    #    e.g. run_kmc(num_sites=..., num_excitons=..., etc.)
    run_kmc_partial = partial(run_kmc, **kmc_kwargs)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=processors) as executor:
        # 2) Submit each trajectory, storing the future->traj_id in a dict
        futures_dict = {}
        for i in range(num_trajectories):
            future = executor.submit(run_kmc_partial)
            futures_dict[future] = i  # associate ID 'i' with this future

        # 3) As each finishes, print progress and store the result
        for future in concurrent.futures.as_completed(futures_dict):
            traj_id = futures_dict[future]
            (times, excitons) = future.result()
            print(f"Trajectory {traj_id+1} done!")  # or just traj_id if you prefer 0-based
            results.append((times, excitons))

    return results
