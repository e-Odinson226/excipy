import numpy as np
import ase 
from time import time
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from tqdm import tqdm
from pyscf import gto, scf, tdscf, lib, dft, lo
from tdens import *
from coupling import *
from excipy.tools import *
from multiprocessing import Pool, cpu_count
from io import StringIO 


def TDENS(mol_i,mol_j):
    xyz_1 = StringIO(); 
    ase.io.write(xyz_1, mol_i, format='xyz')
    frg_1 = xyz_1.getvalue()[3:]
    xyz_2 = StringIO(); 
    ase.io.write(xyz_2, mol_j, format='xyz')
    frg_2 = xyz_2.getvalue()[3:]
    
    #Make Molecule Object
    moli = gto.M(atom=frg_1) # basis='6-31g(d)')
    molj = gto.M(atom=frg_2) # basis='6-31g(d)')
    charge_i = mol_i.get_initial_charges()
    charge_j = mol_j.get_initial_charges()
    Jq_low = coupling_tdchg(charge_i,charge_j,moli.atom_coords(),molj.atom_coords())
   
   
    #print("Molecules:", i,j)
    #print("c_Low = %4.4f\n"%(Jq_low))

    return Jq_low

def TDENS_couplings_fast(molecules, cutoff=20):

    start = time()

    coords = np.array([mol.get_center_of_mass() for mol in molecules])
    kdtree = cKDTree(coords)

    # neighbours_idx[i] will be a list of j-indices within the cutoff,
    # including i itself (distance < 1e-3); query_ball_point is very fast.
    neighbours_idx = kdtree.query_ball_point(coords, cutoff)
    
    a0 = 0.529177   
    num_sites = len(molecules)
    rows, cols, data = [], [], []  
    #V = V.astype(np.float32) # reduce the size of the array. If high accuracy required, pls. use float64, but consider it is memory intensive!!!
    neighbors = {}
    for i in tqdm(range(0, num_sites)):
        nb = []
        center = molecules[i].get_center_of_mass()
        for j in neighbours_idx[i]:    # only j within cutoff
            # if j <= i:        # skip lower triangle & diagonal
            #     continue
            target = molecules[j].get_center_of_mass()
            dist = np.linalg.norm(target-center)
            # print(i,j,np.linalg.norm(target-center)) # for debugging
            if dist < 1e-3:
                cc = 0
            elif dist <= cutoff:
                nb.append(j)
                #dipole approx. calculations
                cc = TDENS(molecules[i], molecules[j])
                
                rows += [i, j]               # symmetric entries
                cols += [j, i]
                data += [cc/2, cc/2]
            else:
                cc = 0     
        neighbors[i] = nb
    V_csr = csr_matrix((data, (rows, cols)),
                       shape=(num_sites, num_sites),
                       dtype=np.float32)
    t = time() - start
    print("Coupling calculation: done.")
    print("Neighbour list: done.")
    print("User time: {} min {:.2f} s".format(int(t/60), t-int(t/60)))
    return neighbors, V_csr 


### NEW CODE with PBC

from scipy.spatial import cKDTree
import numpy as np


def pbc_shift(vec, box):
    return vec - box * np.round(vec / box) if box is not None else vec
# ------------------------------------------------------------------

#NEW TDENS WITH PBC CORRECT VERSION

def TDENS_couplings_fast_PBC(molecules, cutoff=20, box=None):
    
    start = time()
    coords = np.array([mol.get_center_of_mass() for mol in molecules])
    kdtree = cKDTree(coords, boxsize=box)          # PBC search

    neighbours_idx = kdtree.query_ball_point(coords, cutoff)

    num_sites = len(molecules)
    rows, cols, data = [], [], []
    neighbors = {}

    for i in tqdm(range(num_sites)):
        nb = []
        center = coords[i]
        for j in neighbours_idx[i]:
            # ----------------------------------------------------------
            target = coords[j]
            disp   = pbc_shift(target - center, np.array(box) if box else None)
            dist   = np.linalg.norm(disp)
            # ----------------------------------------------------------

            if dist < 1e-3:
                continue
            if dist > cutoff:
                continue

            nb.append(j)

            #NEW: translate molecule j into the image  
            shift_vec = np.round((target - center) / box).astype(int) if box else np.zeros(3, int)
            if shift_vec.any():                      # if we really crossed PBC
                mol_j = molecules[j].copy()
                mol_j.translate(-shift_vec * box)   # bring j into same image as i
            else:
                mol_j = molecules[j]
            # ----------------------------------------------------------

            cc = TDENS(molecules[i], mol_j)

            # symmetric storage 
            rows += [i, j]
            cols += [j, i]
            data += [cc/2, cc/2]

        neighbors[i] = nb

    V_csr = csr_matrix((data, (rows, cols)),
                       shape=(num_sites, num_sites),
                       dtype=np.float32)

    t = time() - start
    print("Coupling calc + neighbour list done in {} min {:.1f} s".format(int(t/60), t % 60))
    return neighbors, V_csr





# these will become globals inside each worker
def _init_worker(coords_, neighbours_idx_, molecules_, box_, cutoff_):
    global coords, neighbours_idx, molecules, neighbours_idx, box, cutoff
    coords         = coords_
    neighbours_idx = neighbours_idx_
    molecules      = molecules_
    box            = box_
    cutoff         = cutoff_

def _work_on_site(i):
    rows_i, cols_i, data_i = [], [], []
    nb_i = []
    center = coords[i]
    for j in neighbours_idx[i]:
        # ----------------------------------------------------------
        target = coords[j]
        disp   = pbc_shift(target - center, np.array(box) if box else None)
        dist   = np.linalg.norm(disp)
        if 1e-3 < dist <= cutoff:
            nb_i.append(j)
            # translate molecule j into the image  
            shift_vec = np.round((target - center) / box).astype(int) if box else np.zeros(3, int)
            if shift_vec.any():                      # if we really crossed PBC
                mol_j = molecules[j].copy()
                mol_j.translate(-shift_vec * box)   # bring j into same image as i
            else:
                mol_j = molecules[j]
            # ----------------------------------------------------------

            cc = TDENS(molecules[i], mol_j)
            rows_i += [i, j]
            cols_i += [j, i]
            data_i += [cc/2, cc/2]
    return i, rows_i, cols_i, data_i, nb_i




def TDENS_couplings_fast_PBC_parallel(molecules, cutoff=20, box=None, nproc=None):
    """
  parallel version with nproc
    """
    start = time()
    coords = np.array([mol.get_center_of_mass() for mol in molecules])
    kdtree = cKDTree(coords, boxsize=box)
    neighbours_idx = kdtree.query_ball_point(coords, cutoff)

    num_sites = len(molecules)
    nproc = nproc or cpu_count()
    with Pool(processes=nproc,
              initializer=_init_worker,
              initargs=(coords, neighbours_idx, molecules, box, cutoff)) as pool:
        # now each task just sends an integer i, not huge arrays
        out = list(tqdm(pool.imap(_work_on_site,
                                  range(num_sites)),
                        total=num_sites))

    # Collect results
    rows, cols, data = [], [], []
    neighbors = {}
    for i, rows_i, cols_i, data_i, nb_i in out:
        rows   .extend(rows_i)
        cols   .extend(cols_i)
        data   .extend(data_i)
        neighbors[i] = nb_i

    # Build sparse matrix
    V_csr = csr_matrix((data, (rows, cols)),
                       shape=(num_sites, num_sites),
                      dtype=np.float32)

    t = time() - start
    print(f"Coupling calc + neighbour list done in {int(t/60)} min {t%60:.1f} s")
    return neighbors, V_csr



def TDM_couplings_fast_PBC(molecules, cutoff=20, box=None):
    """
    box : tuple (Lx, Ly, Lz) in the same units as coords
          or None  →  no periodic wrapping 
    """
    start = time()
    a0 = 0.529177 # Bohr radius, used to convert
    coords = np.array([mol.get_center_of_mass() for mol in molecules])
    kdtree = cKDTree(coords, boxsize=box)        #  enables PBC search

    neighbours_idx = kdtree.query_ball_point(coords, cutoff)

    num_sites = len(molecules)
    rows, cols, data = [], [], []
    neighbors = {}

    for i in tqdm(range(num_sites)):
        nb = []
        center = coords[i]
        for j in neighbours_idx[i]:
            # if j <= i:        #  <<  commented;
            #     continue
            target = coords[j]

            # -------- PBC distance (2 lines) -------------------------
            disp  = pbc_shift(target - center, np.array(box) if box else None)
            dist  = np.linalg.norm(disp)
            # ---------------------------------------------------------

            if dist < 1e-3:
                cc = 0
            elif dist <= cutoff:
                nb.append(j)
                #dipole approx. calculations
                D_i = molecules[i].tdm
                D_j = molecules[j].tdm
                k = ( np.dot(unit_vector(D_i), unit_vector(D_j)) 
                      - 3*np.dot(unit_vector((target-center)), unit_vector(D_i))*np.dot(unit_vector((target-center)), unit_vector(D_j)) )
                cc = k*np.linalg.norm(D_i)*np.linalg.norm(D_j)/((dist/a0)**3)

                # symmetric storage, halved 
                rows += [i, j]
                cols += [j, i]
                data += [cc/2, cc/2]
            else:
                cc = 0
        neighbors[i] = nb

    V_csr = csr_matrix((data, (rows, cols)),
                       shape=(num_sites, num_sites),
                       dtype=np.float32)

    t = time() - start
    print("Coupling calc + neighbour list done in {} min {:.1f} s"
          .format(int(t/60), t % 60))
    return neighbors, V_csr


### NEW CODE FOR KMC WITH PBC


import numpy as np
import random
import concurrent.futures
from functools import partial
from excipy.processes import hopping_rate
import sys

import random, sys, concurrent.futures
import numpy as np
from functools import partial

# ------------------------------------------------------------------
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


