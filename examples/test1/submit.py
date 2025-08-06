import numpy as np 
import excipy
from excipy.processes import couplings_neighbours_fast, hopping_rate, add_exciton_random
from excipy.tools import *
#from excipy.kmc import run_kmc_parallel
from time import time
import ase
from ase.build import separate
from ase.io import read
from ase.visualize import view    
import math
from time import time
from warnings import warn
from excipy.pbc import *


#============================================
""" USER INPUT """
#============================================

NUM_CPU = 8  #number CPUs for KMC calculation
FILENAME = 'PR_clean.pdb'  #input geometry file, *supported formats: vasp, pdb, cif, xyz. NOTE: data should include lattice information!
SUPERCELL = [20,20,20] # whether you want to create a supercell
CUTOFF = 30 # cut-off for interaction in A
mu0 = np.array([2.5634,     -0.0297,     -0.0109])   # Transition dipole moment in a.u.
J_overlap = 1.22      # spectral overlap, can be taken experiment
a0 = 0.529177 # Bohr radius, used to convert positions to a.u. #don't remove! 

#============================================
#============================================
""" Some post-processing procedure... """

for_cell = ase.io.read(FILENAME)  # getting molecule and cell information
print("Initializing calculation: we are using {} CPUs in total. \n".format(NUM_CPU))

#rebuild system with Lowdin charges, and then create supercell!
m = ase.build.separate(for_cell)
charges = np.loadtxt("charges_PR.dat")
for i in range(0,len(m)):
    m[i].set_initial_charges(charges[i])
for_cell = m[0] + m[1] + m[2] + m[3] + m[4] + m[5] + m[6] + m[7]

s = for_cell.repeat(SUPERCELL)
print(SUPERCELL,"supercell created.")
start = time()
mol = separate_molecules_PBI(for_cell, SUPERCELL)            #   mol = ase.build.separate(s)   # separate Atoms into distinct molecules
print("System contains {} molecules.\n".format(len(mol)))
Vol = s.get_volume()*1e-12 # converting micrometr cube, match with experimental units
print("Volume: {:.3f} nm^3".format(Vol*1e9))
print("Lattice parameters [Å]:\n",for_cell.get_cell()[0], "\n",for_cell.get_cell()[1],"\n",for_cell.get_cell()[2],"\n")
t = time() - start
print("Time for molecule separation: {} min {:.2f} s".format(int(t/60), t-int(t/60)*60))

exc_conc = 370000 #excitons per micrometr cube, taken from experimental data

Vol = s.get_volume()*1e-12 # converting micrometr cube, match with experimental units
print("Volume: {:.3f} nm^3".format(Vol*1e9))
NUM_EXC = int(exc_conc*Vol) +1
print("Number of excitons required for {} value in experiment: {}".format(exc_conc, int(exc_conc*Vol)))




a = s.get_cell()[0,0]
b = s.get_cell()[1,1]
c = s.get_cell()[2,2]

mol = add_tdm(mu0, mol) # add rotated TDM to each molecule in the system

neighbors, V = TDM_couplings_fast_PBC(molecules=mol, cutoff=30, box=(a,b,c))

#V_csr1 = csr_matrix(V, dtype=V.dtype)      # keeps only the non–zero entries
#print("dense aray :", V.shape, "   bytes:", V.nbytes / 1e6, "MB")
print("sparse array   :", V.nnz,       "   bytes:", V.data.nbytes / 1e6, "MB")


coords = []
for molecule in mol: 
    coords.append(molecule.get_center_of_mass())
len(coords)


start = time()

all_runs = run_kmc_PBC_parallel(
    num_trajectories=64,
    processors=32,
    system = mol,
    neighbors=neighbors,
    num_excitons=1,  # if exc_list is not None, then ignored!
    exc_list = add_exciton_random(mol, NUM_EXC),
    coupling_matrix = V/1.35,
    overlap = 1.22,
    decay_rate=4*1e9,
    annih_rate=1e15,
    max_time=1.2e-9,
    max_steps=900000,
    box=(a,b,c),
    verbose=True
)


print(len(all_runs), "trajectories in total.")
t = time() - start
print("Calculation finished! Elapsed time: {} min {:.2f} s".format(int(t/60), t-int(t/60)*60))



from excipy.analysis import *

dump_output(all_runs, "output.h5")


from excipy.analysis import plot_average_exciton_population
exc_den = plot_average_exciton_population(all_runs, num_bins=80, x0=0, xf=1)
np.savetxt("density.dat", (exc_den[0], exc_den[1]), delimiter="\t", newline='\n',)
plt.xlim(0,1)
plt.savefig("exciton_density.png", bbox_inches="tight")

t_grid, msd_avg = msd_all_runs(all_runs, coords, box=(a,b,c))
np.savetxt("msd_avg.dat", (t_grid, msd_avg), delimiter="\t", newline='\n',)
plt.figure(1)
plt.plot(t_grid, msd_avg, lw=2, label='avg over runs')
plt.legend(); 
plt.xlim(0, 1)
plt.ylabel("MSD [A^2]")
plt.xlabel("time [ns]")
plt.savefig("msd_test.png")





