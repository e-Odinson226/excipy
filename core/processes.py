import numpy as np
import ase 
from time import time


def dipole_approx(ind_i, ind_j, molecules, cutoff=15):
    """
    Parameters:
    - molecules: molecules list;
    - ind_i, ind_j: molecule indices that will be used in coupling calculation
    
    Return:
    - cc: coupling value in [a.u.] unit
    """
    a0 = 0.529177 # Bohr radius, used to convert positions to a.u.
    dist_ij, emn, vec_ij, o = get_Xmn(molecules,ind_i,ind_j)
    if dist_ij <= 1e-3: 
        cc = 0
    elif dist_ij <= cutoff/a0:
        D_i = molecules[ind_i].tdm
        D_j = molecules[ind_j].tdm
        
        k = ( np.dot(unit_vector(D_i), unit_vector(D_j)) 
              - 3*np.dot(unit_vector(emn), unit_vector(D_i))*np.dot(unit_vector(emn), unit_vector(D_j)) )
        cc = k*np.linalg.norm(D_i)*np.linalg.norm(D_j)/(dist_ij**3)
    else:
        cc = 00
    return cc

def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)


# get distance between i nad j molecules in molecules list given by mol
# Returns intermolecular distance, unit vector, and actual vector [in a.u. units]
def get_Xmn(mol,i,j):
    a0 = 0.529177
    mol_1 = mol[i].get_center_of_mass()/a0  ### converting A to a.u. by deviding a0
    mol_2 = mol[j].get_center_of_mass()/a0
    vec_ij = mol_2 - mol_1
    dist_ij = np.linalg.norm(vec_ij)
    if dist_ij == 0:
        uvec = 0
    else:
        uvec = vec_ij/dist_ij
    return dist_ij, uvec, vec_ij, mol_1

def hopping_rate(i, j, H, mol, J=0.5):
    V = H[i,j]*27.211  # convert au to eV
    hbar = 6.58e-16 #eV*s
    pi = 3.14
    a0 = 0.529177
    a = get_Xmn(mol, i, j)[0]*a0*1e10 # convert A to m
    k = 2*pi*(V**2)*J/hbar
    # print(k) #debugging

    return k # in seconds


def add_exciton_random(mol,num=1):
    N = len(mol)

    site_idx = np.random.choice(N,num)
    return site_idx







            