import numpy as np
import matplotlib.pyplot as plt
import ase 
import ase.build
import math
from time import time

def plot_exciton_density(times, exciton_records, num_sites):
    """
    Plot exciton occupancy vs. time.

    ----------
    times : list of float
        The KMC times after each event (length T).
    exciton_records : list of list of int
        exciton_records[t] is a list of site indices occupied by excitons at time[t].
    num_sites : int
        Number of lattice sites in the simulation.
    """

    # 1) Build a 2D array [time_index, site_index] = number of excitons at site
    T = len(times)
    density = np.zeros((T, num_sites), dtype=int)

    for t_idx, ex_sites in enumerate(exciton_records):
        for site_idx in ex_sites:
            density[t_idx, site_idx] += 1

    # 2) Plot total exciton population vs. time
    total_excitons = density.sum(axis=1)  # sum over all sites at each time

    plt.figure()
    plt.plot(times, total_excitons, label='Total excitons')
    plt.xlabel('Time (ns)')
    plt.ylabel('Number of excitons')
    plt.title('Total Exciton Population vs. Time')
    plt.legend()
    plt.show()

#======================TOOLS FROM spectra_tools.py ====================================
''' Functions for calculation absorption spectra according Frenkel exciton model'''

# calculate center of mass of given molecule
def calculate_center_of_mass(molecule):
    """ Calculate the center of mass of a molecule. """
    positions = molecule.get_positions()
    masses = molecule.get_masses()
    center_of_mass = np.dot(masses, positions) / np.sum(masses)
    return center_of_mass


# sort all given molecules list object according to intermolecular distances. 
# Return molecules list in a sorted manner
def sort_molecules_by_distance(molecules):
    """ Sort molecules based on the distance from the first molecule's center of mass. """
    # Calculate the center of mass for the first molecule
    reference_com = calculate_center_of_mass(molecules[0])

    # Calculate distances from the first molecule
    distances = []
    for molecule in molecules:
        com = calculate_center_of_mass(molecule)
        distance = np.linalg.norm(com - reference_com)
        distances.append(distance)

    # Create a list of indices sorted by distance, except the first one
    sorted_indices = np.argsort(distances)[1:]  # Skip the first as it's the reference

    # Reorder molecules according to sorted indices, placing the first molecule at the start
    sorted_molecules = [molecules[0]] + [molecules[i] for i in sorted_indices]

    return sorted_molecules


#return unit vector of given vector
def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return vector / np.linalg.norm(vector)



# Calculation of rotation matrix that aligns vec1 to vec2
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    param vec1: A 3d "source" vector # this should be DIPOLE VECTOR
    param vec2: A 3d "destination" vector # long axis of the molecule
    return: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = unit_vector(vec1), unit_vector(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


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



# Find two maximum elements value and their index for a given array
def find_two_max_elements(arr):
    # Find the first maximum element and its index
    max1_index = np.argmax(arr)
    max1_value = arr[max1_index]
    
    # Temporarily set the first maximum element to a very small value
    arr[max1_index] = -np.inf
    
    # Find the second maximum element and its index
    max2_index = np.argmax(arr)
    max2_value = arr[max2_index]
    
    # Restore the first maximum element
    arr[max1_index] = max1_value
    
    return max1_value, max2_value, max1_index, max2_index



def angle(v1, v2):
    """Calculate the angle between two vectors"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Ensure the cosine value is within the valid range
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)  # angle in radians
    return np.degrees(angle)  # convert to degrees

# Calculate the angle between the long molecular axis and the y-axis


def vector_plot(vec1,vec2):
    # Define the vectors
    # Create a new figure and an axes set for 2D plotting
    fig, ax = plt.subplots()

    # Adding the vectors to the plot
    #for vector in vectors:
    ax.quiver(0, 0, vec1[1], vec1[2], angles='xy', scale_units='xy', scale=1, color=['r'])
    ax.quiver(0, 0, vec2[1], vec2[2], angles='xy', scale_units='xy', scale=1, color=['b'])

    # Setting the limits of x and y axes
    #ax.set_xlim(-2, 2)
    #ax.set_ylim(-1, 4)
    # Adding grid
    ax.grid(True)
    # Setting axis labels
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    # Adding a title
    ax.set_title('Vector Visualization')
    # Showing the plot
    plt.show()
    
    
    
def vec_space(mol,origins, dipoles):  
    ''' PLOT VECTORS'''


    # Example data
    N = len(mol)  # Number of vectors
    vectors = np.array(dipoles)
    origins = np.array(origins)

    # Create a new figure and an axes set for 2D plotting
    fig, ax = plt.subplots()

    # Adding the vectors to the plot
    for origin, vector in zip(origins, vectors):
        # Only consider x and y components
        ax.quiver(*origin[:2], *vector[:2], angles='xy', scale_units='xy', linewidth=0.5, scale=0.2, color='r')

    # Setting the limits of x and y axes dynamically based on the data
    all_points = origins[:, :2] + vectors[:, :2]  # Only x, y components
    buffer = 20  # Additional space around the vectors
    x_limits = [min(all_points[:, 0]) - buffer, max(all_points[:, 0]) + buffer]
    y_limits = [min(all_points[:, 1]) - buffer, max(all_points[:, 1]) + buffer]
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    # Adding grid
    ax.grid(True)

    # Setting axis labels
    ax.set_xlabel('X [a.u.]')
    ax.set_ylabel('Y [a.u.]')

    # Adding a title
    ax.set_title('vec{d} space')

    # Show the plot
    plt.show()





def rotate_vector_deg(vector, axis, angle_deg):
    """
    Rotate a 3D 'vector' around a 3D 'axis' by an angle 'angle_deg' (in degrees).
    Returns the rotated vector as a NumPy array.

    Uses formula in 3D:
       v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k·v)*(1 - cos(theta))
    where:
       - v is the original vector
       - k is the unit rotation axis
       - theta is the rotation angle in radians
    """
    # Convert angle from degrees to radians
    theta = np.radians(angle_deg)

    # Ensure we have NumPy arrays
    v = np.array(vector, dtype=float)
    k = np.array(axis, dtype=float)
    
    # Normalize the rotation axis
    k /= np.linalg.norm(k)
    
    # Precompute sines/cosines
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Rodrigues' rotation formula
    v_rot = (v * cos_t
             + np.cross(k, v) * sin_t
             + k * np.dot(k, v) * (1.0 - cos_t))
    return v_rot
###############################################################################

def add_tdm(mu0, molecules):
    start = time()
    N = len(molecules)
    for i in range(0,N):
        eigvals, eigvecs = molecules[i].get_moments_of_inertia(vectors=True)
        long_axis_vector = eigvecs[0,:] 
        D = rotation_matrix_from_vectors(mu0, long_axis_vector).dot(mu0)
        molecules[i].tdm = np.array(D)
    print("TDM are aligned with molecules and added with the keyword [tdm]")
    print("User time for add_tdm function:", time() - start, "s!")
    
    return molecules


def neighbour_list(molecules, cutoff=15):

    start = time()
    num_sites = len(molecules)
    neighbors = {}
    for i in range(0, num_sites):
        nb = []
        center = molecules[i].get_center_of_mass()
        for j in range(0, num_sites):
            target = molecules[j].get_center_of_mass()
            # print(i,j,np.linalg.norm(target-center)) # for debugging
            if np.linalg.norm(target-center) < 1e-3:
                continue
            elif np.linalg.norm(target-center) <= cutoff:
                nb.append(j)
            else:
                continue
        neighbors[i] = nb
    t = time() - start
    print("User time for building a neighbour list: {} min {:.2f} s".format(int(t/60), t-int(t/60)))
    return neighbors


from ase import Atoms
from ase.build import separate

def separate_molecules_PBI(atoms, SUPERCELL=[1,1,1]):

    print("We are using pre-defined specific function for PBI unitcell with 8 molecules!\n")
    print("For your system, please make changes accordingly.")
    single = separate(atoms)
    single_cell = single[0]+single[1]+single[2]+single[3]+single[4]+single[5]+single[6]+single[7]
    supcell = single_cell.repeat(SUPERCELL)
    
    mol = []
    for i in range(0,int(len(supcell)/44)):
        single_mol = supcell[i*44:(i+1)*44]
        mol.append(single_mol)

    print(" - System contains {} molecules.\n".format(len(mol)))
    return mol

def separate_molecules_PBI_FULL(atoms, SUPERCELL=[1,1,1]):

    print("We are using pre-defined specific function for PBI unitcell with 8 molecules!\n")
    print("For your system, please make changes accordingly.\n")
    single = separate(atoms, scale=1.4)
    single_cell = single[0]+single[1]+single[2]+single[3]+single[4]+single[5]+single[6]+single[7]
    supcell = single_cell.repeat(SUPERCELL)
    Nmol = len(single[0])
    print(" - Molecule has {} atoms".format(Nmol))
    mol = []
    for i in range(0,int(len(supcell)/Nmol)):
        single_mol = supcell[i*Nmol:(i+1)*Nmol]
        mol.append(single_mol)

    print(" - System contains {} molecules.\n".format(len(mol)))
    return mol






















    

