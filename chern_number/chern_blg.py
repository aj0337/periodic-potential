import os
from mpi4py import MPI
import numpy as np

from tb_hamiltonian.continuum import compute_eigenstuff
from tb_hamiltonian.continuum import GrapheneContinuumModel

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def generate_kpoints(kx_min, kx_max, dkx, ky_min, ky_max, dky):
    # Create grid of kx and ky values
    kx_values = np.arange(kx_min, kx_max + dkx, dkx)
    ky_values = np.arange(ky_min, ky_max + dky, dky)

    # Use np.meshgrid to create the grid and flatten the arrays
    kx_grid, ky_grid = np.meshgrid(kx_values, ky_values, indexing='ij')

    # Stack the flattened kx and ky arrays into pairs of points
    kpoints = np.column_stack([kx_grid.ravel(), ky_grid.ravel()])

    return kpoints

def compute_berry_curvature_wilson(kpoints, dkx, dky, delta, bnd_idx, H_calculator):
    _, psi = compute_eigenstuff(H_calculator, kpoints)
    _, psi_right = compute_eigenstuff(
        H_calculator, [[kx + delta, ky] for kx, ky in kpoints]
    )
    _, psi_up = compute_eigenstuff(H_calculator, [[kx, ky + delta] for kx, ky in kpoints])
    _, psi_diag = compute_eigenstuff(
        H_calculator, [[kx + delta, ky + delta] for kx, ky in kpoints]
    )

    # Extract the wavefunction for the specific band
    psi = psi[:, :, bnd_idx]
    psi_right = psi_right[:, :, bnd_idx]
    psi_up = psi_up[:, :, bnd_idx]
    psi_diag = psi_diag[:, :, bnd_idx]

    # Overlaps between wavefunctions at neighboring k-points
    Ox = np.einsum("ij,ij->i", np.conj(psi), psi_right)
    Oy = np.einsum("ij,ij->i", np.conj(psi_right), psi_diag)
    Ox_reverse = np.einsum("ij,ij->i", np.conj(psi_diag), psi_up)
    Oy_reverse = np.einsum("ij,ij->i", np.conj(psi_up), psi)

    # Wilson loop Berry curvature
    berry_curvature = -np.imag(
        np.log(Ox * Oy * Ox_reverse * Oy_reverse)
    )/(delta**2)

    return berry_curvature


def compute_chern_number(kpoints, dkx, dky, delta, bnd_idx, H_calculator):

    berry_curvature = compute_berry_curvature_wilson(kpoints, dkx, dky, delta, bnd_idx, H_calculator)
    chern_number = np.sum(berry_curvature* dkx * dky) / (2 * np.pi)
    return chern_number


# ### Parallelization over kpoints

def parallel_compute_berry_curvature_and_chern(kpoints, dkx, dky, delta, bnd_idx, H_calculator):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split kpoints among processes more carefully
    num_kpoints = len(kpoints)
    counts = np.array([num_kpoints // size + (1 if i < num_kpoints % size else 0) for i in range(size)])
    displacements = np.cumsum(counts) - counts

    # Assign each process its local kpoints based on counts and displacements
    start = displacements[rank]
    end = start + counts[rank]
    local_kpoints = kpoints[start:end]

    # Each process computes Berry curvature for its local kpoints
    local_berry_curvature = compute_berry_curvature_wilson(local_kpoints, dkx, dky, delta, bnd_idx, H_calculator)

    # Prepare for gathering the results: allocate space for full result on root process
    if rank == 0:
        berry_curvature = np.empty(num_kpoints, dtype=np.float64)
    else:
        berry_curvature = None

    # Gather the results from all processes
    comm.Gatherv(local_berry_curvature, [berry_curvature, counts, displacements, MPI.DOUBLE], root=0)

    # Compute the Chern number on the root process
    if rank == 0:
        berry_curvature = berry_curvature * dkx * dky
        chern_number = np.sum(berry_curvature) / (2 * np.pi)
        return berry_curvature, chern_number
    else:
        return None, None


inputs = dict(
    bond_length=1.425,
    interlayer_hopping=0.22,
    superlattice_potential_periodicity=500,
    superlattice_potential_amplitude=10e-3,
    gate_bias=-5e-3,
    layer_potential_ratio=0.3,
    nearest_neighbor_order=3,
)

model = GrapheneContinuumModel(**inputs)

dkx, dky = 1e-4, 1e-4

if rank == 0:
    a = inputs["superlattice_potential_periodicity"]
    kx_min, kx_max = -np.pi/a, np.pi/a + dkx
    ky_min, ky_max = -2*np.pi/(np.sqrt(3)*a), 2*np.pi/(np.sqrt(3)*a) + dky
    kpoints = generate_kpoints(kx_min, kx_max, dkx, ky_min, ky_max, dky)
else:
    kpoints = None

# Broadcast kpoints to all processes
kpoints = comm.bcast(kpoints, root=0)

# Directory to store Berry curvature files
output_dir = "data/berry_curvature_blg"
chern_output_dir = "data/chern_blg"

if rank == 0:
    # Create directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(chern_output_dir):
        os.makedirs(chern_output_dir)

H_calculator = model.H_total_K
nbands = H_calculator(np.array([0, 0])).shape[0]
mid_band = int(nbands / 2)
# List of band indices you want to compute the Berry curvature for
bnd_indices = np.arange(mid_band-1,mid_band+2,1)  # You can modify this list to compute for other bands

# File to store Chern numbers for each band
chern_filename = os.path.join(chern_output_dir, 'chern_numbers.txt')

# Loop over band indices
for bnd_idx in bnd_indices:
    # Call the existing function to compute the Berry curvature and Chern number
    berry_curvature, chern_number = parallel_compute_berry_curvature_and_chern(
        kpoints, dkx, dky, delta=1e-5, bnd_idx=bnd_idx, H_calculator=H_calculator
    )

    if rank == 0:
        # Prepare data for saving: combine kpoints and berry_curvature
        data_to_save = np.column_stack((kpoints, berry_curvature))

        # Save the kx, ky, Berry curvature values for this band to a file
        output_filename = os.path.join(output_dir, f'berry_curvature_band_{bnd_idx}.npy')
        np.save(output_filename, data_to_save)

        # Write the band index and Chern number to the file
        with open(chern_filename, 'a') as chern_file:
            chern_file.write(f"Band {bnd_idx}: Chern number = {chern_number}\n")
