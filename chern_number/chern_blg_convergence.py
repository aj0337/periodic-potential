import os
from mpi4py import MPI
import numpy as np

from tb_hamiltonian.continuum import compute_eigenstuff
from tb_hamiltonian.continuum import GrapheneContinuumModel

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def generate_kpoints(kx_min, kx_max, dkx, ky_min, ky_max, dky):
    # Calculate the number of points required for kx and ky so that they end exactly at kx_max and ky_max
    num_kx_points = int(np.round((kx_max - kx_min) / dkx)) + 1
    num_ky_points = int(np.round((ky_max - ky_min) / dky)) + 1

    # Create grid of kx and ky values using linspace to ensure the max value is included
    kx_values = np.linspace(kx_min, kx_max, num_kx_points)
    ky_values = np.linspace(ky_min, ky_max, num_ky_points)

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

# Directory to store Berry curvature files
chern_output_dir = "data/chern_blg"

H_calculator = model.H_total_K
nbands = H_calculator(np.array([0, 0])).shape[0]
mid_band = int(nbands / 2)
# List of band indices you want to compute the Berry curvature for
bnd_indices = [mid_band]  # You can modify this list to compute for other bands


# List of values for dkx, dky, and delta to test
dk_values = [5e-5]  # Smaller values for finer k-point grids
delta_values = [1e-10]  # Smaller values for finer Wilson loop discretization

if rank == 0:
    a = inputs["superlattice_potential_periodicity"]
    kx_min, kx_max = -np.pi/a, np.pi/a
    ky_min, ky_max = -2*np.pi/(np.sqrt(3)*a), 2*np.pi/(np.sqrt(3)*a)

    # Create directories if they don't exist
    if not os.path.exists(chern_output_dir):
        os.makedirs(chern_output_dir)

# Open a file to store convergence results
if rank == 0:
    convergence_filename = os.path.join(chern_output_dir, 'chern_convergence.txt')
    with open(convergence_filename, 'w') as conv_file:
        conv_file.write("dkx, dky, delta, band, Chern number\n")

# Loop over grid resolution (dkx, dky) and Wilson loop size (delta)
for dkx, dky in zip(dk_values, dk_values):  # Same dkx and dky for simplicity
    if rank == 0:
        kpoints = generate_kpoints(kx_min, kx_max, dkx, ky_min, ky_max, dky)
    else:
        kpoints = None

    # Broadcast kpoints to all processes
    kpoints = comm.bcast(kpoints, root=0)

    # Loop over delta values for the Wilson loop
    for delta in delta_values:
        # Loop over band indices
        for bnd_idx in bnd_indices:
            # Call the function to compute the Berry curvature and Chern number
            _ , chern_number = parallel_compute_berry_curvature_and_chern(
                kpoints, dkx, dky, delta=delta, bnd_idx=bnd_idx, H_calculator=H_calculator
            )

            if rank == 0:
                # Save the results for convergence analysis
                with open(convergence_filename, 'a') as conv_file:
                    conv_file.write(f"{dkx}, {dky}, {delta}, {bnd_idx}, {chern_number}\n")
