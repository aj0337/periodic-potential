import os
from mpi4py import MPI
import numpy as np
from scipy.interpolate import griddata
from tb_hamiltonian.continuum import compute_eigenstuff
from tb_hamiltonian.continuum import GrapheneContinuumModel

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def generate_kpoints(kx_min, kx_max, dkx, ky_min, ky_max, dky):
    num_kx_points = int(np.round((kx_max - kx_min) / dkx)) + 1
    num_ky_points = int(np.round((ky_max - ky_min) / dky)) + 1

    kx_values = np.linspace(kx_min, kx_max, num_kx_points)
    ky_values = np.linspace(ky_min, ky_max, num_ky_points)

    kx_grid, ky_grid = np.meshgrid(kx_values, ky_values, indexing='ij')
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

    psi = psi[:, :, bnd_idx]
    psi_right = psi_right[:, :, bnd_idx]
    psi_up = psi_up[:, :, bnd_idx]
    psi_diag = psi_diag[:, :, bnd_idx]

    Ox = np.einsum("ij,ij->i", np.conj(psi), psi_right)
    Oy = np.einsum("ij,ij->i", np.conj(psi_right), psi_diag)
    Ox_reverse = np.einsum("ij,ij->i", np.conj(psi_diag), psi_up)
    Oy_reverse = np.einsum("ij,ij->i", np.conj(psi_up), psi)

    berry_curvature = -np.imag(
        np.log(Ox * Oy * Ox_reverse * Oy_reverse)
    )/(delta**2)

    return berry_curvature

def compute_chern_number_from_interpolated(kpoints_fine, interpolated_berry_curvature, dkx_fine, dky_fine):
    chern_number = np.sum(interpolated_berry_curvature * dkx_fine * dky_fine) / (2 * np.pi)
    return chern_number

# ### Parallelization over kpoints
def parallel_compute_berry_curvature(kpoints, dkx, dky, delta, bnd_idx, H_calculator):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_kpoints = len(kpoints)
    counts = np.array([num_kpoints // size + (1 if i < num_kpoints % size else 0) for i in range(size)])
    displacements = np.cumsum(counts) - counts

    start = displacements[rank]
    end = start + counts[rank]
    local_kpoints = kpoints[start:end]

    local_berry_curvature = compute_berry_curvature_wilson(local_kpoints, dkx, dky, delta, bnd_idx, H_calculator)

    if rank == 0:
        berry_curvature = np.empty(num_kpoints, dtype=np.float64)
    else:
        berry_curvature = None

    comm.Gatherv(local_berry_curvature, [berry_curvature, counts, displacements, MPI.DOUBLE], root=0)
    return berry_curvature

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
chern_output_dir = "data/chern_blg"

H_calculator = model.H_total_K
nbands = H_calculator(np.array([0, 0])).shape[0]
mid_band = int(nbands / 2)
bnd_indices = [mid_band]  # Modify this list to compute for other bands

# Values for dkx, dky, and delta
dkx_coarse, dky_coarse = 5e-5, 5e-5  # Coarser grid
dkx_fine, dky_fine = 1e-6, 1e-6  # Finer grid
delta_values = [1e-5]  # Wilson loop discretization

if rank == 0:
    a = inputs["superlattice_potential_periodicity"]
    kx_min, kx_max = -np.pi/a, np.pi/a
    ky_min, ky_max = -2*np.pi/(np.sqrt(3)*a), 2*np.pi/(np.sqrt(3)*a)

    if not os.path.exists(chern_output_dir):
        os.makedirs(chern_output_dir)

# Compute Berry curvature on coarse grid
if rank == 0:
    kpoints_coarse = generate_kpoints(kx_min, kx_max, dkx_coarse, ky_min, ky_max, dky_coarse)
else:
    kpoints_coarse = None

kpoints_coarse = comm.bcast(kpoints_coarse, root=0)

for delta in delta_values:
    for bnd_idx in bnd_indices:
        # Compute Berry curvature on coarse grid
        berry_curvature_coarse = parallel_compute_berry_curvature(
            kpoints_coarse, dkx_coarse, dky_coarse, delta=delta, bnd_idx=bnd_idx, H_calculator=H_calculator
        )

        if rank == 0:
            # Define finer grid for interpolation
            kpoints_fine = generate_kpoints(kx_min, kx_max, dkx_fine, ky_min, ky_max, dky_fine)

            # Interpolate Berry curvature from coarse to fine grid
            interpolated_berry_curvature = griddata(kpoints_coarse, berry_curvature_coarse, kpoints_fine, method='cubic')

            # Compute Chern number on finer grid
            chern_number_fine = compute_chern_number_from_interpolated(kpoints_fine, interpolated_berry_curvature, dkx_fine, dky_fine)

            # Save the interpolated results and Chern number
            with open(os.path.join(chern_output_dir, 'chern_convergence.txt'), 'a') as conv_file:
                conv_file.write(f"{dkx_fine}, {dky_fine}, {delta}, {bnd_idx}, {chern_number_fine}\n")
