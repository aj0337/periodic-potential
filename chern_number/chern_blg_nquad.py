import os
from mpi4py import MPI
import numpy as np
from scipy.interpolate import griddata
from scipy.integrate import nquad
from tb_hamiltonian.continuum import compute_eigenstuff
from tb_hamiltonian.continuum import GrapheneContinuumModel

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def compute_berry_curvature_at_k(kx, ky, delta, bnd_idx, H_calculator):
    """Compute the Berry curvature at a single (kx, ky) point using finite differences."""
    _, psi = compute_eigenstuff(H_calculator, [[kx, ky]])
    _, psi_right = compute_eigenstuff(H_calculator, [[kx + delta, ky]])
    _, psi_up = compute_eigenstuff(H_calculator, [[kx, ky + delta]])
    _, psi_diag = compute_eigenstuff(H_calculator, [[kx + delta, ky + delta]])

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
    berry_curvature = -np.imag(np.log(Ox * Oy * Ox_reverse * Oy_reverse)) / (delta ** 2)

    # Return the curvature for the first eigenstate at the k-point (assuming only one band is being studied)
    return berry_curvature[0]

def compute_berry_curvature_integral(delta, bnd_idx, H_calculator, kx_min, kx_max, ky_min, ky_max):
    """Integrate the Berry curvature over the Brillouin zone using nquad."""
    # Function to integrate: Berry curvature as a function of kx, ky
    def berry_curvature_integrand(kx, ky):
        return compute_berry_curvature_at_k(kx, ky, delta, bnd_idx, H_calculator)

    # Perform the integration using nquad
    result, error = nquad(berry_curvature_integrand, [[kx_min, kx_max], [ky_min, ky_max]])

    return result / (2 * np.pi)

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

# Values for delta and integration bounds
delta_values = [1e-5]  # Wilson loop discretization

if rank == 0:
    a = inputs["superlattice_potential_periodicity"]
    kx_min, kx_max = -np.pi/a, np.pi/a
    ky_min, ky_max = -2*np.pi/(np.sqrt(3)*a), 2*np.pi/(np.sqrt(3)*a)

    if not os.path.exists(chern_output_dir):
        os.makedirs(chern_output_dir)

# Loop over delta values and band indices
for delta in delta_values:
    for bnd_idx in bnd_indices:
        if rank == 0:
            # Compute the Chern number by integrating the Berry curvature
            chern_number = compute_berry_curvature_integral(delta, bnd_idx, H_calculator, kx_min, kx_max, ky_min, ky_max)

            # Save the Chern number
            with open(os.path.join(chern_output_dir, 'chern_nquad.txt'), 'a') as conv_file:
                conv_file.write(f"{delta}, {bnd_idx}, {chern_number}\n")
