#!/usr/bin/env python3
import sys
import numpy as np

from tb_hamiltonian.continuum import GrapheneContinuumModel, compute_eigenstuff
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np


def compute_berry_curvature_wilson(kpoints, dkx, dky, bnd_idx, H_calculator):

    """
    Computes the Berry curvature using the Wilson loop approach over the Brillouin zone.

    Parameters:
    -----------
    kpoints : list of list of float
        List of k-points in the Brillouin zone for which to compute the Berry curvature.
    dkx : float
        The step size in the x-direction in reciprocal space.
    dky : float
        The step size in the y-direction in reciprocal space.
    bnd_idx : int
        The index of the band for which to compute the Berry curvature.
    H_calculator : callable
        A function that calculates the Hamiltonian at a given k-point.

    Returns:
    --------
    total berry_flux : float
        The total Berry curvature using the Wilson loop approach.

    Equation:
    ---------
    The Berry curvature in the Wilson loop approach is computed as:

    F_wilson = ∫ Im(log(Ox * Oy * Ox_reverse * Oy_reverse)) d^2k

    where:

    - \( O_x \) is the overlap between wavefunctions at neighboring k-points in the x-direction:
        O_x = ⟨ψ(kx, ky) | ψ(kx + dkx, ky)⟩

    - \( O_y \) is the overlap between wavefunctions at neighboring k-points after moving in the x-direction, in the y-direction:
        O_y = ⟨ψ(kx + dkx, ky) | ψ(kx + dkx, ky + dky)⟩

    - \( O_x_{\text{reverse}} \) is the overlap between wavefunctions at neighboring k-points after moving in the y-direction, in the reverse x-direction:
        O_x_{\text{reverse}} = ⟨ψ(kx + dkx, ky + dky) | ψ(kx, ky + dky)⟩

    - \( O_y_{\text{reverse}} \) is the overlap between wavefunctions at neighboring k-points after moving in the reverse x-direction, in the reverse y-direction:
        O_y_{\text{reverse}} = ⟨ψ(kx, ky + dky) | ψ(kx, ky)⟩


    The overlaps are computed using the inner product of the wavefunctions at the respective k-points.
    """

    _, psi = compute_eigenstuff(H_calculator, kpoints)
    _, psi_right = compute_eigenstuff(
        H_calculator, [[k[0] + dkx, k[1]] for k in kpoints]
    )
    _, psi_up = compute_eigenstuff(H_calculator, [[k[0], k[1] + dky] for k in kpoints])
    _, psi_diag = compute_eigenstuff(
        H_calculator, [[k[0] + dkx, k[1] + dky] for k in kpoints]
    )

    # Extract the wavefunction for the specific band
    psi = psi[:, :, bnd_idx]
    psi_right = psi_right[:, :, bnd_idx]
    psi_up = psi_up[:, :, bnd_idx]
    psi_diag = psi_diag[:, :, bnd_idx]

    # Overlaps between wavefunctions at neighboring k-points
    Ox = np.einsum("ij,ij->i", np.conj(psi), psi_right)  # Correct
    Oy = np.einsum("ij,ij->i", np.conj(psi_right), psi_diag)  # Now using psi_right for the next step in the y direction
    Ox_reverse = np.einsum("ij,ij->i", np.conj(psi_diag), psi_up)  # Now moving backward in x direction
    Oy_reverse = np.einsum("ij,ij->i", np.conj(psi_up), psi)  # Finally, moving backward in y direction, returning to the start

    # Wilson loop Berry curvature
    berry_flux = np.imag(
        np.log(Ox * Oy * Ox_reverse * Oy_reverse)
    )

    # TODO Not convinced if the area i.e., dkx * dky should be divided or multiplied to normalize berry flux. Also a negative sign may be required depending on convention
    return dkx * dky * np.sum(berry_flux)


def compute_chern_number(kpoints, dkx, dky, bnd_idx, H_calculator):

    """
    Computes the Chern number of a given band in a 2D lattice system.

    Parameters:
    -----------
    kpoints : list of list of float
        List of k-points in the Brillouin zone.
    dkx : float
        The step size in the x-direction in reciprocal space.
    dky : float
        The step size in the y-direction in reciprocal space.
    bnd_idx : int
        The index of the band for which to compute the Chern number.
    H_calculator : callable
        A function that calculates the Hamiltonian at a given k-point.

    Returns:
    --------
    chern_number : float
        The Chern number computed using the finite difference method.
    chern_number_log : float
        The Chern number computed using the logarithmic method.
    chern_number_wilson : float
        The Chern number computed using the Wilson loop method.

    Equation:
    ---------
    The Chern number (C) is computed using the Berry curvature (F) as:

    C = (1 / 2π) F

    """
    # Compute Berry flux using the three different approaches
    berry_flux = compute_berry_curvature_wilson(
        kpoints, dkx, dky, bnd_idx, H_calculator
    )

    chern_number = berry_flux / (2 * np.pi)

    return chern_number


if __name__ == "__main__":

    inputs = dict(
        bond_length=1.425,
        interlayer_hopping=0.22,
        superlattice_potential_periodicity=500,
        superlattice_potential_amplitude=0.020,
        gate_bias=0.024,
        layer_potential_ratio=0.3,
        nearest_neighbor_order=1,
    )

    model = GrapheneContinuumModel(**inputs)

    # Constants for Brillouin zone grid
    nkpts = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    dcc = inputs["bond_length"]
    kx_vals = np.linspace(-np.pi/dcc, np.pi/dcc, nkpts)
    ky_vals = np.linspace(-2*np.pi/(np.sqrt(3)*dcc), 2*np.pi/(np.sqrt(3)*dcc), nkpts)

    dkx = kx_vals[1] - kx_vals[0]
    dky = ky_vals[1] - ky_vals[0]

    H_calculator = model.H_total_K

    bnd_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    kpoints = [[kx, ky] for kx in kx_vals for ky in ky_vals]

    # Split the kx_vals array across ranks
    kpoints_split = np.array_split(kpoints, size)

    # Each rank gets its own subset of kx_vals and ky_vals
    kpoints_local = kpoints_split[rank]

    # Compute the Chern number for this subset of kx and ky values
    chern_local = compute_chern_number(
        kpoints_local, dkx, dky, bnd_idx, H_calculator
    )

    # Sum the local results to rank 0
    chern_total = comm.reduce(chern_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Chern number with Wilson loop: {chern_total}")
