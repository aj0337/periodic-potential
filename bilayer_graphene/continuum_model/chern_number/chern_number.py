#!/usr/bin/env python3

import sys
import numpy as np

from tb_hamiltonian.continuum import GrapheneContinuumModel, compute_eigenstuff

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import numpy as np

def compute_berry_curvature(kpoints, dkx, dky, bnd_idx, H_calculator):
    """
    Computes the Berry curvature using finite differences over the Brillouin zone.

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
    berry_flux : float
        The total Berry curvature over all k-points.

    Equation:
    ---------
    The Berry curvature (F) is computed as:

    F = Im(⟨∂kxψ|∂kyψ⟩ - ⟨∂kyψ|∂kxψ⟩)

    where ψ is the wavefunction of the band at a given k-point, and ∂kx, ∂ky are the derivatives with respect to kx and ky.
    """
    _, psi = compute_eigenstuff(H_calculator, kpoints)
    _, psi_dkx = compute_eigenstuff(H_calculator, [[k[0] + dkx, k[1]] for k in kpoints])
    _, psi_dky = compute_eigenstuff(H_calculator, [[k[0], k[1] + dky] for k in kpoints])

    # Extract the wavefunction for the specific band
    psi = psi[:, :, bnd_idx]
    dpsi_dkx = (psi - psi_dkx[:, :, bnd_idx]) / dkx
    dpsi_dky = (psi - psi_dky[:, :, bnd_idx]) / dky

    # Compute Berry flux using finite difference
    berry_flux = np.imag(
        np.einsum("ij,ij->i", np.conj(dpsi_dkx), dpsi_dky)
        - np.einsum("ij,ij->i", np.conj(dpsi_dky), dpsi_dkx)
    )
    return np.sum(berry_flux)


def compute_berry_curvature_log(kpoints, dkx, dky, bnd_idx, H_calculator):
    """
    Computes the Berry curvature using a logarithmic approach over the Brillouin zone.

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
    berry_flux_log : float
        The total Berry curvature using the logarithmic approach.

    Equation:
    ---------
    The Berry curvature in this approach is computed using a phase comparison of neighboring states:

    F_log = Im(log(Ux * Uy * conj(Ux_dagger) * conj(Uy_dagger)))

    where Ux and Uy are overlaps between wavefunctions at adjacent k-points in the x and y directions, and Ux_dagger, Uy_dagger are their complex conjugates.
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
    Ux = np.einsum("ij,ij->i", np.conj(psi), psi_right)
    Uy = np.einsum("ij,ij->i", np.conj(psi), psi_up)
    Ux_dagger = np.einsum("ij,ij->i", np.conj(psi_up), psi_diag)
    Uy_dagger = np.einsum("ij,ij->i", np.conj(psi_right), psi_diag)

    # Berry flux using the logarithmic method
    berry_flux_log = np.imag(np.log(Ux * Uy * np.conj(Ux_dagger) * np.conj(Uy_dagger)))

    return np.sum(berry_flux_log)


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

    Equation:
    ---------
    The Chern number (C) is computed as the integral of the Berry curvature (F) over the Brillouin zone:

    C = (1 / 2π) ∫ F(kx, ky) d^2k

    This is discretized over a mesh of k-points, with d^2k = dkx * dky.
    """
    # Compute Berry flux using the two different approaches
    berry_flux = compute_berry_curvature(kpoints, dkx, dky, bnd_idx, H_calculator)
    berry_flux_log = compute_berry_curvature_log(
        kpoints, dkx, dky, bnd_idx, H_calculator
    )

    # Normalize the Berry flux to compute the Chern number
    chern_number = berry_flux * dkx * dky / (2 * np.pi)
    chern_number_log = berry_flux_log * dkx * dky / (2 * np.pi)
    return chern_number, chern_number_log


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

    kx_vals = np.linspace(-np.pi, np.pi, nkpts)
    ky_vals = np.linspace(-np.pi, np.pi, nkpts)

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
    chern_local, chern_log_local = compute_chern_number(
        kpoints_local, dkx, dky, bnd_idx, H_calculator
    )

    # Sum the local results to rank 0
    chern_total = comm.reduce(chern_local, op=MPI.SUM, root=0)
    chern_log_total = comm.reduce(chern_log_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Chern number: {chern_total}")
        print(f"Chern number with log formula: {chern_log_total}")
