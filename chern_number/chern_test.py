import numpy as np


def haldane_hamiltonian(k, params):
    t = params["t"]
    t2 = params["t2"]
    M = params["M"]

    # Define the nearest and next-nearest neighbor vectors for the honeycomb lattice
    delta1 = np.array([[1, 0], [-1 / 2, np.sqrt(3) / 2], [-1 / 2, -np.sqrt(3) / 2]])
    delta2 = np.array(
        [[0, np.sqrt(3)], [-3 / 2, -np.sqrt(3) / 2], [3 / 2, -np.sqrt(3) / 2]]
    )
    H = np.zeros((2, 2), dtype=complex)

    # Nearest-neighbor hopping term
    for d in delta1:
        H[0, 1] += t * np.exp(1j * np.dot(k, d))
        H[1, 0] += t * np.exp(-1j * np.dot(k, d))

    # On-site potential (staggered potential)
    H[0, 0] += M
    H[1, 1] += -M

    # Next-nearest-neighbor hopping term with complex phase
    for d in delta2:
        H[0, 0] += 2 * t2 * np.sin(np.dot(k, d))
        H[1, 1] += -2 * t2 * np.sin(np.dot(k, d))
    return H


def compute_eigenstuff(H_calculator, kpoints, params):
    """Compute eigenvalues and eigenvectors for a list of k-points"""
    eigvals = []
    eigvecs = []
    for k in kpoints:
        H = H_calculator(k, params)
        val, vec = np.linalg.eigh(H)
        eigvals.append(val)
        eigvecs.append(vec)
    return np.array(eigvals), np.array(eigvecs)


def generate_kgrid(N):
    """Generate a grid of k-points in the Brillouin zone"""
    kx = np.linspace(-np.pi, np.pi, N)
    ky = np.linspace(-np.pi, np.pi, N)
    kpoints = np.array([[kx[i], ky[j]] for i in range(N) for j in range(N)])
    return kpoints


def generate_kpoints_along_path(path, N=100):
    """Generate k-points along a specified path in the Brillouin zone."""
    kpoints = []
    distances = [0]  # This will track the cumulative distance along the path

    # Generate k-points along the path
    for i in range(len(path) - 1):
        start, end = np.array(path[i]), np.array(path[i + 1])
        segment = np.linspace(start, end, N)
        kpoints.extend(segment)

        # Calculate the cumulative distance for each point in the segment
        delta_k = np.linalg.norm(end - start) / N
        for j in range(1, N + 1):
            distances.append(distances[-1] + delta_k)  # Cumulative distance

    kpoints = np.array(kpoints)
    distances = np.array(distances[: len(kpoints)])  # Ensure matching lengths
    return kpoints, distances


def H_calculator(k, params):
    return haldane_hamiltonian(k, params)


from mpi4py import MPI


def compute_berry_curvature_wilson(kpoints, dkx, dky, bnd_idx, H_calculator, params):
    _, psi = compute_eigenstuff(H_calculator, kpoints, params)
    _, psi_right = compute_eigenstuff(
        H_calculator, [[k[0] + dkx, k[1]] for k in kpoints], params
    )
    _, psi_up = compute_eigenstuff(
        H_calculator, [[k[0], k[1] + dky] for k in kpoints], params
    )
    _, psi_diag = compute_eigenstuff(
        H_calculator, [[k[0] + dkx, k[1] + dky] for k in kpoints], params
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
    berry_flux = -np.imag(np.log(Ox * Oy * Ox_reverse * Oy_reverse))

    return berry_flux


def compute_chern_number_parallel(kpoints, dkx, dky, bnd_idx, H_calculator, params):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide kpoints among the available processes
    num_kpoints = len(kpoints)
    chunk_size = num_kpoints // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank != size - 1 else num_kpoints

    kpoints_subset = kpoints[start:end]

    # Each process computes the Berry curvature for its subset of k-points
    berry_flux_subset = compute_berry_curvature_wilson(
        kpoints_subset, dkx, dky, bnd_idx, H_calculator, params
    )

    # Gather all berry_flux subsets from each process to the root process
    berry_flux = np.zeros_like(kpoints[:, 0])
    comm.Gather(berry_flux_subset, berry_flux, root=0)

    # Compute the total Chern number only on the root process
    if rank == 0:
        # Sum the Berry flux and normalize to compute the Chern number
        chern_number = np.sum(berry_flux) / ((2 * np.pi) * dkx * dky)
        return chern_number
    else:
        return None


# Example usage in your script
if __name__ == "__main__":
    # Define the grid of k-points and Berry flux
    N_grid = 100  # The number of points along each axis of the grid
    kpoints = generate_kgrid(N_grid)
    dkx = 2 * np.pi / N_grid
    dky = 2 * np.pi / N_grid

    params = {
        "t": 1.0,  # Nearest-neighbor hopping
        "t2": 0.10,  # Next-nearest-neighbor hopping
        "M": 0.2,  # Staggered potential
    }

    # Compute the Chern number in parallel
    bnd_idx = 0
    chern_number = compute_chern_number_parallel(
        kpoints, dkx, dky, bnd_idx, H_calculator, params
    )

    # Print the Chern number (only on the root process)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Computed Chern Number: {chern_number}")
