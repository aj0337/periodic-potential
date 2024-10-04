from mpi4py import MPI
import numpy as np


def haldane_hamiltonian(k, params):
    t = params["t"]
    t2 = params["t2"]
    M = params["M"]

    # Define the nearest and next-nearest neighbor vectors for the honeycomb lattice
    delta1 = np.array([[1, 0], [-1/2, np.sqrt(3)/2], [-1/2, -np.sqrt(3)/2]])
    delta2 = np.array([[0, np.sqrt(3)], [-3/2, -np.sqrt(3)/2], [3/2, -np.sqrt(3)/2]])
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
        H[0, 0] += 2 * t2 * np.sin(np.dot(k,d))
        H[1, 1] += -2 * t2 * np.sin(np.dot(k,d))
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

def generate_kgrid(N,a=1):
    """Generate a grid of k-points in the Brillouin zone"""
    kx = np.linspace(-np.pi/a, np.pi/a, N)
    ky = np.linspace(-np.pi/a, np.pi/a, N)
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
    distances = np.array(distances[:len(kpoints)])  # Ensure matching lengths
    return kpoints, distances

def H_calculator(k, params):
    return haldane_hamiltonian(k, params)


# ### Integration as a summation


def generate_kpoints(kx_min, kx_max, dkx, ky_min, ky_max, dky):
    kx_values = np.arange(kx_min, kx_max + dkx, dkx)
    ky_values = np.arange(ky_min, ky_max + dky, dky)

    kpoints = np.array([[kx, ky] for kx in kx_values for ky in ky_values])

    return kpoints

def compute_berry_curvature_wilson(kpoints, dkx, dky, delta, bnd_idx, H_calculator, params):
    _, psi = compute_eigenstuff(H_calculator, kpoints, params)
    _, psi_right = compute_eigenstuff(
        H_calculator, [[kx + delta, ky] for kx, ky in kpoints], params
    )
    _, psi_up = compute_eigenstuff(H_calculator, [[kx, ky + delta] for kx, ky in kpoints], params)
    _, psi_diag = compute_eigenstuff(
        H_calculator, [[kx + delta, ky + delta] for kx, ky in kpoints], params
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


def compute_chern_number(kpoints, dkx, dky, delta, bnd_idx, H_calculator, params):

    berry_curvature = compute_berry_curvature_wilson(kpoints, dkx, dky, delta, bnd_idx, H_calculator, params)
    chern_number = np.sum(berry_curvature* dkx * dky) / (2 * np.pi)
    return chern_number


# ### Parallelization over kpoints

def parallel_compute_chern_number(kpoints, dkx, dky, delta, bnd_idx, H_calculator, params):
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
    local_berry_curvature = compute_berry_curvature_wilson(local_kpoints, dkx, dky, delta, bnd_idx, H_calculator, params)

    # Prepare for gathering the results: allocate space for full result on root process
    if rank == 0:
        berry_curvature = np.empty(num_kpoints, dtype=np.float64)
    else:
        berry_curvature = None

    # Gather the results from all processes
    comm.Gatherv(local_berry_curvature, [berry_curvature, counts, displacements, MPI.DOUBLE], root=0)

    # Compute the Chern number on the root process
    if rank == 0:
        chern_number = np.sum(berry_curvature * dkx * dky) / (2 * np.pi)
        return chern_number
    else:
        return None

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Generate kpoints (only on rank 0)
if rank == 0:
    a = 1
    dkx, dky = 0.01, 0.01
    kx_min, kx_max = -np.pi/a, np.pi/a + dkx
    ky_min, ky_max = -2*np.pi/(np.sqrt(3)*a), 2*np.pi/(np.sqrt(3)*a) + dky
    kpoints = generate_kpoints(kx_min, kx_max, dkx, ky_min, ky_max, dky)
else:
    kpoints = None

# Broadcast kpoints to all processes
kpoints = comm.bcast(kpoints, root=0)

# Compute Chern number in parallel

params = {
    "t": 1.0,
    "t2": 0.05,
    "M": 0.2
}

dkx, dky = 0.01, 0.01
chern_number = parallel_compute_chern_number(kpoints, dkx, dky, delta=1e-2, bnd_idx=0, H_calculator=H_calculator, params=params)

# Print result on rank 0
if rank == 0:
    print("Chern number:", chern_number)
