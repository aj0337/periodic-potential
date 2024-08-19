import sys
from pathlib import Path
from ase.io import read
import numpy as np
from tb_hamiltonian import TBHamiltonian
from tb_hamiltonian.potentials import PotentialFactory
from tb_hamiltonian.utils import get_structure

sys.tracebacklimit = None

# Control parameters
debug = False
use_mpi = True

# Initial structure
structure_label = "BLG"
input_path = Path(".")
initial_structure = read(input_path / "POSCAR", format="vasp")

# Hamiltonian parameters
hopping_parameters = [0.0, -2.7, 0.0, -0.27]
interlayer_coupling = 0.33
nn = 1  # degree of nearest neighbors | don't use 0!<

# Distances
distances = [0.0, 1.425, 2.468, 2.850]

workdir = Path(".")

# lengths
lx = 503.0  # length in x direction (Å)
# ly = 2.7
ly = lx / np.sqrt(3)  # length in y direction (Å) keeping the b/a ratio
lz = 10  # length in z direction (Å)
workdir /= f"len_{lx}x{round(ly,3)}/nn_{nn}"

# # or, repetitions
# nx = 20  # number of repetitions in x direction
# ny = 1  # number of repetitions in y direction
# nz = 1  # number of repetitions in z direction
# workdir /= f"rep_{nx}x{ny}/nn_{nn}"

workdir.mkdir(parents=True, exist_ok=True)

# Define structure
structure = get_structure(
    initial_structure=initial_structure,
    lengths=(lx, ly, lz),
    # repetitions=(nx, ny, nz),
    # structure_filepath=workdir
)

# This will show up at top of Hamiltonian output file
structure.info["label"] = structure_label
structure.write(workdir / "POSCAR", format="vasp")

# Compute H

H = TBHamiltonian(
    structure=structure,
    nearest_neighbor=nn,
    distances=distances,
    hopping_parameters=hopping_parameters,
    interlayer_coupling=interlayer_coupling,
    debug=debug,
)

H.build()

# Apply onsite term
potential_types = ["rectangular"]

for ptype in potential_types:

    for amplitude in np.arange(10e-3, 120e-3, 20e-3):
        potential = PotentialFactory(ptype)
        alpha = (1.0, 0.3)

        path = workdir / potential.name

        if ptype != "null":

            potential.params = {
                "amplitude": amplitude,
                "nx": 2,
                "ny": 2
            }
            path = (
                workdir / potential.name
                / f"amplitude_{potential.params['amplitude']}"
                / f"nx_ny_{potential.params['nx']}_{potential.params['ny']}"
                / f"alpha_{alpha}"
            )

        path.mkdir(parents=True, exist_ok=True)

        H.update_onsite_terms(
            onsite_term=0.0,
            potential=potential,
            alpha=alpha,
        )

        # Write H to file
        H.write_to_file(path, use_mpi=use_mpi)

        # Compute bands
        band_structure = H.get_band_structure(
            high_sym_points={
                "Γ": (0.00000, 0.00000, 0.00000),
                "A": (0.00000, 0.30000, 0.00000),
                "P": (0.00000, 0.33333, 0.00000),
                "B": (0.00000, 0.36666, 0.00000),
                "X": (0.00000, 0.50000, 0.00000),
                "Y": (0.50000, 0.00000, 0.00000),
                "W": (0.50000, 0.50000, 0.00000),
                "K": (1.00000, 0.33333, 0.00000),
                "M": (1.00000, 0.00000, 0.00000),
            },
            path="Γ K M Γ",
            points_per_segment=144,
            use_sparse_solver=True,
            sparse_solver_params={"k": 40, "sigma": 1e-8},
            use_mpi=use_mpi,
            save_data=True,
            outdir=path,
        )

        H.plot_bands(
            band_structure,
            title=ptype,
            mode="scatter",
            plot_params={"s":1,"c":"k"},
            fig_params={"ylim":(-0.025,0.025)},
            save_fig=True,
            outdir=path,
            use_mpi=use_mpi,
        )
