import sys
from pathlib import Path

import numpy as np
from tb_hamiltonian import TBHamiltonian
from tb_hamiltonian.potentials import PotentialFactory
from tb_hamiltonian.utils import get_structure

sys.tracebacklimit = None

nn = 1  # number of nearest neighbors | don't use 0!

workdir = Path(".")

# lengths
lx = 103.0  # length in x direction (Å)
ly = 2.7  # length in y direction (Å) keeping the b/a ratio
lz = 10  # length in z direction (Å)
basepath = workdir / f"len_{lx}x{ly}/nn_{nn}"

# or, repetitions
# nx = 3  # number of repetitions in x direction
# ny = 3  # number of repetitions in y direction
# nz = 1  # number of repetitions in z direction
# basepath = workdir / f"rep_{nx}x{ny}/nn_{nn}"

# basepath = workdir

basepath.mkdir(parents=True, exist_ok=True)

# Define structure

structure = get_structure(
    unit_cell_filepath=workdir / "POSCAR",  # local unit cell file
    lengths=(lx, ly, lz),
    # repetitions=(nx, ny, nz),
    # structure_filepath=workdir
)

structure.info["label"] = "Graphene"  # will show up at top of Hamiltonian output file

structure.write(basepath / "POSCAR", format="vasp")

# Compute H

H = TBHamiltonian(
    structure=structure,
    nearest_neighbor=nn,
    distances=(0.0, 1.425, 2.468, 2.850),
    hopping_parameters=(0.0, -2.7, 0.0, -0.27),
    interlayer_coupling=0.33,
)

H.build()

# Apply onsite term

potential = PotentialFactory("kronig-penney")
potential.params = {
    "amplitude": 0.15,
    # "width": 0.5,
}
H.update_onsite_terms(
    onsite_term=0.0,
    potential=potential,
    alpha=(1.0, ),
)
path = (
    basepath / f"{potential.name}"
    / f"amplitude_{potential.params['amplitude']}"
    # / f"width_{potential.params['width']}"
)
path.mkdir(parents=True, exist_ok=True)

# Write H to file
H.write_to_file(path)

# Plotting
H.plot_bands(
    high_sym_points={
        "A": (-1.00000, 0.33700, 0.00000),
        "P": ( 0.00000, 0.33700, 0.00000),
        "B": ( 1.00000, 0.33700, 0.00000),
    },
    k_path="A P B",
    points_per_segment=120,
    use_sparse_solver=True,
    sparse_solver_params={"k": 10, "sigma": 1e-8},
    use_mpi=True,
    # fig_params={"ylim":(0,0.35)},
    savefig_path=path / "bands.png",
)
