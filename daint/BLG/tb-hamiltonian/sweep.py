import sys
from pathlib import Path
import os
import numpy as np
from tb_hamiltonian import TBHamiltonian
from tb_hamiltonian.potentials import PotentialFactory
from tb_hamiltonian.utils import get_structure

sys.tracebacklimit = None

debug = False
use_mpi = True

nn = 1  # number of nearest neighbors | don't use 0!

workdir = Path(".")

for lx in np.arange(204.,604.,100.):

    # lengths
    lx = lx  # length in x direction (Å)
    ly = 2.7
    # ly = lx / np.sqrt(3)  # length in y direction (Å) keeping the b/a ratio
    lz = 10  # length in z direction (Å)
    basepath = workdir / f"len_{lx}x{round(ly,3)}/nn_{nn}"

    # or, repetitions
    # nx = 20  # number of repetitions in x direction
    # ny = 1  # number of repetitions in y direction
    # nz = 1  # number of repetitions in z direction
    # basepath = workdir / f"rep_{nx}x{ny}/nn_{nn}"

    basepath.mkdir(parents=True, exist_ok=True)

    # Define structure

    structure = get_structure(
        unit_cell_filepath=workdir / "POSCAR",  # local unit cell file
        lengths=(lx, ly, lz),
        # repetitions=(nx, ny, nz),
        # structure_filepath=workdir
    )

    structure.info["label"] = "BLG"  # will show up at top of Hamiltonian output file

    structure.write(basepath / "POSCAR", format="vasp")

    # Compute H

    H = TBHamiltonian(
        structure=structure,
        nearest_neighbor=nn,
        distances=(0.0, 1.425, 2.468, 2.850),
        hopping_parameters=(0.0, -2.7, 0.0, -0.27),
        interlayer_coupling=0.33,
        debug=debug,
    )

    H.build()

    # Apply onsite term

    potential_types = ["null", "kronig-penney", "kronig-penney-break-even-sym", "kronig-penney-break-odd-sym", "sine"]

    for ptype in potential_types:

        if ptype == "null":

            potential = PotentialFactory(ptype)
            H.update_onsite_terms(
                onsite_term=0.0,
                potential=potential,
                alpha=(1.0,1.0)
            )
            path = (
                basepath / f"{potential.name}"
            )

            path.mkdir(parents=True, exist_ok=True)

            # Write H to file
            H.write_to_file(path, use_mpi=use_mpi)

            # Plotting
            H.plot_bands(
                high_sym_points={
                    "Γ": (0.00000, 0.00000, 0.00000),
                    "P": (0.00000, 0.33333, 0.00000),
                    "X": (0.00000, 0.50000, 0.00000),
                    "Y": (0.50000, 0.00000, 0.00000),
                    "K": (1.0000, 0.33333, 0.00000),
                    "M": (1.0000, 0.00000, 0.00000),
                },
                k_path="Γ K M Γ",
                points_per_segment=144,
                use_sparse_solver=True,
                sparse_solver_params={"k": 40, "sigma": 1e-8},
                use_mpi=use_mpi,
                fig_params={"ylim":(-1,1)},
                workdir=path,
                mode="scatter",
                plot_params={"s":5}
            )

        else:

            for amplitude in np.arange(10e-3, 60e-3, 10e-3):
                for alayer2 in np.arange(3e-1,1.0,2e-1):

                    alpha = (1.0, alayer2)
                    potential = PotentialFactory(ptype)
                    potential.params = {
                        "amplitude": amplitude,
                        "width": 0.5,
                    }
                    H.update_onsite_terms(
                        onsite_term=0.0,
                        potential=potential,
                        alpha=alpha,
                    )
                    path = (
                        basepath / f"{potential.name}"
                        / f"amplitude_{potential.params['amplitude']}"
                        / f"width_{potential.params['width']}"
                        / f"alpha_{alpha[1]}"
                    )

                    path.mkdir(parents=True, exist_ok=True)

                    # Write H to file
                    H.write_to_file(path, use_mpi=use_mpi)

                    # Plotting
                    H.plot_bands(
                        high_sym_points={
                            "Γ": (0.00000, 0.00000, 0.00000),
                            "P": (0.00000, 0.33333, 0.00000),
                            "X": (0.00000, 0.50000, 0.00000),
                            "Y": (0.50000, 0.00000, 0.00000),
                            "K": (1.0000, 0.33333, 0.00000),
                            "M": (1.0000, 0.00000, 0.00000),
                        },
                        k_path="Γ K M Γ",
                        points_per_segment=144,
                        use_sparse_solver=True,
                        sparse_solver_params={"k": 40, "sigma": 1e-8},
                        use_mpi=use_mpi,
                        fig_params={"ylim":(-1,1)},
                        workdir=path,
                        mode="scatter",
                        plot_params={"s":5}
                    )
