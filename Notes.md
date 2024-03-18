# Graphene tests

## Using the subutility of WannierTools to create TB Hamiltonian for Graphene in order to use the existing band unfolding capabilities

## General procedure

!!! Note that WannierTools can't build find the bandstructure of a system that has <= 2 atoms due to issues in arpack being unable to find eigenvalues and eigenvectors.

1. checkout graphene-BG-build branch of WannierTools (WT)
2. system.in takes the onsite, first, second and third neighbor hopping parameters as input
3. remember to set use_POSCAR to True to be able to provide your own structure.
4. Also use the sparse = True functionality in system.in
5. iR_cut = 3 is a safe choice. It builds 3 unit cells in the x and y direction around the central unit cell. This should be enough to consider upto third nearest neighbors.
6. Run tgtbgen.
7. The output Hamiltonian in real space will be stored in TG_hr.dat.
8. The first 3 columns are the x,y,z positions (in direct coordinates) of the other unit cell with respect to the central unit cell. Columns 4 and 5 are the atom numbers between which the interactions are being considered. Columns 6 to 11 are the x,y,z coordinates of the two atoms in Angstrom. Columns 12 and 13 are the real and imaginary parts of the hopping parameter.
9. This TG_hr.dat file can be used to build the Hamiltonian in real space and will later be modified by adding space dependent potentials.
10. In order to visualize the bandstructure, we will use the wt.x executable of WT by modifying the wt.in file.
11. Note that wt.x uses TG_hr.dat. In order for the wt.x run to work, TG_hr.dat can't have columns 6 to 11. (Columns 6 to 11 were added to allow for future implementation of space dependent potentials.) <span style="color:red">IMPORTANT! </span>Therefore these columns have to be removed from TG_hr.dat before running wt.x.
12. The primitive unit cell of graphene has two atoms per unit cell. However, in the study of the effect of periodic potentials on graphene, one will create unit cells of graphene that have more than two atoms (supercells). This will create a bandstructure different from that of unit cell graphene. In order to study the effect of periodic potentials on the original cell, one has to perform unfolding of the bands of the supercell graphene onto the Brillouin zone of the unit cell of graphene.
13. WT has the capability to perform band unfolding.
    1. In order to perform the band unfolding, create the primitive unit cell in Vesta and save to .vasp format.
    2. Create the supercell and save to .vasp format.
    3. Use the lattice vectors and direct coordinates of the atomic position in the primitive_cell.vasp as the input of LATTICE_UNFOLD and ATOM_POSITIONS_UNFOLD card of wt.in
    4. Running wt.x should produce a spectrum_unfold_kpath.dat that contains the unfolded band.

## Tests to be considered for comparison to the paper

1. Should the graphene layer be considered a molecule or use periodic boundary conditions? At the moment, I use PBC because the Hamiltonian (plotted in analysis.ipynb) has off diagonal components.
2. Seems like the paper doesn't do unfolding but zooms in at -0.033 eV to 0.033 eV of the folded bands.
3. Figure out the same path of the paper and perhaps stick to their rectangular cell.
