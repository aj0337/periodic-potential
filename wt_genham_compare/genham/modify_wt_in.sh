#!/bin/bash

# Define file paths
template_file="BLG/wt.in-template"
poscar_file="BLG/rep_3x3/nn_1/POSCAR"
output_file="BLG/rep_3x3/nn_1/null/wt.in"

# Extract lattice vectors from POSCAR
lattice_vectors=$(sed -n '3,5p' "$poscar_file")

# Extract number of atoms from POSCAR
num_atoms=$(sed -n '7p' "$poscar_file" | awk '{for(i=1; i<=NF; i++) sum += $i; print sum}')

# Extract coordinate system (Direct or Cartesian) from POSCAR
coordinate_system=$(sed -n '8p' "$poscar_file")

# Extract atom coordinates from POSCAR, starting from line 9 (assuming the coordinate system is on line 8)
sed -n '9,$p' "$poscar_file" | awk '{printf("C    %.6f    %.6f    %.6f\n", $1, $2, $3)}' >/tmp/atom_coordinates.txt

# Read the template file and replace the LATTICE card with the lattice vectors and "Angstrom" line
# Also replace the value under the ATOM_POSITIONS flag with the number of atoms, coordinate system, and rounded coordinates
awk -v lattice_vectors="$lattice_vectors" -v num_atoms="$num_atoms" -v coordinate_system="$coordinate_system" '
BEGIN {replace_lattice = 0; replace_atom_positions = 0; replace_projectors = 0}
{
  if (/LATTICE/) {
    print "LATTICE"
    print "Angstrom"
    print lattice_vectors
    replace_lattice = 1
  } else if (/ATOM_POSITIONS/) {
    print "ATOM_POSITIONS"
    print num_atoms
    print coordinate_system
    while ((getline line < "/tmp/atom_coordinates.txt") > 0) {
      print line
    }
    replace_atom_positions = 1
    close("/tmp/atom_coordinates.txt")
  } else if (/PROJECTORS/) {
    print "PROJECTORS"
    print num_atoms"*1"
    for (i = 0; i < num_atoms; i++) {
      print "C pz"
    }
    replace_projectors = 1
  } else {
    if (replace_lattice && /^ *$/) {
      replace_lattice = 0
      next
    }
    if (replace_atom_positions && /^[0-9]+$/) {
      replace_atom_positions = 0
      next
    }
    if (replace_projectors && /^[0-9]+$/) {
      replace_projectors = 0
      next
    }
    if (!replace_lattice && !replace_atom_positions && !replace_projectors) {
      print $0
    }
  }
}
' "$template_file" >"$output_file"

echo "Modified file created at $output_file"
