#!/usr/bin/env bash

cp $1 POSCAR
rm bulkek.*
/home/anooja/Work/tools/wannier_tools/utility/twisted_graphene_system_tight_binding/tgtbgen
/home/anooja/Work/tools/wannier_tools/src/wt.x
sed -i 's/emin=.*/emin=-2.2/' bulkek.gnu
sed -i 's/emax=.*/emax=2.2/' bulkek.gnu
gnuplot bulkek.gnu
evince bulkek.pdf
