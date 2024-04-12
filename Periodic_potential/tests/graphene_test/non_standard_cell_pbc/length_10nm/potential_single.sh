#!/usr/bin/env bash
value=0
exponent=-1
potential=$(bc <<<"scale=1; $value*10^$exponent")
echo $potential
dirname=potential_$potential
mkdir -p $dirname
cp system.in POSCAR $dirname
sed -i "s/\(^ *potential_height_U0 =\).*/\1 $potential/" $dirname/system.in
(
    cd $dirname
    /home/anooja/Work/tools/wannier_tools/utility/twisted_graphene_system_tight_binding/tgtbgen
)
cp wt.in-line $dirname/wt.in
emin=-2
emax=2
(
    cd $dirname
    /home/anooja/Work/tools/wannier_tools/bin/wt.x
    sed -i "s/\(^ *emin=\).*/\1 $emin/" bulkek.gnu
    sed -i "s/\(^ *emax=\).*/\1 $emax/" bulkek.gnu
    gnuplot bulkek.gnu
)
