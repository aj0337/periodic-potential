#!/usr/bin/env bash
# start=-1
# end=4
# increment=1
# for ((exponent = start; exponent <= end; exponent += increment)); do
start=1
end=3
increment=1
exponent=-1
for ((value = start; value <= end; value += increment)); do
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
    # TODO Automate the modification of wt.in to allow the unfolding process if required
    cp wt.in $dirname/wt.in
    emin=0
    emax=0.5
    (
        cd $dirname
        /home/anooja/Work/tools/wannier_tools/bin/wt.x
        sed -i "s/\(^ *emin=\).*/\1 $emin/" bulkek.gnu
        sed -i "s/\(^ *emax=\).*/\1 $emax/" bulkek.gnu
        gnuplot bulkek.gnu
    )
done
