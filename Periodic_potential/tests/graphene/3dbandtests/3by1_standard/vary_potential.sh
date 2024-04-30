#!/usr/bin/env bash
# start=-1
# end=4
# increment=1
# for ((exponent = start; exponent <= end; exponent += increment)); do
start=0
end=100
increment=20
exponent=-1
for ((value = start; value <= end; value += increment)); do
    potential=$(bc <<<"scale=1; $value*10^$exponent")
    dirname=potential_$potential
    mkdir -p $dirname
    cp system.in POSCAR $dirname
    sed -i "s/\(^ *potential_height_U0 =\).*/\1 $potential/" $dirname/system.in
    (
        cd $dirname
        /home/anooja/Work/tools/wannier_tools/utility/twisted_graphene_system_tight_binding/tgtbgen
    )
    cp wt.in $dirname/wt.in
    (
        cd $dirname
        /home/anooja/Work/tools/wannier_tools/bin/wt.x
    )
done
