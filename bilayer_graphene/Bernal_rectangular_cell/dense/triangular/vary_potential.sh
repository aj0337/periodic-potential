start=0.05
end=1.5
increment=1.6
value=$start
while [ $(echo "$value <= $end" | bc) -eq 1 ]; do
    potential=$(echo "scale=2; $value" | bc)
    dirname=potential_test_$potential
    echo $potential
    mkdir -p $dirname
    cp system.in POSCAR $dirname
    sed -i "s/\(^ *potential_height_U0 =\).*/\1 $potential/" $dirname/system.in
    (
        cd $dirname
        /home/anooja/Work/tools/wannier_tools/utility/twisted_graphene_system_tight_binding/tgtbgen
    )
    # cp wt.in $dirname/wt.in
    # (
    #     cd $dirname
    #     /home/anooja/Work/tools/wannier_tools/src/wt.x
    # )
    value=$(echo "$value + $increment" | bc)
done
