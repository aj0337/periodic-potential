start=10
end=11
increment=10
value=$start
while [ $(echo "$value <= $end" | bc) -eq 1 ]; do
    potential=$(echo "scale=2; $value" | bc)
    dirname=potential_$potential
    mkdir -p $dirname
    cp system.in POSCAR $dirname
    sed -i "s/\(^ *potential_height_U0 =\).*/\1 $potential/" $dirname/system.in
    (
        cd $dirname
        /home/anooja/Work/tools/wannier_tools/bin/tgtbgen
    )
    # cp wt.in $dirname/wt.in
    # (
    #     cd $dirname
    #     /home/anooja/Work/tools/wannier_tools/bin/wt.x
    # )
    value=$(echo "$value + $increment" | bc)
done
