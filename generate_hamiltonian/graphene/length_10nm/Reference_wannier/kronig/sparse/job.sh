#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread
#SBATCH --account="s1267"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anooja.jayaraj@empa.ch
#SBATCH --output=_scheduler-stdout.txt
#SBATCH --error=_scheduler-stderr.txt


module load daint-mc
module load intel-oneapi/2021.3.0

start=0.1
end=1.25
increment=0.2
value=$start
while [ $(echo "$value <= $end" | bc) -eq 1 ]; do
    potential=$(echo "scale=2; $value" | bc)
    dirname=potential_$potential
    mkdir -p $dirname
    cp system.in POSCAR $dirname
    sed -i "s/\(^ *potential_height_U0 =\).*/\1 $potential/" $dirname/system.in
    (
        cd $dirname
        /users/ajayaraj/software/wannier_tools/bin/tgtbgen
    )
    # cp wt.in $dirname/wt.in
    # (
    #     cd $dirname
    #     srun -n 16 /users/ajayaraj/software/wannier_tools/bin/wt.x
    # )
    value=$(echo "$value + $increment" | bc)
done
