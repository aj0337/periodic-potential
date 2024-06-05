#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --job-name="defs1"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
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

E_L=0.005
Vext_values=(0 5 9 11 65 70 75 80)

for Vext in "${Vext_values[@]}"; do
    potential=$(echo "scale=2; $Vext * $E_L" | bc)
    dirname=potential_$potential
    mkdir -p $dirname
    cp wt.in $dirname/wt.in
    (
        cd $dirname
        srun -n 16 /users/ajayaraj/software/wannier_tools/bin/wt.x
    )
done
