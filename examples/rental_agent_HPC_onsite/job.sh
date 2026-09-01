#!/bin/bash
#SBATCH --account=<YOUR-ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00
#SBATCH --mem=1G
#SBATCH -J ac-example-sim
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err

# Usage: sbatch job.sh <x_value> <task_id> <simulation_dir>
# Computes y = x^2 and writes the result to simulation_dir/result_<task_id>.txt

X=$1
TASK_ID=$2
SIMULATION_DIR=$3

RESULT=$(python3 -c "print($X ** 2)")
echo "$RESULT" > "${SIMULATION_DIR}/result_${TASK_ID}.txt"
