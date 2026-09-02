#!/bin/bash
#SBATCH --time=0:05:00
#SBATCH --nodes=1
# TODO: update the two lines below for your HPC system before running.
# Run 'sinfo' to list partitions and 'sacctmgr show user $USER' for your account.
#SBATCH --partition=pdebug
#SBATCH --account=asccasc
#SBATCH --job-name=lf_simulation
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err

# Args passed by manager.py:
#   $1  Argument to lf objective
#   $2  Hero task ID

x=$1
task_id=$2

if [ -z "$x" ]; then
  echo "Error: No objective argument provided."
  echo "Usage: sbatch script_hf_slurm.sh <x> <task-id>"
  exit 1
fi
if [ -z "$task_id" ]; then
  echo "Error: No task-id provided."
  echo "Usage: sbatch script_hf_slurm.sh <x> <task-id>"
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"

# Load environment.
eval "$(conda shell.bash hook)"
conda activate xfoil


# Run mock simulation (replace with your real simulation commands)
echo "Running mock lf simulation with x=$x for task-id=$task_id"
output=$(python lf_simulation.py -x "$x")
echo "$output"

# Extract result from simulation output.
# mock_simulation.py prints a line of the form: conductivity=<value>
result=$(echo "$output" | awk -F: '/^Objective: /{print $2}' | tail -1)
if [ -z "$result" ]; then
    result=-1
fi
echo "Result: performance=$result"

# Write result to a file for the manager to pick up and pass to hero_finalize.
# hero_initialize and hero_finalize are called by the manager on the login node,
# which has outbound internet access. Compute nodes may not.
echo "$result" > "result_${task_id}.txt"
