#!/bin/bash
#SBATCH --job-name=perturb_rnns         # create a short name for your job
#SBATCH --partition=kempner             # partition
#SBATCH --account=kempner_bsabatini_lab # account needed for kempner partition
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks-per-node=1             # total number of tasks per node
#SBATCH --cpus-per-task=8               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                    # number of allocated gpus per node
#SBATCH --mem=32G                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=01:30:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send email when job begins
#SBATCH --mail-type=end                 # send email when job ends

# we need to define the job name directly since it isn't a slurm environment variable
JOB_NAME="perturb-rnns"

# load python and activate our conda environment
module purge
module load python
conda activate networkAlignmentAnalysis

# record the start time
start_time=$(date +%s)

# this is the command that initiates the processes
python intrinsic-excitability/analysis.py --num_trials 128 --jobid 50 --perturb_type random

# record the end time
end_time=$(date +%s)

# measure the time elapsed for the core part of the job in the logfile
total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds"