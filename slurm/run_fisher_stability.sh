#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=8000M
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_fisher_stability.out
#
# CPU-only: compares Fisher (clean p0 vs noisy) per seed/task for ewc.
module load python/3.11.5
source ~/envs/hello_world/bin/activate
cd ${SLURM_SUBMIT_DIR}
python -u analysis/label_noise/fisher_stability.py
