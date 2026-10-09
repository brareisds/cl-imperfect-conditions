#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_phase0.out
# CPU job. Extra args are passed to the script, e.g.: sbatch slurm/run_phase0.sh --smoke
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
cd ${SLURM_SUBMIT_DIR}/src
python ../analysis/label_noise/phase0_recency_wa.py "$@"
