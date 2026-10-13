#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=16000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_nme_eval.out
# Usage (repo root): sbatch slurm/run_nme_eval.sh [--setup intra cross] [--methods icarl ewc] [--legacy-root DIR]
export LD_LIBRARY_PATH=$HOME/cudalibs:$LD_LIBRARY_PATH
export SHUFFLE_CLASSES_FIX=1
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
cd ${SLURM_SUBMIT_DIR}
python analysis/label_noise/nme_eval.py "$@"
