#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_test_corruption.out
#
# Evaluates saved checkpoints (train label noise 0, 0.1, 0.2, 0.3 x seeds 0-4) on
# corrupted test images. LABEL_NOISE must stay unset: noise only affects training.
#
# Variables required, via --export:
#   DATASET   -- organamnist or bloodmnist
#   APPROACH  -- ewc
#
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SHUFFLE_CLASSES_FIX=1
unset LABEL_NOISE
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
REPO_DIR=${SLURM_SUBMIT_DIR}
cd ${REPO_DIR}/src
python -u ../analysis/label_noise/test_corruption_eval.py --datasets ${DATASET} --methods ${APPROACH} \
    --out ../analysis/label_noise/test_corruption_${DATASET}_${APPROACH}.csv
