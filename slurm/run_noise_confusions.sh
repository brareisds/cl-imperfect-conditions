#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_noise_confusions.out
#
# Confusion matrices (task-agnostic, final checkpoint) for the label-noise runs.
# LABEL_NOISE must stay unset here: noise only affects training, the test set
# and the class order are identical to the clean runs.
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
OUT=${REPO_DIR}/analysis/label_noise/confusions
mkdir -p ${OUT}/logs
cd ${REPO_DIR}/src

for P in 0.1 0.2 0.3; do
  for S in 0 1 2 3 4; do
    CK=${REPO_DIR}/results/label_noise/${DATASET}_${APPROACH}_p${P}/${DATASET}_${APPROACH}_fixd_${S}
    echo "== ${DATASET} ${APPROACH} p${P} seed${S}"
    python -u ../analysis/confusion_matrix_check.py --checkpoint-dir ${CK} \
        --dataset ${DATASET} --num-tasks 4 --network resnet18 --seed ${S} \
        --output-csv ${OUT}/${DATASET}_${APPROACH}_p${P}_seed${S}_confusion.csv \
        > ${OUT}/logs/${DATASET}_${APPROACH}_p${P}_seed${S}.log 2>&1
  done
done
