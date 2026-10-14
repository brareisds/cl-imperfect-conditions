#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_confusion_eval.out
#
# Saves confusion matrices (logits and NME) from saved checkpoints and exemplars
# (phase1 iCaRL, phase4 EWC; train label noise 0 and 0.3 x seeds 0-4).
# Existing files in analysis/label_noise/confusions are compared, not overwritten.
#
# Variables required, via --export:
#   DATASET  -- organamnist or bloodmnist
#   METHOD   -- icarl or ewc
# Optional: LEGACY_ROOT, EXTRA (e.g. "--seeds 0 --noise 0.3" for a smoke test)
#
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SHUFFLE_CLASSES_FIX=1
unset LABEL_NOISE
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
REPO_DIR=${SLURM_SUBMIT_DIR}
cd ${REPO_DIR}/src
python -u ../analysis/label_noise/confusion_eval.py --datasets ${DATASET} --methods ${METHOD} \
    ${LEGACY_ROOT:+--legacy-root ${LEGACY_ROOT}} ${EXTRA}
