#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_label_noise.out
export LD_LIBRARY_PATH=$HOME/cudalibs:$LD_LIBRARY_PATH
#
# Same as run_official_savemodel.sh, but with symmetric intra-task
# label noise on the training stream (LABEL_NOISE=p, patched into
# memory_dataset.py; own RNG, seeded per seed via LABEL_NOISE_SEED,
# set inside scripts/script_<dataset>_noise.sh). Test set stays clean.
# Also sets SAVE_FISHER=1 so the Fisher of each task is saved for
# comparison against the clean runs. Results go to a separate folder.
#
# Variables required, via --export:
#   DATASET   -- organamnist or bloodmnist
#   APPROACH  -- ewc or icarl
#   NOISE     -- label flip probability, e.g. 0.1, 0.2, 0.3
#
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SHUFFLE_CLASSES_FIX=1
export SAVE_FISHER=1
export LABEL_NOISE=${NOISE}
echo "Label noise + save-models + shuffle-fix -- dataset=${DATASET} approach=${APPROACH} noise=${NOISE}"
nvidia-smi
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
REPO_DIR=${SLURM_SUBMIT_DIR}
cd ${REPO_DIR}

LOGDIR="${REPO_DIR}/results/label_noise/${DATASET}_${APPROACH}_p${NOISE}"
mkdir -p ${LOGDIR}
LOGNAME="${REPO_DIR}/results/label_noise/lifelonger_label_noise_${DATASET}_${APPROACH}_p${NOISE}_output.log"

bash scripts/script_${DATASET}_noise.sh ${APPROACH} 0 fixd ${LOGDIR} \
    2>&1 | tee ${LOGNAME}
