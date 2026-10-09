#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_phase.out
export LD_LIBRARY_PATH=$HOME/cudalibs:$LD_LIBRARY_PATH
# One job = one (dataset, approach, noise) combination, 5 seeds.
# Required (--export): DATASET, APPROACH, NOISE, OUTROOT
# Optional: MODE (intra|cross), FISHER_NORM, SAVE_EXEMPLARS, NO_CLASS_WEIGHTS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SHUFFLE_CLASSES_FIX=1
export SAVE_FISHER=1
export LABEL_NOISE=${NOISE}
export LABEL_NOISE_MODE=${MODE:-intra}
echo "dataset=${DATASET} approach=${APPROACH} noise=${NOISE} mode=${LABEL_NOISE_MODE} out=${OUTROOT}" \
     "FISHER_NORM=${FISHER_NORM:-} SAVE_EXEMPLARS=${SAVE_EXEMPLARS:-} NO_CLASS_WEIGHTS=${NO_CLASS_WEIGHTS:-}"
nvidia-smi
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
REPO_DIR=${SLURM_SUBMIT_DIR}
cd ${REPO_DIR}

LOGDIR="${REPO_DIR}/${OUTROOT}/${DATASET}_${APPROACH}_p${NOISE}"
mkdir -p ${LOGDIR}
LOGNAME="${REPO_DIR}/${OUTROOT}/log_${DATASET}_${APPROACH}_p${NOISE}_output.log"

bash scripts/script_${DATASET}_noise.sh ${APPROACH} 0 fixd ${LOGDIR} 2>&1 | tee ${LOGNAME}
