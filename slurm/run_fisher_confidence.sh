#!/bin/bash
#SBATCH --time=0-03:00:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_fisher_confidence.out
#
# Required via --export: DATASET (organamnist or bloodmnist)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SHUFFLE_CLASSES_FIX=1
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
cd ${SLURM_SUBMIT_DIR}/src
python -u ../analysis/label_noise/fisher_confidence.py --datasets ${DATASET} \
    --out ../analysis/label_noise/fisher_confidence_${DATASET}.csv \
    | tee ../analysis/label_noise/fisher_confidence_out_${DATASET}.txt
