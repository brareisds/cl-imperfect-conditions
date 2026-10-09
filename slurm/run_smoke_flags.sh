#!/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --account=def-mehraneb
#SBATCH --mem=32000M
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j_smoke.out
export LD_LIBRARY_PATH=$HOME/cudalibs:$LD_LIBRARY_PATH
# Smoke test of the new flags: 2 epochs, tasks 0-1, organamnist, seed 0.
# A: ewc + FISHER_NORM    B: icarl + SAVE_EXEMPLARS + NO_CLASS_WEIGHTS
# C: ewc cross noise      D: icarl cross noise
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SHUFFLE_CLASSES_FIX=1
module load python/3.11.5 cuda cudnn
source ~/envs/hello_world/bin/activate
REPO_DIR=${SLURM_SUBMIT_DIR}
cd ${REPO_DIR}
OUT=${REPO_DIR}/results/smoke
rm -rf ${OUT}; mkdir -p ${OUT} logs

run() {  # $1=case $2=approach
  PYTHONPATH=src python -u src/main_incremental.py --exp-name smoke_$1 \
    --datasets organamnist --num-tasks 4 --network resnet18 --seed 0 \
    --nepochs 2 --batch-size 32 --results-path ${OUT} --approach $2 --gpu 0 \
    --num-exemplars 200 --exemplar-selection herding --lr 0.001 \
    --stop-at-task 2 --save-models > ${OUT}/log_$1.txt 2>&1
  rc=$?
  echo "case $1 exit code: $rc"
  return $rc
}
fail=0

( export LABEL_NOISE=0.3 LABEL_NOISE_SEED=0 LABEL_NOISE_MODE=intra FISHER_NORM=1e-6 SAVE_FISHER=1; run A ewc ) || fail=1
( export LABEL_NOISE=0.3 LABEL_NOISE_SEED=0 LABEL_NOISE_MODE=intra SAVE_EXEMPLARS=1 NO_CLASS_WEIGHTS=1; run B icarl ) || fail=1
( export LABEL_NOISE=0.3 LABEL_NOISE_SEED=0 LABEL_NOISE_MODE=cross; run C ewc ) || fail=1
( export LABEL_NOISE=0.3 LABEL_NOISE_SEED=0 LABEL_NOISE_MODE=cross; run D icarl ) || fail=1

for c in A B C D; do
  echo "--- case $c"
  echo "error lines: $(grep -c -E 'Traceback|Error' ${OUT}/log_$c.txt)"
  grep -h "label-noise" ${OUT}/log_$c.txt | head -3
  grep -A3 "TAg Acc" ${OUT}/log_$c.txt | tail -4
done
echo "--- files"
ls ${OUT}/organamnist_ewc_smoke_A/fisher 2>/dev/null
ls ${OUT}/organamnist_icarl_smoke_B/exemplars 2>/dev/null
for c in A B C D; do grep -q -E "Traceback|Error" ${OUT}/log_$c.txt && fail=1; done
[ -f ${OUT}/organamnist_ewc_smoke_A/fisher/task1_raw_unnorm.pt ] || fail=1
[ -f ${OUT}/organamnist_icarl_smoke_B/exemplars/task1.pt ] || fail=1
echo "smoke result: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
