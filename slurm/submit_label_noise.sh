#!/bin/bash
for D in organamnist bloodmnist; do
  for A in ewc icarl; do
    for P in 0.1 0.2 0.3; do
      sbatch --job-name=ln_${D}_${A}_${P} \
        --export=ALL,DATASET=${D},APPROACH=${A},NOISE=${P} \
        slurm/run_label_noise.sh
    done
  done
done
