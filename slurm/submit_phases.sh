#!/bin/bash
# Usage (repo root): [DRY=1] bash slurm/submit_phases.sh phase1|phase2|phase3|phase4|phase5|phase6|ablation
# phase3 needs FNORM_ORGANAMNIST and FNORM_BLOODMNIST.
# DEP=<jobid>: jobs start only if that job finishes with exit code 0.
sub() {  # $1=job name, $2=--export string
  cmd="sbatch ${DEP:+--dependency=afterok:$DEP} ${3:+--time=$3} --job-name=$1 --export=ALL,$2 slurm/run_phase.sh"
  if [ -n "$DRY" ]; then echo "$cmd"; else eval "$cmd"; fi
}

phase1() {  # iCaRL saving exemplars, intra noise (incl. p=0)
  for D in organamnist bloodmnist; do for P in 0 0.1 0.2 0.3; do
    sub p1_${D}_icarl_${P} "DATASET=$D,APPROACH=icarl,NOISE=$P,OUTROOT=results/phase1_icarl,SAVE_EXEMPLARS=1"
  done; done
}
phase2() {  # cross-task noise
  for D in organamnist bloodmnist; do for A in ewc icarl; do for P in 0.1 0.3; do
    extra=""; [ "$A" = "icarl" ] && extra=",SAVE_EXEMPLARS=1"
    sub p2_${D}_${A}_${P} "DATASET=$D,APPROACH=$A,NOISE=$P,MODE=cross,OUTROOT=results/phase2_cross${extra}"
  done; done; done
}
phase3() {  # EWC with normalized Fisher
  if [ -z "$FNORM_ORGANAMNIST" ] || [ -z "$FNORM_BLOODMNIST" ]; then
    echo "set FNORM_ORGANAMNIST and FNORM_BLOODMNIST"; exit 1
  fi
  for D in organamnist bloodmnist; do
    if [ "$D" = "organamnist" ]; then C=$FNORM_ORGANAMNIST; else C=$FNORM_BLOODMNIST; fi
    for P in 0 0.3; do
      sub p3_${D}_ewc_${P} "DATASET=$D,APPROACH=ewc,NOISE=$P,OUTROOT=results/phase3_fisher_norm,FISHER_NORM=$C"
    done
  done
}
ablation() {  # iCaRL without class weights
  for P in 0 0.3; do
    sub ab_organamnist_icarl_${P} "DATASET=organamnist,APPROACH=icarl,NOISE=$P,OUTROOT=results/ablation_noweight,NO_CLASS_WEIGHTS=1"
  done
}
phase4() {  # EWC saving exemplars, intra noise (incl. p=0)
  for D in organamnist bloodmnist; do for P in 0 0.1 0.2 0.3; do
    sub p4_${D}_ewc_${P} "DATASET=$D,APPROACH=ewc,NOISE=$P,OUTROOT=results/phase4_ewc_exemplars,SAVE_EXEMPLARS=1"
  done; done
}
phase5() {  # EWC saving exemplars, cross-task noise
  for D in organamnist bloodmnist; do for P in 0.1 0.3; do
    sub p5_${D}_ewc_${P} "DATASET=$D,APPROACH=ewc,NOISE=$P,MODE=cross,OUTROOT=results/phase5_ewc_cross_exemplars,SAVE_EXEMPLARS=1"
  done; done
}
phase6() {  # EWC and iCaRL saving exemplars, no label noise, PathMNIST and TissueMNIST
  for D in pathmnist tissuemnist; do for A in icarl ewc; do
    if [ "$D" = "tissuemnist" ]; then T=0-12:00:00; else T=0-06:00:00; fi
    sub p6_${D}_${A} "DATASET=$D,APPROACH=$A,NOISE=0,OUTROOT=results/phase6_other_datasets,SAVE_EXEMPLARS=1" $T
  done; done
}
case "$1" in phase1|phase2|phase3|phase4|phase5|phase6|ablation) "$1";; *) echo "usage: $0 phase1|phase2|phase3|phase4|phase5|phase6|ablation"; exit 1;; esac
