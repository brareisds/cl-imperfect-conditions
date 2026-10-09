#!/bin/bash
# Usage (repo root): [DRY=1] bash slurm/submit_phases.sh phase1|phase2|phase3|ablation
# phase3 needs FNORM_ORGANAMNIST and FNORM_BLOODMNIST.
# DEP=<jobid>: jobs start only if that job finishes with exit code 0.
sub() {  # $1=job name, $2=--export string
  cmd="sbatch ${DEP:+--dependency=afterok:$DEP} --job-name=$1 --export=ALL,$2 slurm/run_phase.sh"
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
case "$1" in phase1|phase2|phase3|ablation) "$1";; *) echo "usage: $0 phase1|phase2|phase3|ablation"; exit 1;; esac
