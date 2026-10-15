#!/usr/bin/env bash
# Redo only the CPN-S part of the degree study after switching CPN-S to its
# released truncation rule N = N(tol) (previously N was fixed to 50).
#
# For every experiment and degree p in {3, 4, 5}:
#   1. plots/p<p> (fixed-N results) is moved to
#      plots/__old__/__old__10-15-2026_fixedN50/p<p>   (kept for reference)
#   2. plots/p<p> is recreated, seeded with the budget selection and the
#      r-sweep of the archived folder (identical numbers and timings for all
#      methods other than CPN-S), and the cpn / noise / plots / field /
#      report phases are rerun.
#
# Usage (pixi environment, from anywhere in the repo):
#   bash experiments/rerun_cpn_with_N_rule.sh               # all three
#   bash experiments/rerun_cpn_with_N_rule.sh burgers2d     # one experiment
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then EXPS=(lin_adv_diff burgers2d boussinesq); fi
PY=${PY:-"pixi run python"}
ARCH="__old__/__old__10-15-2026_fixedN50"

for exp in "${EXPS[@]}"; do
  cd "$HERE/$exp"
  echo "=============== $exp ==============="
  mkdir -p "plots/$ARCH"
  for p in 3 4 5; do
    if [ -d "plots/p$p" ] && [ ! -d "plots/$ARCH/p$p" ]; then
      mv "plots/p$p" "plots/$ARCH/p$p"
    fi
    if [ ! -d "plots/$ARCH/p$p" ]; then
      echo "no previous plots/p$p to seed from; running the full pipeline"
      $PY run_experiment.py all --tag "p$p" --cpn-p "$p"
    else
      if [ "$p" = 3 ]; then
        $PY run_experiment.py all --tag "p$p" --cpn-p "$p" --seed-from "$ARCH/p$p"
      else
        $PY run_experiment.py all --tag "p$p" --cpn-p "$p" --seed-from "$ARCH/p$p" --field-from p3
      fi
    fi
  done
done
echo "all done"
