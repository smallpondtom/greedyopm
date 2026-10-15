#!/usr/bin/env bash
# Rerun the three experiments with
#   * Plain Greedy (p=3) on all degree-2 and degree-3 monomials,
#   * CPN-S on its released hyperbolic-cross index set at degrees 3, 4, 5,
# and write each degree's results, figures and timing tables to
# experiments/<exp>/plots/p{3,4,5}/.
#
# The p4 and p5 runs reuse the budget selection and the r-sweep of p3
# (--seed-from p3), so every method other than CPN-S has identical numbers in
# the three folders; CPN-S, the noise study, the field panels, the report and
# the timing table are recomputed per degree.
#
# Usage (from anywhere inside the repo, pixi environment):
#   bash experiments/run_cpn_degree_study.sh               # all three
#   bash experiments/run_cpn_degree_study.sh burgers2d     # one experiment
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then EXPS=(lin_adv_diff burgers2d boussinesq); fi
PY=${PY:-"pixi run python"}

for exp in "${EXPS[@]}"; do
  cd "$HERE/$exp"
  echo "=============== $exp ==============="
  $PY run_experiment.py all --tag p3 --cpn-p 3
  $PY run_experiment.py all --tag p4 --cpn-p 4 --seed-from p3 --field-from p3
  $PY run_experiment.py all --tag p5 --cpn-p 5 --seed-from p3 --field-from p3
done
echo "all done"
