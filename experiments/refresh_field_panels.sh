#!/usr/bin/env bash
# Redraw the field-reconstruction panels of the degree study so that every
# panel shows the SAME fitted model as the error-vs-n curves:
#   * p3: refit the n = R_FIELD models once, write their errors back into the
#         sweep entries, cache the reconstructions, draw the panels;
#   * p4, p5: reuse p3's models for every method but CPN-S (--field-from p3),
#         refit only CPN-S at its own tolerance, draw the panels;
#   * then redraw the other figures, the report and the timing table of each
#     folder from the updated checkpoints.
# Panels carry no relative-error labels any more.
#
# Usage (pixi environment, from anywhere in the repo):
#   bash experiments/refresh_field_panels.sh               # all three
#   bash experiments/refresh_field_panels.sh boussinesq    # one experiment
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then EXPS=(lin_adv_diff burgers2d boussinesq); fi
PY=${PY:-"pixi run python"}

for exp in "${EXPS[@]}"; do
  cd "$HERE/$exp"
  echo "=============== $exp ==============="
  rm -f plots/p3/${exp}_field_models.npz plots/p4/${exp}_field_models.npz plots/p5/${exp}_field_models.npz
  $PY run_experiment.py field --tag p3 --cpn-p 3
  $PY run_experiment.py plots --tag p3 --cpn-p 3
  for p in 4 5; do
    $PY run_experiment.py field --tag "p$p" --cpn-p "$p" --field-from p3
    $PY run_experiment.py plots --tag "p$p" --cpn-p "$p"
  done
done
echo "all done"
