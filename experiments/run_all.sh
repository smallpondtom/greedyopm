#!/usr/bin/env bash
# Reproduce every numerical result of the paper from scratch.
#
# For each experiment (lin_adv_diff, burgers2d, boussinesq) the script
#
#   1. archives an existing plots/p3, plots/p4, plots/p5 to
#      plots/__old__/__old__<date>_<time>/ (nothing is ever deleted);
#   2. runs the complete pipeline with CPN-S at degree p = 3
#      (budget selection -> r-sweep -> CPN-S tolerance sweep -> noise study
#       -> field panels -> figures, report, timing table);
#   3. runs CPN-S at p = 4 and p = 5 on top of the p = 3 results
#      (--seed-from p3 reuses the budget selection and the r-sweep of every
#       method other than CPN-S, so those numbers are identical in the three
#       folders; --field-from p3 reuses their fitted n = 10 models for the
#       field panels);
#   4. copies the figures used in the paper to $PAPER_FIGS when that
#      directory exists.
#
# Consistency of the field panels with the error-vs-n curves: the field
# phase fits every method once at n = 10, writes that fit's errors back into
# the sweep entry <method>@r=10 and caches the reconstructions
# (<exp>_field_models.npz).  The "plots" phase is therefore run again AFTER
# the field phase, so that the error-vs-n figure, the report and the timing
# table show the same n = 10 models as the panels.
#
# Usage (from anywhere inside the repo, pixi environment):
#   bash experiments/run_all.sh                 # all three experiments
#   bash experiments/run_all.sh burgers2d       # one experiment
#   PY="python" bash experiments/run_all.sh     # other interpreter
#
# The CPN-S tolerance sweeps dominate the runtime.  Every phase is
# checkpointed in plots/p<k>/<exp>_results.json; the script always starts
# from a clean plots/p<k> (the previous one is archived), so to resume an
# interrupted run move the archived folder back to plots/p<k> and call
# run_experiment.py directly with the same arguments as below.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY=${PY:-"pixi run python"}
PAPER_FIGS=${PAPER_FIGS:-"$ROOT/paper/69a9b9faac5cd168cde518ef/figures"}
STAMP="$(date +%Y-%m-%d_%H%M)"

EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then EXPS=(lin_adv_diff burgers2d boussinesq); fi

# figure folder of each experiment inside the paper
paper_dir() {
  case "$1" in
    lin_adv_diff) echo "lin_adv_diff" ;;
    burgers2d)    echo "burgers" ;;
    boussinesq)   echo "boussinesq" ;;
    *)            echo "$1" ;;
  esac
}

# field panel shown in the paper (the test parameter outside the training range)
paper_field() {
  case "$1" in
    boussinesq) echo "boussinesq_speed_test_extrap" ;;
    *)          echo "${1}_field_test_extrap" ;;
  esac
}

for exp in "${EXPS[@]}"; do
  cd "$HERE/$exp"
  echo
  echo "=============================== $exp ==============================="

  # -- 1. archive previous results ------------------------------------------
  for p in 3 4 5; do
    if [ -d "plots/p$p" ]; then
      mkdir -p "plots/__old__/__old__$STAMP"
      mv "plots/p$p" "plots/__old__/__old__$STAMP/p$p"
      echo "[$exp] archived plots/p$p -> plots/__old__/__old__$STAMP/p$p"
    fi
  done

  # -- 2. full pipeline with CPN-S (p = 3) -----------------------------------
  $PY run_experiment.py all   --tag p3 --cpn-p 3
  $PY run_experiment.py plots --tag p3 --cpn-p 3     # redraw with the field-phase n = 10 fits

  # -- 3. CPN-S (p = 4) and (p = 5) on the same fits of the other methods ----
  for p in 4 5; do
    $PY run_experiment.py all   --tag "p$p" --cpn-p "$p" --seed-from p3 --field-from p3
    $PY run_experiment.py plots --tag "p$p" --cpn-p "$p"
  done

  # -- 4. figures used in the paper -----------------------------------------
  dst="$PAPER_FIGS/$(paper_dir "$exp")"
  if [ -d "$dst" ]; then
    cp "plots/p3/${exp}_error_vs_r.pdf"        "$dst/${exp}_error_vs_r_p3.pdf"
    cp "plots/p5/${exp}_error_vs_r.pdf"        "$dst/${exp}_error_vs_r_p5.pdf"
    cp "plots/p3/${exp}_active_features.pdf"   "$dst/"
    cp "plots/p3/${exp}_budget_study.pdf"      "$dst/"
    cp "plots/p3/$(paper_field "$exp").pdf"    "$dst/"
    echo "[$exp] paper figures updated in $dst"
  else
    echo "[$exp] $dst not found; paper figures not copied"
  fi
done

echo
echo "all done ($STAMP)"
