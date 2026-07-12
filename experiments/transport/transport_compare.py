"""Linear-transport (advecting pulse) comparison experiment.

Runs POD, Greedy-QM, Geelen-QM, Greedy-Fixed-OPM (p=2), Greedy-OPM (p=2), and
Greedy-OPM (p=[2,3]) on an advecting Gaussian pulse.  Snapshots are split by a
fixed-seed random permutation into 50% train / 25% validation / 25% test (see
experiments/exp_split.py); the validation set tunes the Greedy-OPM sparsity
parameter gamma_sp and the test set is held out for the reported errors.  For
the space-time field images each split's columns are re-sorted into time order.

The r-sweep is resumable (checkpoint JSON); rerun until it prints DONE.  Once
the sweep is complete it writes the error-vs-r curves, the gamma_sp study
(validation error / feature count vs gamma_sp), and the field reconstruction /
absolute-error panels (space-time images) for train and test.

Phases:  sweep | curves | active | study | field | all.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                          # experiments/
from exp_methods import run_sweep, build_model, METHOD_NAMES   # noqa: E402
from exp_plots import (error_vs_r, field_panels,               # noqa: E402
                       field_panels_horizontal, gamma_sp_study,
                       active_features_table, active_features_bar)
from exp_split import split_indices                            # noqa: E402

PLOTS = _HERE / "plots"
CKPT = PLOTS / "transport_compare_results.json"

# ---- problem + experiment configuration ------------------------------------
NX, NT = 1024, 1200
PULSE_W, X0, SPEED, TFINAL = 1.0e-3, 0.1, 5.0, 0.2
R_MAX = 10
R_FIELD = 8
RS = list(range(1, R_MAX + 1))
CFG = dict(
    s_pool=100,
    gamma_qm=1e-13,
    gamma_opm=1e-13,
    gamma_geelen=1e-13,
    geelen_rel=0.20,
    geelen_iters=400,
    tau=1e-6,
    gamma_sp_points=24,   # gamma_sp grid resolution for the validation sweep
    gamma_sp_tol=0.01,     # accept sparsest gamma_sp within 2% of best val error
)
TIME_BUDGET = 100.0


def generate():
    """Generate the advecting Gaussian pulse data matrix S (shape d=NX by n=NT)."""
    x = np.linspace(0, 1, NX)
    t = np.linspace(0, TFINAL, NT)
    S = np.empty((NX, NT))
    for j, tj in enumerate(t):
        S[:, j] = np.exp(-((x - (X0 + SPEED * tj)) ** 2) / PULSE_W) \
            / np.sqrt(PULSE_W * np.pi)
    return S, x, t


def main(phase="all"):
    S, x, t = generate()
    itr, iva, ite = split_indices(S.shape[1])
    S_train, S_val, S_test = S[:, itr], S[:, iva], S[:, ite]
    print(f"[transport] d={S.shape[0]} n={S.shape[1]} "
          f"(train {S_train.shape[1]} / val {S_val.shape[1]} / "
          f"test {S_test.shape[1]})  phase={phase}")

    if phase in ("all", "sweep"):
        results, done = run_sweep(jnp.asarray(S_train), jnp.asarray(S_val),
                                  jnp.asarray(S_test), RS, CFG, str(CKPT),
                                  time_budget=TIME_BUDGET)
        if not done:
            print("[transport] sweep incomplete -- rerun to continue")
            return
        if phase == "sweep":
            print("SWEEP DONE")
            return

    import json
    results = json.load(open(CKPT))

    if phase in ("all", "curves"):
        error_vs_r(results, RS, METHOD_NAMES,
                   str(PLOTS / "transport_error_vs_r"),
                   "Advecting pulse — reconstruction error vs reduced dimension")

    if phase in ("all", "study"):
        gamma_sp_study(results, RS, METHOD_NAMES,
                       str(PLOTS / "transport_gamma_sp_study"),
                       "Advecting pulse — gamma_sp validation study")

    if phase in ("all", "active"):
        print("\n[transport] number of active nonlinear features:")
        active_features_table(results, RS, METHOD_NAMES)
        active_features_bar(results, RS, METHOD_NAMES,
                            str(PLOTS / "transport_active_features"),
                            "Advecting pulse — active nonlinear features vs reduced dimension")

    if phase in ("all", "field"):
        # Models are fit on the (shuffled) train columns; for the space-time
        # images each split's columns are re-sorted into time order so the
        # random subset renders as a coherent picture across the full window.
        disp = {"train": np.sort(itr), "test": np.sort(ite)}
        Sd = {sp: S[:, cols] for sp, cols in disp.items()}
        ext = {sp: [t[cols][0], t[cols][-1], x[0], x[-1]]
               for sp, cols in disp.items()}
        recon = {"train": {}, "test": {}}
        rel = {"train": {}, "test": {}}
        for nm in METHOD_NAMES:
            gsp = (results.get(f"{nm}@r={R_FIELD}") or {}).get("gamma_sp")
            model = build_model(nm, jnp.asarray(S_train), R_FIELD, CFG,
                                gamma_sp=gsp)
            for split in ("train", "test"):
                Smat = jnp.asarray(Sd[split])
                recon[split][nm] = np.asarray(model.reconstruct(Smat))
                rel[split][nm] = model.reconstruction_error(Smat, relative=True)
        for split, tag in [("train", "Training"), ("test", "Test")]:
            title = f"Advecting pulse — space-time reconstruction ({tag}, r={R_FIELD})"
            field_panels(
                Sd[split], recon[split], METHOD_NAMES,
                str(PLOTS / f"transport_field_{split}"), title,
                cmap="viridis", extent=ext[split], xlabel="time $t$",
                ylabel="space $x$", aspect="auto", rel_errs=rel[split])
            field_panels_horizontal(
                Sd[split], recon[split], METHOD_NAMES,
                str(PLOTS / f"transport_field_{split}_horizontal"), title,
                cmap="viridis", extent=ext[split], xlabel="time $t$",
                ylabel="space $x$", aspect="auto", rel_errs=rel[split])
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[transport] elapsed {time.time()-t0:.1f}s")
