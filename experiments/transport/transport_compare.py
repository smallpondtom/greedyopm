"""Linear-transport (advecting pulse) comparison experiment.

Runs POD, Greedy-QM, Geelen-QM, Greedy-Fixed-OPM (p=2), Greedy-OPM (p=2), and
Greedy-OPM (p=[2,3]) on an advecting Gaussian pulse, with even snapshots for
training and odd snapshots for testing.

The r-sweep is resumable (checkpoint JSON); rerun until it prints DONE.  Once
the sweep is complete it writes the error-vs-r curves and the field
reconstruction / absolute-error panels (space-time images) for train and test.
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
                       active_features_table, active_features_bar)

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
    tau=1e-6
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
    S_train, S_test = S[:, ::2], S[:, 1::2]
    t_train, t_test = t[::2], t[1::2]
    print(f"[transport] d={S.shape[0]} n={S.shape[1]} "
          f"(train {S_train.shape[1]} / test {S_test.shape[1]})  phase={phase}")

    if phase in ("all", "sweep"):
        results, done = run_sweep(jnp.asarray(S_train), jnp.asarray(S_test),
                                  RS, CFG, str(CKPT), time_budget=TIME_BUDGET)
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

    if phase in ("all", "active"):
        print("\n[transport] number of active nonlinear features:")
        active_features_table(results, RS, METHOD_NAMES)
        active_features_bar(results, RS, METHOD_NAMES,
                            str(PLOTS / "transport_active_features"),
                            "Advecting pulse — active nonlinear features vs reduced dimension")

    if phase in ("all", "field"):
        # Build each model ONCE at r=R_FIELD and reconstruct both splits.
        ext = {"train": [t_train[0], t_train[-1], x[0], x[-1]],
               "test": [t_test[0], t_test[-1], x[0], x[-1]]}
        recon = {"train": {}, "test": {}}
        rel = {"train": {}, "test": {}}
        for nm in METHOD_NAMES:
            model = build_model(nm, jnp.asarray(S_train), R_FIELD, CFG)
            for split, Smat in [("train", S_train), ("test", S_test)]:
                recon[split][nm] = np.asarray(model.reconstruct(jnp.asarray(Smat)))
                rel[split][nm] = model.reconstruction_error(jnp.asarray(Smat),
                                                            relative=True)
        for split, Smat, tag in [("train", S_train, "Training"),
                                 ("test", S_test, "Test")]:
            field_panels(
                np.asarray(Smat), recon[split], METHOD_NAMES,
                str(PLOTS / f"transport_field_{split}"),
                f"Advecting pulse — space-time reconstruction ({tag}, r={R_FIELD})",
                cmap="viridis", extent=ext[split], xlabel="time $t$",
                ylabel="space $x$", aspect="auto", rel_errs=rel[split])
    print("DONE")


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print(f"[transport] elapsed {time.time()-t0:.1f}s")
