#!/usr/bin/env python3
"""Generate (and cache) the Boussinesq snapshot data for every split.

Runs the solver once per ``(Re, store_every)`` pair of :mod:`problem` and
writes one ``.npz`` per pair under ``data/``, then a ``data/manifest.json``
with the parameters, shapes and resolution diagnostics of each run.  Cached
runs are skipped, so the script is resumable.

Usage
-----
    python generate_data.py             # all five splits
    python generate_data.py train       # one or more named splits
    python generate_data.py --re 1500   # one Reynolds number, test stride
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import problem as P
from boussinesq import diagnostics

HERE = Path(__file__).resolve().parent


def run_one(Re: float, se: int) -> dict:
    t0 = time.time()
    S, info = P.solve_boussinesq(Re, se)
    wall = time.time() - t0
    d = diagnostics(S[:, ::max(1, S.shape[1] // 20)],
                    dict(Nx=P.NX, Ny=P.NY, Lx=P.LX, Ly=P.LY, Y=info["Y"]))
    rec = dict(
        Re=float(Re), nu=float(P.viscosity(Re)), store_every=int(se),
        n_snap=int(S.shape[1]), d=int(S.shape[0]), wall_s=round(wall, 1),
        finite=bool(np.isfinite(S).all()),
        mean_theta_drift=float(abs(d["mean_theta"][-1] / d["mean_theta"][0] - 1)),
        theta_var_monotone=bool(np.all(np.diff(d["theta_var"]) <= 1e-12)),
        max_tail_theta=float(d["tail_theta"].max()),
        max_tail_w=float(d["tail_w"].max()),
        max_asym=float(d["asym"].max()),
        max_speed=float(d["max_speed"].max()),
        final_y_c=float(d["y_c"][-1]),
    )
    print(f"  Re={Re:8.2f}  se={se:3d}  n={rec['n_snap']:4d}  "
          f"wall={wall:6.1f}s  tail(theta,w)=({rec['max_tail_theta']:.1e},"
          f"{rec['max_tail_w']:.1e})  <theta> drift={rec['mean_theta_drift']:.1e}  "
          f"asym={rec['max_asym']:.1e}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=list(P.SPLITS))
    ap.add_argument("--re", type=float, default=None,
                    help="generate a single Reynolds number (test stride)")
    args = ap.parse_args()

    manifest_path = P.DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["config"] = dict(
        NX=P.NX, NY=P.NY, LX=P.LX, LY=P.LY, Y0=P.Y0, R=P.R, PR=P.PR, DT=P.DT,
        T_END=P.T_END, RE_MIN=P.RE_MIN, RE_MAX=P.RE_MAX,
        N_PARAMS_TRAIN=P.N_PARAMS_TRAIN, EXTRAP_FRAC=P.EXTRAP_FRAC,
        VAL_EXTRAP_FRAC=P.VAL_EXTRAP_FRAC, SEED=P.SEED,
    )
    runs = manifest.setdefault("runs", {})

    if args.re is not None:
        rec = run_one(args.re, P.STORE_EVERY_TEST)
        runs[f"Re{args.re:.6g}_se{P.STORE_EVERY_TEST}"] = rec
    else:
        res = P.all_reynolds()
        for split in args.splits:
            se = P.store_every(split)
            print(f"[{split}]  Re = {res[split]}  store_every = {se}")
            for Re in res[split]:
                rec = run_one(Re, se)
                rec["split"] = split
                runs[f"Re{Re:.6g}_se{se}"] = rec
    P.DATA.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
