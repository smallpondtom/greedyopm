#!/usr/bin/env python3
"""Verification of the Boussinesq solver, independent of the cached data.

Five checks, each printed with a pass/fail against a tolerance and written
to ``plots/validation.json`` (numbers) and ``plots/validation.png`` (curves):

1. **Invariants and energy budget** (``128 x 192``, every step stored).
   The mean temperature is conserved to round-off (advection is a divergence
   and the ``k = 0`` mode of a dealiased product is exact), the thermal
   variance decreases monotonically, and the kinetic energy obeys

       KE(T) - KE(0) = int_0^T ( <theta v> - nu <w^2> ) dt

   with the right-hand side integrated by Simpson's rule on the stored
   trajectory.  The 2/3 rule makes the advective term energy-conserving to
   round-off, so the residual is the time-integration error only.
2. **Mirror symmetry** about ``x = Lx/2`` is preserved to round-off (the
   initial condition is symmetric; asymmetry can only come from round-off
   amplified by an instability, which flags a too-large ``Re``).
3. **Time-step convergence** on the production grid at the largest
   Reynolds number used: ``dt`` against ``dt/2``, relative Frobenius
   difference of the final state (RK4: expect ``~ 1e-7`` or smaller).
4. **Grid convergence** at the largest Reynolds number used: the production
   grid against one twice as fine, compared on the coarse points at several
   times.  This is the check that the *extrapolation* test is resolved.
5. **Rise law**: the theta-weighted centroid height of a 2D thermal grows
   like ``t^{1/2}`` once the buoyant vortex pair has formed (buoyancy per
   unit length is conserved and there is no other scale).  Reported, not
   thresholded (viscosity and the finite box both bend it).

Usage
-----
    python validate_solver.py            # everything (grid check ~ minutes)
    python validate_solver.py --quick    # skip the fine-grid run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

import problem as P                                 # noqa: E402
from boussinesq import generate_boussinesq, diagnostics, split_fields  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"

TOL = dict(mean_theta=1e-12, budget=1e-6, asym=1e-10, dt=1e-6, grid=3e-3)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _simpson(y, h):
    """Composite Simpson on a uniform grid (odd number of points required)."""
    n = y.shape[0]
    if n % 2 == 0:
        y = y[:-1]
        n -= 1
    return h / 3.0 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def check_invariants(Re: float):
    Nx, Ny = P.NX // 2, P.NY // 2
    dt, t_end = 2 * P.DT, 4.0
    t0 = time.time()
    S, info = generate_boussinesq(Nx=Nx, Ny=Ny, Lx=P.LX, Ly=P.LY, Re=Re,
                                  Pr=P.PR, dt=dt, t_end=t_end, store_every=1,
                                  y0=P.Y0, R=P.R)
    d = diagnostics(S, info)
    rhs = d["buoyancy_flux"] - info["nu"] * d["enstrophy"]
    dKE = d["KE"][-1] - d["KE"][0]
    integral = _simpson(rhs, dt)
    res = dict(
        Re=Re, grid=[Nx, Ny], t_end=t_end, wall_s=round(time.time() - t0, 1),
        mean_theta_drift=float(np.abs(d["mean_theta"] / d["mean_theta"][0] - 1).max()),
        theta_var_monotone=bool(np.all(np.diff(d["theta_var"]) <= 0)),
        KE_final=float(d["KE"][-1]),
        budget_residual=float(abs(dKE - integral) / abs(dKE)),
        max_asym=float(d["asym"].max()),
    )
    res["pass_mean_theta"] = res["mean_theta_drift"] < TOL["mean_theta"]
    res["pass_budget"] = res["budget_residual"] < TOL["budget"]
    res["pass_asym"] = res["max_asym"] < TOL["asym"]
    return res, d, info["tspan"]


def check_timestep(Re: float):
    t0 = time.time()
    S1, _ = generate_boussinesq(Nx=P.NX, Ny=P.NY, Lx=P.LX, Ly=P.LY, Re=Re,
                                Pr=P.PR, dt=P.DT, t_end=P.T_END,
                                store_every=int(round(P.T_END / P.DT)),
                                y0=P.Y0, R=P.R)
    S2, _ = generate_boussinesq(Nx=P.NX, Ny=P.NY, Lx=P.LX, Ly=P.LY, Re=Re,
                                Pr=P.PR, dt=0.5 * P.DT, t_end=P.T_END,
                                store_every=int(round(P.T_END / (0.5 * P.DT))),
                                y0=P.Y0, R=P.R)
    err = _rel(S1[:, -1], S2[:, -1])
    return dict(Re=Re, dt=P.DT, rel_diff_final=err, wall_s=round(time.time() - t0, 1),
                pass_dt=err < TOL["dt"])


def check_grid(Re: float, n_times: int = 5):
    t0 = time.time()
    se_c = int(round(P.T_END / P.DT)) // n_times
    Sc, ic = generate_boussinesq(Nx=P.NX, Ny=P.NY, Lx=P.LX, Ly=P.LY, Re=Re,
                                 Pr=P.PR, dt=P.DT, t_end=P.T_END,
                                 store_every=se_c, y0=P.Y0, R=P.R)
    Sf, inf = generate_boussinesq(Nx=2 * P.NX, Ny=2 * P.NY, Lx=P.LX, Ly=P.LY,
                                  Re=Re, Pr=P.PR, dt=0.5 * P.DT, t_end=P.T_END,
                                  store_every=2 * se_c, y0=P.Y0, R=P.R)
    errs = []
    for j in range(1, Sc.shape[1]):
        uc, vc, tc = split_fields(Sc[:, j], P.NX, P.NY)
        uf, vf, tf = split_fields(Sf[:, j], 2 * P.NX, 2 * P.NY)
        sub = (slice(None, None, 2), slice(None, None, 2))
        errs.append(dict(t=float(ic["tspan"][j]),
                         theta=_rel(tc, tf[sub]),
                         u=_rel(np.stack([uc, vc]), np.stack([uf[sub], vf[sub]]))))
    worst = max(max(e["theta"], e["u"]) for e in errs)
    tails = diagnostics(Sc, dict(Nx=P.NX, Ny=P.NY, Lx=P.LX, Ly=P.LY, Y=ic["Y"]))
    return dict(Re=Re, coarse=[P.NX, P.NY], fine=[2 * P.NX, 2 * P.NY],
                errors=errs, worst=worst, wall_s=round(time.time() - t0, 1),
                max_tail_theta=float(tails["tail_theta"].max()),
                max_tail_w=float(tails["tail_w"].max()),
                pass_grid=worst < TOL["grid"])


def rise_law(d, t, t_min: float = 5.0):
    m = t >= t_min
    h = d["y_c"][m] - P.Y0
    p = np.polyfit(np.log(t[m]), np.log(h), 1)
    return float(p[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the grid check")
    ap.add_argument("--re", type=float, default=None,
                    help="Reynolds number for the checks (default: test_extrap)")
    args = ap.parse_args()
    PLOTS.mkdir(exist_ok=True)
    Re = args.re if args.re is not None else P.test_extrap_reynolds()
    print(f"validating at Re = {Re:g} (largest Reynolds number used)")

    out = {}
    print("[1,2] invariants, energy budget, symmetry ...", flush=True)
    out["invariants"], d_inv, t_inv = check_invariants(Re)
    for k in ("mean_theta_drift", "theta_var_monotone", "budget_residual",
              "max_asym"):
        print(f"      {k:20s} {out['invariants'][k]}")

    print("[3]   time-step convergence ...", flush=True)
    out["timestep"] = check_timestep(Re)
    print(f"      rel diff (dt vs dt/2) {out['timestep']['rel_diff_final']:.2e}")

    if not args.quick:
        print("[4]   grid convergence (this runs the fine grid, minutes) ...", flush=True)
        out["grid"] = check_grid(Re)
        for e in out["grid"]["errors"]:
            print(f"      t={e['t']:5.1f}  theta {e['theta']:.2e}  u {e['u']:.2e}")
        print(f"      tails on production grid: theta {out['grid']['max_tail_theta']:.1e}"
              f"  w {out['grid']['max_tail_w']:.1e}")

    print("[5]   rise law ...", flush=True)
    S, info = P.solve_boussinesq(Re, P.STORE_EVERY_TEST)
    dd = diagnostics(S[:, ::4], dict(Nx=P.NX, Ny=P.NY, Lx=P.LX, Ly=P.LY,
                                     Y=info["Y"]))
    tt = info["tspan"][::4]
    out["rise_exponent"] = rise_law(dd, tt)
    print(f"      centroid height ~ t^{out['rise_exponent']:.2f} on t >= 5 "
          f"(2D thermal similarity: 0.5)")

    passed = all(v for k, v in
                 {**out["invariants"], **out["timestep"], **out.get("grid", {})}.items()
                 if k.startswith("pass_"))
    out["all_passed"] = bool(passed)
    (PLOTS / "validation.json").write_text(json.dumps(out, indent=2))
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED",
          f"-> {PLOTS / 'validation.json'}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ax = axes[0]
    rhs = d_inv["buoyancy_flux"] - (1.0 / Re) * d_inv["enstrophy"]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rhs[1:] + rhs[:-1])) * (t_inv[1] - t_inv[0])])
    ax.plot(t_inv, d_inv["KE"] - d_inv["KE"][0], label="KE(t) - KE(0)")
    ax.plot(t_inv, cum, "--", label=r"$\int (\langle\theta v\rangle - \nu\langle\omega^2\rangle)$")
    ax.set_title("energy budget"); ax.set_xlabel("$t$"); ax.legend(fontsize=8)
    ax = axes[1]
    if "grid" in out:
        e = out["grid"]["errors"]
        ax.semilogy([x["t"] for x in e], [x["theta"] for x in e], "o-", label=r"$\theta$")
        ax.semilogy([x["t"] for x in e], [x["u"] for x in e], "s-", label=r"$(u, v)$")
        ax.set_title(f"grid {P.NX}x{P.NY} vs {2*P.NX}x{2*P.NY}, Re={Re:g}")
        ax.set_xlabel("$t$"); ax.legend(fontsize=8)
    else:
        ax.set_title("grid check skipped (--quick)")
    ax = axes[2]
    m = tt > 0
    ax.loglog(tt[m], dd["y_c"][m] - P.Y0, label="centroid rise")
    ax.loglog(tt[m], (dd["y_c"][-1] - P.Y0) * np.sqrt(tt[m] / tt[-1]), "k--",
              lw=1, label=r"$\propto t^{1/2}$")
    ax.set_title(f"rise law, fitted exponent {out['rise_exponent']:.2f}")
    ax.set_xlabel("$t$"); ax.legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "validation.png", dpi=140)
    print(f"figure -> {PLOTS / 'validation.png'}")


if __name__ == "__main__":
    main()
