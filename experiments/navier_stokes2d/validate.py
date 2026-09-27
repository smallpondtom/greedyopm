#!/usr/bin/env python3
"""Validate the SU2 angle sweep against Kurtulus (2015): steady/unsteady regime,
Strouhal number of the lift signal, mean lift and drag.

Reference: D. F. Kurtulus, "On the unsteady behavior of the flow around NACA 0012
airfoil with steady external conditions at Re = 1000", Int. J. Micro Air
Vehicles 7(3), 301-326, 2015.  Body-fitted finite-volume (Fluent), far field at
15c, 300 nodes around the airfoil.  Findings used here: the flow is steady
below alpha = 8 deg and periodic above; St = f c / U_inf of the lift signal is
0.876 at 8 deg, 0.89 at 9 deg, then decreases (about 0.60 at 15 deg, 0.50 at
20 deg, 0.45 at 23 deg, 0.23 at 25 deg); mean C_d is about 0.12 at 0 deg and
0.69 at 26 deg; C_l/C_d peaks at 2.55 near 11 deg.

Writes ``plots/validation.md`` and ``plots/validation.png``.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
T0 = 30.0

# figure readings from Kurtulus (2015); the 8 and 9 deg values are stated in the text
KURTULUS_ST = {8: 0.876, 9: 0.89, 15: 0.60, 20: 0.50, 23: 0.45, 25: 0.23}
KURTULUS_STEADY_BELOW = 8


def peak_frequency(sig, dt):
    sig = sig - sig.mean()
    if sig.std() < 1e-7:
        return 0.0
    w = np.hanning(len(sig))
    F = np.abs(np.fft.rfft(sig * w)); fr = np.fft.rfftfreq(len(sig), d=dt); F[0] = 0
    i = int(np.argmax(F))
    if 0 < i < len(F) - 1:
        a, b, c = np.log(F[i - 1] + 1e-30), np.log(F[i] + 1e-30), np.log(F[i + 1] + 1e-30)
        den = a - 2 * b + c
        di = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(fr[i] + di * (fr[1] - fr[0]))
    return float(fr[i])


def summarize(path):
    d = np.load(path)
    t, cl, cd = d["force_t"], d["Cl"], d["Cd"]
    m = t >= T0
    dt = float(t[1] - t[0])
    return dict(alpha=float(d["alpha_deg"]), Cl=float(cl[m].mean()), Cl_rms=float(cl[m].std()),
                Cd=float(cd[m].mean()), Cd_rms=float(cd[m].std()), St=peak_frequency(cl[m], dt),
                unsteady=bool(cl[m].std() > 1e-3), wall_h=float(d["wall_hours"]))


def main():
    files = sorted(glob.glob(os.path.join(DATA, "naca_a*.npz")))
    rows = [summarize(f) for f in files]
    rows.sort(key=lambda r: r["alpha"])
    os.makedirs(PLOTS, exist_ok=True)
    lines = ["# SU2 sweep vs Kurtulus (2015)", "",
             f"Window t >= {T0:g}. St from the lift spectrum (Hann window, parabolic peak).", "",
             "| alpha | regime (ours) | regime (Kurtulus) | St (ours) | St (Kurtulus) | mean Cl | Cl rms | mean Cd | Cd rms | wall h |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        a = int(round(r["alpha"]))
        ref_reg = "steady" if a < KURTULUS_STEADY_BELOW else "periodic"
        ref_st = KURTULUS_ST.get(a)
        lines.append(f"| {a} | {'periodic' if r['unsteady'] else 'steady'} | {ref_reg} | "
                     f"{r['St']:.3f} | {ref_st if ref_st is not None else '-'} | {r['Cl']:.4f} | "
                     f"{r['Cl_rms']:.2e} | {r['Cd']:.4f} | {r['Cd_rms']:.2e} | {r['wall_h']:.2f} |")
    open(os.path.join(PLOTS, "validation.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    al = [r["alpha"] for r in rows]
    axes[0].plot(al, [r["St"] if r["unsteady"] else np.nan for r in rows], "o-", label="SU2 (this work)")
    axes[0].plot(list(KURTULUS_ST), list(KURTULUS_ST.values()), "s--", color="C3", label="Kurtulus (2015)")
    axes[0].set_xlabel(r"$\alpha$ (deg)"); axes[0].set_ylabel("St"); axes[0].legend(frameon=False)
    axes[1].errorbar(al, [r["Cl"] for r in rows], yerr=[r["Cl_rms"] for r in rows], fmt="o-")
    axes[1].set_xlabel(r"$\alpha$ (deg)"); axes[1].set_ylabel(r"$\overline{C_l}$ ($\pm$ rms)")
    axes[2].errorbar(al, [r["Cd"] for r in rows], yerr=[r["Cd_rms"] for r in rows], fmt="o-")
    axes[2].set_xlabel(r"$\alpha$ (deg)"); axes[2].set_ylabel(r"$\overline{C_d}$ ($\pm$ rms)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "validation.png"), dpi=150)
    print("wrote plots/validation.md and validation.png")


if __name__ == "__main__":
    main()
