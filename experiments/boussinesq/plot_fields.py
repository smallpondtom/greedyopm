#!/usr/bin/env python3
"""Plot the cached Boussinesq snapshots: field panels, diagnostics, GIFs.

Everything is drawn from the *stored* snapshot matrices under ``data/`` (so
what you see is the data the experiment consumes, not a fresh run).

Figures (under ``plots/``)
--------------------------
* ``fields_Re<Re>.png``   -- temperature, vorticity and speed at six times
  for one Reynolds number (default: the two extreme training numbers and
  the extrapolation test).
* ``compare_Re.png``      -- temperature at the final time for every cached
  Reynolds number side by side (what the parameter changes).
* ``diagnostics.png``     -- rise curve, kinetic energy, thermal variance
  and spectral tails against time for every cached Reynolds number.
* ``anim_Re<Re>.gif``     -- animation of temperature and vorticity
  (``--gif``).

Usage
-----
    python plot_fields.py                 # panels for min/max train + extrap
    python plot_fields.py --re 600 1500   # panels for these Reynolds numbers
    python plot_fields.py --gif --re 900  # animation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import problem as P                                          # noqa: E402
from boussinesq import diagnostics, vorticity                # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
EXTENT = [0.0, P.LX, 0.0, P.LY]


def _load(Re: float):
    """Cached snapshots for ``Re`` at whichever stride exists (finest first)."""
    for se in sorted({P.STORE_EVERY_TEST, P.STORE_EVERY_TRAIN, P.STORE_EVERY_VAL}):
        f = P.DATA / f"boussinesq_N{P.NX}x{P.NY}_Re{Re:.6g}_se{se}.npz"
        if f.exists():
            z = np.load(f)
            return z["S"], z["tspan"]
    raise FileNotFoundError(f"no cached run for Re={Re:g}; run generate_data.py")


def _cached_reynolds():
    res = []
    for f in sorted(P.DATA.glob(f"boussinesq_N{P.NX}x{P.NY}_Re*_se*.npz")):
        res.append(float(f.stem.split("_Re")[1].split("_se")[0]))
    return sorted(set(res))


def _panel(ax, F, cmap, vmin, vmax):
    im = ax.imshow(F.T, origin="lower", extent=EXTENT, cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    return im


def plot_fields(Re: float, n_times: int = 6):
    S, t = _load(Re)
    idx = np.linspace(0, S.shape[1] - 1, n_times).round().astype(int)
    idx[0] = max(idx[0], np.searchsorted(t, 1.0))   # skip the trivial t=0 frame
    wmax = max(np.abs(vorticity(*P.fields(S[:, i])[:2], P.LX, P.LY)).max()
               for i in idx) * 0.8
    smax = max(P.speed(S[:, i]).max() for i in idx)

    fig, axes = plt.subplots(3, n_times, figsize=(1.55 * n_times, 7.2))
    for j, i in enumerate(idx):
        u, v, th = P.fields(S[:, i])
        w = vorticity(u, v, P.LX, P.LY)
        _panel(axes[0, j], th, "inferno", 0.0, 1.0)
        _panel(axes[1, j], w, "RdBu_r", -wmax, wmax)
        _panel(axes[2, j], np.sqrt(u ** 2 + v ** 2), "viridis", 0.0, smax)
        axes[0, j].set_title(f"$t = {t[i]:.1f}$", fontsize=10)
    for ax, lab in zip(axes[:, 0], (r"$\theta$", r"$\omega$", r"$|\mathbf{u}|$")):
        ax.set_ylabel(lab, fontsize=12)
    fig.suptitle(rf"Rising thermal, $\mathrm{{Re}} = {Re:g}$", fontsize=12)
    fig.tight_layout()
    out = PLOTS / f"fields_Re{Re:g}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def plot_compare(res, time: float = P.T_END):
    fig, axes = plt.subplots(1, len(res), figsize=(1.7 * len(res), 3.0))
    for ax, Re in zip(np.atleast_1d(axes), res):
        S, t = _load(Re)
        i = int(np.argmin(np.abs(t - time)))
        _panel(ax, P.temperature(S[:, i]), "inferno", 0.0, 1.0)
        ax.set_title(f"Re = {Re:g}", fontsize=9)
    fig.suptitle(rf"$\theta$ at $t = {time:g}$ across the parameter range", fontsize=11)
    fig.tight_layout()
    out = PLOTS / "compare_Re.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"wrote {out}")


def plot_diagnostics(res):
    info = dict(Nx=P.NX, Ny=P.NY, Lx=P.LX, Ly=P.LY,
                Y=np.load(next(P.DATA.glob("boussinesq_*.npz")))["Y"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    cmap = plt.get_cmap("viridis")
    for k, Re in enumerate(res):
        S, t = _load(Re)
        step = max(1, S.shape[1] // 50)
        d = diagnostics(S[:, ::step], info)
        tt = t[::step]
        c = cmap(k / max(1, len(res) - 1))
        axes[0, 0].plot(tt, d["y_c"] - P.Y0, color=c, label=f"Re={Re:g}")
        axes[0, 1].plot(tt, d["KE"], color=c)
        axes[0, 2].plot(tt, d["theta_var"] / d["theta_var"][0], color=c)
        axes[1, 0].semilogy(tt[1:], d["tail_theta"][1:] + 1e-20, color=c)
        axes[1, 1].semilogy(tt[1:], d["tail_w"][1:] + 1e-20, color=c)
        axes[1, 2].semilogy(tt, np.abs(d["mean_theta"] / d["mean_theta"][0] - 1) + 1e-17,
                            color=c)
    # 2D thermal similarity law: height ~ t^{1/2} at late times
    tt = np.linspace(3.0, P.T_END, 50)
    S, t = _load(res[-1])
    d = diagnostics(S[:, ::max(1, S.shape[1] // 50)], info)
    ref = (d["y_c"][-1] - P.Y0) * np.sqrt(tt / t[-1])
    axes[0, 0].plot(tt, ref, "k--", lw=1, label=r"$\propto t^{1/2}$")
    axes[0, 0].set_title(r"centroid rise $y_c - y_0$"); axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("kinetic energy")
    axes[0, 2].set_title(r"$\|\theta\|^2 / \|\theta_0\|^2$ (must decrease)")
    axes[1, 0].set_title(r"spectral tail of $\theta$")
    axes[1, 1].set_title(r"spectral tail of $\omega$")
    axes[1, 2].set_title(r"$|\langle\theta\rangle/\langle\theta_0\rangle - 1|$")
    for ax in axes.ravel():
        ax.set_xlabel("$t$"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PLOTS / "diagnostics.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def animate(Re: float, fps: int = 12, dpi: int = 80, scale: float = 1.0,
            every: int = 1):
    """Temperature / vorticity animation.  ``scale`` enlarges the figure
    (5.2 x 4.2 in at ``scale=1``) and ``dpi`` the rasterisation, so the pixel
    size is ``scale * dpi * (5.2, 4.2)``; ``every`` keeps every k-th frame."""
    S, t = _load(Re)
    S, t = S[:, ::every], t[::every]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(5.2 * scale, 4.2 * scale))
    u, v, th = P.fields(S[:, -1])
    wmax = 0.8 * np.abs(vorticity(u, v, P.LX, P.LY)).max()
    im0 = _panel(a0, th, "inferno", 0.0, 1.0)
    im1 = _panel(a1, vorticity(u, v, P.LX, P.LY), "RdBu_r", -wmax, wmax)
    fs = 10 * scale
    a0.set_title(r"$\theta$", fontsize=1.2 * fs); a1.set_title(r"$\omega$", fontsize=1.2 * fs)
    title = fig.suptitle("", fontsize=fs)
    fig.tight_layout()

    def update(i):
        u, v, th = P.fields(S[:, i])
        im0.set_data(th.T); im1.set_data(vorticity(u, v, P.LX, P.LY).T)
        title.set_text(f"Re = {Re:g},  t = {t[i]:.2f}")
        return im0, im1, title

    anim = FuncAnimation(fig, update, frames=S.shape[1], blit=False)
    tag = "" if (scale == 1.0 and dpi == 80) else f"_{int(round(5.2 * scale * dpi))}px"
    out = PLOTS / f"anim_Re{Re:g}{tag}.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, nargs="*", default=None)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--no-diag", action="store_true")
    ap.add_argument("--fps", type=int, default=12, help="GIF frame rate")
    ap.add_argument("--dpi", type=int, default=80, help="GIF rasterisation dpi")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="GIF figure enlargement factor (pixels scale with dpi*scale)")
    ap.add_argument("--every", type=int, default=1, help="keep every k-th frame")
    args = ap.parse_args()
    PLOTS.mkdir(exist_ok=True)

    cached = _cached_reynolds()
    if args.re is None:
        res = P.all_reynolds()
        wanted = [res["train"][0], res["train"][-1], res["test_extrap"][0]]
        args.re = [Re for Re in wanted if Re in cached] or cached[-1:]
    for Re in args.re:
        if args.gif:
            animate(Re, fps=args.fps, dpi=args.dpi, scale=args.scale,
                    every=args.every)
        else:
            plot_fields(Re)
    if not args.gif and not args.no_diag and cached:
        plot_compare(cached)
        plot_diagnostics(cached)


if __name__ == "__main__":
    main()
