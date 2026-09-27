#!/usr/bin/env python3
"""Animate the cached 2D Burgers snapshots as a GIF, for visual inspection.

Four panels per frame, all from the *stored* snapshot matrix (so what you see
is the data the experiment actually consumes, not a fresh run):

* ``u``          -- first velocity component.
* ``v``          -- second velocity component.
* ``|U|``        -- speed, with a quiver overlay of the velocity field and a
  marker advected at exactly the mean-flow velocity ``(c, c)``.  If the
  solver is right, the pattern must travel *with* that marker: the marker is
  drawn at ``(x0 + c t, y0 + c t) mod 2 pi``, independent of the data.
* vorticity      -- ``omega = v_x - u_y`` (spectral), which shows the thin
  internal layers most clearly.

Colour limits are fixed over the whole trajectory so that motion, steepening
and viscous decay are all visible rather than being rescaled away.

Usage
-----
    python animate.py                 # test_interp and test_extrap
    python animate.py all             # every cached viscosity
    python animate.py 580 265         # by Reynolds number
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402

import burgers2d as B                                # noqa: E402
import problem as P                                  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"

FPS = 15
DPI = 64

#: Colour limits are fixed over the trajectory (so the viscous decay is
#: visible rather than rescaled away), but taken from a high percentile
#: rather than the maximum: the sharp-front spike near ``t ~ 1.5`` is a
#: narrow outlier that would otherwise wash out every other frame.
CLIP_PCT = 99.7


def vorticity(u, v, KX, KY):
    """``v_x - u_y`` by spectral differentiation."""
    vx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v)))
    uy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u)))
    return vx - uy


def animate(nu: float, store_every: int, out: Path, normalize: bool = False):
    """Write one GIF.

    ``normalize=False`` fixes the colour limits over the whole trajectory, so
    the viscous decay of the perturbation is visible directly.  Because the
    sharp-front phase near ``t ~ 1.5`` sets those limits, the late frames then
    look nearly uniform -- which is honest but hides the shape of the
    late-time field.  ``normalize=True`` rescales every frame by its own
    amplitude instead, so the *structure* stays legible at all times; the
    decay is then reported numerically in the title rather than shown.
    """
    S, info = P.solve_burgers(nu, store_every)
    X, Y, t = info["X"], info["Y"], np.asarray(info["tspan"])
    N = P.N
    NN = N * N
    _, _, KX, KY = B._grids(N)
    n_frames = S.shape[1]

    U = S[:NN].T.reshape(n_frames, N, N)
    V = S[NN:].T.reshape(n_frames, N, N)
    spd = np.sqrt(U ** 2 + V ** 2)
    vor = np.stack([vorticity(U[j], V[j], KX, KY) for j in range(n_frames)])

    # fixed limits over the whole trajectory (see CLIP_PCT)
    ulim = np.percentile(np.abs(U - P.C_MEAN), CLIP_PCT)
    vlim = np.percentile(np.abs(V - P.C_MEAN), CLIP_PCT)
    wlim = np.percentile(np.abs(vor), CLIP_PCT)
    slim = (np.percentile(spd, 100 - CLIP_PCT), np.percentile(spd, CLIP_PCT))

    # perturbation amplitude relative to t=0, reported per frame so the
    # viscous decay is quantified and not just eyeballed
    pert = np.linalg.norm(S - P.C_MEAN, axis=0)
    pert = pert / pert[0]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.8), constrained_layout=True)
    ax_u, ax_v, ax_s, ax_w = axes.ravel()
    ext = (0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi)

    def im(ax, data, cmap, vmin, vmax, title):
        # transpose: rows of the arrays index x, columns index y
        h = ax.imshow(data.T, origin="lower", extent=ext, cmap=cmap,
                      vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        fig.colorbar(h, ax=ax, fraction=0.046)
        return h

    h_u = im(ax_u, U[0], "RdBu_r", P.C_MEAN - ulim, P.C_MEAN + ulim, "$u$")
    h_v = im(ax_v, V[0], "RdBu_r", P.C_MEAN - vlim, P.C_MEAN + vlim, "$v$")
    h_s = im(ax_s, spd[0], "viridis", slim[0], slim[1], r"$|U|$ + quiver")
    h_w = im(ax_w, vor[0], "PuOr_r", -wlim, wlim, r"$\omega = v_x - u_y$")

    q = max(1, N // 20)
    quiv = ax_s.quiver(X[::q, ::q], Y[::q, ::q], U[0, ::q, ::q], V[0, ::q, ::q],
                       color="w", scale=30.0, width=0.004)
    # marker advected at exactly the mean flow (c, c) -- data-independent
    x0 = y0 = np.pi
    (mark,) = ax_s.plot([x0], [y0], "o", mfc="none", mec="r", mew=2.2, ms=13)

    Re = P.reynolds(nu)
    sup = fig.suptitle("", fontsize=10)

    def rescale(j):
        """Per-frame symmetric limits about the mean flow / about zero."""
        a = max(np.percentile(np.abs(U[j] - P.C_MEAN), CLIP_PCT), 1e-9)
        b = max(np.percentile(np.abs(V[j] - P.C_MEAN), CLIP_PCT), 1e-9)
        w = max(np.percentile(np.abs(vor[j]), CLIP_PCT), 1e-9)
        h_u.set_clim(P.C_MEAN - a, P.C_MEAN + a)
        h_v.set_clim(P.C_MEAN - b, P.C_MEAN + b)
        h_w.set_clim(-w, w)
        lo = np.percentile(spd[j], 100 - CLIP_PCT)
        hi = np.percentile(spd[j], CLIP_PCT)
        h_s.set_clim(lo, hi if hi > lo else lo + 1e-9)

    def update(j):
        h_u.set_data(U[j].T); h_v.set_data(V[j].T)
        h_s.set_data(spd[j].T); h_w.set_data(vor[j].T)
        if normalize:
            rescale(j)
        quiv.set_UVC(U[j, ::q, ::q], V[j, ::q, ::q])
        mark.set_data([(x0 + P.C_MEAN * t[j]) % (2 * np.pi)],
                      [(y0 + P.C_MEAN * t[j]) % (2 * np.pi)])
        sup.set_text(f"2D Burgers  $Re={Re:.0f}$  "
                     f"$t={t[j]:.2f}$  frame {j + 1}/{n_frames}  "
                     f"pert. {100 * pert[j]:.0f}%"
                     + ("  [per-frame scale]" if normalize else "  [fixed scale]"))
        return h_u, h_v, h_s, h_w, quiv, mark, sup

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    PLOTS.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)

    # re-encode with an adaptive palette; PillowWriter writes full frames
    from PIL import Image
    im = Image.open(out)
    frames = []
    try:
        while True:
            frames.append(im.convert("RGB").quantize(colors=224,
                                                     method=Image.MEDIANCUT))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)

    size = out.stat().st_size / 1e6
    print(f"  wrote {out.name}  ({n_frames} frames, {size:.1f} MB, "
          f"Re={Re:.0f}, t in [0, {t[-1]:.3f}], "
          f"perturbation decays to {100 * pert[-1]:.0f}%)")


def main(argv):
    normalize = "--norm" in argv
    argv = [a for a in argv if a != "--norm"]
    jobs = []
    if not argv:
        jobs = [(P.viscosity(P.test_interp_reynolds()), P.STORE_EVERY_TEST),
                (P.viscosity(P.test_extrap_reynolds()), P.STORE_EVERY_TEST)]
    elif argv == ["all"]:
        for Re in P.training_reynolds():
            jobs.append((P.viscosity(Re), P.STORE_EVERY_TRAIN))
        for Re in P.val_interp_reynolds():
            jobs.append((P.viscosity(Re), P.STORE_EVERY_VAL))
        jobs.append((P.viscosity(P.val_extrap_reynolds()), P.STORE_EVERY_VAL))
        jobs.append((P.viscosity(P.test_interp_reynolds()), P.STORE_EVERY_TEST))
        jobs.append((P.viscosity(P.test_extrap_reynolds()), P.STORE_EVERY_TEST))
    else:
        for a in argv:
            Re = float(a)
            se = (P.STORE_EVERY_TRAIN
                  if any(abs(Re - r) < 1e-6 for r in P.training_reynolds())
                  else P.STORE_EVERY_TEST)
            jobs.append((P.viscosity(Re), se))

    for nu, se in jobs:
        suffix = "_norm" if normalize else ""
        out = PLOTS / f"burgers2d_field_Re{P.reynolds(nu):.0f}{suffix}.gif"
        print(f"animating nu={nu:.6g} (Re={P.reynolds(nu):.0f}), stride {se}")
        animate(nu, se, out, normalize=normalize)


if __name__ == "__main__":
    main(sys.argv[1:])
