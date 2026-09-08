"""Journal-quality figures for the standard-vs-GMG OpInf comparison."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1 import ImageGrid  # noqa: E402

from methods import METHOD_STYLES  # noqa: E402  (../methods.py)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "lines.linewidth": 2.2,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

KIND_TITLES = {
    "plain": "Standard OpInf",
    "gmg": "Geometric manifold-Galerkin OpInf",
}
KIND_SHORT = {"plain": "Standard\nOpInf", "gmg": "GMG\nOpInf"}
_CAP = 10.0          # display cap: errors at/above this are diverged rollouts


def _save(fig, out_prefix: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _clip(v: float) -> float:
    v = float(v)
    return min(v, _CAP) if np.isfinite(v) else _CAP


# ---------------------------------------------------------------------------
# ROM state error vs reduced dimension (one figure per ROM kind)
# ---------------------------------------------------------------------------


def error_vs_r(results, rs, methods, kind: str, out_prefix: str):
    """Two panels (training median, interpolation test) of ROM error vs r."""
    panels = [("train_median", "Training speeds (median)"),
              ("interp", "Interpolation test")]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharey=True)
    for ax, (field, title) in zip(axes, panels):
        for name in methods:
            xs = [r for r in rs if f"{name}@r={r}" in results]
            ys = [_clip(results[f"{name}@r={r}"][kind][field]) for r in xs]
            st = METHOD_STYLES[name]
            ax.semilogy(xs, ys, st["fmt"], label=name, color=st["color"],
                        lw=st["lw"], ms=st["ms"])
        ax.axhline(_CAP, color="black", ls="--", lw=1.4, alpha=0.7)
        ax.set_title(title, pad=10)
        ax.set_xlabel(r"Reduced dimension $r$")
        ax.set_xticks(list(rs))
        ax.grid(True, which="both", alpha=0.25)
        _despine(ax)
    axes[0].set_ylabel("Relative state error")
    axes[0].text(rs[0] + 0.1, _CAP * 0.82, "diverged (clipped to cap)",
                 ha="left", va="top", fontsize=13, color="black")
    fig.suptitle(KIND_TITLES.get(kind, kind), fontsize=21, y=1.03)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.10), frameon=False, handlelength=2.6,
               columnspacing=1.4)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Decoded space-time fields at the interpolated speed (truth + every method)
# ---------------------------------------------------------------------------


def _spacetime(S: np.ndarray, n_show: int = 500) -> np.ndarray:
    """Subsample a (d, N) trajectory in time for display."""
    step = max(1, S.shape[1] // n_show)
    return np.asarray(S)[:, ::step]


def fields(data, results, methods, Ls, out_prefix: str, *, r_field=None):
    """Space-time diagrams ``u(x, t)`` of the decoded interp-speed ROM
    trajectories: one column per method (plus the truth), one row per ROM kind.

    The manifolds are refit at ``r_field`` (default: the ``r`` where the
    GMG Greedy-OPM (p=3) interpolation error is smallest) and re-simulated.
    """
    from opinf_methods import ManifoldROM
    from greedyopm import opinf
    from run_opinf import _learn_all, _ops_for, KINDS

    if r_field is None:
        cand = [(results[k]["gmg"]["interp"], results[k]["r"])
                for k in results if k.startswith("Greedy-OPM (p=3)@")]
        r_field = min(cand)[1] if cand else 6
    dt = data["dt"]
    speeds = data["train"]["speeds"]
    blocks = data["train"]["blocks"]
    S_int = data["interp"]["S"]
    c_int = data["interp"]["c"]
    truth = _spacetime(S_int)

    dec = {kind: {} for kind in KINDS}
    errs = {kind: {} for kind in KINDS}
    from run_opinf import _build_mrom
    for name in methods:
        L = Ls.get(name)
        if isinstance(L, dict) and "L_ref" not in L:
            L = L.get(int(r_field), L.get(str(int(r_field))))
        mrom = _build_mrom(name, int(r_field), data, L)
        for kind in KINDS:
            learned = _learn_all(mrom, data, kind)
            ops_i = _ops_for(mrom, learned, data, c_int, kind)
            S_dec = mrom.simulate(ops_i, S_int, dt, kind)
            errs[kind][name] = opinf.relative_state_error(S_int, S_dec)
            dec[kind][name] = _spacetime(np.nan_to_num(
                S_dec, nan=1e6, posinf=1e6, neginf=-1e6))

    vmin = float(np.min(truth))
    vmax = float(np.max(truth))
    n_cols = len(methods) + 1
    t_end = float(data["t"][-1])
    extent = [0.0, t_end, 0.0, 1.0]
    xticks = [0.0, round(0.5 * t_end, 2)]
    im_kw = dict(origin="lower", aspect="auto", cmap="viridis",
                 vmin=vmin, vmax=vmax, extent=extent)

    def _wrap(name: str) -> str:
        return name.replace(" (", "\n(")

    fig, axes = plt.subplots(
        2, n_cols, figsize=(2.35 * n_cols + 1.4, 7.6),
        sharex=True, sharey=True,
        gridspec_kw=dict(wspace=0.10, hspace=0.34,
                         left=0.09, right=0.90, top=0.82, bottom=0.09))
    im = None
    for i, kind in enumerate(KINDS):
        ax0 = axes[i, 0]
        im = ax0.imshow(np.clip(truth, vmin, vmax), **im_kw)
        ax0.set_title("Truth\n", fontsize=13)
        ax0.set_ylabel(f"{KIND_SHORT[kind]}\n$x$", fontsize=14)
        for j, name in enumerate(methods):
            ax = axes[i, 1 + j]
            ax.imshow(np.clip(dec[kind][name], vmin, vmax), **im_kw)
            e = errs[kind][name]
            tag = "diverged" if (not np.isfinite(e) or e > _CAP) \
                else f"err {e:.2e}"
            ax.set_title(f"{_wrap(name)}\n{tag}", fontsize=11.5)
        for j in range(n_cols):
            axj = axes[i, j]
            axj.set_xlim(0.0, t_end)
            axj.set_xticks(xticks)
            axj.tick_params(labelsize=11)
            if i == 1:
                axj.set_xlabel("$t$", fontsize=14)
    cax = fig.add_axes([0.915, 0.09, 0.014, 0.73])
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=12)
    fig.suptitle(rf"Decoded ROM trajectories at the interpolated speed "
                 rf"$c={c_int:.3g}$  ($r={int(r_field)}$)", fontsize=18,
                 y=0.965)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", bbox_inches=None)
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")
