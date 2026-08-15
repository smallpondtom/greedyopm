"""Plotting for the advection-diffusion comparison.

All figures are written as ``.png`` and ``.pdf`` next to the checkpoint.

* :func:`error_vs_r`          -- 1-by-3 error-vs-r curves
  (train / unseen-parameter test / forecast test).
* :func:`active_features_bar` -- grouped bars of active nonlinear feature
  counts; Greedy-OPM (p=4) is drawn as a degree-split stacked bar.
* :func:`budget_study`        -- reconstruction error, active features, and
  attained certificate versus the Lipschitz budget L.
* :func:`field_panels`        -- space-time field reconstructions and
  pointwise absolute errors (one column per method).
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.axes_grid1 import ImageGrid  # noqa: E402

from methods import METHOD_STYLES  # noqa: E402

SPLIT_LABELS = {
    "train": "Training",
    "test_param": "Test — unseen wave speed",
    "test_forecast": "Test — forecast",
}

#: Method drawn as a degree-split stacked bar (order 2 / 3 / 4).
_SPLIT_METHOD = "Greedy-OPM (p=4)"


def _save(fig, out_prefix: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")


def _lighten(color, amount: float):
    """Blend ``color`` toward white by ``amount`` in [0, 1]."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)


# ---------------------------------------------------------------------------
# 1x3 error-vs-r curves
# ---------------------------------------------------------------------------


def error_vs_r(results, rs, methods, out_prefix, suptitle):
    """1-by-3 semilogy of relative error vs r for the three splits."""
    splits = ["train", "test_param", "test_forecast"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)
    for ax, split in zip(axes, splits):
        for name in methods:
            ys = [results[f"{name}@r={r}"][split] for r in rs]
            st = METHOD_STYLES[name]
            ax.semilogy(rs, ys, st["fmt"], label=name, color=st["color"],
                        lw=st["lw"], ms=st["ms"])
        ax.set_title(SPLIT_LABELS[split], fontsize=18, pad=8)
        ax.set_xlabel(r"Reduced dimension $r$", fontsize=16)
        ax.set_xticks(list(rs))
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=13)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Relative reconstruction error", fontsize=16)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=14,
               bbox_to_anchor=(0.5, -0.08), frameon=False, handlelength=3)
    fig.suptitle(suptitle, fontsize=20, y=1.02)
    plt.tight_layout()
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Active nonlinear features (degree-split bar for p=4)
# ---------------------------------------------------------------------------


def _by_degree(results, name, r):
    rec = results.get(f"{name}@r={r}") or {}
    bd = rec.get("n_active_by_degree")
    return {int(k): int(v) for k, v in bd.items()} if bd else None


def active_features_bar(results, rs, methods, out_prefix, suptitle):
    """Grouped bars of active nonlinear feature counts vs r.

    ``Greedy-OPM (p=4)`` is drawn as a stacked bar split by polynomial
    degree (quadratic solid at the bottom, cubic and quartic hatched above)
    so the composition of its active set is visible.
    """
    rs = list(rs)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(rs))
    nm = len(methods)
    width = 0.8 / nm
    hatches = {2: None, 3: "///", 4: "xxx"}
    lighten = {2: 0.0, 3: 0.45, 4: 0.7}
    handles = []

    for i, name in enumerate(methods):
        xpos = x + (i - (nm - 1) / 2) * width
        st = METHOD_STYLES[name]
        breakdown = [_by_degree(results, name, r) for r in rs]
        if name == _SPLIT_METHOD and all(b is not None for b in breakdown):
            bottom = np.zeros(len(rs))
            for q in (2, 3, 4):
                vals = np.array([b.get(q, 0) for b in breakdown], dtype=float)
                col = _lighten(st["color"], lighten[q])
                bars = ax.bar(xpos, vals, width, bottom=bottom, color=col,
                              edgecolor="white", lw=0.6, hatch=hatches[q])
                bottom += vals
                handles.append(Patch(facecolor=col, edgecolor="white",
                                     hatch=hatches[q],
                                     label=f"{name} — order {q}"))
            ax.bar_label(bars, labels=[str(int(v)) for v in bottom],
                         fontsize=8, padding=1)
        else:
            ys = [int(results[f"{name}@r={r}"]["n_active"]) for r in rs]
            bars = ax.bar(xpos, ys, width, color=st["color"],
                          edgecolor="white", lw=0.5)
            ax.bar_label(bars, fontsize=8, padding=1)
            handles.append(Patch(facecolor=st["color"], edgecolor="white",
                                 label=name))

    ax.set_xticks(x)
    ax.set_xticklabels(rs)
    ax.set_xlabel(r"Reduced dimension $r$", fontsize=15)
    ax.set_ylabel("Active nonlinear features", fontsize=15)
    ax.set_title(suptitle, fontsize=17, pad=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=handles, fontsize=11, ncol=2, frameon=False)
    plt.tight_layout()
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Budget study: error / features / certificate vs L
# ---------------------------------------------------------------------------


def budget_study(study, out_prefix, suptitle):
    """Reconstruction error, active features, and certificate vs budget L.

    ``study`` maps method name -> list of records with keys ``L``,
    ``train``, ``test_param``, ``test_forecast``, ``n_active``,
    ``certificate`` (full refits at each L).
    """
    splits = ["train", "test_param", "test_forecast"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.6))

    for ax, split in zip(axes[:3], splits):
        for name, recs in study.items():
            st = METHOD_STYLES[name]
            Ls = [rec["L"] for rec in recs]
            ys = [rec[split] for rec in recs]
            ax.loglog(Ls, ys, st["fmt"], label=name, color=st["color"],
                      lw=st["lw"], ms=st["ms"])
        ax.set_title(SPLIT_LABELS[split], fontsize=16, pad=8)
        ax.set_xlabel(r"Lipschitz budget $L$", fontsize=15)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Relative reconstruction error", fontsize=15)

    # Right panel: active features (solid) and attained certificate check.
    ax = axes[3]
    for name, recs in study.items():
        st = METHOD_STYLES[name]
        Ls = [rec["L"] for rec in recs]
        ax.semilogx(Ls, [rec["n_active"] for rec in recs], st["fmt"],
                    color=st["color"], lw=st["lw"], ms=st["ms"], label=name)
    ax.set_xlabel(r"Lipschitz budget $L$", fontsize=15)
    ax.set_ylabel("Active nonlinear features", fontsize=15)
    ax.set_title("Active set growth with $L$", fontsize=16, pad=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(study), fontsize=13,
               bbox_to_anchor=(0.5, -0.09), frameon=False, handlelength=3)
    fig.suptitle(suptitle, fontsize=19, y=1.03)
    plt.tight_layout()
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Space-time field panels
# ---------------------------------------------------------------------------


def field_panels(truth_img, recon_imgs, names, out_prefix, suptitle,
                 extent=None, xlabel="time $t$", ylabel="space $x$",
                 rel_errs=None, cell=2.6):
    """One column per method: reconstruction (top) and |error| (bottom).

    A shared colorbar per row sits at the right edge; the ground truth sets
    the color range of the reconstruction row.
    """
    n = len(names)
    vmin, vmax = float(np.min(truth_img)), float(np.max(truth_img))
    errs = {nm: np.abs(recon_imgs[nm] - truth_img) for nm in names}
    # Cap the shared error scale at the truth's peak so a diverging method
    # (e.g. Greedy-QM out of distribution) does not wash out the others.
    emax = min(max(float(np.max(errs[nm])) for nm in names), vmax) or 1.0
    im_kw = dict(extent=extent, origin="lower", aspect="auto")

    fig = plt.figure(figsize=(cell * n + 1.4, cell * 2 + 1.3))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, n), axes_pad=(0.08, 0.30),
                     share_all=True, cbar_mode="edge", cbar_location="right",
                     cbar_size="6%", cbar_pad=0.10)

    im_recon = im_err = None
    for j, nm in enumerate(names):
        a0, a1 = grid[j], grid[n + j]
        im_recon = a0.imshow(recon_imgs[nm], cmap="viridis",
                             vmin=vmin, vmax=vmax, **im_kw)
        tag = nm + (f"\n(rel.err {rel_errs[nm]:.2e})" if rel_errs else "")
        a0.set_title(tag, fontsize=10)
        im_err = a1.imshow(errs[nm], cmap="RdBu_r", vmin=0, vmax=emax, **im_kw)
        a1.set_xlabel(xlabel, fontsize=10)
        a0.tick_params(labelsize=8)
        a1.tick_params(labelsize=8)
        if j == 0:
            a0.set_ylabel(f"Reconstruction\n{ylabel}", fontsize=11)
            a1.set_ylabel(f"Absolute error\n{ylabel}", fontsize=11)

    grid.cbar_axes[0].colorbar(im_recon).ax.tick_params(labelsize=8)
    grid.cbar_axes[1].colorbar(im_err).ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=15)
    _save(fig, out_prefix)
