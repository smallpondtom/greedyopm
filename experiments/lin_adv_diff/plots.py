"""Plotting for the advection-diffusion comparison (journal-quality).

All figures are written as ``.png`` (300 dpi) and vector ``.pdf`` next to the
checkpoints, with no figure titles (report-ready): the quantitative context
that used to live in the titles is written to ``*_report.json`` instead.

* :func:`error_vs_r`             -- relative error vs r, one panel per split.
* :func:`active_features_bar`    -- active nonlinear feature counts vs r.
* :func:`budget_study`           -- interp / extrap error, validation
  objective and active-set size vs the Lipschitz budget L.
* :func:`certificate_vs_budget`  -- the attained Lipschitz certificate vs the
  set budget across r (primal-dual validity check).
* :func:`field_panels`           -- field reconstructions and |error| panels.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.axes_grid1 import ImageGrid  # noqa: E402

from methods import METHOD_STYLES  # noqa: E402

# ---------------------------------------------------------------------------
# Journal style
# ---------------------------------------------------------------------------

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
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
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

SPLIT_LABELS = {
    "train": "Training",
    "val_interp": "Validation (interp.)",
    "val_extrap": "Validation (extrap.)",
    "test_interp": "Interpolation test",
    "test_extrap": "Extrapolation test",
}

#: Method drawn as a degree-split stacked bar.
_SPLIT_METHOD = "Greedy-OPM (p=3)"


def _save(fig, out_prefix: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}")
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")


def _lighten(color, amount: float):
    """Blend ``color`` toward white by ``amount`` in [0, 1]."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---------------------------------------------------------------------------
# error-vs-r curves (one panel per split)
# ---------------------------------------------------------------------------


def error_vs_r(results, rs, methods, out_prefix, suptitle=None,
               splits=("train", "test_interp", "test_extrap")):
    """Semilogy of relative reconstruction error vs r, one panel per split."""
    splits = list(splits)
    fig, axes = plt.subplots(1, len(splits), figsize=(6.6 * len(splits), 6.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, split in zip(axes, splits):
        for name in methods:
            ys = [results[f"{name}@r={r}"][split] for r in rs]
            st = METHOD_STYLES[name]
            ax.semilogy(rs, ys, st["fmt"], label=name, color=st["color"],
                        lw=st["lw"], ms=st["ms"])
        ax.set_title(SPLIT_LABELS.get(split, split), pad=10)
        ax.set_xlabel(r"Reduced dimension $r$")
        ax.set_xticks(list(rs))
        ax.grid(True, which="both", alpha=0.25)
        _despine(ax)
    axes[0].set_ylabel("Relative reconstruction error")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods),
               bbox_to_anchor=(0.5, -0.04), frameon=False, handlelength=2.6,
               columnspacing=1.4)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Active nonlinear features (degree-split bar for p=3)
# ---------------------------------------------------------------------------


def _by_degree(results, name, r):
    rec = results.get(f"{name}@r={r}") or {}
    bd = rec.get("n_active_by_degree")
    return {int(k): int(v) for k, v in bd.items()} if bd else None


def active_features_bar(results, rs, methods, out_prefix, suptitle=None):
    """Grouped bars of active nonlinear feature counts vs r."""
    rs = list(rs)
    fig, ax = plt.subplots(figsize=(13, 6.2))
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
            for q in (2, 3):
                vals = np.array([b.get(q, 0) for b in breakdown], dtype=float)
                col = _lighten(st["color"], lighten[q])
                bars = ax.bar(xpos, vals, width, bottom=bottom, color=col,
                              edgecolor="white", lw=0.6, hatch=hatches[q])
                bottom += vals
                handles.append(Patch(facecolor=col, edgecolor="white",
                                     hatch=hatches[q],
                                     label=f"{name} — order {q}"))
            ax.bar_label(bars, labels=[str(int(v)) for v in bottom],
                         fontsize=10, padding=1)
        else:
            ys = [int(results[f"{name}@r={r}"]["n_active"]) for r in rs]
            bars = ax.bar(xpos, ys, width, color=st["color"],
                          edgecolor="white", lw=0.5)
            ax.bar_label(bars, fontsize=10, padding=1)
            handles.append(Patch(facecolor=st["color"], edgecolor="white",
                                 label=name))

    ax.set_xticks(x)
    ax.set_xticklabels(rs)
    ax.set_xlabel(r"Reduced dimension $r$")
    ax.set_ylabel("Active nonlinear features")
    ax.grid(True, axis="y", alpha=0.25)
    _despine(ax)
    ax.legend(handles=handles, ncol=2, frameon=False)
    plt.tight_layout()
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Budget study: error / objective / features vs L
# ---------------------------------------------------------------------------


def budget_study(study, out_prefix, suptitle=None):
    """Grid-adaptive per-variant budget selection (four panels vs ``L``)."""
    panels = [
        ("test_interp", "Interpolation test error", True),
        ("test_extrap", "Extrapolation test error", True),
        ("objective", r"Validation objective $\sqrt{e_{\mathrm{in}}\,"
                      r"e_{\mathrm{ex}}}$", True),
        ("n_active", "Active nonlinear features", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6.0))

    for ax, (key, title, logy) in zip(axes, panels):
        for name, rec in study.items():
            st = METHOD_STYLES[name]
            curve = rec["curve"]
            Ls = [c["L"] for c in curve]
            ys = [c[key] for c in curve]
            plot = ax.loglog if logy else ax.semilogx
            plot(Ls, ys, st["fmt"], label=name, color=st["color"],
                 lw=st["lw"], ms=st["ms"] - 1)
        for name, rec in study.items():
            st = METHOD_STYLES[name]
            ax.axvline(rec["L_star"], color=st["color"], ls=":", lw=2.0,
                       alpha=0.9)
        ax.set_title(title, pad=10)
        ax.set_xlabel(r"Lipschitz budget $L$")
        ax.grid(True, which="both", alpha=0.25)
        _despine(ax)
    axes[0].set_ylabel("Relative reconstruction error")
    axes[-1].set_ylabel("Active nonlinear features")

    handles = [Line2D([], [], color=METHOD_STYLES[n]["color"],
                      lw=METHOD_STYLES[n]["lw"], marker="D", ms=7,
                      label=f"{n}   (selected $L^* = {rec['L_star']:g}$, "
                            "dotted line)")
               for n, rec in study.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(study),
               bbox_to_anchor=(0.5, -0.03), frameon=False, handlelength=2.6)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Certificate vs budget (primal-dual validity)
# ---------------------------------------------------------------------------


def certificate_vs_budget(results, rs, opm_names, L_star, out_prefix,
                          suptitle=None):
    """Attained Lipschitz certificate vs the set budget across ``r``.

    Left: the certificate ``L_hat_Gamma(delta)`` (markers/solid) and the set
    budget ``L*`` (dashed) for each variant, log scale.  Right: the ratio
    ``L_hat_Gamma / L*``, which must stay <= 1 for every ``r`` -- the
    guarantee produced by the closed-form budget calibration.
    """
    rs = list(rs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0))

    for name in opm_names:
        st = METHOD_STYLES[name]
        L = float(L_star[name])
        certs = [float(results[f"{name}@r={r}"]["certificate"]) for r in rs]
        deg = name.split("(")[-1].rstrip(")")
        ax1.plot(rs, certs, st["fmt"], color=st["color"], lw=st["lw"],
                 ms=st["ms"] + 1,
                 label=rf"{deg}: certificate $\hat L_\Gamma(\delta)$")
        ax1.axhline(L, color=st["color"], ls="--", lw=2.0,
                    label=rf"{deg}: budget $L^* = {L:g}$")
        ax2.plot(rs, [c / L for c in certs], st["fmt"], color=st["color"],
                 lw=st["lw"], ms=st["ms"] + 1, label=deg)

    ax1.set_yscale("log")
    ax1.set_xlabel(r"Reduced dimension $r$")
    ax1.set_ylabel(r"Lipschitz constant")
    ax1.set_xticks(rs)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(frameon=False, loc="lower right", fontsize=14, handlelength=2.4)
    _despine(ax1)

    ax2.axhspan(0.0, 1.0, color="#2ca02c", alpha=0.07, zorder=0)
    ax2.axhline(1.0, color="black", ls=":", lw=1.8)
    ax2.set_ylim(0.0, 1.09)
    ax2.set_xlim(rs[0] - 0.4, rs[-1] + 0.4)
    ax2.set_xticks(rs)
    ax2.set_xlabel(r"Reduced dimension $r$")
    ax2.set_ylabel(r"$\hat L_\Gamma(\delta)\, /\, L^*$")
    ax2.text(rs[0] - 0.2, 1.005,
             r"feasible region  $\hat L_\Gamma(\delta) \leq L^*$",
             fontsize=14, va="bottom", ha="left")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, loc="lower right", title="Greedy-OPM",
               title_fontsize=14)
    _despine(ax2)

    plt.tight_layout()
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Field panels
# ---------------------------------------------------------------------------


def field_panels(truth_img, recon_imgs, names, out_prefix, suptitle=None,
                 extent=None, xlabel="time $t$", ylabel="space $x$",
                 rel_errs=None, cell=2.8):
    """One column per method: reconstruction (top) and |error| (bottom)."""
    n = len(names)
    vmin, vmax = float(np.min(truth_img)), float(np.max(truth_img))
    errs = {nm: np.abs(recon_imgs[nm] - truth_img) for nm in names}
    emax = min(max(float(np.max(errs[nm])) for nm in names), vmax) or 1.0
    im_kw = dict(extent=extent, origin="lower", aspect="auto")

    fig = plt.figure(figsize=(cell * n + 1.6, cell * 2 + 1.2))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, n), axes_pad=(0.10, 0.34),
                     share_all=True, cbar_mode="edge", cbar_location="right",
                     cbar_size="6%", cbar_pad=0.12)

    im_recon = im_err = None
    for j, nm in enumerate(names):
        a0, a1 = grid[j], grid[n + j]
        im_recon = a0.imshow(recon_imgs[nm], cmap="viridis",
                             vmin=vmin, vmax=vmax, **im_kw)
        tag = nm + (f"\n(rel. err {rel_errs[nm]:.2e})" if rel_errs else "")
        a0.set_title(tag, fontsize=14)
        im_err = a1.imshow(errs[nm], cmap="RdBu_r", vmin=0, vmax=emax, **im_kw)
        a1.set_xlabel(xlabel, fontsize=15)
        a0.tick_params(labelsize=12)
        a1.tick_params(labelsize=12)
        if j == 0:
            a0.set_ylabel(f"Reconstruction\n{ylabel}", fontsize=15)
            a1.set_ylabel(f"Absolute error\n{ylabel}", fontsize=15)

    grid.cbar_axes[0].colorbar(im_recon).ax.tick_params(labelsize=12)
    grid.cbar_axes[1].colorbar(im_err).ax.tick_params(labelsize=12)
    _save(fig, out_prefix)
