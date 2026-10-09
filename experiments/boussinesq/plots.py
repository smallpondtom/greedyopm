"""Plotting for the comparison experiments (journal-quality, shared).

All figures are written as ``.png`` (300 dpi) and vector ``.pdf`` next to the
checkpoints, with no figure titles (report-ready): the quantitative context
that used to live in the titles is written to ``*_report.json`` instead.

* :func:`error_vs_r`             -- relative error vs r (1 x 2: within /
  outside the training parameter range).
* :func:`active_features_bar`    -- active nonlinear feature counts vs r
  (quadratic bars plain, cubic bars hatched).
* :func:`budget_study`           -- validation objective and active-set
  size vs the Lipschitz budget L (1 x 2).
* :func:`certificate_vs_budget`  -- the attained Lipschitz certificate vs the
  set budget across r (primal-dual validity check).
* :func:`field_panels`           -- field reconstructions and |error| panels.

Legend text comes from ``methods.LEGEND_LABELS`` (display names); the
internal method names (JSON keys) are unchanged.
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

from methods import LEGEND_LABELS, METHOD_STYLES  # noqa: E402

# ---------------------------------------------------------------------------
# Journal style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "xtick.labelsize": 19,
    "ytick.labelsize": 19,
    "legend.fontsize": 19,
    "axes.linewidth": 1.3,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6.5,
    "ytick.major.size": 6.5,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "lines.linewidth": 3.0,
    "hatch.linewidth": 1.2,
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
    "test_interp": "Within training parameter range",
    "test_extrap": "Outside training parameter range",
}

#: Methods drawn as a degree-split stacked bar (quadratic + hatched cubic).
_SPLIT_METHODS = ("Greedy-OPM (p=3)", "CPN-S")
#: Methods whose nonlinear features are all cubic (hatched bars).
_CUBIC_METHODS = ("Greedy-CM",)
#: Hatch pattern marking degree-3 features.
_CUBIC_HATCH = "///"

GRID_KW = dict(which="major", alpha=0.3, lw=0.9)


def _label(name: str) -> str:
    return LEGEND_LABELS.get(name, name)


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


def _major_grid_only(ax):
    ax.grid(True, **GRID_KW)
    ax.grid(False, which="minor")


def _row_major(n: int, ncol: int) -> list:
    """Handle order that makes a column-filled legend read row by row."""
    nrows = int(np.ceil(n / ncol))
    order = []
    for i in range(ncol):
        for j in range(nrows):
            k = j * ncol + i
            if k < n:
                order.append(k)
    return order


def _series(results, name, rs, key):
    """``(xs, ys)`` of ``results[name@r=.][key]`` over the available ``r``."""
    xs, ys = [], []
    for r in rs:
        rec = results.get(f"{name}@r={r}")
        if rec is not None and rec.get(key) is not None:
            xs.append(r)
            ys.append(rec[key])
    return xs, ys


# ---------------------------------------------------------------------------
# error-vs-r curves (within / outside the training parameter range)
# ---------------------------------------------------------------------------


def error_vs_r(results, rs, methods, out_prefix, suptitle=None,
               splits=("test_interp", "test_extrap")):
    """Semilogy of relative reconstruction error vs r, one panel per split."""
    splits = list(splits)
    fig, axes = plt.subplots(1, len(splits), figsize=(7.5 * len(splits), 5.5),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, split in zip(axes, splits):
        for name in methods:
            xs, ys = _series(results, name, rs, split)
            if not xs:
                continue
            st = METHOD_STYLES[name]
            ax.semilogy(xs, ys, st["fmt"], label=_label(name),
                        color=st["color"], lw=st["lw"], ms=st["ms"],
                        markeredgecolor="white", markeredgewidth=0.8)
        ax.set_title(SPLIT_LABELS.get(split, split), pad=10)
        ax.set_xlabel(r"Reduced dimension $n$")
        ax.set_xticks(list(rs))
        _major_grid_only(ax)
        _despine(ax)
    axes[0].set_ylabel("Relative\nreconstruction error")
    h, l = axes[0].get_legend_handles_labels()
    ncol = 4 if len(h) > 5 else len(h)
    nrows = int(np.ceil(len(h) / ncol))
    order = _row_major(len(h), ncol)
    fig.legend([h[k] for k in order], [l[k] for k in order],
               loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, 0.01),
               frameon=False, handlelength=2.6, columnspacing=1.6,
               handletextpad=0.6)
    plt.tight_layout(rect=(0, 0.09 * nrows, 1, 1))
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Active nonlinear features (degree-split bar for p=3, hatched cubic)
# ---------------------------------------------------------------------------


def _by_degree(results, name, r):
    rec = results.get(f"{name}@r={r}") or {}
    bd = rec.get("n_active_by_degree")
    return {int(k): int(v) for k, v in bd.items()} if bd else None


def active_features_bar(results, rs, methods, out_prefix, suptitle=None,
                        logy=False):
    """Grouped bars of active nonlinear feature counts vs r.

    Quadratic features are plain bars; cubic (degree-3) features are hatched.
    ``Greedy-OPM (p=3)`` (and ``CPN-S``, if listed) are drawn as one stacked
    bar each (quadratic part plain, cubic part hatched) in a single color.
    ``logy`` puts the counts on a log axis.
    """
    rs = list(rs)
    fig, ax = plt.subplots(figsize=(15, 6.0))
    x = np.arange(len(rs))
    nm = len(methods)
    span = 0.92                       # group width in units of the r spacing
    slot = span / nm
    width = 0.84 * slot               # bar width (the rest is spacing)
    handles = []

    for i, name in enumerate(methods):
        # r values for which this method has a record (CPN-S starts at n >= 2)
        have = [r for r in rs if f"{name}@r={r}" in results]
        xpos = np.array([x[rs.index(r)] for r in have]) + (i - (nm - 1) / 2) * slot
        st = METHOD_STYLES[name]
        col = st["color"]
        breakdown = [_by_degree(results, name, r) for r in have]
        if name in _SPLIT_METHODS and all(b is not None for b in breakdown):
            bottom = np.zeros(len(have))
            bars = None
            for q in (2, 3):
                vals = np.array([b.get(q, 0) for b in breakdown], dtype=float)
                bars = ax.bar(xpos, vals, width, bottom=bottom, color=col,
                              edgecolor="white", lw=0.8,
                              hatch=_CUBIC_HATCH if q == 3 else None)
                bottom += vals
            ax.bar_label(bars, labels=[str(int(v)) for v in bottom],
                         fontsize=12, padding=2)
            handles.append(Patch(facecolor=col, edgecolor="white",
                                 label=_label(name)))
        else:
            ys = [int(results[f"{name}@r={r}"]["n_active"]) for r in have]
            hatch = _CUBIC_HATCH if name in _CUBIC_METHODS else None
            bars = ax.bar(xpos, ys, width, color=col, edgecolor="white",
                          lw=0.8, hatch=hatch)
            ax.bar_label(bars, fontsize=12, padding=2)
            handles.append(Patch(facecolor=col, edgecolor="white",
                                 hatch=hatch, label=_label(name)))

    handles.append(Patch(facecolor="#b0b0b0", edgecolor="white",
                         hatch=_CUBIC_HATCH, label="cubic features"))
    ax.set_xticks(x)
    ax.set_xticklabels(rs)
    ax.tick_params(axis="x", length=0)          # labels only, no tick marks
    ax.set_xlabel(r"Reduced dimension $n$")
    ax.set_ylabel("Active nonlinear features")
    if logy:
        ax.set_yscale("log")
        ymax = max(p.get_height() + p.get_y() for p in ax.patches)
        ax.set_ylim(0.6, ymax * 1.8)
    ax.grid(True, axis="y", **GRID_KW)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper left",
              handlelength=1.8, handleheight=1.3)
    plt.tight_layout()
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Budget study: validation objective / active features vs L
# ---------------------------------------------------------------------------


def budget_study(study, out_prefix, suptitle=None):
    """Grid-adaptive per-variant budget selection (two panels vs ``L``)."""
    panels = [
        ("objective", r"Validation objective $\sqrt{\mathcal{E}_{\mathrm{in}}\,"
                      r"\mathcal{E}_{\mathrm{ex}}}$", True),
        ("n_active", "Active nonlinear features", False),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 6.2))

    for ax, (key, title, logy) in zip(axes, panels):
        for name, rec in study.items():
            st = METHOD_STYLES[name]
            curve = rec["curve"]
            Ls = [c["L"] for c in curve]
            ys = [c[key] for c in curve]
            plot = ax.loglog if logy else ax.semilogx
            plot(Ls, ys, st["fmt"], label=_label(name), color=st["color"],
                 lw=st["lw"], ms=st["ms"] - 1, markeredgecolor="white",
                 markeredgewidth=0.8)
        for name, rec in study.items():
            st = METHOD_STYLES[name]
            ax.axvline(rec["L_star"], color=st["color"], ls=":", lw=2.6,
                       alpha=0.9)
        ax.set_title(title, pad=10)
        ax.set_xlabel(r"Lipschitz budget $P$")
        _major_grid_only(ax)
        _despine(ax)
    axes[0].set_ylabel("Relative\nreconstruction error")
    axes[-1].set_ylabel("Active nonlinear features")

    handles = [Line2D([], [], color=METHOD_STYLES[n]["color"],
                      lw=METHOD_STYLES[n]["lw"],
                      linestyle=METHOD_STYLES[n]["fmt"][:-1],
                      marker=METHOD_STYLES[n]["fmt"][-1],
                      ms=METHOD_STYLES[n]["ms"] - 1, markeredgecolor="white",
                      label=f"{_label(n)}   (selected $P^* = {rec['L_star']:g}$, "
                            "dotted line)")
               for n, rec in study.items()]
    fig.legend(handles=handles, loc="lower center", ncol=1,
               bbox_to_anchor=(0.5, -0.1), frameon=False, handlelength=2.6,
               columnspacing=1.6)
    fig.suptitle('Lipschitz Budget Tuning with Validation Data Set', fontweight='bold', fontsize=25)
    plt.tight_layout(rect=(0, 0.08, 1, 1), w_pad=4.0)
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Certificate vs budget (primal-dual validity)
# ---------------------------------------------------------------------------


def certificate_vs_budget(results, rs, opm_names, L_star, out_prefix,
                          suptitle=None):
    """Attained Lipschitz certificate vs the set budget across ``r``.

    Left: the certificate ``L_hat_Gamma(delta)`` (markers/solid) and the set
    budget ``P*`` (dashed) for each variant, log scale.  Right: the ratio
    ``L_hat_Gamma / P*``, which must stay <= 1 for every ``r`` -- the
    guarantee produced by the closed-form budget calibration.
    """
    rs = list(rs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.0))

    for name in opm_names:
        st = METHOD_STYLES[name]
        L = float(L_star[name])
        certs = [float(results[f"{name}@r={r}"]["certificate"]) for r in rs]
        deg = name.split("(")[-1].rstrip(")")
        ax1.plot(rs, certs, st["fmt"], color=st["color"], lw=st["lw"],
                 ms=st["ms"] + 1, markeredgecolor="white",
                 label=rf"{deg}: certificate $\hat L_\Gamma(\delta)$")
        ax1.axhline(L, color=st["color"], ls="--", lw=2.4,
                    label=rf"{deg}: budget $P^* = {L:g}$")
        ax2.plot(rs, [c / L for c in certs], st["fmt"], color=st["color"],
                 lw=st["lw"], ms=st["ms"] + 1, markeredgecolor="white",
                 label=deg)

    ax1.set_yscale("log")
    ax1.set_xlabel(r"Reduced dimension $n$")
    ax1.set_ylabel(r"Lipschitz constant")
    ax1.set_xticks(rs)
    _major_grid_only(ax1)
    ax1.legend(frameon=False, loc="lower right", fontsize=16, handlelength=2.4)
    _despine(ax1)

    ax2.axhspan(0.0, 1.0, color="#2ca02c", alpha=0.07, zorder=0)
    ax2.axhline(1.0, color="black", ls=":", lw=2.0)
    ax2.set_ylim(0.0, 1.09)
    ax2.set_xlim(rs[0] - 0.4, rs[-1] + 0.4)
    ax2.set_xticks(rs)
    ax2.set_xlabel(r"Reduced dimension $n$")
    ax2.set_ylabel(r"$\hat L_\Gamma(\delta)\, /\, P^*$")
    ax2.text(rs[0] - 0.2, 1.005,
             r"feasible region  $\hat L_\Gamma(\delta) \leq P^*$",
             fontsize=16, va="bottom", ha="left")
    _major_grid_only(ax2)
    ax2.legend(frameon=False, loc="lower right", title="Greedy-OPM",
               title_fontsize=16)
    _despine(ax2)

    plt.tight_layout()
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Field panels
# ---------------------------------------------------------------------------


def field_panels(truth_img, recon_imgs, names, out_prefix, suptitle=None,
                 extent=None, xlabel="time $t$", ylabel="space $x$",
                 rel_errs=None, cell=3.3, aspect="auto", cmap="viridis",
                 transpose=False, pair_pad=0.12, block_gap=1.25):
    """4 x 4 panel: ground truth top-left, then per method the
    reconstruction (upper row of a pair) and |error| (lower row).

    Slot layout (``T`` = truth, ``k`` = k-th method, ``e`` = its error):

        T  1  2  3
        .  e  e  e
        4  5  6  7
        e  e  e  e

    The two row pairs are separate image grids: ``pair_pad`` (inches) is
    the vertical gap between a field and its error panel, ``block_gap``
    (inches) the gap between the second and third rows that holds the titles
    of the third row.  ``transpose`` draws arrays stored as ``(nx, ny)`` with
    ``x`` horizontal; ``aspect="equal"`` sizes the panels by ``extent``.
    """
    names = list(names)
    if len(names) > 7:
        raise ValueError("field_panels draws at most 7 methods in a 4 x 4 grid")
    if transpose:
        truth_img = np.asarray(truth_img).T
        recon_imgs = {nm: np.asarray(v).T for nm, v in recon_imgs.items()}
    vmin, vmax = float(np.min(truth_img)), float(np.max(truth_img))
    errs = {nm: np.abs(recon_imgs[nm] - truth_img) for nm in names}
    emax = min(max(float(np.max(errs[nm])) for nm in names), vmax) or 1.0
    im_kw = dict(extent=extent, origin="lower", aspect=aspect)
    ratio = 1.0
    if aspect == "equal" and extent is not None:
        ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])

    ncol = 4
    panel_h = cell * ratio
    col_pad, cbar_frac, cbar_pad = 0.12, 0.06, 0.12
    top_pad, bot_pad = 0.9, 0.9                  # title row / x-label row
    left_pad, right_pad = 0.9, 0.3
    # the figure is sized so that each image grid fills its rect exactly
    # (panels of cell x cell*ratio inches), which keeps the vertical gaps
    # at the requested values
    grid_w = ncol * cell + (ncol - 1) * col_pad + cbar_frac * cell + cbar_pad
    fig_w = left_pad + grid_w + right_pad
    fig_h = 4 * panel_h + 2 * pair_pad + block_gap + top_pad + bot_pad
    fig = plt.figure(figsize=(fig_w, fig_h))
    block_h = (2 * panel_h + pair_pad) / fig_h   # height of one row pair

    def make_grid(bottom):
        return ImageGrid(fig, (left_pad / fig_w, bottom, grid_w / fig_w, block_h),
                         nrows_ncols=(2, ncol), axes_pad=(col_pad, pair_pad),
                         share_all=True, cbar_mode="edge",
                         cbar_location="right", cbar_size=f"{100 * cbar_frac:g}%",
                         cbar_pad=cbar_pad)

    grid_bot = make_grid(bot_pad / fig_h)
    grid_top = make_grid((bot_pad + 2 * panel_h + pair_pad + block_gap) / fig_h)
    grids = (grid_top, grid_bot)
    for g in grids:
        for ax in g:
            ax.set_visible(False)

    def panel(row, col):
        g = grids[row // 2]
        ax = g[(row % 2) * ncol + col]
        ax.set_visible(True)
        ax.tick_params(labelsize=22)
        return ax

    t_fs, l_fs = 24, 22
    ax = panel(0, 0)
    im_recon = ax.imshow(truth_img, cmap=cmap, vmin=vmin, vmax=vmax, **im_kw)
    ax.set_title("Ground truth", fontsize=t_fs)
    ax.set_ylabel(ylabel, fontsize=l_fs)

    slots = [(0, 1), (0, 2), (0, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
    im_err = None
    for nm, (row, col) in zip(names, slots):
        a0, a1 = panel(row, col), panel(row + 1, col)
        im_recon = a0.imshow(recon_imgs[nm], cmap=cmap, vmin=vmin, vmax=vmax,
                             **im_kw)
        tag = _label(nm) + (f"\n(rel. err {rel_errs[nm]:.2e})" if rel_errs else "")
        a0.set_title(tag, fontsize=t_fs)
        im_err = a1.imshow(errs[nm], cmap="RdBu_r", vmin=0, vmax=emax, **im_kw)
        if row + 1 == 3:
            a1.set_xlabel(xlabel, fontsize=l_fs)
        if col == 0 or (row == 0 and col == 1):
            a1.set_ylabel(f"Absolute error\n{ylabel}", fontsize=l_fs)
            if col != 0:      # slot (1, 0) is empty: show the row label here
                a1.axis["left"].toggle(label=True)
                a1.tick_params(left=True, labelleft=True)
        if col == 0:
            a0.set_ylabel(ylabel, fontsize=l_fs)
    for g in grids:
        g.cbar_axes[0].colorbar(im_recon).ax.tick_params(labelsize=20)
        g.cbar_axes[1].colorbar(im_err).ax.tick_params(labelsize=20)
    # x tick labels only on the last row; drop the tick at the right edge so
    # neighbouring panels do not run their labels together
    for k in range(ncol):
        grid_top[ncol + k].axis["bottom"].toggle(ticklabels=False)
    if extent is not None:
        # drop the tick at the upper end of each axis so the labels of
        # neighbouring panels (x: "8" and "0", y: "0" and "14") do not touch
        xt = [t for t in grid_bot[0].get_xticks()
              if extent[0] <= t < extent[1] - 1e-9 * (extent[1] - extent[0])]
        yt = [t for t in grid_bot[0].get_yticks()
              if extent[2] <= t < extent[3] - 1e-9 * (extent[3] - extent[2])]
        for g in grids:
            for ax in g:
                ax.set_xticks(xt)
                ax.set_yticks(yt)
    _save(fig, out_prefix)


# ---------------------------------------------------------------------------
# Noise robustness: error vs the noise-to-signal ratio sigma
# ---------------------------------------------------------------------------


def noise_study(rec, methods, out_prefix, suptitle=None):
    """Additive white noise on the states at noise-to-signal ratio ``sigma``:
    relative error of the reconstruction against the clean states (left) and
    against the noisy inputs (right).  ``rec`` is the ``_noise`` record of
    the results checkpoint.
    """
    sigmas = list(rec["sigmas"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4))
    for ax, key in ((ax1, "err_clean"), (ax2, "err_noisy")):
        for name in methods:
            ys = rec[key].get(name)
            if ys is None:
                continue
            st = METHOD_STYLES[name]
            ax.loglog(sigmas, ys, st["fmt"], label=_label(name),
                      color=st["color"], lw=st["lw"], ms=st["ms"],
                      markeredgecolor="white", markeredgewidth=0.8)
        ax.set_xlabel(r"Noise-to-signal ratio $\sigma$")
        _major_grid_only(ax)
        _despine(ax)
    ax1.set_ylabel("Relative\nreconstruction error")
    ax1.set_title("Error against the clean states", pad=10)
    ax2.set_title("Error against the noisy inputs", pad=10)
    h, l = ax1.get_legend_handles_labels()
    ncol = 4 if len(h) > 5 else len(h)
    nrows = int(np.ceil(len(h) / ncol))
    order = _row_major(len(h), ncol)
    fig.legend([h[k] for k in order], [l[k] for k in order],
               loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.06),
               frameon=False, handlelength=2.6, columnspacing=1.6,
               handletextpad=0.6)
    plt.tight_layout(rect=(0, 0.1 * nrows, 1, 1), w_pad=3.0)
    _save(fig, out_prefix)
