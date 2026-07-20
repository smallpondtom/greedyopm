"""Shared plotting: error-vs-r curves and field reconstruction / error panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import ImageGrid

from exp_methods import METHOD_STYLES

# Methods whose active nonlinear set spans more than one polynomial degree and
# should therefore be drawn as a degree-split (stacked) bar.
_SPLIT_METHOD = "Greedy-OPM (p=[2,3])"


def _lighten(color, amount=0.55):
    """Blend ``color`` toward white by ``amount`` in [0, 1]."""
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1.0 - c) * amount)


def error_vs_r(results, rs, methods, out_prefix, suptitle):
    """Side-by-side train/test semilogy of relative error vs r."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), sharey=True)
    for ax, split, sub in zip(axes, ["train", "test"], ["Training", "Test"]):
        for name in methods:
            pts = [(r, results[f"{name}@r={r}"][split]) for r in rs
                   if f"{name}@r={r}" in results]
            if not pts:
                continue
            st = METHOD_STYLES[name]
            ax.semilogy([p[0] for p in pts], [p[1] for p in pts], st["fmt"],
                        label=name, color=st["color"], lw=st["lw"],
                        ms=st["ms"])
        ax.set_title(f"{sub} data", fontsize=20, pad=8)
        ax.set_xlabel(r"Reduced dimension $r$", fontsize=18)
        ax.set_xticks(rs)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=14)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Relative reconstruction error", fontsize=18)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=15,
               bbox_to_anchor=(0.5, -0.09), frameon=False, handlelength=3)
    fig.suptitle(suptitle, fontsize=22, y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")


def _active_count(results, name, r):
    """Number of active nonlinear features for (method, r).

    The checkpoint stores ``n_active = -1`` for methods that don't track
    feature selection: POD (no nonlinear features -> 0) and Greedy-QM (keeps
    the FULL quadratic shell -> r(r+1)/2).  Fill those analytically so the
    table/plot is complete and comparable.
    """
    rec = results.get(f"{name}@r={r}")
    if rec is None:
        return None
    na = int(rec["n_active"])
    if na >= 0:
        return na
    if name == "POD":
        return 0
    return r * (r + 1) // 2          # Greedy-QM: all quadratic monomials


def _active_by_degree(results, name, r):
    """`{degree(int): count}` breakdown of the active nonlinear features.

    Returns ``None`` when the checkpoint has no per-degree record for this
    (method, r), in which case the caller falls back to a single bar.
    """
    rec = results.get(f"{name}@r={r}")
    if rec is None:
        return None
    bd = rec.get("n_active_by_degree")
    if not bd:
        return None
    return {int(k): int(v) for k, v in bd.items()}


def active_features_table(results, rs, methods):
    """Return (header, rows) and print a number-of-active-features table."""
    header = ["r"] + methods
    rows = []
    for r in rs:
        rows.append([r] + [_active_count(results, m, r) for m in methods])

    w = [max(len(str(h)), 3) for h in header]
    for row in rows:
        w = [max(wi, len(str(c))) for wi, c in zip(w, row)]
    line = "  ".join(str(h).rjust(wi) for h, wi in zip(header, w))
    print(line)
    print("  ".join("-" * wi for wi in w))
    for row in rows:
        print("  ".join(str(c).rjust(wi) for c, wi in zip(row, w)))
    return header, rows


def active_features_bar(results, rs, methods, out_prefix, suptitle):
    """Grouped bar chart: number of active nonlinear features vs r.

    ``Greedy-OPM (p=[2,3])`` is drawn as a degree-split (stacked) bar --
    quadratic (solid, bottom) + cubic (hatched, top) separated by a white
    divider -- so the quad/cubic composition of its active set is visible.  All
    other methods stay single bars.  Falls back to a single bar for any entry
    lacking a per-degree breakdown in the checkpoint.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    rs = list(rs)
    x = np.arange(len(rs))
    nm = len(methods)
    width = 0.8 / nm
    handles = []                                    # ordered legend proxies
    for i, name in enumerate(methods):
        xpos = x + (i - (nm - 1) / 2) * width
        st = METHOD_STYLES[name]
        # Per-r degree breakdown (quad/cubic) if available for the split method.
        breakdown = [_active_by_degree(results, name, r) for r in rs]
        do_split = (name == _SPLIT_METHOD
                    and all(b is not None for b in breakdown))
        if do_split:
            quad = np.array([b.get(2, 0) for b in breakdown], dtype=float)
            cubic = np.array([b.get(3, 0) for b in breakdown], dtype=float)
            col_c = _lighten(st["color"], 0.5)
            bq = ax.bar(xpos, quad, width, color=st["color"],
                        edgecolor="white", lw=0.6)
            bc = ax.bar(xpos, cubic, width, bottom=quad, color=col_c,
                        edgecolor="white", lw=0.6, hatch="///")
            ax.bar_label(bq, labels=[str(int(v)) if v else "" for v in quad],
                         label_type="center", fontsize=7, color="white")
            ax.bar_label(bc, labels=[str(int(v)) if v else "" for v in cubic],
                         label_type="center", fontsize=7)
            ax.bar_label(bc, labels=[str(int(q + c)) for q, c in
                                     zip(quad, cubic)], fontsize=8, padding=1)
            handles.append(Patch(facecolor=st["color"], edgecolor="white",
                                 label=f"{name} — quadratic"))
            handles.append(Patch(facecolor=col_c, edgecolor="white",
                                 hatch="///", label=f"{name} — cubic"))
        else:
            ys = [(_active_count(results, name, r) or 0) for r in rs]
            bars = ax.bar(xpos, ys, width, color=st["color"],
                          edgecolor="white", lw=0.5)
            ax.bar_label(bars, fontsize=8, padding=1)
            handles.append(Patch(facecolor=st["color"], edgecolor="white",
                                 label=name))
    ax.set_xticks(x)
    ax.set_xticklabels(rs)
    ax.set_xlabel(r"Reduced dimension $r$", fontsize=16)
    ax.set_ylabel("Number of active nonlinear features", fontsize=16)
    ax.set_title(suptitle, fontsize=18, pad=10)
    ax.tick_params(labelsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=handles, fontsize=12, ncol=2, frameon=False)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")


def gamma_sp_study(results, rs, methods, out_prefix, suptitle):
    """Two-panel gamma_sp study built from the stored validation sweeps.

    Left  : validation reconstruction error vs gamma_sp (one curve per r).
    Right : number of active nonlinear features vs gamma_sp.

    Curves are colour-graded by reduced dimension r (viridis colorbar); the two
    Greedy-OPM variants are distinguished by line style, and the selected
    gamma_sp (sparsest within tolerance of the best validation error) is marked
    with a star on each curve.  Only entries carrying a ``gamma_sp_curve`` in
    the checkpoint are drawn, so this is a no-op if the sweep stored none.
    """
    opm = [m for m in methods if m in
           ("Greedy-OPM (p=2)", _SPLIT_METHOD) and any(
               (results.get(f"{m}@r={r}") or {}).get("gamma_sp_curve")
               for r in rs)]
    if not opm:
        print(f"    [gamma_sp_study] no validation curves to plot; skipped")
        return

    rs = list(rs)
    ls_for = {_SPLIT_METHOD: "-", "Greedy-OPM (p=2)": "--"}
    norm = mcolors.Normalize(vmin=min(rs), vmax=max(rs))
    cmap = plt.cm.viridis

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7),
                                   constrained_layout=True)
    for name in opm:
        ls = ls_for.get(name, "-")
        for r in rs:
            rec = results.get(f"{name}@r={r}")
            if not rec or not rec.get("gamma_sp_curve"):
                continue
            curve = rec["gamma_sp_curve"]
            g = np.array([c["gamma_sp"] for c in curve])
            e = np.array([c["val_err"] for c in curve])
            na = np.array([c["n_active"] for c in curve])
            m = g > 0
            if not np.any(m):
                continue
            color = cmap(norm(r))
            axL.loglog(g[m], e[m], ls, color=color, lw=1.7, alpha=0.9)
            axR.semilogx(g[m], na[m], ls, color=color, lw=1.7, alpha=0.9)
            gsel = rec.get("gamma_sp")
            if gsel is not None:
                j = int(np.argmin(np.abs(g - gsel)))
                gx = g[j] if g[j] > 0 else g[m].min()
                axL.plot(gx, e[j], "*", color=color, ms=13, mec="k",
                         mew=0.6, zorder=5)
                axR.plot(gx, na[j], "*", color=color, ms=13, mec="k",
                         mew=0.6, zorder=5)
    for ax in (axL, axR):
        ax.set_xlabel(r"Sparsity parameter $\gamma_{\mathrm{sp}}$", fontsize=16)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=13)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axL.set_ylabel("Validation reconstruction error", fontsize=16)
    axR.set_ylabel("Number of active nonlinear features", fontsize=16)
    axL.set_title("Validation error vs $\\gamma_{\\mathrm{sp}}$", fontsize=16)
    axR.set_title("Feature count vs $\\gamma_{\\mathrm{sp}}$", fontsize=16)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=(axL, axR), fraction=0.04, pad=0.02)
    cbar.set_label(r"Reduced dimension $r$", fontsize=14)
    handles = [Line2D([0], [0], color="gray", ls=ls_for.get(m, "-"),
                      lw=2, label=m) for m in opm]
    handles.append(Line2D([0], [0], marker="*", color="w", mec="k",
                          ms=12, label=r"selected $\gamma_{\mathrm{sp}}$"))
    axL.legend(handles=handles, fontsize=12, frameon=False, loc="best")
    fig.suptitle(suptitle, fontsize=20)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_prefix}.png/.pdf")


def field_panels(truth_img, recon_imgs, names, out_prefix, suptitle,
                 cmap="viridis", extent=None, xlabel="", ylabel="",
                 aspect="auto", rel_errs=None, formats=("png",), figsize=None):
    """Grid: row 0 = ground truth; each following row = (recon, |error|).

    ``figsize`` overrides the default (useful to shrink horizontal whitespace
    for square, equal-aspect fields -- pass a narrower width).
    """
    n = len(names)
    fig, axes = plt.subplots(n + 1, 2,
                             figsize=figsize or (11, 3.0 * (n + 1)))
    vmin, vmax = float(np.min(truth_img)), float(np.max(truth_img))
    errs = {nm: np.abs(recon_imgs[nm] - truth_img) for nm in names}
    emax = max(float(np.max(errs[nm])) for nm in names) or 1.0
    im_kw = dict(extent=extent, aspect=aspect, origin="lower")

    # Ground truth on the top row (left), right cell shows the colorbar ref.
    ax = axes[0, 0]
    im0 = ax.imshow(truth_img, cmap=cmap, vmin=vmin, vmax=vmax, **im_kw)
    ax.set_title("Ground truth", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    axes[0, 1].axis("off")

    for i, nm in enumerate(names, start=1):
        a0, a1 = axes[i, 0], axes[i, 1]
        im = a0.imshow(recon_imgs[nm], cmap=cmap, vmin=vmin, vmax=vmax, **im_kw)
        tag = nm + (f"\n(rel.err {rel_errs[nm]:.2e})" if rel_errs else "")
        a0.set_title(tag, fontsize=12)
        a0.set_ylabel(ylabel, fontsize=11)
        fig.colorbar(im, ax=a0, fraction=0.046, pad=0.04)
        ie = a1.imshow(errs[nm], cmap="RdBu_r", vmin=0, vmax=emax, **im_kw)
        a1.set_title(f"|error|  ({nm})", fontsize=12)
        fig.colorbar(ie, ax=a1, fraction=0.046, pad=0.04)
        if i == n:
            a0.set_xlabel(xlabel, fontsize=12)
            a1.set_xlabel(xlabel, fontsize=12)
    fig.suptitle(suptitle, fontsize=18, y=1.005)
    plt.tight_layout()
    for ext in formats:
        fig.savefig(f"{out_prefix}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_prefix} [{','.join(formats)}]")


def field_panels_horizontal(truth_img, recon_imgs, names, out_prefix, suptitle,
                            cmap="viridis", extent=None, xlabel="", ylabel="",
                            aspect="auto", rel_errs=None, formats=("png",),
                            cell=2.6, axes_pad=(0.08, 0.30)):
    """Horizontal grid: one column per method, row 0 = reconstruction, row 1 =
    |error|.

    A single shared colorbar per row (recon colormap / error colormap) sits at
    the right edge, and the panels are packed with ``ImageGrid`` so square,
    equal-aspect fields stay tight (use a small ``axes_pad`` to remove the white
    gutters between plots).  ``aspect="equal"`` preserves the data aspect ratio
    (square fields); ``aspect="auto"`` lets each image fill its cell (space-time).
    """
    n = len(names)
    vmin, vmax = float(np.min(truth_img)), float(np.max(truth_img))
    errs = {nm: np.abs(recon_imgs[nm] - truth_img) for nm in names}
    emax = max(float(np.max(errs[nm])) for nm in names) or 1.0
    im_kw = dict(extent=extent, origin="lower")

    fig = plt.figure(figsize=(cell * n + 1.4, cell * 2 + 1.3))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, n), axes_pad=axes_pad,
                     aspect=(aspect == "equal"), share_all=True,
                     cbar_mode="edge", cbar_location="right",
                     cbar_size="6%", cbar_pad=0.10)

    im_recon = im_err = None
    for j, nm in enumerate(names):
        a0, a1 = grid[j], grid[n + j]                 # row 0, row 1
        im_recon = a0.imshow(recon_imgs[nm], cmap=cmap, vmin=vmin, vmax=vmax,
                             aspect=aspect, **im_kw)
        tag = nm + (f"\n(rel.err {rel_errs[nm]:.2e})" if rel_errs else "")
        a0.set_title(tag, fontsize=10)
        im_err = a1.imshow(errs[nm], cmap="RdBu_r", vmin=0, vmax=emax,
                           aspect=aspect, **im_kw)
        a1.set_xlabel(xlabel, fontsize=10)
        a0.tick_params(labelsize=8)
        a1.tick_params(labelsize=8)
        if j == 0:
            a0.set_ylabel(f"Reconstruction\n{ylabel}", fontsize=11)
            a1.set_ylabel(f"Absolute error\n{ylabel}", fontsize=11)

    cb0 = grid.cbar_axes[0].colorbar(im_recon)        # row 0: field colormap
    cb1 = grid.cbar_axes[1].colorbar(im_err)          # row 1: error colormap
    cb0.ax.tick_params(labelsize=8)
    cb1.ax.tick_params(labelsize=8)
    fig.suptitle(suptitle, fontsize=16)
    for ext in formats:
        fig.savefig(f"{out_prefix}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out_prefix} [{','.join(formats)}]")
