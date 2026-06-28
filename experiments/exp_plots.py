"""Shared plotting: error-vs-r curves and field reconstruction / error panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from exp_methods import METHOD_STYLES


def error_vs_r(results, rs, methods, out_prefix, suptitle):
    """Side-by-side train/test semilogy of relative error vs r."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5), sharey=True)
    for ax, split, sub in zip(axes, ["train", "test"], ["Training", "Test"]):
        for name in methods:
            ys = [results[f"{name}@r={r}"][split] for r in rs]
            st = METHOD_STYLES[name]
            ax.semilogy(rs, ys, st["fmt"], label=name, color=st["color"],
                        lw=st["lw"], ms=st["ms"])
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
    """Grouped bar chart: number of active nonlinear features vs r."""
    fig, ax = plt.subplots(figsize=(14, 7))
    rs = list(rs)
    x = np.arange(len(rs))
    nm = len(methods)
    width = 0.8 / nm
    for i, name in enumerate(methods):
        ys = [_active_count(results, name, r) for r in rs]
        st = METHOD_STYLES[name]
        bars = ax.bar(x + (i - (nm - 1) / 2) * width, ys, width,
                      label=name, color=st["color"], edgecolor="white", lw=0.5)
        ax.bar_label(bars, fontsize=8, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(rs)
    ax.set_xlabel(r"Reduced dimension $r$", fontsize=16)
    ax.set_ylabel("Number of active nonlinear features", fontsize=16)
    ax.set_title(suptitle, fontsize=18, pad=10)
    ax.tick_params(labelsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=12, ncol=2, frameon=False)
    plt.tight_layout()
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
