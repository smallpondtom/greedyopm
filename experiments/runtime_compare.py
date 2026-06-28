"""Cross-experiment runtime comparison.

Loads the resumable-sweep checkpoints written by the three comparison
experiments (linear transport, 2D shallow water, transient Navier-Stokes) and
compiles the per-method, per-reduced-dimension fit runtime into

    * one figure with three panels (runtime vs r, one curve per method), and
    * LaTeX tables (one detailed table per experiment + a compact summary)

so the timings can be dropped straight into the writeup.

Each checkpoint already records the wall time of every ``(method, r)`` fit:
``fit_secs`` (fit only; written by newer runs) is preferred, falling back to
``secs`` (fit + reconstruction-error evaluation) for older checkpoints.  Run
this *after* the three experiment sweeps are complete:

    python runtime_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp_methods import METHOD_NAMES, METHOD_STYLES

_HERE = Path(__file__).resolve().parent
PLOTS = _HERE / "plots"

# (key, pretty title, checkpoint path)
EXPERIMENTS = [
    ("transport", "Linear transport",
     _HERE / "transport" / "plots" / "transport_compare_results.json"),
    ("swe", "Shallow water (2D)",
     _HERE / "shallow_water" / "plots" / "swe_compare_results.json"),
    ("nse", r"Navier--Stokes (K\'arm\'an, Re=30)",
     _HERE / "nse" / "plots" / "nse_compare_results.json"),
]
# plain-text titles for the matplotlib panels (no LaTeX escaping)
PLAIN_TITLE = {
    "transport": "Linear transport",
    "swe": "Shallow water (2D)",
    "nse": "Navier–Stokes (Kármán, Re=30)",
}


def _runtime(rec):
    """Fit runtime for one checkpoint entry (prefer fit-only timing)."""
    if rec is None:
        return None
    v = rec.get("fit_secs", rec.get("secs"))
    return None if v is None else float(v)


def load(path):
    """Return {method: {r: secs}} and the sorted list of r present."""
    res = json.load(open(path))
    rs = sorted({rec["r"] for rec in res.values()})
    table = {nm: {} for nm in METHOD_NAMES}
    for nm in METHOD_NAMES:
        for r in rs:
            table[nm][r] = _runtime(res.get(f"{nm}@r={r}"))
    return table, rs


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def plot(data):
    fig, axes = plt.subplots(1, len(EXPERIMENTS), figsize=(20, 6.2), sharey=True)
    for ax, (key, _ltitle, _path) in zip(axes, EXPERIMENTS):
        table, rs = data[key]
        for nm in METHOD_NAMES:
            ys = [table[nm].get(r) for r in rs]
            st = METHOD_STYLES[nm]
            ax.semilogy(rs, ys, st["fmt"], label=nm, color=st["color"],
                        lw=st["lw"], ms=st["ms"])
        ax.set_title(PLAIN_TITLE[key], fontsize=19, pad=8)
        ax.set_xlabel(r"Reduced dimension $r$", fontsize=17)
        ax.set_xticks(rs)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=13)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Fit runtime [s]", fontsize=17)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=14,
               bbox_to_anchor=(0.5, -0.06), frameon=False, handlelength=3)
    fig.suptitle("Algorithm runtime vs reduced dimension", fontsize=23, y=1.03)
    plt.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "runtime_compare"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved {out}.png/.pdf")


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------

def _fmt(v):
    return "--" if v is None else f"{v:.2f}"


def detailed_table(key, ltitle, table, rs):
    """rows = methods, columns = r, entries = fit runtime [s]."""
    head = " & ".join([r"\textbf{Method}"] + [f"$r={r}$" for r in rs]) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Fit runtime [s] vs reduced dimension $r$ --- " + ltitle + r".}",
        r"\label{tab:runtime-" + key + r"}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(rs) + r"}",
        r"\toprule", head, r"\midrule",
    ]
    for nm in METHOD_NAMES:
        row = " & ".join([nm] + [_fmt(table[nm].get(r)) for r in rs])
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def summary_table(data):
    """rows = methods, columns = experiment, entries = mean fit runtime [s]."""
    head = " & ".join([r"\textbf{Method}"] + [ltitle for _, ltitle, _ in EXPERIMENTS])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Mean fit runtime [s] over $r=1,\dots,r_{\max}$.}",
        r"\label{tab:runtime-summary}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(EXPERIMENTS) + r"}",
        r"\toprule", head + r" \\", r"\midrule",
    ]
    for nm in METHOD_NAMES:
        cells = []
        for key, _ltitle, _path in EXPERIMENTS:
            table, rs = data[key]
            vals = [table[nm][r] for r in rs if table[nm].get(r) is not None]
            cells.append(f"{np.mean(vals):.2f}" if vals else "--")
        lines.append(" & ".join([nm] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    data = {}
    for key, _ltitle, path in EXPERIMENTS:
        if not path.exists():
            raise SystemExit(f"missing checkpoint {path} -- run the sweep first")
        data[key] = load(path)

    plot(data)

    blocks = [summary_table(data)]
    for key, ltitle, _path in EXPERIMENTS:
        table, rs = data[key]
        blocks.append(detailed_table(key, ltitle, table, rs))
    tex = "\n\n".join(blocks) + "\n"

    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / "runtime_tables.tex"
    out.write_text(tex)
    print(f"    saved {out}\n")
    print(tex)


if __name__ == "__main__":
    main()
