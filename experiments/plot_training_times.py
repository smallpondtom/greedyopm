"""Standalone training-time-vs-r plotter for the greedyopm experiments.

Reads an experiment's results checkpoint (the JSON written by ``run_experiment``
/ ``run_pert``, whose keys are ``"<method>@r=<r>"`` and whose records carry a
``secs`` field) and draws training time versus the reduced dimension ``r`` for
every method.  Works for ``lin_adv_diff``, ``burgers2d`` and ``nse`` without
touching their drivers.

Usage:
    python plot_training_times.py <results.json> [out_basename]

Example:
    python plot_training_times.py experiments/lin_adv_diff/plots/lin_adv_diff_results.json \
        experiments/lin_adv_diff/plots/lin_adv_diff_timing
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# Colors/markers consistent with the experiments' METHOD_STYLES.
STYLES = {
    "POD":              dict(fmt="-o",  color="black",   lw=2.6, ms=7),
    "Greedy-QM":        dict(fmt="--^", color="#1f77b4", lw=2.0, ms=7),
    "Greedy-CM":        dict(fmt="--s", color="#2ca02c", lw=2.0, ms=7),
    "Geelen-QM":        dict(fmt="--v", color="#ff7f0e", lw=2.0, ms=7),
    "Greedy-OPM (p=2)": dict(fmt="-P",  color="#9467bd", lw=2.0, ms=8),
    "Greedy-OPM (p=3)": dict(fmt="-D",  color="#d62728", lw=2.0, ms=7),
}
_DEFAULT = dict(fmt="-o", color="gray", lw=2.0, ms=6)

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.labelsize": 17, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 12, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 2.2, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

_KEY = re.compile(r"^(?P<name>.+)@r=(?P<r>\d+)$")


def collect(results: dict):
    times = defaultdict(dict)
    for key, rec in results.items():
        m = _KEY.match(key)
        if not m or not isinstance(rec, dict) or "secs" not in rec:
            continue
        times[m.group("name")][int(m.group("r"))] = float(rec["secs"])
    return times


def main(path, out=None):
    results = json.loads(Path(path).read_text())
    times = collect(results)
    if not times:
        raise SystemExit(f"no '<method>@r=<r>' records with 'secs' in {path}")
    order = [n for n in STYLES if n in times] + \
            [n for n in times if n not in STYLES]
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    for name in order:
        st = STYLES.get(name, _DEFAULT)
        rs = sorted(times[name])
        ys = [times[name][r] for r in rs]
        ax.plot(rs, ys, st["fmt"], color=st["color"], lw=st["lw"],
                ms=st["ms"], label=name)
    ax.set_yscale("log")
    ax.set_xlabel(r"reduced dimension $r$")
    ax.set_ylabel("training time (s)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best", framealpha=0.92)
    out = out or str(Path(path).with_name(Path(path).stem + "_timing"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + ".png"); fig.savefig(out + ".pdf")
    print(f"saved {out}.png / .pdf  ({len(times)} methods)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
