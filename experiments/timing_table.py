"""Training / inference timing records and the LaTeX table for the paper.

Shared by the three experiment drivers (``lin_adv_diff``, ``burgers2d``,
``boussinesq``).  Every driver records, per method and reduced dimension
``r``,

* ``secs``                     -- wall time of the fit (training), and
* ``infer_secs``               -- wall time of one encode-decode pass over
                                  the ``test_interp`` split (median of
                                  ``INFER_REPEATS`` timed passes after one
                                  warm-up pass, so JIT compilation is
                                  excluded), together with
* ``infer_secs_per_snapshot``  -- ``infer_secs`` divided by the number of
                                  snapshots in that split.

:func:`time_inference` is the measurement, :func:`collect_timings` gathers
the records from a results checkpoint, :func:`write_timings` writes them as
JSON and CSV, and :func:`latex_table` renders the booktabs table (one column
pair "training / inference" per method, one row per ``r``).
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

import jax

INFER_REPEATS = 3


def time_inference(model, X) -> dict:
    """Median wall time of ``model.reconstruct(X)`` after a warm-up pass."""
    out = model.reconstruct(X)
    jax.block_until_ready(out)                       # warm-up (JIT, no timing)
    times = []
    for _ in range(INFER_REPEATS):
        tic = time.perf_counter()
        out = model.reconstruct(X)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - tic)
    secs = float(statistics.median(times))
    n = int(X.shape[1])
    return dict(infer_secs=secs, infer_secs_per_snapshot=secs / n,
                infer_n_snapshots=n, infer_repeats=INFER_REPEATS)


def collect_timings(results: dict, methods: list, rs: list,
                    cpn_name: str | None = None) -> dict:
    """``{method: {r: {training_s, inference_s, inference_s_per_snapshot}}}``
    from a results checkpoint (``CPN-S`` rows missing where its tolerance
    sweep did not reach that ``n``)."""
    names = list(methods) + ([cpn_name] if cpn_name else [])
    out = {}
    for name in names:
        rows = {}
        for r in rs:
            rec = results.get(f"{name}@r={r}")
            if rec is None:
                continue
            rows[str(r)] = dict(
                training_s=rec.get("secs"),
                inference_s=rec.get("infer_secs"),
                inference_s_per_snapshot=rec.get("infer_secs_per_snapshot"),
                inference_n_snapshots=rec.get("infer_n_snapshots"),
            )
        out[name] = rows
    return out


def write_timings(timings: dict, out_prefix: str, rs: list, labels: dict,
                  meta: dict | None = None) -> None:
    """Write ``<out_prefix>.json``, ``<out_prefix>.csv`` and
    ``<out_prefix>_table.tex``."""
    path = Path(out_prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(
        dict(meta=meta or {}, timings=timings), indent=1))
    with path.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "r", "training_s", "inference_s",
                    "inference_s_per_snapshot", "inference_n_snapshots"])
        for name, rows in timings.items():
            for r in rs:
                row = rows.get(str(r))
                if row is None:
                    continue
                w.writerow([name, r, row["training_s"], row["inference_s"],
                            row["inference_s_per_snapshot"],
                            row["inference_n_snapshots"]])
    (path.parent / (path.name + "_table.tex")).write_text(
        latex_table(timings, rs, labels, meta or {}))


def _fmt(x, kind: str) -> str:
    if x is None:
        return "--"
    if kind == "train":                               # seconds
        return f"{x:.2f}" if x >= 0.1 else f"{x:.2e}"
    # inference: milliseconds per snapshot
    ms = 1e3 * x
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 10:
        return f"{ms:.1f}"
    return f"{ms:.2f}" if ms >= 1 else f"{ms:.3f}"


def latex_table(timings: dict, rs: list, labels: dict, meta: dict) -> str:
    """Booktabs table: methods across (training [s] / inference [ms per
    snapshot]), reduced dimension down.  Paste into the journal draft; the
    header uses only standard LaTeX (booktabs, multicolumn)."""
    names = [n for n in timings if timings[n]]
    ncol = 1 + 2 * len(names)
    lines = []
    lines.append("% training time in seconds (fit at the selected budget / "
                 "tolerance); inference time in milliseconds per snapshot")
    lines.append("%% (median of %d encode-decode passes over the %s split "
                 "after one warm-up pass)" % (
                     meta.get("infer_repeats") or INFER_REPEATS,
                     meta.get("infer_split", "test_interp")))
    if meta.get("machine"):
        lines.append("% machine: " + str(meta["machine"]))
    lines.append(r"\begin{table}[tb]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{%s. Training time (s) and inference time "
                 r"(ms per snapshot) against the reduced dimension \( n \).}"
                 % meta.get("title", "Timings"))
    lines.append(r"  \label{tab:%s_timings}" % meta.get("experiment", "exp"))
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(r"  \begin{tabular}{r" + "rr" * len(names) + "}")
    lines.append(r"    \toprule")
    head = [r"\multirow{2}{*}{\( n \)}"] + [
        r"\multicolumn{2}{c}{%s}" % labels.get(n, n) for n in names]
    lines.append("    " + " & ".join(head) + r" \\")
    cm = " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i)
                  for i in range(len(names)))
    lines.append("    " + cm)
    sub = [""] + ["training & inference"] * len(names)
    lines.append("    " + " & ".join(sub) + r" \\")
    lines.append(r"    \midrule")
    for r in rs:
        cells = [str(r)]
        for n in names:
            row = timings[n].get(str(r))
            if row is None:
                cells += ["--", "--"]
            else:
                cells += [_fmt(row["training_s"], "train"),
                          _fmt(row["inference_s_per_snapshot"], "infer")]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    assert ncol == 1 + 2 * len(names)
    return "\n".join(lines) + "\n"
