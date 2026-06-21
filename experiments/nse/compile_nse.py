"""Compile the transient Navier-Stokes (Karman vortex, Re=30) snapshots.

Reads the index JSON (sc_30_0-12_2000.json), loads the 2000 velocity .npy
snapshots in time order (restricted to interior DOFs ``invinds``), and assembles

  * V  : state matrix (d x n), columns ordered by increasing time,
  * C  : sensor / output matrix (6 x d) so that y(t) = C v(t),
  * trange : the time stamps.

Everything is saved to a single compiled .npz.  Mirrors load_nsevel_data() in
the source repo's load_or_comp.py but without the (optional) dolfin preview.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

SRC = Path("/sessions/loving-gallant-allen/mnt/transient_ns_pbenner")
JSON = SRC / "simulation_nse" / "sc_30_0-12_2000.json"
OUT = SRC / "nse_snapshots.npz"          # one compiled file


def main():
    t0 = time.time()
    with open(JSON) as f:
        d = json.load(f)
    vstr = d["velstrs"]
    invinds = np.asarray(d["invinds"], dtype=np.int64)
    times = list(vstr.keys())
    tvals = np.array([float(x) for x in times])
    order = np.argsort(tvals)                       # ensure time order
    tvals = tvals[order]
    times = [times[i] for i in order]
    if not np.all(tvals[:-1] <= tvals[1:]):
        raise RuntimeError("time stamps not sorted")
    n = len(times)
    dloc = invinds.shape[0]
    print(f"[compile] n={n} snapshots, d={dloc} interior DOFs, "
          f"t in [{tvals[0]}, {tvals[-1]}]")

    V = np.empty((dloc, n), dtype=np.float32)
    for j, ct in enumerate(times):
        a = np.load(str(SRC / "simulation_nse" / (vstr[ct] + ".npy")))
        V[:, j] = a[invinds, 0].astype(np.float32)
        if j % 400 == 0:
            print(f"  loaded {j}/{n}  ({time.time()-t0:.1f}s)")

    # sensor matrix C = cmat[:, invinds]  (6 x d), stored dense (tiny)
    cm = d["cmat"]
    assert cm["format"] == "csr"
    C = sps.csr_matrix((cm["data"], cm["indices"], cm["indptr"]),
                       shape=tuple(cm["shape"]))[:, invinds].toarray()
    print(f"[compile] C shape {C.shape}  (n_outputs={C.shape[0]})")

    np.savez(OUT, V=V, trange=tvals, C=C.astype(np.float64),
             invinds=invinds)
    print(f"[compile] saved {OUT}  ({OUT.stat().st_size/1e6:.0f} MB)  "
          f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
