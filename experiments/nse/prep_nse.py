"""Compress the NSE state losslessly onto the training POD span (method of
snapshots), so the d=41682 problem is cheap for repeated method fits.

TEMPORAL split (see experiments/exp_split.py): the first 70% of the
snapshots train the models, the next 15% (near future) are the validation
window used to tune the Greedy-OPM hyperparameters, and the final 15% (far
future) are the held-out test window, so testing is a genuine
forward-in-time extrapolation.  The 70% training fraction is chosen so the
training window reaches the saturated vortex-shedding limit cycle (t ~ 8):
with a 50% fraction the training span misses ~40% of the future snapshot
energy and NO reconstruction method can recover it, whereas at 70% the
future is reconstructible (span floor ~1-2%) yet still out-of-distribution
(the shedding amplitude and phase keep evolving).

Let X = V_train - mean.  Via the small Gram matrix G = X^T X
(n_train x n_train) we get X = U diag(s) W^T with W, s from eig(G); the
training column space is span(U) and the POD coordinates are simply

    A_train = U^T X = diag(s) W^T,                       (K x n_train, exact)
    A_split = U^T (V_split - mean) = diag(1/s) W^T (X^T X_split).

Because U is orthonormal and K = rank(X), the training representation is
exact (floor_train ~ 0); the val/test floors are the (small) energy of those
snapshots outside the training span.  No d x K basis is ever formed.

Outputs: y = y_ref + C_red A with C_red = C U, y_ref = C mean; true outputs Y=C V.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                 # experiments/
from exp_split import split_indices                   # noqa: E402

DATA_DIR = _HERE / "data"
RAW = DATA_DIR / "nse_snapshots.npz"
OUT = DATA_DIR / "nse_compressed.npz"
ENERGY_TOL = 1e-13          # drop POD modes below this relative singular energy


def main():
    t0 = time.time()
    dat = np.load(RAW)
    V = dat["V"].astype(np.float64)          # (d, 2000)
    trange = dat["trange"]
    C = dat["C"]                             # (6, d)
    d, n = V.shape

    itr, iva, ite = split_indices(n, train_frac=0.7, val_frac=0.15)
    V_tr, V_va, V_te = V[:, itr], V[:, iva], V[:, ite]
    t_tr, t_va, t_te = trange[itr], trange[iva], trange[ite]
    s_ref = V_tr.mean(axis=1, keepdims=True)
    Xtr = V_tr - s_ref
    print(f"[prep] temporal split: train {Xtr.shape[1]} / val {V_va.shape[1]}"
          f" / test {V_te.shape[1]}; method-of-snapshots")

    # --- training POD via Gram matrix (n_train x n_train) -------------------
    G = Xtr.T @ Xtr                          # (ntr, ntr)
    w, W = np.linalg.eigh(G)                 # ascending
    w = w[::-1]; W = W[:, ::-1]              # descending
    w = np.clip(w, 0.0, None)
    sig = np.sqrt(w)
    tot = float(np.sum(w))
    keep = sig > (sig[0] * np.sqrt(ENERGY_TOL))
    K = int(np.count_nonzero(keep))
    sig = sig[:K]; W = W[:, :K]
    cum = np.cumsum(w[:K]) / tot
    for kk in (10, 25, 50, 100, 200, 400, K):
        if kk <= K:
            print(f"    energy captured @K={kk:4d}: {cum[kk-1]:.8f} "
                  f"(rel.floor {np.sqrt(max(1-cum[kk-1],0)):.2e})")

    # A_train = diag(sig) W^T ;  A_split = diag(1/sig) W^T (Xtr^T Xsplit)
    A_tr = sig[:, None] * W.T                              # (K, ntr)
    blocks = {}
    for tag, Vs in (("val", V_va), ("test", V_te)):
        Xs = Vs - s_ref
        A = (W.T @ (Xtr.T @ Xs)) / sig[:, None]
        nrm = float(np.linalg.norm(Xs))
        floor = float(np.sqrt(max(nrm**2 - np.sum(A**2), 0.0)))
        blocks[tag] = (A, nrm, floor)
        print(f"[prep] {tag}: floor(rel)={floor/nrm:.2e}")
    nrm_tr = float(np.linalg.norm(Xtr))
    floor_tr = float(np.sqrt(max(nrm_tr**2 - np.sum(A_tr**2), 0.0)))
    print(f"[prep] K={K}  floor_train(rel)={floor_tr/nrm_tr:.2e}")

    # C_red = C U = C Xtr W diag(1/sig) ;  y_ref = C mean ; Y = C V
    C_red = ((C @ Xtr) @ W) / sig[None, :]                 # (6, K)
    y_ref = (C @ s_ref).reshape(-1)

    np.savez(OUT,
             A_train=A_tr, A_val=blocks["val"][0], A_test=blocks["test"][0],
             Y_train=C @ V_tr, Y_val=C @ V_va, Y_test=C @ V_te,
             C_red=C_red, y_ref=y_ref, sigma=sig,
             trange_train=t_tr, trange_val=t_va, trange_test=t_te,
             floor_train=floor_tr, floor_val=blocks["val"][2],
             floor_test=blocks["test"][2],
             nrm_train=nrm_tr, nrm_val=blocks["val"][1],
             nrm_test=blocks["test"][1], K=K)
    print(f"[prep] saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)  "
          f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
