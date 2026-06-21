"""Compress the NSE state losslessly onto the training POD span (method of
snapshots), so the d=41682 problem is cheap for repeated method fits.

Even snapshots -> training, odd -> test.  Let X = V_train - mean.  Via the small
Gram matrix G = X^T X (n_train x n_train) we get X = U diag(s) W^T with W, s from
eig(G); the training column space is span(U) and the POD coordinates are simply

    A_train = U^T X = diag(s) W^T,                       (K x n_train, exact)
    A_test  = U^T (V_test - mean) = diag(1/s) W^T (X^T X_test).

Because U is orthonormal and K = rank(X) = n_train, the training representation
is exact (floor_train = 0); the test floor is the (small) energy of the test
snapshots outside the training span.  No d x K basis is ever formed.

Outputs: y = y_ref + C_red A with C_red = C U, y_ref = C mean; true outputs Y=C V.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

SRC = Path("/sessions/loving-gallant-allen/mnt/transient_ns_pbenner")
RAW = SRC / "nse_snapshots.npz"
OUT = Path(__file__).resolve().parent / "nse_compressed.npz"
ENERGY_TOL = 1e-13          # drop POD modes below this relative singular energy


def main():
    t0 = time.time()
    dat = np.load(RAW)
    V = dat["V"].astype(np.float64)          # (d, 2000)
    trange = dat["trange"]
    C = dat["C"]                             # (6, d)
    d, n = V.shape

    V_tr, V_te = V[:, ::2], V[:, 1::2]
    t_tr, t_te = trange[::2], trange[1::2]
    s_ref = V_tr.mean(axis=1, keepdims=True)
    Xtr = V_tr - s_ref
    Xte = V_te - s_ref
    ntr = Xtr.shape[1]
    print(f"[prep] train {Xtr.shape}, test {Xte.shape}; method-of-snapshots")

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

    # A_train = diag(sig) W^T ;  A_test = diag(1/sig) W^T (Xtr^T Xte)
    A_tr = sig[:, None] * W.T                          # (K, ntr)
    A_te = (W.T @ (Xtr.T @ Xte)) / sig[:, None]        # (K, nte)

    nrm_tr = float(np.linalg.norm(Xtr))
    nrm_te = float(np.linalg.norm(Xte))
    floor_tr = float(np.sqrt(max(nrm_tr**2 - np.sum(A_tr**2), 0.0)))
    floor_te = float(np.sqrt(max(nrm_te**2 - np.sum(A_te**2), 0.0)))
    print(f"[prep] K={K}  floor_train(rel)={floor_tr/nrm_tr:.2e}  "
          f"floor_test(rel)={floor_te/nrm_te:.2e}")

    # C_red = C U = C Xtr W diag(1/sig) ;  y_ref = C mean ; Y = C V
    C_red = ((C @ Xtr) @ W) / sig[None, :]            # (6, K)
    y_ref = (C @ s_ref).reshape(-1)
    Y_tr = C @ V_tr
    Y_te = C @ V_te

    np.savez(OUT, A_train=A_tr, A_test=A_te, Y_train=Y_tr, Y_test=Y_te,
             C_red=C_red, y_ref=y_ref, sigma=sig,
             trange_train=t_tr, trange_test=t_te,
             floor_train=floor_tr, floor_test=floor_te,
             nrm_train=nrm_tr, nrm_test=nrm_te, K=K)
    print(f"[prep] saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)  "
          f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
