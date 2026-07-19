"""Shared train / validation / test split (temporal) for the experiments.

Snapshots are assigned to splits by CONTIGUOUS TIME WINDOWS: the first 50% of
the timesteps are used for training, the next 25% for validation, and the
final 25% for testing.  Both validation and test therefore lie in the *future*
of the training window, so the test set is a genuinely out-of-distribution
extrapolation task -- exactly the regime the stability-regularized
Greedy-OPM is designed for.

The validation window (the *near* future) tunes the Greedy-OPM
hyperparameters -- the sparsity parameter ``gamma_sp`` and the stability
weight ``mu`` -- so that the selected features are the ones that transfer
forward in time; the test window (the *far* future) is held out for the
reported errors.  Note that an interleaved or randomly permuted split would
make train and test nearly identical and flatter generalisation; we avoid it
deliberately.

Model fitting (SVD, mean, greedy selection) is invariant to column order.
With the temporal split each block is already in time order, so space-time
visualisations can use the split columns directly.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

TRAIN_FRAC = 0.5   # first 50% of the timesteps
VAL_FRAC = 0.25    # next 25% (near future) -- hyperparameter tuning


def split_indices(
    n: int, *, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(idx_train, idx_val, idx_test)`` contiguous column indices.

    ``idx_train`` covers the first ``train_frac`` of the columns,
    ``idx_val`` the following ``val_frac``, and ``idx_test`` the remainder
    (assuming the raw columns are ordered by timestep).
    """
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    idx = np.arange(n)
    return (
        idx[:n_train],
        idx[n_train:n_train + n_val],
        idx[n_train + n_val:],
    )


def train_val_test_split(*arrays, train_frac: float = TRAIN_FRAC,
                         val_frac: float = VAL_FRAC):
    """Split each array (along its LAST axis) into (train, val, test) blocks.

    Every input is split at the same temporal indices, so snapshot matrices of
    shape ``(d, n)`` and companion 1-D arrays of length ``n`` (e.g. a time
    vector) stay aligned.  Returns a flat tuple
    ``(a0_tr, a0_val, a0_te, a1_tr, a1_val, a1_te, ...)``.
    """
    if not arrays:
        raise ValueError("train_val_test_split needs at least one array")
    n = np.asarray(arrays[0]).shape[-1]
    itr, iva, ite = split_indices(n, train_frac=train_frac, val_frac=val_frac)
    out = []
    for a in arrays:
        a = np.asarray(a)
        if a.shape[-1] != n:
            raise ValueError(
                f"array last-axis length {a.shape[-1]} != {n}; all arrays must "
                "share the same number of columns")
        out.extend((a[..., itr], a[..., iva], a[..., ite]))
    return tuple(out)
