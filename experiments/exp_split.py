"""Shared train / validation / test split (random, seeded) for the experiments.

Snapshots are assigned to splits by a fixed-seed random permutation: 50%
train, 25% validation, 25% test.  This is deliberately different from the two
alternatives we rejected:

* the old even/odd split -- train and test are interleaved and nearly
  identical, so the test error flatters generalisation; and
* a contiguous first-half / second-half split -- validation and test land in an
  unseen *future* regime, so the training-fit nonlinear manifold does not
  transfer and the gamma_sp tuning prunes every nonlinear feature (Greedy-OPM
  collapses to POD).

A random split gives genuinely held-out validation/test sets that are still
IN-distribution, so the nonlinear features generalise and the validation-tuned
gamma_sp selects a meaningful active set.  The validation set tunes the
Greedy-OPM sparsity parameter gamma_sp; the test set is held out for the
reported errors.

Model fitting (SVD, mean, greedy selection) is invariant to column order, so
using the shuffled train columns directly is fine.  For time-ordered
*visualisations* (space-time images, y(t) trajectories), sort the split's
columns back into time order first (``np.sort(idx)`` when the raw columns are
already time-ordered).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

SEED = 0   # fixed split seed -- reproducible across phases and reruns


def split_indices(
    n: int, *, seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(idx_train, idx_val, idx_test)`` column indices (50/25/25).

    A fixed-seed permutation of ``range(n)`` split into 50% / 25% / 25%.
    """
    perm = np.random.default_rng(seed).permutation(n)
    n_train = n // 2
    n_val = (n - n_train) // 2
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def train_val_test_split(*arrays, seed: int = SEED):
    """Split each array (along its LAST axis) into (train, val, test) blocks.

    Every input is split at the same random indices, so snapshot matrices of
    shape ``(d, n)`` and companion 1-D arrays of length ``n`` (e.g. a time
    vector) stay aligned.  Returns a flat tuple
    ``(a0_tr, a0_val, a0_te, a1_tr, a1_val, a1_te, ...)``.
    """
    if not arrays:
        raise ValueError("train_val_test_split needs at least one array")
    n = np.asarray(arrays[0]).shape[-1]
    itr, iva, ite = split_indices(n, seed=seed)
    out = []
    for a in arrays:
        a = np.asarray(a)
        if a.shape[-1] != n:
            raise ValueError(
                f"array last-axis length {a.shape[-1]} != {n}; all arrays must "
                "share the same number of columns")
        out.extend((a[..., itr], a[..., iva], a[..., ite]))
    return tuple(out)
