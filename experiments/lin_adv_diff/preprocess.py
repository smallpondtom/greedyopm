"""Snapshot preprocessing: centering + row-wise min-max normalization.

All methods are fit on preprocessed snapshots

    N = (S - mu - shift) / scale ,

where ``mu`` is the row-wise training mean (centering) and ``shift`` /
``scale`` map each centered row onto ``[-1, 1]`` from its row-wise training
minimum and maximum.  At reconstruction the chain is inverted -- the model
output is un-normalized and then un-centered -- so all reported errors live
in the original physical units.

The statistics are computed on the training split only and reused verbatim
for the test splits (out-of-sample values may leave ``[-1, 1]`` slightly;
that is the point of the extrapolation margin ``delta``).
"""

from __future__ import annotations

import jax.numpy as jnp


class RowMinMaxNormalizer:
    """Center by the training row mean, then scale each row to [-1, 1].

    Arguments
    ---------
    eps : float
        Floor on the row half-ranges, guarding constant rows.
    """

    def __init__(self, eps: float = 1e-30):
        self.eps = float(eps)

    def fit(self, S: jnp.ndarray) -> "RowMinMaxNormalizer":
        """Compute centering and scaling statistics from training data.

        Arguments
        ---------
        S : (d, n) raw training snapshots.
        """
        S = jnp.asarray(S)
        self.mu_ = jnp.mean(S, axis=1, keepdims=True)       # centering
        A = S - self.mu_
        lo = jnp.min(A, axis=1, keepdims=True)
        hi = jnp.max(A, axis=1, keepdims=True)
        self.shift_ = 0.5 * (hi + lo)                       # row midrange
        self.scale_ = jnp.maximum(0.5 * (hi - lo), self.eps)
        return self

    def transform(self, S: jnp.ndarray) -> jnp.ndarray:
        """Center and normalize raw snapshots (rows of the training split
        land exactly in [-1, 1])."""
        return (jnp.asarray(S) - self.mu_ - self.shift_) / self.scale_

    def inverse_transform(self, N: jnp.ndarray) -> jnp.ndarray:
        """Un-normalize and then un-center model output."""
        return jnp.asarray(N) * self.scale_ + self.shift_ + self.mu_


class PreprocessedModel:
    """Wrap a fitted model so its API operates in the original units.

    ``reconstruct`` transforms the input, applies the inner model, and
    inverts the preprocessing; ``reconstruction_error`` is therefore a
    physical-space error, with the relative variant measured against the
    centered raw data.  All other attributes (certificate, active features,
    selected modes, ...) are forwarded to the inner model.
    """

    def __init__(self, model, normalizer: RowMinMaxNormalizer):
        self.model = model
        self.normalizer = normalizer

    def reconstruct(self, S: jnp.ndarray) -> jnp.ndarray:
        """Encode-decode in normalized space, mapped back to raw units."""
        N_hat = self.model.reconstruct(self.normalizer.transform(S))
        return self.normalizer.inverse_transform(N_hat)

    def reconstruction_error(
        self, S: jnp.ndarray, *, relative: bool = True,
    ) -> float:
        """Frobenius error in the original units (relative to the
        centered raw snapshots)."""
        S = jnp.asarray(S)
        err = float(jnp.linalg.norm(S - self.reconstruct(S), "fro"))
        if relative:
            return err / float(jnp.linalg.norm(S - self.normalizer.mu_, "fro"))
        return err

    def __getattr__(self, name):
        return getattr(self.model, name)
