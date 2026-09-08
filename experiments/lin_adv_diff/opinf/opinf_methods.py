"""Manifold + ROM adapters for the OpInf-on-manifold experiment.

Every method contributes a *manifold* -- fit once, jointly, on the training
snapshot matrix of the reconstruction experiment, with the **same settings**
(configuration and, for Greedy-OPM, the same selected Lipschitz budgets) as the
manifold-learning results -- and a *nonlinear feature map* shared by the
manifold decoder and the reduced model.  Two reduced models are built on every
manifold (both from :mod:`greedyopm.opinf`, see ``notes/opinf_opm.tex``):

* **Standard (plain latent) OpInf**: ``shat' = c + A shat + N p(shat)``.
  Deliberately evaluated even where it diverges -- the point of the experiment
  is to show that it does.
* **Geometric manifold-Galerkin (GMG) OpInf**:
  ``shat' = M(shat)^{-1} (a(shat) + J_p^T b(shat))`` with mass matrix
  ``M = I + J_p^T Vbar^T Vbar J_p``, whose conditioning is bounded by the
  manifold's Lipschitz budget (``cond(M) <= L^2``).

For POD both models coincide with the stable linear latent ROM.
"""

from __future__ import annotations

import os

from typing import Dict, Optional

import jax
import numpy as np

from methods import build_model, CFG                    # noqa: F401  (../)
from baselines import quadratic_features, cubic_features  # noqa: F401
from greedyopm import opinf

OPINF_METHODS = [
    "POD",
    "Greedy-QM",
    "Greedy-CM",
    "Geelen-QM",
    "Greedy-OPM (p=2)",
    "Greedy-OPM (p=3)",
]
OPM_NAMES = ("Greedy-OPM (p=2)", "Greedy-OPM (p=3)")

ROM_DEGREES = {
    "POD": "constant + linear",
    "Greedy-QM": "constant + linear + quadratic",
    "Greedy-CM": "constant + linear + cubic",
    "Geelen-QM": "constant + linear + quadratic (sparse)",
    "Greedy-OPM (p=2)": "constant + linear + quadratic",
    "Greedy-OPM (p=3)": "constant + linear + quadratic + cubic",
}

#: Order-wise Tikhonov weights: negligible on the empirically orthonormal OPM
#: feature block; a light ridge on the (ill-conditioned) monomial blocks.
LAM_ORTHO = dict(lam_const=0.0, lam_lin=1e-10, lam_nl=1e-10)
LAM_MONO = dict(lam_const=0.0, lam_lin=1e-10, lam_nl=1e-8)

#: Certificate collar for the *reduced-model* experiment.  The manifold
#: experiment certifies the decoder on a small collar (delta = 0.1); a ROM
#: trajectory drifts further off the training coordinate cloud than an
#: encode-decode pass ever does, so the budget is enforced over a dilated
#: collar (delta = 1.0, as in the note's experiments) -- taming the features
#: over the region the ROM actually visits.
#: Certificate collar for the ROM's manifolds.  The reduced trajectory
#: visits a dilated neighborhood of the fit snapshots, so the gentle regime
#: enforces the budget over a wide collar (delta = 1).  The "manifold"
#: (harsh) regime keeps the reconstruction experiment's delta = 0.1 -- the
#: value its archived budgets and results were produced with.
ROM_DELTA = float(os.environ.get(
    "OPINF_ROM_DELTA",
    "0.1" if os.environ.get("OPINF_REGIME", "gentle") == "manifold" else "1.0"))
ROM_CFG = dict(CFG, delta=ROM_DELTA)


# ---------------------------------------------------------------------------
# Monomial feature maps (experiment-local: they live in baselines)
# ---------------------------------------------------------------------------


def monomial_feature_fn(kind: str, active=None):
    """Full (or active-column) quadratic / cubic monomial features."""
    import jax.numpy as jnp
    base = quadratic_features if kind == "quadratic" else cubic_features
    if active is None:
        def feats(s):
            return base(s.reshape(-1, 1))[:, 0]
    else:
        act = jnp.asarray(np.asarray(active))

        def feats(s):
            return base(s.reshape(-1, 1))[act, 0]
    return feats


def monomial_feature_jac_fn(kind: str, active=None):
    """Monomial features and their Jacobian ``(p, J_p)`` by forward-mode AD."""
    feats = monomial_feature_fn(kind, active)

    def feats_jac(s):
        return feats(s), jax.jacfwd(feats)(s)
    return feats_jac


def _feature_and_jac(name: str, inner):
    """Return ``(feature_fn, feature_jac_fn, Vbar)`` for a fitted manifold."""
    if name == "POD":
        return None, None, None
    if name in OPM_NAMES:
        return (opinf.opm_feature_fn(inner),
                opinf.opm_feature_jac_fn(inner),
                np.asarray(inner.Vbar_nl_))
    if name == "Greedy-QM":
        return (monomial_feature_fn("quadratic"),
                monomial_feature_jac_fn("quadratic"),
                np.asarray(inner.Vbar_))
    if name == "Greedy-CM":
        return (monomial_feature_fn("cubic"),
                monomial_feature_jac_fn("cubic"),
                np.asarray(inner.Vbar_))
    if name == "Geelen-QM":
        act = inner.active_
        return (monomial_feature_fn("quadratic", active=act),
                monomial_feature_jac_fn("quadratic", active=act),
                np.asarray(inner.Vbar_))
    raise ValueError(f"Unknown method {name!r}")


class ManifoldROM:
    """A fitted manifold with both a plain and a geometric (GMG) reduced model."""

    def __init__(self, name: str, r: int, S_fit: np.ndarray,
                 *, L: Optional[float] = None, cfg: dict = ROM_CFG):
        self.name = name
        self.r = int(r)
        self.pm = build_model(name, S_fit, r, cfg=cfg, L=L)
        self.inner = self.pm.model
        self.normalizer = self.pm.normalizer
        self.feature_fn, self.feature_jac_fn, self.Vbar = \
            _feature_and_jac(name, self.inner)
        self.has_manifold = self.Vbar is not None
        self.lam = LAM_ORTHO if name in OPM_NAMES else LAM_MONO
        self.G_lift = (self.Vbar.T @ self.Vbar) if self.has_manifold else None
        self.n_features = int(getattr(self.inner, "n_active_nonlinear", 0))
        self.certificate = float(getattr(self.inner, "certificate_", np.nan))

    def set_budget(self, L: float):
        """Recalibrate an OPM manifold's budget; refresh the ROM feature data."""
        self.inner.set_budget(float(L))
        self.feature_fn, self.feature_jac_fn, self.Vbar = \
            _feature_and_jac(self.name, self.inner)
        self.G_lift = (self.Vbar.T @ self.Vbar) if self.has_manifold else None
        self.n_features = int(getattr(self.inner, "n_active_nonlinear", 0))
        self.certificate = float(getattr(self.inner, "certificate_", np.nan))
        return self

    # -- coordinate maps ----------------------------------------------------

    def encode(self, S_phys: np.ndarray) -> np.ndarray:
        N = self.normalizer.transform(np.asarray(S_phys))
        return np.asarray(self.inner.encode(N))

    def _normalized(self, S_phys: np.ndarray) -> np.ndarray:
        return np.asarray(self.normalizer.transform(np.asarray(S_phys)))

    def decode(self, latent: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp
        N_hat = self.inner.decode(jnp.asarray(np.asarray(latent, dtype=float)))
        return np.asarray(self.normalizer.inverse_transform(N_hat))

    def features(self, Shat: np.ndarray) -> Optional[np.ndarray]:
        return opinf.eval_features_matrix(self.feature_fn, Shat)

    def manifold_reconstruction_error(self, S_phys: np.ndarray) -> float:
        """Encode-decode reconstruction error (the ROM's representation floor)."""
        return float(self.pm.reconstruction_error(np.asarray(S_phys),
                                                  relative=True))

    # -- Operator Inference from one trajectory -----------------------------

    def learn_plain(self, S_block: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        """Standard latent-OpInf operators ``(c, A, N)`` from one trajectory."""
        Shat = self.encode(S_block)
        dShat = opinf.fbct4(Shat, dt)
        P_nl = self.features(Shat)
        return opinf.infer_operators(Shat, dShat, P_nl, **self.lam)

    def learn_gmg(self, S_block: np.ndarray, dt: float) -> Dict[str, Dict[str, np.ndarray]]:
        """GMG operators ``a = V^T f`` and ``b = Vbar^T f`` (``b`` absent for POD)."""
        Shat = self.encode(S_block)
        dShat = opinf.fbct4(Shat, dt)
        P_nl = self.features(Shat)
        a_ops = opinf.infer_operators(Shat, dShat, P_nl, **self.lam)
        if not self.has_manifold:
            return {"a": a_ops}
        q = self.Vbar.T @ self._normalized(S_block)     # (m, N) lifting coords
        dq = opinf.fbct4(q, dt)                         # Vbar^T Ndot
        b_ops = opinf.infer_operators(Shat, dq, P_nl, **self.lam)
        return {"a": a_ops, "b": b_ops}

    def learn(self, S_block: np.ndarray, dt: float, kind: str):
        return (self.learn_plain if kind == "plain" else self.learn_gmg)(S_block, dt)

    # -- Affine-in-parameter Operator Inference (exact velocities) ----------

    def _dN(self, dS_block: np.ndarray) -> np.ndarray:
        """Normalized-space velocity: ``N = (S - mu - shift)/scale`` gives
        ``dN/dt = dS/dt / scale`` (the affine offsets drop out)."""
        return np.asarray(np.asarray(dS_block) / self.normalizer.scale_)

    def learn_affine(self, blocks, dblocks, speeds, kind: str) -> Dict:
        """Pooled affine-in-parameter operators from exact velocities.

        The FOM operator is affine in the wave speed, ``A(c) = -c D1 + nu D2``,
        so the *intrusive* reduced operators are exactly ``O(c) = O0 + tau O1``
        with ``tau = (c - c_mid)/c_half`` the centered/scaled speed.  One
        regression pools all training speeds (per-snapshot design ``[d; tau d]``,
        targets the exact projected velocities), replacing the per-speed fits
        and the entrywise spline interpolation entirely.  (The *per-speed
        learned* operators are NOT affine in ``c`` -- each fit projects the
        closure error along its own trajectory's data distribution, which is
        what made the entrywise splines noisy; the pooled fit estimates the one
        affine family the parametric structure supports.)
        """
        speeds = [float(c) for c in speeds]
        c_mid = 0.5 * (max(speeds) + min(speeds))
        c_half = max(0.5 * (max(speeds) - min(speeds)), 1e-12)
        taus = [(c - c_mid) / c_half for c in speeds]
        V = np.asarray(self.inner.U_r_)
        Shat_list = [self.encode(blk) for blk in blocks]
        P_list = [self.features(Shat) for Shat in Shat_list]
        dN_list = [self._dN(dblk) for dblk in dblocks]
        dShat_list = [V.T @ dN for dN in dN_list]
        a0, a1 = opinf.infer_operators_affine(
            Shat_list, dShat_list, P_list, taus, **self.lam)
        learned = dict(c_mid=c_mid, c_half=c_half, a=(a0, a1))
        if kind == "gmg" and self.has_manifold:
            dq_list = [self.Vbar.T @ dN for dN in dN_list]
            b0, b1 = opinf.infer_operators_affine(
                Shat_list, dq_list, P_list, taus, **self.lam)
            learned["b"] = (b0, b1)
        return learned

    def ops_at(self, learned: Dict, c: float, kind: str):
        """Evaluate the affine operator family at speed ``c`` in the format
        the ``simulate`` paths expect."""
        tau = (float(c) - learned["c_mid"]) / learned["c_half"]
        a = opinf.affine_ops_at(*learned["a"], tau)
        if kind == "plain":
            return a
        ops = {"a": a}
        if "b" in learned:
            ops["b"] = opinf.affine_ops_at(*learned["b"], tau)
        return ops

    # -- Simulate + decode --------------------------------------------------

    def simulate_plain(self, ops: Dict[str, np.ndarray], S_truth: np.ndarray,
                       dt: float, *, method: str = "rk4") -> np.ndarray:
        s0 = self.encode(S_truth[:, :1])[:, 0]
        n_steps = S_truth.shape[1] - 1
        latent = opinf.simulate(ops, self.feature_fn, s0, n_steps, dt,
                                method=method)
        return self.decode(latent)

    def simulate_gmg(self, gmg_ops: Dict[str, Dict[str, np.ndarray]],
                     S_truth: np.ndarray, dt: float) -> np.ndarray:
        s0 = self.encode(S_truth[:, :1])[:, 0]
        n_steps = S_truth.shape[1] - 1
        if not self.has_manifold:                       # POD: linear latent ROM
            latent = opinf.simulate(gmg_ops["a"], None, s0, n_steps, dt,
                                    method="rk4")
        else:
            latent = opinf.simulate_gmg(
                gmg_ops["a"], gmg_ops["b"], self.feature_jac_fn,
                self.G_lift, s0, n_steps, dt)
        return self.decode(latent)

    def simulate(self, ops, S_truth: np.ndarray, dt: float, kind: str) -> np.ndarray:
        if kind == "plain":
            return self.simulate_plain(ops, S_truth, dt)
        return self.simulate_gmg(ops, S_truth, dt)
