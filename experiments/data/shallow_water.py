"""Faithful Fourier pseudo-spectral solver for the 2D shallow-water equations.

Reproduces the system documented in quickpde
(https://algopaul.github.io/quickpde/ShallowWater2d.html):

    d_t h   = -( d_x(h d_x phi) + d_y(h d_y phi) )
    d_t phi = -1/2 (d_x phi)^2 - 1/2 (d_y phi)^2 - h

on the periodic domain [-pi, pi)^2, with the default perturbed-flat IC

    h0(x, y) = 1 + 0.33 exp(-alpha (x^2 + y^2)),   phi0 = 0,

where alpha is the ``ic_sharpness``.  The initial bump disperses into
outward-propagating waves (transport + nonlinear, slow Kolmogorov n-width).

The spatial derivatives are computed spectrally (FFT); the quadratic
nonlinearities are formed in physical space and 2/3-dealiased.  Time stepping is
explicit RK4.  Everything is JIT-compiled and the time loop uses ``lax.scan``.

This module is standalone (run it to dump an ``.npz``) and importable
(``generate_shallow_water``).
"""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial


def _grids(N: int):
    """Periodic grid on [-pi, pi)^2 and integer wavenumbers."""
    L = 2.0 * np.pi
    x = -np.pi + L * np.arange(N) / N            # periodic, excludes +pi
    X, Y = np.meshgrid(x, y := x, indexing="ij")
    k1d = np.fft.fftfreq(N, d=L / N) * L          # = integer wavenumbers
    KX, KY = np.meshgrid(k1d, k1d, indexing="ij")
    return X, Y, KX, KY


@partial(jax.jit, static_argnums=())
def _rhs(h, phi, KX, KY, mask):
    """Right-hand side of the shallow-water system (spectral, dealiased)."""
    iKX, iKY = 1j * KX, 1j * KY
    phi_hat = jnp.fft.fft2(phi)
    phx = jnp.real(jnp.fft.ifft2(iKX * phi_hat))   # d_x phi
    phy = jnp.real(jnp.fft.ifft2(iKY * phi_hat))   # d_y phi

    # Flux divergence  d_x(h phx) + d_y(h phy), dealiased.
    fx_hat = mask * jnp.fft.fft2(h * phx)
    fy_hat = mask * jnp.fft.fft2(h * phy)
    div = jnp.real(jnp.fft.ifft2(iKX * fx_hat + iKY * fy_hat))
    dh = -div

    # Kinetic term 1/2 (phx^2 + phy^2), dealiased.
    kin_hat = mask * jnp.fft.fft2(0.5 * (phx * phx + phy * phy))
    kin = jnp.real(jnp.fft.ifft2(kin_hat))
    dphi = -kin - h
    return dh, dphi


@partial(jax.jit, static_argnums=())
def _rk4_step(h, phi, dt, KX, KY, mask):
    k1h, k1p = _rhs(h, phi, KX, KY, mask)
    k2h, k2p = _rhs(h + 0.5 * dt * k1h, phi + 0.5 * dt * k1p, KX, KY, mask)
    k3h, k3p = _rhs(h + 0.5 * dt * k2h, phi + 0.5 * dt * k2p, KX, KY, mask)
    k4h, k4p = _rhs(h + dt * k3h, phi + dt * k3p, KX, KY, mask)
    h = h + (dt / 6.0) * (k1h + 2 * k2h + 2 * k3h + k4h)
    phi = phi + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    return h, phi


def hamiltonian(h, phi, KX, KY):
    """Conserved Hamiltonian H = int 1/2 h(|grad phi|^2) + 1/2 h^2 dA."""
    phi_hat = jnp.fft.fft2(phi)
    phx = jnp.real(jnp.fft.ifft2(1j * KX * phi_hat))
    phy = jnp.real(jnp.fft.ifft2(1j * KY * phi_hat))
    dA = (2 * np.pi / h.shape[0]) ** 2
    return float(jnp.sum(0.5 * h * (phx**2 + phy**2) + 0.5 * h**2) * dA)


def generate_shallow_water(
    N: int = 64,
    dt: float = 2.0e-3,
    t_end: float = 8.0,
    store_every: int = 20,
    ic_sharpness: float = 3.0,
    bump_amp: float = 0.33,
    seed_fields: bool = True,
):
    """Integrate the SWE2d system and return stacked [h; phi] snapshots.

    Returns
    -------
    S : (2*N*N, n_snap) float64
        Snapshot matrix; rows 0:N*N are ``h``, rows N*N: are ``phi``.
    info : dict
        Grids, times, and the field shape for reshaping/plotting.
    """
    X, Y, KX, KY = _grids(N)
    KXj, KYj = jnp.asarray(KX), jnp.asarray(KY)
    kmax = N // 3
    mask = jnp.asarray((np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)).astype(jnp.float64)

    h0 = 1.0 + bump_amp * np.exp(-ic_sharpness * (X**2 + Y**2))
    phi0 = np.zeros((N, N))
    h, phi = jnp.asarray(h0), jnp.asarray(phi0)

    n_steps = int(round(t_end / dt))
    n_store = n_steps // store_every

    def scan_body(carry, _):
        h, phi = carry

        def inner(c, __):
            h, phi = c
            return _rk4_step(h, phi, dt, KXj, KYj, mask), None

        (h, phi), _ = jax.lax.scan(inner, (h, phi), None, length=store_every)
        snap = jnp.concatenate([h.reshape(-1), phi.reshape(-1)])
        return (h, phi), snap

    (h, phi), snaps = jax.lax.scan(scan_body, (h, phi), None, length=n_store)
    # snaps: (n_store, 2*N*N) -> include the initial state as first column
    S0 = jnp.concatenate([jnp.asarray(h0).reshape(-1), jnp.asarray(phi0).reshape(-1)])
    S = jnp.concatenate([S0[None, :], snaps], axis=0).T   # (2*N*N, n_store+1)

    tspan = dt * store_every * np.arange(n_store + 1)
    info = dict(N=N, X=X, Y=Y, KX=KX, KY=KY, tspan=tspan,
                field_shape=(N, N), dt=dt, store_every=store_every,
                ic_sharpness=ic_sharpness)
    return np.asarray(S), info


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "swe2d.npz"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    S, info = generate_shallow_water(N=N)
    KXj, KYj = jnp.asarray(info["KX"]), jnp.asarray(info["KY"])
    NN = N * N
    h0 = S[:NN, 0].reshape(N, N); p0 = S[NN:, 0].reshape(N, N)
    hT = S[:NN, -1].reshape(N, N); pT = S[NN:, -1].reshape(N, N)
    H0 = hamiltonian(jnp.asarray(h0), jnp.asarray(p0), KXj, KYj)
    HT = hamiltonian(jnp.asarray(hT), jnp.asarray(pT), KXj, KYj)
    print(f"S shape {S.shape}, finite={np.isfinite(S).all()}")
    print(f"h range [{S[:NN].min():.3f},{S[:NN].max():.3f}]  "
          f"phi range [{S[NN:].min():.3f},{S[NN:].max():.3f}]")
    print(f"Hamiltonian: H0={H0:.6f}  HT={HT:.6f}  drift={abs(HT-H0)/abs(H0):.2e}")
    np.savez_compressed(out, S=S, **{k: v for k, v in info.items()
                                     if k not in ("KX", "KY")})
    print(f"saved {out}")
