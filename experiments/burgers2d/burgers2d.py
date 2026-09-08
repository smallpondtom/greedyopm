"""Fourier pseudo-spectral solver for the 2D viscous Burgers equations.

Solves, on the periodic square ``[0, 2 pi)^2``,

    u_t + u u_x + v u_y = nu (u_xx + u_yy) ,
    v_t + u v_x + v v_y = nu (v_xx + v_yy) ,

from a smooth vortical initial condition superposed on a uniform mean flow
``(c, c)``:

    u0 = c + a ( sin x cos y + 0.5 sin 2x cos 3y ) ,
    v0 = c + a ( -cos x sin y + 0.4 cos 3x sin 2y ) .

The nonlinearity steepens the initial harmonics into thin internal layers
while the viscosity ``nu`` sets their thickness, and the mean flow sweeps
the whole pattern diagonally across the box.  That combination is the point
of the model: transport of thin moving structures gives a slowly decaying
Kolmogorov n-width (so a linear subspace struggles), while a strictly
positive viscosity keeps the problem physical and well resolved -- unlike a
purely advective system, whose fronts never stop sharpening.

Because a constant mean flow only translates the solution (Galilean
invariance), the Reynolds number ``Re = a * 2 pi / nu`` is the parameter
that actually changes the shape of the solution, and it is what the
parametric experiment varies.

Spatial derivatives are spectral (FFT); the quadratic terms are formed in
physical space and 2/3-dealiased.  Time stepping is explicit RK4 with the
viscous term carried in the right-hand side, which is stable for the
``(nu, dt, N)`` combinations used here.  Everything is JIT-compiled and the
time loop uses ``lax.scan``.

This module is standalone (run it to dump an ``.npz``) and importable
(``generate_burgers2d``).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _grids(N: int):
    """Periodic grid on ``[0, 2 pi)^2`` and integer wavenumbers."""
    x = 2.0 * np.pi * np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    k1d = np.fft.fftfreq(N, d=1.0 / N)               # integer wavenumbers
    KX, KY = np.meshgrid(k1d, k1d, indexing="ij")
    return X, Y, KX, KY


@jax.jit
def _rhs(u, v, KX, KY, mask, nu):
    """Right-hand side of the 2D Burgers system (spectral, dealiased)."""
    iKX, iKY = 1j * KX, 1j * KY
    K2 = KX ** 2 + KY ** 2

    u_hat, v_hat = jnp.fft.fft2(u), jnp.fft.fft2(v)
    ux = jnp.real(jnp.fft.ifft2(iKX * u_hat))
    uy = jnp.real(jnp.fft.ifft2(iKY * u_hat))
    vx = jnp.real(jnp.fft.ifft2(iKX * v_hat))
    vy = jnp.real(jnp.fft.ifft2(iKY * v_hat))

    # advective terms, dealiased in spectral space
    adv_u = jnp.real(jnp.fft.ifft2(mask * jnp.fft.fft2(u * ux + v * uy)))
    adv_v = jnp.real(jnp.fft.ifft2(mask * jnp.fft.fft2(u * vx + v * vy)))

    lap_u = jnp.real(jnp.fft.ifft2(-K2 * u_hat))
    lap_v = jnp.real(jnp.fft.ifft2(-K2 * v_hat))
    return -adv_u + nu * lap_u, -adv_v + nu * lap_v


@jax.jit
def _rk4_step(u, v, dt, KX, KY, mask, nu):
    k1u, k1v = _rhs(u, v, KX, KY, mask, nu)
    k2u, k2v = _rhs(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v, KX, KY, mask, nu)
    k3u, k3v = _rhs(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v, KX, KY, mask, nu)
    k4u, k4v = _rhs(u + dt * k3u, v + dt * k3v, KX, KY, mask, nu)
    u = u + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return u, v


def initial_condition(X, Y, c: float, amp: float):
    """Mean flow ``(c, c)`` plus a two-scale vortical perturbation."""
    u0 = c + amp * (np.sin(X) * np.cos(Y)
                    + 0.5 * np.sin(2 * X) * np.cos(3 * Y))
    v0 = c + amp * (-np.cos(X) * np.sin(Y)
                    + 0.4 * np.cos(3 * X) * np.sin(2 * Y))
    return u0, v0


def generate_burgers2d(
    N: int = 128,
    nu: float = 5.0e-2,
    dt: float = 1.0e-3,
    t_end: float = 2.0 * np.pi,
    store_every: int = 8,
    c: float = 1.0,
    amp: float = 1.0,
):
    """Integrate 2D viscous Burgers and return stacked ``[u; v]`` snapshots.

    Arguments
    ---------
    N : int
        Grid points per direction (state dimension ``d = 2 N^2``).
    nu : float
        Kinematic viscosity; ``Re = amp * 2 pi / nu``.
    dt, t_end, store_every :
        Time step, final time (default one full domain traversal at ``c``)
        and snapshot stride.
    c : float
        Uniform mean flow in both directions.
    amp : float
        Amplitude of the vortical perturbation.

    Returns
    -------
    S : (2 N^2, n_snap) float64
        Snapshot matrix; rows ``0:N*N`` are ``u``, rows ``N*N:`` are ``v``.
    info : dict
        Grids, times, field shape and the parameters.
    """
    X, Y, KX, KY = _grids(N)
    KXj, KYj = jnp.asarray(KX), jnp.asarray(KY)
    kmax = N // 3
    mask = jnp.asarray((np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)
                       ).astype(jnp.float64)

    u0, v0 = initial_condition(X, Y, c, amp)
    u, v = jnp.asarray(u0), jnp.asarray(v0)

    n_steps = int(round(t_end / dt))
    n_store = n_steps // store_every

    def scan_body(carry, _):
        u, v = carry

        def inner(cc, __):
            u, v = cc
            return _rk4_step(u, v, dt, KXj, KYj, mask, nu), None

        (u, v), _ = jax.lax.scan(inner, (u, v), None, length=store_every)
        return (u, v), jnp.concatenate([u.reshape(-1), v.reshape(-1)])

    (u, v), snaps = jax.lax.scan(scan_body, (u, v), None, length=n_store)
    S0 = jnp.concatenate([jnp.asarray(u0).reshape(-1),
                          jnp.asarray(v0).reshape(-1)])
    S = jnp.concatenate([S0[None, :], snaps], axis=0).T  # (2 N^2, n_store+1)

    tspan = dt * store_every * np.arange(n_store + 1)
    info = dict(N=N, X=X, Y=Y, tspan=tspan, field_shape=(N, N), dt=dt,
                store_every=store_every, nu=nu, c=c, amp=amp,
                Re=amp * 2.0 * np.pi / nu)
    return np.asarray(S), info


def spectrum_tail(S_col, N: int) -> float:
    """Relative energy above 2/3 of the Nyquist wavenumber (resolution check).

    A well-resolved pseudo-spectral field puts a negligible fraction of its
    energy near the dealiasing cutoff; a large value means the internal
    layers are thinner than the grid can carry.
    """
    NN = N * N
    tail = 0.0
    tot = 0.0
    _, _, KX, KY = _grids(N)
    kmax = N // 3
    hi = (np.abs(KX) > 0.75 * kmax) | (np.abs(KY) > 0.75 * kmax)
    for blk in (S_col[:NN], S_col[NN:]):
        F = np.abs(np.fft.fft2(np.asarray(blk).reshape(N, N))) ** 2
        tail += float(F[hi].sum())
        tot += float(F.sum())
    return tail / tot


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "burgers2d.npz"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    nu = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0e-2
    S, info = generate_burgers2d(N=N, nu=nu)
    NN = N * N
    print(f"S shape {S.shape}, finite={np.isfinite(S).all()}, Re={info['Re']:.1f}")
    print(f"u range [{S[:NN].min():.3f}, {S[:NN].max():.3f}]  "
          f"v range [{S[NN:].min():.3f}, {S[NN:].max():.3f}]")
    print(f"spectral tail (final): {spectrum_tail(S[:, -1], N):.2e}")
    np.savez_compressed(out, S=S, **{k: v for k, v in info.items()})
    print(f"saved {out}")
