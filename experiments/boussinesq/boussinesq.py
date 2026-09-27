"""Fourier pseudo-spectral solver for the 2D Boussinesq equations (rising thermal).

Solves, on the periodic rectangle ``[0, Lx) x [0, Ly)``, the incompressible
Boussinesq equations in vorticity-streamfunction form

    w_t  + u w_x  + v w_y  = nu    (w_xx  + w_yy)  + theta_x ,
    th_t + u th_x + v th_y = kappa (th_xx + th_yy) ,
    u = psi_y ,  v = -psi_x ,  -(psi_xx + psi_yy) = w ,

started from a quiescent fluid with a hot Gaussian blob

    theta0 = exp(-((x - x0)^2 + (y - y0)^2) / R^2) .

Non-dimensionalization
----------------------
Lengths are scaled by the blob radius ``R`` and velocities by the buoyancy
velocity ``U = sqrt(g alpha dtheta R)``, so the buoyancy term is ``+theta_x``
with unit coefficient and the only free parameters are

    Re = U R / nu ,   Pr = nu / kappa      (Ra = Re^2 Pr) .

The blob accelerates upward, rolls up into a mushroom cap (a buoyant vortex
pair) that thins its own stem and, as ``Re`` grows, sheds secondary rolls
along the cap and stem.  The reduced coordinates therefore combine a long
transport (the cap rises many radii) with roll-up structure that sharpens
with ``Re`` -- a slowly decaying Kolmogorov n-width whose shape changes
with the parameter.

Why the vorticity form on a periodic box
----------------------------------------
Vorticity-streamfunction eliminates the pressure exactly, so there is no
Leray projection and no pressure Poisson solve; the streamfunction is one
spectral division.  The ``k = 0`` velocity mode is not represented, which
is the same as saying the mean buoyancy ``<theta>`` is balanced by a
hydrostatic pressure gradient (only ``theta_x`` enters the vorticity
equation, and it kills the mean).  Physically the box is a frame in which
the fluid at large has no net vertical momentum.

Numerics
--------
Spectral derivatives, products in physical space with 2/3 dealiasing, and
the two diffusion terms integrated exactly by an integrating factor
(``exp(-nu k^2 dt)``) inside a classical RK4 stage structure (IF-RK4).  The
time step is set by the advective CFL alone; diffusion never restricts it.
Everything is JIT-compiled and the time loop is a ``lax.scan``.

This module is standalone (run it to dump an ``.npz``) and importable
(:func:`generate_boussinesq`).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


# ---- grids -------------------------------------------------------------------


def grids(Nx: int, Ny: int, Lx: float, Ly: float):
    """Periodic grid on ``[0, Lx) x [0, Ly)`` and angular wavenumbers.

    Returns ``X, Y`` (shape ``(Nx, Ny)``, ``ij`` indexing) and ``KX, KY``.
    """
    x = Lx * np.arange(Nx) / Nx
    y = Ly * np.arange(Ny) / Ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=Lx / Nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=Ly / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return X, Y, KX, KY


def dealias_mask(Nx: int, Ny: int) -> np.ndarray:
    """2/3-rule mask (1 inside ``|k| <= N/3`` in each direction, 0 outside)."""
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return ((np.abs(KX) <= Nx // 3) & (np.abs(KY) <= Ny // 3)).astype(np.float64)


class Operators:
    """Spectral operators for one grid, held as JAX arrays."""

    def __init__(self, Nx, Ny, Lx, Ly):
        X, Y, KX, KY = grids(Nx, Ny, Lx, Ly)
        self.X, self.Y = X, Y
        self.KX = jnp.asarray(KX)
        self.KY = jnp.asarray(KY)
        K2 = KX ** 2 + KY ** 2
        self.K2 = jnp.asarray(K2)
        K2_inv = np.zeros_like(K2)
        K2_inv[K2 > 0] = 1.0 / K2[K2 > 0]        # psi_hat[0, 0] = 0
        self.K2_inv = jnp.asarray(K2_inv)
        self.mask = jnp.asarray(dealias_mask(Nx, Ny))


# ---- right-hand side ---------------------------------------------------------


def _velocity(w_hat, KX, KY, K2_inv):
    """``(u, v)`` in physical space from the vorticity spectrum."""
    psi_hat = w_hat * K2_inv
    u = jnp.real(jnp.fft.ifft2(1j * KY * psi_hat))
    v = jnp.real(jnp.fft.ifft2(-1j * KX * psi_hat))
    return u, v


@jax.jit
def _nonlinear(w_hat, th_hat, KX, KY, K2_inv, mask):
    """Advection (dealiased) plus buoyancy, in spectral space.

    Returns ``(N_w, N_th)`` with ``N_w = -F[u.grad w] + i kx th_hat`` and
    ``N_th = -F[u.grad th]``.
    """
    u, v = _velocity(w_hat, KX, KY, K2_inv)
    iKX, iKY = 1j * KX, 1j * KY
    wx = jnp.real(jnp.fft.ifft2(iKX * w_hat))
    wy = jnp.real(jnp.fft.ifft2(iKY * w_hat))
    thx = jnp.real(jnp.fft.ifft2(iKX * th_hat))
    thy = jnp.real(jnp.fft.ifft2(iKY * th_hat))
    N_w = -mask * jnp.fft.fft2(u * wx + v * wy) + iKX * th_hat
    N_th = -mask * jnp.fft.fft2(u * thx + v * thy)
    return N_w, N_th


@jax.jit
def _ifrk4_step(w_hat, th_hat, Ew, Ew2, Et, Et2, dt, KX, KY, K2_inv, mask):
    """One integrating-factor RK4 step.

    ``Ew = exp(-nu k^2 dt/2)``, ``Ew2 = Ew^2`` (and ``Et``, ``Et2`` for the
    temperature).  With the diffusion factored out the remaining ODE is
    integrated by classical RK4; the stage values are propagated by the
    half- and full-step exponentials.
    """
    aw, at = _nonlinear(w_hat, th_hat, KX, KY, K2_inv, mask)
    w1 = Ew * (w_hat + 0.5 * dt * aw)
    t1 = Et * (th_hat + 0.5 * dt * at)
    bw, bt = _nonlinear(w1, t1, KX, KY, K2_inv, mask)
    w2 = Ew * w_hat + 0.5 * dt * bw
    t2 = Et * th_hat + 0.5 * dt * bt
    cw, ct = _nonlinear(w2, t2, KX, KY, K2_inv, mask)
    w3 = Ew2 * w_hat + dt * Ew * cw
    t3 = Et2 * th_hat + dt * Et * ct
    dw, dt_ = _nonlinear(w3, t3, KX, KY, K2_inv, mask)
    w_new = Ew2 * w_hat + (dt / 6.0) * (Ew2 * aw + 2.0 * Ew * (bw + cw) + dw)
    th_new = Et2 * th_hat + (dt / 6.0) * (Et2 * at + 2.0 * Et * (bt + ct) + dt_)
    return w_new, th_new


# ---- initial condition and driver --------------------------------------------


def initial_temperature(X, Y, x0: float, y0: float, R: float = 1.0):
    """Gaussian hot blob of radius ``R`` centred at ``(x0, y0)``."""
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / R ** 2)


def generate_boussinesq(
    Nx: int = 256,
    Ny: int = 512,
    Lx: float = 8.0,
    Ly: float = 16.0,
    Re: float = 1000.0,
    Pr: float = 1.0,
    dt: float = 5.0e-3,
    t_end: float = 10.0,
    store_every: int = 20,
    y0: float = 3.0,
    R: float = 1.0,
):
    """Integrate the rising thermal and return stacked ``[u; v; theta]`` snapshots.

    Arguments
    ---------
    Nx, Ny : int
        Grid points; the state dimension is ``d = 3 Nx Ny``.
    Lx, Ly : float
        Box size in blob radii (``Ly / Ny`` should equal ``Lx / Nx``).
    Re, Pr : float
        Reynolds number ``U R / nu`` and Prandtl number ``nu / kappa``.
    dt, t_end, store_every :
        Time step, final time and snapshot stride.
    y0, R : float
        Initial height and radius of the blob (centred in ``x``).

    Returns
    -------
    S : (3 Nx Ny, n_snap) float64
        Snapshot matrix; row blocks ``[0:NN]``, ``[NN:2NN]``, ``[2NN:]`` are
        ``u``, ``v``, ``theta`` with ``NN = Nx Ny``, each flattened in
        C order from shape ``(Nx, Ny)``.
    info : dict
        Grids, times, field shape and the parameters.
    """
    ops = Operators(Nx, Ny, Lx, Ly)
    nu = 1.0 / Re
    kappa = nu / Pr

    th0 = initial_temperature(ops.X, ops.Y, 0.5 * Lx, y0, R)
    th_hat = jnp.fft.fft2(jnp.asarray(th0))
    w_hat = jnp.zeros_like(th_hat)

    Ew = jnp.exp(-nu * ops.K2 * 0.5 * dt)
    Et = jnp.exp(-kappa * ops.K2 * 0.5 * dt)
    Ew2, Et2 = Ew * Ew, Et * Et

    n_steps = int(round(t_end / dt))
    n_store = n_steps // store_every
    NN = Nx * Ny

    def fields(w_hat, th_hat):
        u, v = _velocity(w_hat, ops.KX, ops.KY, ops.K2_inv)
        th = jnp.real(jnp.fft.ifft2(th_hat))
        return jnp.concatenate([u.reshape(-1), v.reshape(-1), th.reshape(-1)])

    def scan_body(carry, _):
        w_hat, th_hat = carry

        def inner(cc, __):
            return _ifrk4_step(*cc, Ew, Ew2, Et, Et2, dt, ops.KX, ops.KY,
                               ops.K2_inv, ops.mask), None

        (w_hat, th_hat), _ = jax.lax.scan(inner, (w_hat, th_hat), None,
                                          length=store_every)
        return (w_hat, th_hat), fields(w_hat, th_hat)

    (w_hat, th_hat), snaps = jax.lax.scan(scan_body, (w_hat, th_hat), None,
                                          length=n_store)
    S0 = fields(jnp.zeros_like(th_hat), jnp.fft.fft2(jnp.asarray(th0)))
    S = jnp.concatenate([S0[None, :], snaps], axis=0).T   # (3 NN, n_store+1)

    tspan = dt * store_every * np.arange(n_store + 1)
    info = dict(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, X=ops.X, Y=ops.Y, tspan=tspan,
                field_shape=(Nx, Ny), dt=dt, store_every=store_every,
                Re=Re, Pr=Pr, nu=nu, kappa=kappa, y0=y0, R=R)
    return np.asarray(S), info


# ---- helpers for analysis ------------------------------------------------------


def split_fields(S_col: np.ndarray, Nx: int, Ny: int):
    """``(u, v, theta)`` as ``(Nx, Ny)`` arrays from one snapshot column."""
    NN = Nx * Ny
    return (np.asarray(S_col[:NN]).reshape(Nx, Ny),
            np.asarray(S_col[NN:2 * NN]).reshape(Nx, Ny),
            np.asarray(S_col[2 * NN:]).reshape(Nx, Ny))


def vorticity(u: np.ndarray, v: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """``w = v_x - u_y`` by spectral differentiation."""
    Nx, Ny = u.shape
    _, _, KX, KY = grids(Nx, Ny, Lx, Ly)
    vx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v)))
    uy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u)))
    return vx - uy


def spectrum_tail(field: np.ndarray, frac: float = 0.75) -> float:
    """Relative energy above ``frac`` of the dealiasing cutoff (resolution check).

    A well-resolved pseudo-spectral field puts a negligible fraction of its
    energy near the 2/3 cutoff; a large value means the thinnest structures
    are not carried by the grid.
    """
    Nx, Ny = field.shape
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx)
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    hi = (np.abs(KX) > frac * (Nx // 3)) | (np.abs(KY) > frac * (Ny // 3))
    F = np.abs(np.fft.fft2(field)) ** 2
    tot = F.sum()
    return float(F[hi].sum() / tot) if tot > 0 else 0.0


def diagnostics(S: np.ndarray, info: dict) -> dict:
    """Integral diagnostics along the trajectory (all as arrays over time).

    ``mean_theta`` should be conserved to round-off, ``theta_var`` must
    decrease monotonically, ``KE`` follows the budget
    ``d/dt KE = <theta v> - nu <w^2>``, ``y_c`` is the theta-weighted
    centroid height (the thermal's rise curve).
    """
    Nx, Ny, Lx, Ly = info["Nx"], info["Ny"], info["Lx"], info["Ly"]
    dA = (Lx / Nx) * (Ly / Ny)
    Y = info["Y"]
    out = {k: [] for k in ("mean_theta", "theta_var", "KE", "enstrophy",
                           "buoyancy_flux", "y_c", "max_speed",
                           "tail_theta", "tail_w", "asym")}
    for j in range(S.shape[1]):
        u, v, th = split_fields(S[:, j], Nx, Ny)
        w = vorticity(u, v, Lx, Ly)
        out["mean_theta"].append(th.sum() * dA)
        out["theta_var"].append((th ** 2).sum() * dA)
        out["KE"].append(0.5 * (u ** 2 + v ** 2).sum() * dA)
        out["enstrophy"].append((w ** 2).sum() * dA)
        out["buoyancy_flux"].append((th * v).sum() * dA)
        out["y_c"].append((th * Y).sum() / th.sum())
        out["max_speed"].append(np.sqrt(u ** 2 + v ** 2).max())
        out["tail_theta"].append(spectrum_tail(th))
        out["tail_w"].append(spectrum_tail(w))
        # mirror symmetry about x = Lx/2: theta(x) = theta(Lx - x)
        out["asym"].append(np.abs(th - np.roll(th[::-1], 1, axis=0)).max()
                           / np.abs(th).max())
    return {k: np.asarray(v) for k, v in out.items()}


if __name__ == "__main__":
    import sys
    import time

    out = sys.argv[1] if len(sys.argv) > 1 else "boussinesq.npz"
    Re = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
    Nx = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    t0 = time.time()
    S, info = generate_boussinesq(Nx=Nx, Ny=2 * Nx, Re=Re)
    print(f"S shape {S.shape}, finite={np.isfinite(S).all()}, "
          f"Re={info['Re']:.0f}, wall {time.time() - t0:.1f} s")
    d = diagnostics(S[:, ::10], info)
    print(f"mean theta drift {abs(d['mean_theta'][-1] / d['mean_theta'][0] - 1):.2e}, "
          f"final tail theta {d['tail_theta'][-1]:.2e}, w {d['tail_w'][-1]:.2e}, "
          f"asym {d['asym'].max():.2e}")
    np.savez_compressed(out, S=S, **{k: v for k, v in info.items()})
    print(f"saved {out}")
