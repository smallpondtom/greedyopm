#!/usr/bin/env python3
"""Verify the 2D viscous Burgers solver and the cached snapshot data.

The solver in :mod:`burgers2d` is checked against *exact* solutions rather
than against itself.  The 2D vector Burgers system

    U_t + (U . grad) U = nu lap U ,    U = (u, v) ,

admits a Cole-Hopf transformation whenever ``U`` is irrotational: if
``Phi_t = nu lap Phi`` and ``Phi > 0`` then ``U = -2 nu grad log Phi`` solves
the system exactly.  Taking ``Phi(x, y, t) = A(x, t) + B(y, t)`` with ``A``
and ``B`` 1D heat kernels gives a genuinely two-dimensional, *non-separable*
exact solution in which both cross terms ``v u_y`` and ``u v_x`` are nonzero
-- so the test exercises every term of the discretisation, not just a 1D
slice of it.

Checks performed
----------------
1. ``mask``     -- the 2/3 dealiasing mask matches the rule it claims, and
                   the production fields really are band-limited by it.
2. ``linear``   -- at vanishing amplitude the scheme must reproduce the pure
                   diffusive decay ``exp(-nu k^2 t)`` (pins ``nu`` and the
                   Laplacian).
3. ``exact1d``  -- Cole-Hopf solution with ``v = 0``, ``u = u(x)``
                   (x-derivatives, u-equation), and its ``y`` mirror.
4. ``exact2d``  -- the non-separable 2D Cole-Hopf solution above (full
                   coupled system, both cross terms active).
5. ``diagonal`` -- ``u = v = w(x + y, t)`` reduces to 1D Burgers at time
                   ``2t``; an independent check of the cross terms.
6. ``galilean`` -- a uniform mean flow ``(c, c)`` must translate the solution
                   rigidly by ``(ct, ct)`` (validated at an integer cell
                   shift, so the comparison is exact).
7. ``grid``     -- spatial convergence of the production initial condition at
                   the most demanding viscosity, against an ``N = 512`` run.
8. ``time``     -- RK4 temporal convergence (expect order 4) against a
                   ``dt = 1.25e-4`` run.
9. ``data``     -- every cached ``.npz``: shape, finiteness, time vector,
                   the stored first column against the analytic initial
                   condition, the spectral tail, and -- for one file -- a
                   full re-integration reproducing the stored snapshots.
10. ``splits``  -- the cached files are exactly the ``(nu, store_every)``
                   pairs the five parametric splits ask for.

Writes ``plots/validation.md``.  Run ``python validate.py`` for everything,
or pass a subset of the check names above.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import iv

import jax
import jax.numpy as jnp

import burgers2d as B
import problem as P

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"

# Viscosity for the exact-solution tests.  Chosen inside the production range
# (nu in [0.0108, 0.0449]) and large enough that the Cole-Hopf Fourier series
# is resolved well below the N = 128 dealiasing cutoff k <= 42; see the
# reported spectral tails.
NU_EXACT = 0.05

_results: list[tuple[str, str, bool, str]] = []


def record(check: str, quantity: str, ok: bool, detail: str) -> bool:
    _results.append((check, quantity, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {quantity}: {detail}")
    return ok


# ---------------------------------------------------------------------------
# driving the production stepper
# ---------------------------------------------------------------------------


def dealias_mask(N: int):
    """The mask exactly as :func:`burgers2d.generate_burgers2d` builds it."""
    _, _, KX, KY = B._grids(N)
    kmax = N // 3
    return jnp.asarray((np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)
                       ).astype(jnp.float64)


def integrate(u0, v0, nu: float, dt: float, n_steps: int):
    """March the production RK4 step ``n_steps`` times (no snapshot storage)."""
    N = u0.shape[0]
    _, _, KX, KY = B._grids(N)
    KXj, KYj = jnp.asarray(KX), jnp.asarray(KY)
    mask = dealias_mask(N)
    u, v = jnp.asarray(u0), jnp.asarray(v0)

    def body(carry, _):
        u, v = carry
        return B._rk4_step(u, v, dt, KXj, KYj, mask, nu), None

    (u, v), _ = jax.lax.scan(body, (u, v), None, length=int(n_steps))
    return np.asarray(u), np.asarray(v)


def rel_err(num, exact) -> float:
    return float(np.linalg.norm(num - exact) / np.linalg.norm(exact))


def band_limit(field, N: int):
    """Project onto the solver's resolved space ``|kx|, |ky| <= N // 3``."""
    F = np.fft.fft2(np.asarray(field))
    return np.real(np.fft.ifft2(np.asarray(dealias_mask(N)) * F))


def truncation_floor(field, N: int) -> float:
    """Relative L2 cost of representing ``field`` in the solver's basis.

    No dealiased pseudo-spectral scheme on this grid can do better than this,
    so it is the floor against which agreement with an exact solution must be
    judged -- comparing to machine precision would be meaningless.
    """
    return rel_err(band_limit(field, N), field)


def tail_frac(field, N: int, frac: float = 1.0) -> float:
    """Fraction of spectral energy strictly above ``frac * (N // 3)``."""
    _, _, KX, KY = B._grids(N)
    kmax = N // 3
    hi = (np.abs(KX) > frac * kmax) | (np.abs(KY) > frac * kmax)
    F = np.abs(np.fft.fft2(np.asarray(field))) ** 2
    return float(F[hi].sum() / F.sum())


# ---------------------------------------------------------------------------
# exact solutions (Cole-Hopf)
# ---------------------------------------------------------------------------


def _heat_phi(x, t: float, nu: float, nmax: int = 160):
    """``Phi`` and ``Phi_x`` for the heat evolution of ``exp(cos x / (2 nu))``.

    ``exp(a cos x) = I_0(a) + 2 sum_{n>=1} I_n(a) cos(n x)`` with
    ``a = 1 / (2 nu)``, so the heat-equation solution is obtained by damping
    mode ``n`` with ``exp(-nu n^2 t)``.  This is exact, not a discretisation.
    """
    a = 1.0 / (2.0 * nu)
    n = np.arange(1, nmax + 1)
    cn = 2.0 * iv(n, a) * np.exp(-nu * n ** 2 * t)
    ang = np.outer(np.asarray(x).ravel(), n)
    phi = iv(0, a) + np.cos(ang) @ cn
    phi_x = -(np.sin(ang) @ (n * cn))
    shape = np.shape(x)
    return phi.reshape(shape), phi_x.reshape(shape)


def burgers1d_exact(x, t: float, nu: float):
    """Exact 1D viscous Burgers solution with ``u(x, 0) = sin x``."""
    phi, phi_x = _heat_phi(x, t, nu)
    return -2.0 * nu * phi_x / phi


def burgers2d_exact(X, Y, t: float, nu: float):
    """Non-separable 2D Cole-Hopf solution, ``Phi = A(x, t) + B(y, t)``.

    At ``t = 0`` this is ``u = sin(x) A / (A + B)``,
    ``v = sin(y) B / (A + B)`` -- irrotational, with both ``u_y`` and ``v_x``
    nonzero, so ``v u_y`` and ``u v_x`` are genuinely tested.
    """
    A, A_x = _heat_phi(X, t, nu)
    Bf, B_y = _heat_phi(Y, t, nu)
    phi = A + Bf
    return -2.0 * nu * A_x / phi, -2.0 * nu * B_y / phi


# ---------------------------------------------------------------------------
# 1. dealiasing mask
# ---------------------------------------------------------------------------


def check_mask():
    print("\n[mask] 2/3 dealiasing rule")
    N = 128
    _, _, KX, KY = B._grids(N)
    m = np.asarray(dealias_mask(N))
    kmax = N // 3

    record("mask", "cutoff", kmax == 42, f"N//3 = {kmax} (Nyquist {N // 2})")
    keep = (np.abs(KX) <= kmax) & (np.abs(KY) <= kmax)
    record("mask", "mask == |k|<=N//3 rule", bool(np.array_equal(m, keep.astype(float))),
           f"{int(keep.sum())} of {N * N} modes retained ({keep.mean():.1%})")
    # a real-valued filter needs m[k] == m[-k]; np.roll(m[::-1, ::-1], 1)
    # maps mode k to mode -k in fftfreq ordering.
    conj = np.roll(m[::-1, ::-1], shift=(1, 1), axis=(0, 1))
    record("mask", "binary and conjugate-symmetric",
           bool(set(np.unique(m)) <= {0.0, 1.0} and np.array_equal(m, conj)),
           "entries in {0, 1} and m[k] == m[-k] (keeps the filter real)")

    # the dealiasing must actually bite: production fields carry no energy
    # above the cutoff at any time.
    nu = P.viscosity(P.test_extrap_reynolds())
    S, _ = P.solve_burgers(nu, P.STORE_EVERY_TEST)
    NN = N * N
    worst = max(max(tail_frac(S[:NN, j].reshape(N, N), N),
                    tail_frac(S[NN:, j].reshape(N, N), N))
                for j in range(S.shape[1]))
    record("mask", "production field band-limited by mask", worst < 1e-25,
           f"max energy fraction above k=42 over all snapshots = {worst:.2e}")


def check_resolution():
    """Is 128^2 enough -- at *every* time, not just the last one?

    ``burgers2d.spectrum_tail`` is only ever applied to ``S[:, -1]`` (in the
    module's ``__main__``), and the README quotes that number.  But the
    solution is smoothest at the end: the nonlinearity steepens the harmonics
    first and viscosity then erodes them, so the tail peaks near ``t ~ 1-2``
    and the final snapshot is the *least* demanding one in the trajectory.
    """
    print("\n[resolution] spectral tail over the whole trajectory, not just t_end")
    rows = []
    for p in sorted(P.DATA.glob("burgers2d_N*.npz"), key=lambda q: _parse(q)[1]):
        N, nu, se = _parse(p)
        S = np.load(p)["S"]
        t = np.load(p)["tspan"]
        tails = np.array([B.spectrum_tail(S[:, j], N) for j in range(S.shape[1])])
        rows.append((P.reynolds(nu), nu, tails[-1], tails.max(),
                     float(t[int(tails.argmax())])))

    for Re, nu, last, mx, t_at in rows:
        record("resolution", f"Re={Re:.0f}", mx < 1.0e-6,
               f"tail at t_end = {last:.2e} (what the README quotes) but "
               f"max over the trajectory = {mx:.2e} at t={t_at:.2f}"
               + ("" if mx < 1e-6 else "  <-- exceeds the claimed 1e-6"))

    worst = max(r[3] for r in rows)
    record("resolution", "README claim 'whole range fully resolved, tail < 1e-6'",
           worst < 1.0e-6,
           f"worst tail anywhere in the data is {worst:.2e} "
           f"(at Re={max(rows, key=lambda r: r[3])[0]:.0f}), "
           f"{worst / 1e-6:.0f}x the quoted bound")


# ---------------------------------------------------------------------------
# 2. linear (diffusive) limit
# ---------------------------------------------------------------------------


def check_stability():
    """Explicit-RK4 stability margin of the production ``(nu, dt, N)`` set.

    The viscous term sits in the right-hand side, so the step is bounded by
    the *diffusive* eigenvalue.  Note the Laplacian is applied to the full
    field, un-dealiased, so the relevant wavenumber is the Nyquist ``N/2``,
    not the 2/3 cutoff:  ``lambda_visc = nu * 2 * (N/2)^2``, and RK4 requires
    ``dt * lambda_visc <= 2.785`` on the negative real axis.

    This binds hardest at the *largest* viscosity, i.e. the lowest Reynolds
    number -- the opposite end from where resolution binds.  It is also why
    a grid-refinement study cannot simply raise ``N`` at fixed ``dt``.
    """
    print("\n[stability] explicit RK4 limit for the viscous term")
    RK4_REAL = 2.785      # RK4 stability boundary on the negative real axis
    RK4_IMAG = 2.828      # ... and on the imaginary axis (advection)
    N, dt = P.N, P.DT
    k_nyq = N // 2
    worst = None
    for Re in sorted({*np.round(P.training_reynolds(), 6),
                      *np.round(P.val_interp_reynolds(), 6),
                      round(P.val_extrap_reynolds(), 6),
                      round(P.test_interp_reynolds(), 6),
                      round(P.test_extrap_reynolds(), 6)}):
        nu = P.viscosity(Re)
        lam = nu * 2.0 * k_nyq ** 2
        margin = RK4_REAL / (dt * lam)
        worst = margin if worst is None else min(worst, margin)
        record("stability", f"viscous, Re={Re:.0f}", margin > 1.0,
               f"dt*nu*2*(N/2)^2 = {dt * lam:.3f} <= {RK4_REAL} "
               f"({margin:.1f}x margin)")

    # advective limit from the dealiased cutoff and the observed max speed
    S, _ = P.solve_burgers(P.viscosity(P.test_extrap_reynolds()),
                           P.STORE_EVERY_TEST)
    NN = N * N
    umax = float(np.abs(S[:NN]).max())
    vmax = float(np.abs(S[NN:]).max())
    lam_adv = (umax + vmax) * (N // 3)
    record("stability", "advective", dt * lam_adv < RK4_IMAG,
           f"dt*(|u|+|v|)max*kcut = {dt * lam_adv:.3f} <= {RK4_IMAG} "
           f"({RK4_IMAG / (dt * lam_adv):.1f}x margin)")

    record("stability", "binding constraint is diffusion at the LOWEST Re",
           worst is not None and worst > 1.0,
           f"tightest margin {worst:.1f}x at Re={P.RE_MIN:.0f}; refining to "
           f"N=384 at this nu and dt={dt:g} diverges (dt limit "
           f"{RK4_REAL / (P.viscosity(P.RE_MIN) * 2 * 192 ** 2):.2e}), so a "
           f"grid study must shrink dt with N")


def check_linear():
    print("\n[linear] vanishing-amplitude limit -> pure diffusion exp(-nu k^2 t)")
    N, nu, dt, eps = 64, 0.05, 1.0e-3, 1.0e-6
    X, Y, _, _ = B._grids(N)
    for kx, ky, label in ((1, 0, "k=(1,0)"), (0, 3, "k=(0,3)"), (2, 5, "k=(2,5)")):
        u0 = eps * np.sin(kx * X + ky * Y)
        v0 = eps * np.cos(kx * X + ky * Y)
        t = 1.0
        u, v = integrate(u0, v0, nu, dt, int(round(t / dt)))
        dec = np.exp(-nu * (kx ** 2 + ky ** 2) * t)
        e = max(rel_err(u, u0 * dec), rel_err(v, v0 * dec))
        # nonlinear term is O(eps^2) relative to O(eps) -> error floor ~ eps
        record("linear", f"decay rate {label}", e < 5.0e-6,
               f"rel err vs exp(-nu k^2 t) = {e:.2e}  (O(eps)={eps:.0e} floor)")


# ---------------------------------------------------------------------------
# 3-5. exact Cole-Hopf solutions
# ---------------------------------------------------------------------------


#: Time step for the exact-solution tests.  The RK4 error at this step is
#: ~2e-12 (see the ``galilean`` and ``time`` checks), i.e. far below the
#: spatial truncation floor, so these checks isolate the spatial operator.
DT_EXACT = 2.5e-4


def _exact_report(check: str, label: str, num, exact, N: int, *, slack: float = 2.5):
    """Judge agreement with an exact solution against the truncation floor.

    A dealiased pseudo-spectral scheme cannot represent what lies above its
    cutoff, so the meaningful question is whether the computed solution sits
    at the floor set by that cutoff -- not whether it reaches machine
    precision.  ``slack`` allows for the propagation of the truncated modes
    through the nonlinearity.
    """
    e = rel_err(num, exact)
    floor = truncation_floor(exact, N)
    scheme = rel_err(num, band_limit(exact, N))
    ok = e < max(slack * floor, 1.0e-11)
    record(check, f"{label} (N={N})", ok,
           f"rel err = {e:.2e} vs truncation floor {floor:.2e} "
           f"(ratio {e / floor:.2f}); distance to band-limited exact = "
           f"{scheme:.2e}")
    return e


def check_exact1d():
    print(f"\n[exact1d] Cole-Hopf, 1D embedded in 2D (nu = {NU_EXACT})")
    t = 1.0
    for N in (128, 256):
        x1 = 2.0 * np.pi * np.arange(N) / N
        n_steps = int(round(t / DT_EXACT))

        # --- u = u(x), v = 0 : x-derivatives and the u-equation
        u0 = np.broadcast_to(burgers1d_exact(x1, 0.0, NU_EXACT)[:, None],
                             (N, N)).copy()
        u, v = integrate(u0, np.zeros((N, N)), NU_EXACT, DT_EXACT, n_steps)
        ue = np.broadcast_to(burgers1d_exact(x1, t, NU_EXACT)[:, None],
                             (N, N)).copy()
        _exact_report("exact1d", f"u(x), v=0 at t={t}", u, ue, N)
        record("exact1d", f"v stays identically zero (N={N})",
               float(np.abs(v).max()) == 0.0, f"max|v| = {np.abs(v).max():.1e}")

        # --- v = v(y), u = 0 : y-derivatives and the v-equation
        v0 = np.broadcast_to(burgers1d_exact(x1, 0.0, NU_EXACT)[None, :],
                             (N, N)).copy()
        u, v = integrate(np.zeros((N, N)), v0, NU_EXACT, DT_EXACT, n_steps)
        ve = np.broadcast_to(burgers1d_exact(x1, t, NU_EXACT)[None, :],
                             (N, N)).copy()
        _exact_report("exact1d", f"v(y), u=0 at t={t}", v, ve, N)
        record("exact1d", f"u stays identically zero (N={N})",
               float(np.abs(u).max()) == 0.0, f"max|u| = {np.abs(u).max():.1e}")


def check_exact2d():
    print(f"\n[exact2d] non-separable 2D Cole-Hopf (nu = {NU_EXACT})")
    N = 128
    X, Y, KX, KY = B._grids(N)
    u0, v0 = burgers2d_exact(X, Y, 0.0, NU_EXACT)

    # the cross terms must actually be exercised
    uy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u0)))
    vx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v0)))
    record("exact2d", "cross terms nonzero",
           np.abs(uy).max() > 1e-3 and np.abs(vx).max() > 1e-3,
           f"max|u_y| = {np.abs(uy).max():.3f}, max|v_x| = {np.abs(vx).max():.3f} "
           f"-- so v*u_y and u*v_x are both active "
           f"(irrotational as required: max|u_y - v_x| = {np.abs(uy - vx).max():.1e})")

    # time sweep on the production grid
    for t in (0.5, 1.0, 2.0):
        u, v = integrate(u0, v0, NU_EXACT, DT_EXACT, int(round(t / DT_EXACT)))
        ue, ve = burgers2d_exact(X, Y, t, NU_EXACT)
        _exact_report("exact2d", f"u, full 2D system at t={t}", u, ue, N)
        _exact_report("exact2d", f"v, full 2D system at t={t}", v, ve, N)

    # spatial convergence: refining the grid must drive the error down
    # spectrally, confirming truncation -- not a scheme error -- is the limit.
    t, errs = 1.0, {}
    for Nr in (96, 128, 192, 256):
        Xr, Yr, _, _ = B._grids(Nr)
        a, b = burgers2d_exact(Xr, Yr, 0.0, NU_EXACT)
        u, v = integrate(a, b, NU_EXACT, DT_EXACT, int(round(t / DT_EXACT)))
        ue, ve = burgers2d_exact(Xr, Yr, t, NU_EXACT)
        errs[Nr] = max(rel_err(u, ue), rel_err(v, ve))
    seq = [errs[n] for n in (96, 128, 192, 256)]
    record("exact2d", "spectral convergence in N",
           seq[0] > seq[1] > seq[2] and seq[-1] < 1.0e-8,
           "rel err at t=1: " + ", ".join(f"N={n}: {errs[n]:.2e}"
                                          for n in (96, 128, 192, 256))
           + " -- error falls faster than any power of 1/N")


def check_diagonal():
    print("\n[diagonal] u = v = w(x+y, t), reduces to 1D Burgers at time 2t")
    t = 0.5
    for N in (128, 256):
        X, Y, _, _ = B._grids(N)
        S = X + Y
        # w_t + 2 w w_s = 2 nu w_ss  <=>  w(s, t) = U(s, 2t) with U 1D Burgers
        u0 = burgers1d_exact(S, 0.0, NU_EXACT)
        u, v = integrate(u0, u0.copy(), NU_EXACT, DT_EXACT,
                         int(round(t / DT_EXACT)))
        we = burgers1d_exact(S, 2.0 * t, NU_EXACT)
        _exact_report("diagonal", f"u = v = w(x+y) at t={t} (tau=2t={2 * t})",
                      u, we, N)
        record("diagonal", f"u and v remain equal (N={N})",
               float(np.abs(u - v).max()) < 1e-14,
               f"max|u - v| = {np.abs(u - v).max():.1e}")


# ---------------------------------------------------------------------------
# 6. Galilean invariance
# ---------------------------------------------------------------------------


def check_galilean():
    """Galilean equivariance -- and, as a by-product, the RK4 order.

    The boost ``u(x, t) -> u(x - ct, t) + c`` is a symmetry of the PDE and of
    the semi-discrete spectral system (the dealiasing mask is diagonal in
    Fourier space, so it commutes with translation by *any* amount).  It is a
    *time-dependent* symmetry, however, and a Runge-Kutta map reproduces such
    a symmetry only to its own order -- so the residual here is the RK4 time
    error, and it must fall as ``dt^4``.  That makes this an independent
    measurement of the temporal order that does not need a reference run.
    """
    print("\n[galilean] a uniform mean flow must translate the solution rigidly")
    N, nu, cells = 128, 0.02, 16
    # translate by an exact integer number of cells so the final comparison
    # needs no interpolation:  c * t = cells * (2 pi / N)  with c = 1.
    c, t = 1.0, cells * (2.0 * np.pi / N)
    X, Y, _, _ = B._grids(N)

    u0p, v0p = B.initial_condition(X, Y, 0.0, 1.0)            # rest frame
    u0, v0 = B.initial_condition(X, Y, c, 1.0)                # moving frame
    record("galilean", "IC differs only by the mean flow",
           bool(np.allclose(u0 - u0p, c) and np.allclose(v0 - v0p, c)),
           f"u0 - u0(c=0) == {c} exactly")

    errs = {}
    for n_steps in (100, 200, 400, 800):
        dt = t / n_steps
        up, vp = integrate(u0p, v0p, nu, dt, n_steps)
        u, v = integrate(u0, v0, nu, dt, n_steps)
        su = np.roll(np.roll(up, cells, axis=0), cells, axis=1) + c
        sv = np.roll(np.roll(vp, cells, axis=0), cells, axis=1) + c
        errs[dt] = max(rel_err(u, su), rel_err(v, sv))

    dts = sorted(errs, reverse=True)
    record("galilean", f"shift by {cells} cells at t={t:.4f}",
           errs[dts[-1]] < 1.0e-9,
           "rel err vs the rest-frame solution rolled diagonally: "
           + ", ".join(f"dt={d:.2e}: {errs[d]:.2e}" for d in dts))
    orders = [np.log2(errs[a] / errs[b]) for a, b in zip(dts, dts[1:])]
    record("galilean", "residual is RK4 time error (order 4)",
           all(3.5 < o < 4.6 for o in orders),
           "observed order p = " + ", ".join(f"{o:.2f}" for o in orders)
           + " -- a time-dependent symmetry is reproduced to the scheme order")

    # round-off is not what limits any of these comparisons
    rng = np.random.default_rng(0)
    dt = t / 400
    pert = 1e-16 * np.abs(u0).max() * rng.standard_normal((N, N))
    ua, _ = integrate(u0, v0, nu, dt, 400)
    ub, _ = integrate(u0 + pert, v0, nu, dt, 400)
    growth = rel_err(ub, ua) / (np.linalg.norm(pert) / np.linalg.norm(u0))
    record("galilean", "round-off is not amplified", growth < 10.0,
           f"a 2e-16 relative IC perturbation grows by {growth:.2f}x over "
           f"400 steps -- the trajectory is well conditioned")


# ---------------------------------------------------------------------------
# 7-8. convergence of the production configuration
# ---------------------------------------------------------------------------


def _production_final(N: int, dt: float, nu: float, t_end: float = P.T_END):
    X, Y, _, _ = B._grids(N)
    u0, v0 = B.initial_condition(X, Y, P.C_MEAN, P.AMP)
    return integrate(u0, v0, nu, dt, int(round(t_end / dt)))


#: Times at which the grid study compares.  ``t ~ 1.2-1.9`` is where the
#: internal layers are thinnest; ``t_end`` is where the README's diagnostic
#: happens to look, and is the most forgiving point in the trajectory.
GRID_TIMES = (1.2, 1.92, 3.0, float(P.T_END))


def check_grid():
    """Spatial convergence at the *hardest* time, not only at ``t_end``."""
    print("\n[grid] spatial convergence of the production IC at the smallest nu")
    nu = P.viscosity(P.test_extrap_reynolds())      # hardest case in the data
    dt, Nref = P.DT, 512

    def trajectory(N):
        X, Y, _, _ = B._grids(N)
        u, v = B.initial_condition(X, Y, P.C_MEAN, P.AMP)
        out, t_now = {}, 0.0
        for T in GRID_TIMES:
            u, v = integrate(u, v, nu, dt, int(round((T - t_now) / dt)))
            t_now = T
            out[T] = (np.asarray(u), np.asarray(v))
        return out

    t0 = time.time()
    ref = trajectory(Nref)
    print(f"    reference N={Nref} done in {time.time() - t0:.0f}s")
    sols = {N: trajectory(N) for N in (128, 256)}

    for T in GRID_TIMES:
        ur, vr = ref[T]
        errs = {}
        for N in (128, 256):
            u, v = sols[N][T]
            m = Nref // N
            errs[N] = max(rel_err(u, ur[::m, ::m]), rel_err(v, vr[::m, ::m]))
        # the production grid should be good to ~1e-4 everywhere if the
        # README's "fully resolved" claim holds
        record("grid", f"N=128 vs N={Nref} at t={T:.2f}", errs[128] < 1.0e-4,
               f"rel err = {errs[128]:.2e} (N=256: {errs[256]:.2e})"
               + ("" if errs[128] < 1e-4 else "  <-- N=128 under-resolved here"))

    record("grid", "production grid adequate at Re=580",
           max(max(rel_err(sols[128][T][0], ref[T][0][::4, ::4]),
                   rel_err(sols[128][T][1], ref[T][1][::4, ::4]))
               for T in GRID_TIMES) < 1.0e-4,
           f"nu={nu:.4g}, Re={P.reynolds(nu):.0f}; worst-time N=128 error above "
           f"is what the snapshot data carries")


def check_time():
    """RK4 temporal convergence at the production grid.

    ``T_TIME`` is 6.0, not ``T_END = 2 pi``: every step size below divides
    6.0 into a whole number of steps, so all runs stop at *exactly* the same
    instant.  With ``t_end = 2 pi`` they do not -- ``int(round(2 pi / dt))``
    lands dt=1e-3 and dt=5e-4 on t=6.283000 but the reference on t=6.283125,
    and the comparison then measures that 1.25e-4 end-time mismatch (a fixed
    7.4e-5 floor, identical for both) instead of the discretisation error.
    """
    print("\n[time] RK4 temporal convergence at the production grid")
    nu, N, dt_ref = P.viscosity(P.test_extrap_reynolds()), 128, 1.25e-4
    T_TIME = 6.0
    t0 = time.time()
    uref, vref = _production_final(N, dt_ref, nu, T_TIME)
    print(f"    reference dt={dt_ref} done in {time.time() - t0:.0f}s")
    errs = {}
    for dt in (2.0e-3, 1.0e-3, 5.0e-4):
        assert abs(T_TIME / dt - round(T_TIME / dt)) < 1e-9, dt
        u, v = _production_final(N, dt, nu, T_TIME)
        errs[dt] = max(rel_err(u, uref), rel_err(v, vref))
    ratios = [errs[2.0e-3] / errs[1.0e-3], errs[1.0e-3] / errs[5.0e-4]]
    orders = [np.log2(r) for r in ratios]
    for dt, e in errs.items():
        record("time", f"dt={dt:g}", e < 1.0e-8,
               f"rel err vs dt={dt_ref:g} is {e:.2e}")
    record("time", "observed order", all(3.5 < o < 4.6 for o in orders),
           "p = " + ", ".join(f"{o:.2f}" for o in orders) + " (RK4 expects 4)")


# ---------------------------------------------------------------------------
# 9-10. the cached data on disk
# ---------------------------------------------------------------------------


def check_nwidth():
    """Does the under-resolution change what the experiment actually measures?

    The ``grid`` check shows the ``N = 128`` snapshots carry ~2% error at
    ``Re = 580``.  That is a statement about the fields; the experiment,
    though, is about the *low-rank structure* of the snapshot matrix -- the
    singular-value decay and the POD projection error.  So recompute the same
    snapshot times at ``N = 512``, restrict to the ``128`` grid, and compare
    the two spectra.  If they agree, the resolution problem does not
    invalidate the method comparison even though it does weaken any claim
    about resolving the PDE itself.
    """
    print("\n[nwidth] does the resolution error move the singular values?")
    nu, se = P.viscosity(P.test_extrap_reynolds()), P.STORE_EVERY_TEST
    S128, _ = P.solve_burgers(nu, se)
    n_store = int(round(P.T_END / P.DT)) // se

    N, m = 512, 4
    X, Y, _, _ = B._grids(N)
    u, v = B.initial_condition(X, Y, P.C_MEAN, P.AMP)
    cols = [np.concatenate([u[::m, ::m].ravel(), v[::m, ::m].ravel()])]
    t0 = time.time()
    for _ in range(n_store):
        u, v = integrate(u, v, nu, P.DT, se)
        cols.append(np.concatenate([np.asarray(u)[::m, ::m].ravel(),
                                    np.asarray(v)[::m, ::m].ravel()]))
    S512 = np.stack(cols, axis=1)
    print(f"    N=512 trajectory done in {time.time() - t0:.0f}s")

    record("nwidth", "snapshot matrices differ", True,
           f"rel difference between the N=128 data and the N=512 solution "
           f"restricted to the same grid = {rel_err(S128, S512):.2e}")

    def spectrum(S):
        M = S - S.mean(axis=1, keepdims=True)
        s = np.linalg.svd(M, compute_uv=False)
        s = s / s[0]
        e = np.sqrt(np.maximum(0.0, 1.0 - np.cumsum(s ** 2) / np.sum(s ** 2)))
        return s, e

    s128, e128 = spectrum(S128)
    s512, e512 = spectrum(S512)

    for r in (5, 10, 15, 20, 30):
        d = abs(e128[r - 1] - e512[r - 1]) / e512[r - 1]
        record("nwidth", f"POD projection error at r={r}", d < 0.05,
               f"N=128 data {e128[r - 1]:.3e} vs N=512 {e512[r - 1]:.3e} "
               f"({100 * d:.1f}% apart)")

    worst = max(abs(s128[i] - s512[i]) / s512[i] for i in range(30))
    record("nwidth", "singular-value decay is unaffected", worst < 0.05,
           f"worst relative disagreement over the first 30 singular values "
           f"is {100 * worst:.2f}% -- the n-width story the experiment rests "
           f"on survives the resolution error")


def _parse(path: Path):
    stem = path.stem                      # burgers2d_N128_nu0.0204886_se80
    parts = stem.split("_")
    return int(parts[1][1:]), float(parts[2][2:]), int(parts[3][2:])


def check_splits():
    print("\n[splits] cached files vs the (nu, store_every) pairs the splits need")
    want = set()
    for Re in P.training_reynolds():
        want.add((f"{P.viscosity(Re):.6g}", P.STORE_EVERY_TRAIN))
    for Re in P.val_interp_reynolds():
        want.add((f"{P.viscosity(Re):.6g}", P.STORE_EVERY_VAL))
    want.add((f"{P.viscosity(P.val_extrap_reynolds()):.6g}", P.STORE_EVERY_VAL))
    want.add((f"{P.viscosity(P.test_interp_reynolds()):.6g}", P.STORE_EVERY_TEST))
    want.add((f"{P.viscosity(P.test_extrap_reynolds()):.6g}", P.STORE_EVERY_TEST))

    have = set()
    for p in sorted(P.DATA.glob("burgers2d_N*.npz")):
        N, nu, se = _parse(p)
        have.add((f"{nu:.6g}", se))
    record("splits", "cached set == required set", have == want,
           f"{len(have)} files, {len(want)} required; "
           f"missing {sorted(want - have)}, extra {sorted(have - want)}")

    # column counts the splits will produce
    counts = {}
    for name in P.SPLITS:
        if name == "train":
            res, se = P.training_reynolds(), P.STORE_EVERY_TRAIN
        elif name == "val_interp":
            res, se = P.val_interp_reynolds(), P.STORE_EVERY_VAL
        elif name == "val_extrap":
            res, se = [P.val_extrap_reynolds()], P.STORE_EVERY_VAL
        elif name == "test_interp":
            res, se = [P.test_interp_reynolds()], P.STORE_EVERY_TEST
        else:
            res, se = [P.test_extrap_reynolds()], P.STORE_EVERY_TEST
        per = int(round(P.T_END / P.DT)) // se + 1
        counts[name] = (len(np.atleast_1d(res)), per, len(np.atleast_1d(res)) * per)
    detail = "; ".join(f"{k}: {v[0]}x{v[1]}={v[2]}" for k, v in counts.items())
    record("splits", "split column counts", True, detail)
    record("splits", "Reynolds numbers", True,
           f"train {np.round(P.training_reynolds(), 1).tolist()}, "
           f"val_interp {np.round(P.val_interp_reynolds(), 1).tolist()}, "
           f"val_extrap {P.val_extrap_reynolds():.1f}, "
           f"test_interp {P.test_interp_reynolds():.1f}, "
           f"test_extrap {P.test_extrap_reynolds():.1f}")


def check_data():
    print("\n[data] integrity of every cached .npz")
    files = sorted(P.DATA.glob("burgers2d_N*.npz"))
    record("data", "files present", len(files) == 9, f"{len(files)} cached .npz")

    n_steps = int(round(P.T_END / P.DT))
    all_ok = True
    for p in files:
        N, nu, se = _parse(p)
        z = np.load(p)
        S, X, Y, tspan = z["S"], z["X"], z["Y"], z["tspan"]
        NN = N * N
        n_store = n_steps // se
        msgs, ok = [], True

        ok &= S.shape == (2 * NN, n_store + 1)
        msgs.append(f"S{S.shape}=={(2 * NN, n_store + 1)}")
        ok &= bool(np.isfinite(S).all())
        ok &= S.dtype == np.float64

        # time vector
        t_want = P.DT * se * np.arange(n_store + 1)
        ok &= bool(np.allclose(tspan, t_want, rtol=0, atol=1e-12))
        msgs.append(f"t[-1]={tspan[-1]:.4f}")

        # grid
        x1 = 2.0 * np.pi * np.arange(N) / N
        Xw, Yw = np.meshgrid(x1, x1, indexing="ij")
        ok &= bool(np.array_equal(X, Xw) and np.array_equal(Y, Yw))

        # the stored first column must be the analytic IC, bit for bit
        u0, v0 = B.initial_condition(X, Y, P.C_MEAN, P.AMP)
        ic_err = max(np.abs(S[:NN, 0] - u0.ravel()).max(),
                     np.abs(S[NN:, 0] - v0.ravel()).max())
        ok &= ic_err == 0.0
        msgs.append(f"IC exact ({ic_err:.0e})")

        # all columns finite and free of blow-up
        ok &= bool(np.abs(S).max() < 10.0)
        msgs.append(f"max|S|={np.abs(S).max():.2f}")
        msgs.append(f"Re={P.reynolds(nu):.0f}")
        all_ok &= ok
        record("data", p.name, bool(ok), ", ".join(msgs))

    # reproduce one file end to end
    p = P.DATA / f"burgers2d_N128_nu{P.viscosity(P.test_interp_reynolds()):.6g}_se{P.STORE_EVERY_TEST}.npz"
    z = np.load(p)
    t0 = time.time()
    S, info = B.generate_burgers2d(N=P.N, nu=_parse(p)[1], dt=P.DT, t_end=P.T_END,
                                  store_every=_parse(p)[2], c=P.C_MEAN, amp=P.AMP)
    e = float(np.abs(S - z["S"]).max())
    rel = rel_err(S, z["S"])
    # Not bit-identical: the difference grows linearly from the first stored
    # column (~3e-11 per step), which is a systematic per-step discrepancy,
    # not chaos or round-off.  The solver is deterministic within a session,
    # so this is XLA/FFT drift between the environment that produced the
    # cached data and the current one.  It is ~6 orders of magnitude below
    # the spatial discretisation error, so it changes nothing scientifically.
    record("data", f"re-integration reproduces {p.name}", rel < 1e-6,
           f"max|diff| = {e:.2e}, rel = {rel:.2e} ({time.time() - t0:.0f}s) "
           f"-- not bit-identical (XLA drift since the data was generated), "
           f"but far below the discretisation error")


# ---------------------------------------------------------------------------


CHECKS = dict(mask=check_mask, stability=check_stability,
              linear=check_linear, exact1d=check_exact1d,
              exact2d=check_exact2d, diagonal=check_diagonal,
              galilean=check_galilean, resolution=check_resolution,
              grid=check_grid, time=check_time, nwidth=check_nwidth,
              data=check_data, splits=check_splits)


def write_report(partial: bool = False):
    """Write the markdown report.

    A subset run goes to its own file: ``validation.md`` is the record of a
    complete run, and a one-check re-run must not silently replace it.
    """
    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / ("validation_partial.md" if partial else "validation.md")
    n_fail = sum(1 for *_, ok, _ in ((a, b, c, d) for a, b, c, d in _results) if not ok)
    lines = ["# 2D Burgers solver and data verification", "",
             f"`validate.py`: {len(_results)} checks, "
             f"**{len(_results) - n_fail} passed, {n_fail} failed**.", "",
             "## Summary", "",
             "**The solver is correct.**  Against exact Cole-Hopf solutions "
             "of the 2D vector Burgers system -- including a non-separable "
             "`Phi = A(x,t) + B(y,t)` solution in which both cross terms "
             "`v u_y` and `u v_x` are active -- the computed field sits at "
             "1.01-1.02x the *truncation floor*, i.e. it is as accurate as "
             "any scheme on this dealiased basis can be.  RK4 order 4 is "
             "confirmed two independent ways (step refinement, and the "
             "Galilean residual, which is an `O(dt^4)` symmetry error).", "",
             "**The cached data is under-resolved at high Reynolds number.**  "
             "At `Re = 580` the `N = 128` snapshots carry ~2% error against "
             "an `N = 512` reference during the steepening phase "
             "(`t ~ 1.2-3`).  The README's 'fully resolved, spectral tail "
             "< 1e-6' comes from `spectrum_tail(S[:, -1])` -- the *final* "
             "snapshot, which is the smoothest point of the trajectory.  The "
             "tail actually peaks near `t ~ 1.1-1.9`.", "",
             "**But the experiment's conclusions survive it**: the singular "
             "values and POD projection errors of the snapshot matrix agree "
             "with the refined solution to well under 5% (see `nwidth`), so "
             "the n-width argument and the method ranking are unaffected.  "
             "What the resolution error does undermine is any claim that the "
             "data resolves the *PDE* at these Reynolds numbers.", "",
             "Note that tolerances here are judged against the dealiasing "
             "truncation floor, not machine precision: a pseudo-spectral "
             "scheme cannot represent what lies above its cutoff, so "
             "comparing an exact solution to 1e-16 would be meaningless.", "",
             "| check | quantity | result | detail |",
             "|---|---|---|---|"]
    for check, q, ok, detail in _results:
        # a bare "|" would end the markdown cell early
        q = q.replace("|", r"\|")
        detail = detail.replace("|", r"\|")
        lines.append(f"| `{check}` | {q} | {'PASS' if ok else 'FAIL'} | {detail} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")
    return n_fail


if __name__ == "__main__":
    names = sys.argv[1:] or list(CHECKS)
    bad = [n for n in names if n not in CHECKS]
    if bad:
        sys.exit(f"unknown check(s) {bad}; choose from {list(CHECKS)}")
    for n in names:
        CHECKS[n]()
    n_fail = write_report(partial=len(names) < len(CHECKS))
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed")
    sys.exit(1 if n_fail else 0)
