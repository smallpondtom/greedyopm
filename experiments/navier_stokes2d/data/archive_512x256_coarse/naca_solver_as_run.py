"""NACA 0012 vortex-shedding solver: 2-D incompressible Navier-Stokes past a
NACA 0012 airfoil, Fourier pseudo-spectral with 2/3 dealiasing, rotational-form
nonlinearity, Leray projection, SSP-RK3 advection, exact spectral diffusion,
Brinkman/volume penalization for the airfoil, a downstream fringe/sponge, Strang
splitting, JAX + XLA.

Physics is identical to the prepared reference code; this version runs headless
and exposes a few run parameters through environment variables so the same file
serves a quick verification run and the full data generation:

  NS_NX, NS_NY     grid (default 512 x 256)
  NS_TEND          end time, convective units (default 40)
  NS_SNAPDT        snapshot spacing (default 0.20)
  NS_SAVEVEL       "1"/"0" store u, v in the npz (default 1)
  NS_OUT           output npz path (default naca0012_Re1000.npz)
  NS_GIF           vorticity gif path ("" = skip)
  NS_GIF_STRIDE    use every k-th snapshot in the gif (default 1)
"""

import os
import time

import jax
import jax.numpy as jnp
from jax import lax

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _env(name, default):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return type(default)(v)


# ============================================================
# User parameters
# ============================================================

RE = 1000.0
U_INF = 1.0
CHORD = 1.0
ALPHA_DEG = _env("NS_ALPHA", 15.0)
THICKNESS = 0.12

XMIN, XMAX = -4.0, 12.0
YMIN, YMAX = -4.0, 4.0

NX = _env("NS_NX", 512)
NY = _env("NS_NY", 256)

T_END = _env("NS_TEND", 40.0)
DT = 0.005
SNAPSHOT_DT = _env("NS_SNAPDT", 0.20)

ETA = 5.0e-4
MASK_WIDTH_CELLS = 1.5

FRINGE_START = 9.0
FRINGE_STRENGTH = 2.0
FRINGE_TRANSITION = 0.30

SAVE_VELOCITY = bool(_env("NS_SAVEVEL", 1))
OUTPUT_FILE = _env("NS_OUT", "naca0012_Re1000.npz")
FIGURE_FILE = _env("NS_FIG", "final_vorticity.png")
GIF_FILE = _env("NS_GIF", "")
GIF_STRIDE = _env("NS_GIF_STRIDE", 1)


# ============================================================
# JAX setup
# ============================================================

jax.config.update("jax_enable_x64", False)
DTYPE = jnp.float32

print("JAX devices:", jax.devices())


# ============================================================
# Grid
# ============================================================

LX, LY = XMAX - XMIN, YMAX - YMIN
DX, DY = LX / NX, LY / NY

x = XMIN + DX * jnp.arange(NX, dtype=DTYPE)
y = YMIN + DY * jnp.arange(NY, dtype=DTYPE)
X, Y = jnp.meshgrid(x, y, indexing="xy")


# ============================================================
# Fourier wavenumbers
# ============================================================

kx = (2.0 * jnp.pi * jnp.fft.fftfreq(NX, d=DX)).astype(DTYPE)
ky = (2.0 * jnp.pi * jnp.fft.fftfreq(NY, d=DY)).astype(DTYPE)
KX, KY = jnp.meshgrid(kx, ky, indexing="xy")
K2 = KX**2 + KY**2
K2_SAFE = jnp.where(K2 == 0.0, 1.0, K2)


# ============================================================
# 2/3-rule dealiasing mask
# ============================================================

mx = jnp.fft.fftfreq(NX) * NX
my = jnp.fft.fftfreq(NY) * NY
MX, MY = jnp.meshgrid(mx, my, indexing="xy")
DEALIAS = ((jnp.abs(MX) <= NX / 3) & (jnp.abs(MY) <= NY / 3)).astype(DTYPE)


# ============================================================
# NACA 0012 geometry
# ============================================================

def make_naca00xx_mask(X, Y, chord, thickness, alpha_deg, x_le=0.0,
                       y_center=0.0, smooth_width=0.02):
    """Smooth indicator (~1 inside solid, ~0 outside) for a NACA 00xx airfoil,
    rotated about the quarter chord."""
    alpha = jnp.deg2rad(alpha_deg)
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    x_ref = x_le + 0.25 * chord
    y_ref = y_center
    dx = X - x_ref
    dy = Y - y_ref
    xb = ca * dx + sa * dy + 0.25 * chord
    yb = -sa * dx + ca * dy
    xi = xb / chord
    xi_clip = jnp.clip(xi, 0.0, 1.0)
    yt = (5.0 * thickness * chord * (
        0.2969 * jnp.sqrt(xi_clip) - 0.1260 * xi_clip - 0.3516 * xi_clip**2
        + 0.2843 * xi_clip**3 - 0.1036 * xi_clip**4))
    shape_mask = 0.5 * (1.0 + jnp.tanh((yt - jnp.abs(yb)) / smooth_width))
    left_window = 0.5 * (1.0 + jnp.tanh(xb / smooth_width))
    right_window = 0.5 * (1.0 + jnp.tanh((chord - xb) / smooth_width))
    return shape_mask * left_window * right_window


mask_width = MASK_WIDTH_CELLS * min(DX, DY)
CHI = make_naca00xx_mask(X, Y, chord=CHORD, thickness=THICKNESS,
                         alpha_deg=ALPHA_DEG, x_le=0.0, y_center=0.0,
                         smooth_width=mask_width).astype(DTYPE)


# ============================================================
# Fringe / sponge
# ============================================================

SIGMA = (0.5 * FRINGE_STRENGTH * (
    1.0 + jnp.tanh((X - FRINGE_START) / FRINGE_TRANSITION))).astype(DTYPE)


# ============================================================
# Spectral operations
# ============================================================

@jax.jit
def project_velocity(u, v):
    uh = jnp.fft.fft2(u)
    vh = jnp.fft.fft2(v)
    k_dot_u = KX * uh + KY * vh
    uh_proj = uh - KX * k_dot_u / K2_SAFE
    vh_proj = vh - KY * k_dot_u / K2_SAFE
    uh_proj = uh_proj.at[0, 0].set(uh[0, 0])
    vh_proj = vh_proj.at[0, 0].set(vh[0, 0])
    u_proj = jnp.fft.ifft2(uh_proj).real
    v_proj = jnp.fft.ifft2(vh_proj).real
    return u_proj.astype(DTYPE), v_proj.astype(DTYPE)


@jax.jit
def vorticity(u, v):
    uh = jnp.fft.fft2(u)
    vh = jnp.fft.fft2(v)
    omega_hat = 1j * KX * vh - 1j * KY * uh
    return jnp.fft.ifft2(omega_hat).real.astype(DTYPE)


@jax.jit
def nonlinear_rhs(u, v):
    uh = jnp.fft.fft2(u)
    vh = jnp.fft.fft2(v)
    omega_hat = 1j * KX * vh - 1j * KY * uh
    omega = jnp.fft.ifft2(omega_hat).real
    fu = v * omega
    fv = -u * omega
    fuh = jnp.fft.fft2(fu) * DEALIAS
    fvh = jnp.fft.fft2(fv) * DEALIAS
    k_dot_f = KX * fuh + KY * fvh
    fuh = fuh - KX * k_dot_f / K2_SAFE
    fvh = fvh - KY * k_dot_f / K2_SAFE
    fuh = fuh.at[0, 0].set(0.0 + 0.0j)
    fvh = fvh.at[0, 0].set(0.0 + 0.0j)
    return jnp.fft.ifft2(fuh).real.astype(DTYPE), jnp.fft.ifft2(fvh).real.astype(DTYPE)


@jax.jit
def nonlinear_rk3(u, v, dt):
    ru, rv = nonlinear_rhs(u, v)
    u1 = u + dt * ru
    v1 = v + dt * rv
    ru1, rv1 = nonlinear_rhs(u1, v1)
    u2 = 0.75 * u + 0.25 * (u1 + dt * ru1)
    v2 = 0.75 * v + 0.25 * (v1 + dt * rv1)
    ru2, rv2 = nonlinear_rhs(u2, v2)
    u3 = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * ru2)
    v3 = (1.0 / 3.0) * v + (2.0 / 3.0) * (v2 + dt * rv2)
    return u3.astype(DTYPE), v3.astype(DTYPE)


NU = DTYPE(1.0 / RE)


@jax.jit
def linear_relaxation_step(u, v, dt):
    uh = jnp.fft.fft2(u)
    vh = jnp.fft.fft2(v)
    diffusion_factor = jnp.exp(-NU * K2 * dt)
    uh = uh * diffusion_factor
    vh = vh * diffusion_factor
    u = jnp.fft.ifft2(uh).real
    v = jnp.fft.ifft2(vh).real
    body_decay = jnp.exp(-CHI * dt / ETA)
    u = body_decay * u
    v = body_decay * v
    fringe_decay = jnp.exp(-SIGMA * dt)
    u = U_INF + fringe_decay * (u - U_INF)
    v = fringe_decay * v
    u, v = project_velocity(u, v)
    return u, v


@jax.jit
def timestep(state):
    u, v = state
    half_dt = DTYPE(0.5 * DT)
    full_dt = DTYPE(DT)
    u, v = linear_relaxation_step(u, v, half_dt)
    u, v = nonlinear_rk3(u, v, full_dt)
    u, v = linear_relaxation_step(u, v, half_dt)
    return u, v


# ============================================================
# Snapshot cadence
# ============================================================

STEPS_PER_SNAPSHOT = int(round(SNAPSHOT_DT / DT))
if STEPS_PER_SNAPSHOT < 1:
    raise ValueError("SNAPSHOT_DT must be >= DT.")
ACTUAL_SNAPSHOT_DT = STEPS_PER_SNAPSHOT * DT
NSTEPS = int(round(T_END / DT))
NSNAP = NSTEPS // STEPS_PER_SNAPSHOT


@jax.jit
def advance_to_next_snapshot(state):
    return lax.fori_loop(0, STEPS_PER_SNAPSHOT, lambda _, s: timestep(s), state)


@jax.jit
def diagnostics(u, v):
    speed = jnp.sqrt(u**2 + v**2)
    umax = jnp.max(speed)
    kinetic_energy = 0.5 * jnp.mean(u**2 + v**2)
    uh = jnp.fft.fft2(u)
    vh = jnp.fft.fft2(v)
    div_hat = 1j * KX * uh + 1j * KY * vh
    div = jnp.fft.ifft2(div_hat).real
    div_rms = jnp.sqrt(jnp.mean(div**2))
    return umax, kinetic_energy, div_rms


# ============================================================
# Initial condition
# ============================================================

u0 = jnp.ones((NY, NX), dtype=DTYPE) * U_INF
v0 = jnp.zeros((NY, NX), dtype=DTYPE)
u0, v0 = linear_relaxation_step(u0, v0, DTYPE(DT))
state = (u0, v0)


# ============================================================
# Allocate host snapshot arrays
# ============================================================

snapshot_shape = (NSNAP + 1, NY, NX)
if SAVE_VELOCITY:
    u_snapshots = np.empty(snapshot_shape, dtype=np.float32)
    v_snapshots = np.empty(snapshot_shape, dtype=np.float32)
omega_snapshots = np.empty(snapshot_shape, dtype=np.float32)
times = np.arange(NSNAP + 1, dtype=np.float32) * ACTUAL_SNAPSHOT_DT


# ============================================================
# Compile
# ============================================================

print("=" * 60)
print(f"Re={RE} alpha={ALPHA_DEG} grid={NX}x{NY} domain=[{XMIN},{XMAX}]x[{YMIN},{YMAX}]")
print(f"dx,dy={DX:.5f},{DY:.5f} dt={DT} t_end={T_END} steps={NSTEPS} "
      f"snap_dt={ACTUAL_SNAPSHOT_DT} nsnap={NSNAP + 1} save_vel={SAVE_VELOCITY}")
print("=" * 60)
print("JIT compiling...", flush=True)
t_compile = time.perf_counter()
state = advance_to_next_snapshot(state)
state[0].block_until_ready()
print(f"compiled+first block in {time.perf_counter() - t_compile:.2f}s", flush=True)

state = (u0, v0)   # reset

if SAVE_VELOCITY:
    u_snapshots[0] = np.asarray(state[0])
    v_snapshots[0] = np.asarray(state[1])
omega_snapshots[0] = np.asarray(vorticity(*state))


# ============================================================
# Simulation
# ============================================================

print("Running...", flush=True)
wall_start = time.perf_counter()
for k in range(1, NSNAP + 1):
    state = advance_to_next_snapshot(state)
    if SAVE_VELOCITY:
        u_snapshots[k] = np.asarray(jax.device_get(state[0]))
        v_snapshots[k] = np.asarray(jax.device_get(state[1]))
    omega_snapshots[k] = np.asarray(jax.device_get(vorticity(*state)))
    if k == 1 or k % 10 == 0 or k == NSNAP:
        umax, ke, div_rms = jax.device_get(diagnostics(*state))
        print(f"t={times[k]:7.3f}  |u|max={float(umax):7.4f}  KE={float(ke):9.6f}  "
              f"div_rms={float(div_rms):.2e}  wall={time.perf_counter() - wall_start:7.2f}s",
              flush=True)
print(f"finished in {time.perf_counter() - wall_start:.2f}s", flush=True)


# ============================================================
# Save
# ============================================================

x_np = np.asarray(jax.device_get(x))
y_np = np.asarray(jax.device_get(y))
chi_np = np.asarray(jax.device_get(CHI))
save_dict = dict(x=x_np, y=y_np, t=times, omega=omega_snapshots, chi=chi_np,
                 Re=np.float32(RE), alpha_deg=np.float32(ALPHA_DEG),
                 dt=np.float32(DT), chord=np.float32(CHORD), U_inf=np.float32(U_INF))
if SAVE_VELOCITY:
    save_dict["u"] = u_snapshots
    save_dict["v"] = v_snapshots
print(f"saving {OUTPUT_FILE} ...", flush=True)
np.savez_compressed(OUTPUT_FILE, **save_dict)
print("saved.", flush=True)


# ============================================================
# Final vorticity figure
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))
vort_limit = 8.0
pcm = ax.pcolormesh(x_np, y_np, omega_snapshots[-1], shading="auto",
                    cmap="RdBu_r", vmin=-vort_limit, vmax=vort_limit)
ax.contour(x_np, y_np, chi_np, levels=[0.5], colors="k", linewidths=1.5)
fig.colorbar(pcm, ax=ax, label=r"$\omega c/U_\infty$")
ax.set_xlim(-1.0, 8.0); ax.set_ylim(-2.0, 2.0)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$x/c$"); ax.set_ylabel(r"$y/c$")
ax.set_title(rf"NACA 0012, $Re_c={RE:.0f}$, $\alpha={ALPHA_DEG:.0f}^\circ$, $t={T_END:.1f}$")
fig.tight_layout(); fig.savefig(FIGURE_FILE, dpi=200); plt.close(fig)
print(f"saved {FIGURE_FILE}", flush=True)


# ============================================================
# Vorticity GIF
# ============================================================

if GIF_FILE:
    import matplotlib.animation as animation
    stride = max(1, int(GIF_STRIDE))
    idx = range(0, NSNAP + 1, stride)
    figg, axg = plt.subplots(figsize=(11, 5.0))
    vl = 8.0
    pcm = axg.pcolormesh(x_np, y_np, omega_snapshots[0], shading="auto",
                         cmap="RdBu_r", vmin=-vl, vmax=vl)
    axg.contour(x_np, y_np, chi_np, levels=[0.5], colors="k", linewidths=1.2)
    axg.set_xlim(-1.5, 9.0); axg.set_ylim(-2.5, 2.5)
    axg.set_aspect("equal", adjustable="box")
    axg.set_xlabel(r"$x/c$"); axg.set_ylabel(r"$y/c$")
    figg.colorbar(pcm, ax=axg, label=r"$\omega c/U_\infty$", fraction=0.025)
    ttl = axg.set_title("")

    def _frame(k):
        pcm.set_array(omega_snapshots[k].ravel())
        ttl.set_text(rf"NACA 0012 $Re={RE:.0f}$ $\alpha={ALPHA_DEG:.0f}^\circ$  $t={times[k]:.1f}$")
        return pcm, ttl

    anim = animation.FuncAnimation(figg, _frame, frames=list(idx), blit=False)
    anim.save(GIF_FILE, writer=animation.PillowWriter(fps=15))
    plt.close(figg)
    print(f"saved {GIF_FILE}", flush=True)
