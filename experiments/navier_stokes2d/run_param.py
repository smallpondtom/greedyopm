"""Parametric NACA 0012 sweep over angle of attack (Re fixed at 1000).

Runs the spectral solver once per angle as an isolated subprocess (fresh JIT),
saving the full u, v, omega snapshots to data/naca_aXX.npz and a vorticity GIF to
plots/naca_aXX.gif. Angles span mild incidence to deep stall so the parametric
solution manifold is transport-rich (the point of the exercise: slow n-width).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(HERE, "naca_solver.py")
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(DATA, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

# Full set used by the interpolation/extrapolation experiment (problem.py):
# train {4,8,12,16}, val_interp {6,14}, test_interp {10}, val_extrap {20},
# test_extrap {24}.  (The original 6-angle sv_decay set was {4,8,12,16,20,24}.)
ANGLES = [4, 6, 8, 10, 12, 14, 16, 20, 24]
T_END = "30"
NX, NY = "512", "256"

for a in ANGLES:
    tag = f"a{a:02d}"
    out = os.path.join(DATA, f"naca_{tag}.npz")
    gif = os.path.join(PLOTS, f"naca_{tag}.gif")
    fig = os.path.join(PLOTS, f"final_{tag}.png")
    if os.path.exists(out):
        print(f"[{tag}] exists, skipping", flush=True)
        continue
    env = dict(os.environ,
               NS_ALPHA=str(float(a)), NS_NX=NX, NS_NY=NY, NS_TEND=T_END,
               NS_SAVEVEL="1", NS_OUT=out, NS_GIF=gif, NS_FIG=fig,
               NS_GIF_STRIDE="2")
    t0 = time.time()
    print(f"\n===== alpha = {a} deg =====", flush=True)
    r = subprocess.run([sys.executable, SOLVER], env=env)
    if r.returncode != 0:
        print(f"[{tag}] FAILED (rc={r.returncode})", flush=True)
        break
    print(f"[{tag}] done in {time.time() - t0:.0f}s", flush=True)

print("\nPARAM SWEEP DONE", flush=True)
