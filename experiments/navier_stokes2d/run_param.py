#!/usr/bin/env python3
"""Parametric NACA 0012 sweep over angle of attack with SU2 (laminar, Re_c = 1000).

For each angle this script writes a configuration with ``su2_lam_config.make_config``,
runs SU2_CFD (MPI), converts the binary restart files SU2 writes every
``write_every`` time steps into one compact ``data/naca_aXX.npz`` and deletes
the restart files.  The mesh is shared by all angles (the angle of attack enters
through the free-stream direction), so every npz has the same node ordering.

npz contents
------------
  x, y            node coordinates (chords), the airfoil nodes come first
  t               snapshot times (c/U_inf)
  u, v, p         (n_t, n_nodes) float32; velocity in units of U_inf, pressure
                  in units of rho U_inf^2 (SU2 incompressible, p_inf = 0)
  force_t, Cl, Cd per-time-step aerodynamic coefficients (chord-based)
  tris, quads     element connectivity for plotting
  airfoil_nodes   indices of the wall nodes (closed loop, counter-clockwise)
  Re, alpha_deg, dt, chord, U_inf   scalars

Usage:  python run_param.py [--angles 4 8 ...] [--np 2] [--t-end 60] [--dt 0.025]
        (set SU2_CFD=/path/to/SU2_CFD if it is not on PATH)
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from su2_lam_config import make_config  # noqa: E402

DATA = os.path.join(HERE, "data")
MESH = os.path.join(DATA, "naca0012_lam.su2")
CASES = os.path.join(HERE, "cases")
SU2_CFD = os.environ.get("SU2_CFD", "SU2_CFD")


def read_su2_mesh(path):
    """Nodes, triangles, quads and the airfoil node loop of an SU2 mesh."""
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    tris, quads, nodes, airfoil = [], [], [], []
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("NELEM"):
            n = int(ln.split("=")[1])
            for k in range(n):
                p = lines[i + 1 + k].split()
                if p[0] == "5":
                    tris.append([int(v) for v in p[1:4]])
                elif p[0] == "9":
                    quads.append([int(v) for v in p[1:5]])
            i += n + 1
        elif ln.startswith("NPOIN"):
            n = int(ln.split("=")[1])
            nodes = np.array([[float(v) for v in lines[i + 1 + k].split()[:2]] for k in range(n)])
            i += n + 1
        elif ln.startswith("MARKER_TAG") and "airfoil" in ln:
            n = int(lines[i + 1].split("=")[1])
            edges = [[int(v) for v in lines[i + 2 + k].split()[1:3]] for k in range(n)]
            airfoil = [e[0] for e in edges]
            i += n + 2
        else:
            i += 1
    return nodes, np.asarray(tris), np.asarray(quads), np.asarray(airfoil)


def read_restart(path):
    raw = open(path, "rb").read()
    hdr = np.frombuffer(raw[:20], dtype=np.int32)
    nfields, npoints = int(hdr[1]), int(hdr[2])
    names = [raw[20 + 33 * i:20 + 33 * (i + 1)].split(b"\x00")[0].decode() for i in range(nfields)]
    off = 20 + 33 * nfields
    data = np.frombuffer(raw[off:off + 8 * nfields * npoints], dtype=np.float64).reshape(npoints, nfields)
    return dict(zip(names, data.T))


def read_history(path):
    """Per-time-step (last inner iteration) time, CL, CD from history.csv."""
    import csv
    rows = list(csv.reader(open(path)))
    hdr = [h.strip().strip('"') for h in rows[0]]
    it = hdr.index("Time_Iter"); ct = hdr.index("Cur_Time")
    cl = hdr.index("CL"); cd = hdr.index("CD")
    last = {}
    for r in rows[1:]:
        if len(r) < len(hdr):
            continue
        last[int(r[it])] = (float(r[ct]), float(r[cl]), float(r[cd]))
    ks = sorted(last)
    arr = np.array([last[k] for k in ks])
    return arr[:, 0], arr[:, 1], arr[:, 2]


def run_angle(alpha, args, mesh_info):
    tag = f"a{alpha:02d}"
    out = os.path.join(DATA, f"naca_{tag}.npz")
    if os.path.exists(out) and not args.force:
        print(f"[{tag}] exists, skipping", flush=True)
        return
    cdir = os.path.join(CASES, tag)
    os.makedirs(cdir, exist_ok=True)
    local_mesh = os.path.join(cdir, os.path.basename(MESH))
    if not os.path.exists(local_mesh):
        os.symlink(MESH, local_mesh)
    cfg = make_config(os.path.basename(MESH), float(alpha), Re=args.Re, dt=args.dt,
                      t_end=args.t_end, inner_iter=args.inner_iter, cfl=args.cfl,
                      write_every=args.write_every, inner_conv=args.inner_conv)
    cfg = cfg.replace("OUTPUT_FILES= ( RESTART, CSV )", "OUTPUT_FILES= ( RESTART )")
    open(os.path.join(cdir, "case.cfg"), "w").write(cfg)

    print(f"\n===== alpha = {alpha} deg =====", flush=True)
    t0 = time.time()
    with open(os.path.join(cdir, "su2.log"), "w") as log:
        cmd = ["mpirun", "-n", str(args.np), SU2_CFD, "case.cfg"] if args.np > 1 else [SU2_CFD, "case.cfg"]
        rc = subprocess.run(cmd, cwd=cdir, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    print(f"[{tag}] SU2 rc={rc} in {wall / 3600:.2f} h", flush=True)

    # ---- collect snapshots ----
    files = sorted(glob.glob(os.path.join(cdir, "restart_flow_*.dat")))
    iters = np.array([int(os.path.basename(f)[13:18]) for f in files])
    # SU2 writes at (iter + 1) % write_every == 0, plus the initial field and the
    # last two levels for BDF2 restarts; keep only the regular series
    keep = (iters + 1) % args.write_every == 0
    files = [f for f, k in zip(files, keep) if k]; iters = iters[keep]
    order = np.argsort(iters); files = [files[k] for k in order]; iters = iters[order]
    nodes, tris, quads, airfoil = mesh_info
    n_nodes = len(nodes)
    u = np.empty((len(files), n_nodes), np.float32)
    v = np.empty_like(u); p = np.empty_like(u)
    for k, f in enumerate(files):
        d = read_restart(f)
        if k == 0:
            assert np.allclose(d["x"], nodes[:, 0]) and np.allclose(d["y"], nodes[:, 1]), \
                "restart node order differs from the mesh"
        u[k] = d["Velocity_x"]; v[k] = d["Velocity_y"]; p[k] = d["Pressure"]
    t = iters * args.dt
    ft, Cl, Cd = read_history(os.path.join(cdir, "history.csv"))
    np.savez_compressed(
        out, x=nodes[:, 0], y=nodes[:, 1], t=t.astype(np.float32), u=u, v=v, p=p,
        force_t=ft, Cl=Cl, Cd=Cd, tris=tris, quads=quads, airfoil_nodes=airfoil,
        Re=np.float32(args.Re), alpha_deg=np.float32(alpha), dt=np.float32(args.dt),
        chord=np.float32(1.0), U_inf=np.float32(1.0), su2_rc=np.int32(rc),
        wall_hours=np.float32(wall / 3600))
    print(f"[{tag}] saved {out}: {len(files)} snapshots, {n_nodes} nodes", flush=True)
    if not args.keep_restarts:
        for f in files:
            os.remove(f)
        for f in glob.glob(os.path.join(cdir, "restart_flow.dat")):
            os.remove(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, nargs="+", default=[8, 12, 16, 20, 24, 14, 10, 18, 22])
    ap.add_argument("--np", type=int, default=2)
    ap.add_argument("--Re", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--t-end", type=float, default=60.0)
    ap.add_argument("--inner-iter", type=int, default=25)
    ap.add_argument("--inner-conv", type=float, default=-7.5)
    ap.add_argument("--cfl", type=float, default=50.0)
    ap.add_argument("--write-every", type=int, default=8)
    ap.add_argument("--keep-restarts", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    if not os.path.exists(MESH):
        print("mesh not found; generating it with make_mesh.py", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "make_mesh.py"), "--out", MESH,
                        "--preview", os.path.join(DATA, "mesh_preview.png")], check=True)
    mesh_info = read_su2_mesh(MESH)
    if not os.path.exists(os.path.join(DATA, "mesh.npz")):
        nodes, tris, quads, airfoil = mesh_info
        np.savez_compressed(os.path.join(DATA, "mesh.npz"), x=nodes[:, 0], y=nodes[:, 1],
                            tris=tris, quads=quads, airfoil_nodes=airfoil)
    for a in args.angles:
        run_angle(a, args, mesh_info)
    print("\nPARAM SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
