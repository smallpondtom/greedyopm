#!/usr/bin/env python3
"""Body-fitted hybrid mesh around a NACA 0012 for laminar SU2 runs at Re_c ~ 1000.

Structure
---------
* an O-type structured quad boundary layer extruded from the airfoil along
  smoothed surface normals (geometric growth from a first-cell height
  ``h1``), with a fan of cells at the sharp trailing edge;
* an unstructured triangle region between the outer boundary-layer curve and a
  circular far field of radius ``R_far``, with refinement boxes covering the
  wake for angles of attack up to ~25 deg (the angle of attack is set through the
  free-stream direction in SU2, so one mesh serves every angle and the node
  ordering is identical across the parameter sweep).

The mesh is written in SU2 native format with markers ``airfoil`` and
``farfield``.  All lengths are in chords; the leading edge is at the origin and
the chord lies on the x axis.

Usage:  python make_mesh.py [--out naca0012_lam.su2] [--preview mesh.png]
"""
from __future__ import annotations

import argparse
import math

import numpy as np


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------

def naca0012(n_surf: int, closed: bool = True) -> np.ndarray:
    """Closed-loop surface points, counter-clockwise starting and ending at the
    trailing edge (TE appears once; the loop is implicit).  Cosine clustering."""
    a4 = 0.1036 if closed else 0.1015
    beta = np.linspace(0.0, math.pi, n_surf // 2 + 1)
    xc = 0.5 * (1.0 - np.cos(beta))                     # 0 .. 1, clustered at both ends
    yt = 0.6 * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                + 0.2843 * xc**3 - a4 * xc**4)
    yt[0] = 0.0
    yt[-1] = 0.0
    # counter-clockwise: upper surface TE -> LE, then lower surface LE -> TE
    upper = np.column_stack([xc[::-1], yt[::-1]])
    lower = np.column_stack([xc[1:], -yt[1:]])
    pts = np.vstack([upper, lower])[:-1]                 # drop the duplicate TE at the end
    return pts


def loop_normals(pts: np.ndarray, smooth_passes: int = 6) -> np.ndarray:
    """Outward unit normals of a closed counter-clockwise polygon (averaged over
    the two adjacent segments, then lightly smoothed along the loop)."""
    n = len(pts)
    seg = np.roll(pts, -1, axis=0) - pts                 # segment i: pts[i] -> pts[i+1]
    seg_n = np.column_stack([seg[:, 1], -seg[:, 0]])     # outward for a CCW loop
    seg_n /= np.linalg.norm(seg_n, axis=1, keepdims=True)
    nrm = seg_n + np.roll(seg_n, 1, axis=0)              # node i: segments i-1 and i
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    # smooth along the open chain 1..n-1; the trailing edge (node 0) is a sharp
    # corner handled by a fan and must not pull its neighbours' normals
    for _ in range(smooth_passes):
        left = np.roll(nrm, 1, axis=0); right = np.roll(nrm, -1, axis=0)
        left[1] = nrm[1]; right[n - 1] = nrm[n - 1]
        new = 0.5 * nrm + 0.25 * (left + right)
        new[0] = nrm[0]
        nrm = new / np.linalg.norm(new, axis=1, keepdims=True)
    return nrm, seg_n


def polygon_self_intersects(pts: np.ndarray) -> int:
    """Number of crossing segment pairs of a closed polygon (O(n^2))."""
    n = len(pts); a = pts; b = np.roll(pts, -1, axis=0)
    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])
    count = 0
    for i in range(n):
        j = np.arange(i + 2, n if i > 0 else n - 1)
        if len(j) == 0:
            continue
        p1, q1 = a[i], b[i]; p2, q2 = a[j], b[j]
        d1 = orient(p1, q1, p2); d2 = orient(p1, q1, q2)
        d3 = orient(p2, q2, p1); d4 = orient(p2, q2, q1)
        count += int(np.sum((d1 * d2 < 0) & (d3 * d4 < 0)))
    return count


def layer_heights(h1: float, ratio: float, n_layers: int) -> np.ndarray:
    return h1 * ratio ** np.arange(n_layers)


# ----------------------------------------------------------------------------
# structured boundary layer with a trailing-edge fan
# ----------------------------------------------------------------------------

def build_boundary_layer(surf: np.ndarray, h1: float, ratio: float,
                         n_layers: int, n_fan: int):
    """Return (nodes, quads, tris, airfoil_edges, outer_ring) where

    * ``nodes``          (N, 2) coordinates; the first len(surf) are the airfoil
    * ``quads``          (Q, 4) node indices, counter-clockwise
    * ``tris``           (T, 3) first-layer fan triangles at the trailing edge
    * ``airfoil_edges``  (E, 2) wall boundary segments
    * ``outer_ring``     list of node indices of the outer boundary-layer curve,
                         counter-clockwise, closed loop implicit
    """
    n_s = len(surf)
    nrm, seg_n = loop_normals(surf)
    # trailing edge is node 0; the fan rays sweep counter-clockwise around the
    # back of the airfoil, from the normal of the last segment (lower surface
    # arriving at the TE) to that of the first segment (upper surface leaving it).
    n_lo = seg_n[-1]
    n_up = seg_n[0]
    a_lo = math.atan2(n_lo[1], n_lo[0])
    a_up = math.atan2(n_up[1], n_up[0])
    if a_up < a_lo:
        a_up += 2.0 * math.pi
    fan_angles = np.linspace(a_lo, a_up, n_fan + 1)      # lower side -> upper side
    fan_dirs = np.column_stack([np.cos(fan_angles), np.sin(fan_angles)])

    heights = np.cumsum(layer_heights(h1, ratio, n_layers))
    # ring ordering (counter-clockwise, matching the surface loop): the surface
    # loop starts at the TE and goes along the lower surface; the fan rays are
    # inserted at the TE going from the upper-side ray to the lower-side ray,
    # which is the CCW direction.  Ring index layout per layer:
    #   [fan ray 0 (lower side) ... fan ray n_fan (upper side), surf 1, ..., surf n_s-1]
    dirs = np.vstack([fan_dirs, nrm[1:]])
    base = np.vstack([np.repeat(surf[:1], n_fan + 1, axis=0), surf[1:]])
    n_ring = len(dirs)                                     # = n_fan + n_s

    nodes = [surf]                                         # layer 0: surface nodes
    ring_ids = []                                          # per layer >= 1, list of node ids
    next_id = n_s
    for j in range(n_layers):
        pts = base + heights[j] * dirs
        nodes.append(pts)
        ring_ids.append(np.arange(next_id, next_id + n_ring))
        next_id += n_ring
    nodes = np.vstack(nodes)

    # layer 0 ring: fan rays all map to the TE node (0), surface nodes to themselves
    ring0 = np.concatenate([np.zeros(n_fan + 1, dtype=int), np.arange(1, n_s)])

    quads, tris = [], []
    prev = ring0
    for j in range(n_layers):
        cur = ring_ids[j]
        for k in range(n_ring):
            k2 = (k + 1) % n_ring
            a, b, c, d = prev[k], prev[k2], cur[k2], cur[k]
            if a == b:                                     # degenerate at the TE fan (layer 0)
                tris.append((a, c, d))
            else:
                quads.append((a, b, c, d))
        prev = cur
    airfoil_edges = [(i, (i + 1) % n_s) for i in range(n_s)]
    outer_ring = list(ring_ids[-1])
    return nodes, np.asarray(quads), np.asarray(tris), airfoil_edges, outer_ring


# ----------------------------------------------------------------------------
# unstructured exterior with gmsh
# ----------------------------------------------------------------------------

def build_exterior(ring_pts: np.ndarray, R_far: float, h_far: float,
                   boxes, h_ring: float, verbose: bool = False):
    """Triangulate the region between the closed polygon ``ring_pts`` and a
    circle of radius ``R_far`` centered at (0.5, 0).  Returns (nodes, tris,
    farfield_edges, ring_tag_map) where ring_tag_map[i] is the gmsh node tag of
    ring point i and farfield_edges are pairs of gmsh node tags."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("exterior")
    occ = gmsh.model.geo

    n_ring = len(ring_pts)
    ptags = [occ.addPoint(float(x), float(y), 0.0) for x, y in ring_pts]
    ltags = [occ.addLine(ptags[i], ptags[(i + 1) % n_ring]) for i in range(n_ring)]
    inner_loop = occ.addCurveLoop(ltags)

    cx, cy = 0.5, 0.0
    c0 = occ.addPoint(cx, cy, 0.0)
    cp = [occ.addPoint(cx + R_far, cy, 0.0), occ.addPoint(cx, cy + R_far, 0.0),
          occ.addPoint(cx - R_far, cy, 0.0), occ.addPoint(cx, cy - R_far, 0.0)]
    arcs = [occ.addCircleArc(cp[i], c0, cp[(i + 1) % 4]) for i in range(4)]
    outer_loop = occ.addCurveLoop(arcs)
    surf = occ.addPlaneSurface([outer_loop, inner_loop])
    occ.synchronize()

    # no new nodes on the ring polygon: one element per segment
    for lt in ltags:
        gmsh.model.mesh.setTransfiniteCurve(lt, 2)

    # size fields: graded from the ring outward, plus wake refinement boxes
    fields = []
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", ltags)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_ring)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 0.6 * R_far)
    fields.append(f_thr)
    for (x0, x1, y0, y1, h) in boxes:
        f_box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f_box, "VIn", h)
        gmsh.model.mesh.field.setNumber(f_box, "VOut", h_far)
        gmsh.model.mesh.field.setNumber(f_box, "XMin", x0)
        gmsh.model.mesh.field.setNumber(f_box, "XMax", x1)
        gmsh.model.mesh.field.setNumber(f_box, "YMin", y0)
        gmsh.model.mesh.field.setNumber(f_box, "YMax", y1)
        gmsh.model.mesh.field.setNumber(f_box, "Thickness", 1.5)
        fields.append(f_box)
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)            # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    for a in arcs:
        gmsh.model.mesh.setTransfiniteCurve(a, int(0.5 * math.pi * R_far / h_far) + 1)

    gmsh.model.addPhysicalGroup(1, arcs, name="farfield")
    gmsh.model.addPhysicalGroup(1, ltags, name="ring")
    gmsh.model.addPhysicalGroup(2, [surf], name="fluid")
    gmsh.model.mesh.generate(2)

    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :2]
    tag2row = {int(t): i for i, t in enumerate(tags)}
    etypes, etags, enodes = gmsh.model.mesh.getElements(2)
    tris = None
    for et, en in zip(etypes, enodes):
        if int(et) == 2:                                   # 3-node triangle
            tris = np.asarray(en, dtype=int).reshape(-1, 3)
    if tris is None:
        raise RuntimeError(f"gmsh produced no triangles (element types {list(etypes)})")
    far_edges = []
    for a in arcs:
        et, _, en = gmsh.model.mesh.getElements(1, a)
        for t, n in zip(et, en):
            if t == 1:
                far_edges.extend(np.asarray(n, dtype=int).reshape(-1, 2).tolist())
    ring_tags = []
    for pt in ptags:
        t, _, _ = gmsh.model.mesh.getNodes(0, pt, includeBoundary=True)
        ring_tags.append(int(t[0]))
    gmsh.finalize()
    return coords, tag2row, tris, far_edges, ring_tags


# ----------------------------------------------------------------------------
# assembly and SU2 output
# ----------------------------------------------------------------------------

def signed_area(p):
    x, y = p[:, :, 0], p[:, :, 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def write_su2(path, nodes, tris, quads, airfoil_edges, far_edges):
    with open(path, "w") as f:
        f.write("NDIME= 2\n")
        f.write(f"NELEM= {len(tris) + len(quads)}\n")
        k = 0
        for t in tris:
            f.write(f"5 {t[0]} {t[1]} {t[2]} {k}\n"); k += 1
        for q in quads:
            f.write(f"9 {q[0]} {q[1]} {q[2]} {q[3]} {k}\n"); k += 1
        f.write(f"NPOIN= {len(nodes)}\n")
        for i, (x, y) in enumerate(nodes):
            f.write(f"{x:.16e} {y:.16e} {i}\n")
        f.write("NMARK= 2\n")
        f.write("MARKER_TAG= airfoil\n")
        f.write(f"MARKER_ELEMS= {len(airfoil_edges)}\n")
        for a, b in airfoil_edges:
            f.write(f"3 {a} {b}\n")
        f.write("MARKER_TAG= farfield\n")
        f.write(f"MARKER_ELEMS= {len(far_edges)}\n")
        for a, b in far_edges:
            f.write(f"3 {a} {b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="naca0012_lam.su2")
    ap.add_argument("--n-surf", type=int, default=320)
    ap.add_argument("--h1", type=float, default=0.002)
    ap.add_argument("--ratio", type=float, default=1.12)
    ap.add_argument("--n-layers", type=int, default=22)
    ap.add_argument("--n-fan", type=int, default=16)
    ap.add_argument("--R-far", type=float, default=30.0)
    ap.add_argument("--h-far", type=float, default=2.5)
    ap.add_argument("--h-near", type=float, default=0.035)
    ap.add_argument("--h-wake", type=float, default=0.06)
    ap.add_argument("--preview", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    surf = naca0012(args.n_surf)
    bl_nodes, quads, tris_bl, airfoil_edges, outer_ring = build_boundary_layer(
        surf, args.h1, args.ratio, args.n_layers, args.n_fan)
    ring_pts = bl_nodes[outer_ring]
    n_x = polygon_self_intersects(ring_pts)
    print(f"outer ring self-intersections: {n_x}")
    assert n_x == 0, "outer boundary-layer curve self-intersects; reduce layers or ratio"
    h_ring = float(np.median(np.linalg.norm(np.roll(ring_pts, -1, axis=0) - ring_pts, axis=1)))
    print(f"boundary layer: {len(bl_nodes)} nodes, {len(quads)} quads, {len(tris_bl)} fan "
          f"triangles, outer ring {len(outer_ring)} nodes, median ring spacing {h_ring:.4f}")

    # wake boxes: near wake fine, far wake coarser; they cover the wake for
    # angles of attack from 0 to ~25 deg (the wake leaves along the free stream)
    boxes = [(-0.6, 4.0, -1.3, 2.4, args.h_near),
             (4.0, 9.0, -1.3, 4.0, args.h_wake)]
    coords, tag2row, tris_g, far_edges_tags, ring_tags = build_exterior(
        ring_pts, args.R_far, args.h_far, boxes, h_ring, args.verbose)

    # global numbering: boundary-layer nodes first, then gmsh nodes not on the ring
    n_bl = len(bl_nodes)
    gmsh_to_global = {}
    for i, t in enumerate(ring_tags):
        gmsh_to_global[t] = outer_ring[i]
    new_nodes = []
    for t, row in tag2row.items():
        if t not in gmsh_to_global:
            gmsh_to_global[t] = n_bl + len(new_nodes)
            new_nodes.append(coords[row])
    nodes = np.vstack([bl_nodes, np.asarray(new_nodes)])
    lut = np.full(max(gmsh_to_global) + 1, -1, dtype=int)
    for t, g in gmsh_to_global.items():
        lut[t] = g
    tris_ext = lut[np.asarray(tris_g, dtype=int)]
    assert tris_ext.min() >= 0
    far_edges = [(int(lut[a]), int(lut[b])) for a, b in far_edges_tags]
    tris = np.vstack([tris_bl, tris_ext]) if len(tris_bl) else tris_ext

    # drop nodes not referenced by any element (e.g. the circle-centre point gmsh
    # creates) and renumber consecutively
    used = np.zeros(len(nodes), dtype=bool)
    used[np.unique(tris)] = True; used[np.unique(quads)] = True
    if not used.all():
        remap = -np.ones(len(nodes), dtype=int); remap[used] = np.arange(used.sum())
        nodes = nodes[used]; tris = remap[tris]; quads = remap[quads]
        airfoil_edges = [(int(remap[a]), int(remap[b])) for a, b in airfoil_edges]
        far_edges = [(int(remap[a]), int(remap[b])) for a, b in far_edges]
        print(f"dropped {(~used).sum()} unused nodes")

    # orientation and quality checks
    a_t = signed_area(nodes[tris]); a_q = signed_area(nodes[quads])
    flip_t = a_t < 0; flip_q = a_q < 0
    tris[flip_t] = tris[flip_t][:, ::-1]; quads[flip_q] = quads[flip_q][:, ::-1]
    a_t = np.abs(a_t); a_q = np.abs(a_q)
    print(f"total: {len(nodes)} nodes, {len(tris)} triangles, {len(quads)} quads, "
          f"{len(tris) + len(quads)} elements; min areas tri {a_t.min():.2e} quad {a_q.min():.2e}; "
          f"flipped {flip_t.sum()} tris, {flip_q.sum()} quads")
    assert a_t.min() > 0 and a_q.min() > 0

    write_su2(args.out, nodes, tris, quads, airfoil_edges, far_edges)
    print(f"wrote {args.out}")

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        views = [(-0.6, 9.5, -2.0, 4.0), (-0.3, 1.3, -0.5, 0.5),
                 (0.85, 1.15, -0.12, 0.12), (-0.05, 0.12, -0.08, 0.08)]
        for ax, (x0, x1, y0, y1) in zip(axes.ravel(), views):
            sel_t = np.all((nodes[tris][:, :, 0] > x0 - 0.5) & (nodes[tris][:, :, 0] < x1 + 0.5)
                           & (nodes[tris][:, :, 1] > y0 - 0.5) & (nodes[tris][:, :, 1] < y1 + 0.5), axis=1)
            sel_q = np.all((nodes[quads][:, :, 0] > x0 - 0.5) & (nodes[quads][:, :, 0] < x1 + 0.5)
                           & (nodes[quads][:, :, 1] > y0 - 0.5) & (nodes[quads][:, :, 1] < y1 + 0.5), axis=1)
            ax.add_collection(PolyCollection(nodes[tris[sel_t]], facecolor="none",
                                             edgecolor="0.3", linewidth=0.25))
            ax.add_collection(PolyCollection(nodes[quads[sel_q]], facecolor="none",
                                             edgecolor="C3", linewidth=0.25))
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig(args.preview, dpi=130)
        print(f"wrote {args.preview}")


if __name__ == "__main__":
    main()
