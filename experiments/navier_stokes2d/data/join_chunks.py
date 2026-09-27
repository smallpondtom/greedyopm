#!/usr/bin/env python3
"""Reassemble data/naca_aXX.npz from data/_chunks/naca_aXX.npz.partNN (then delete _chunks/)."""
import glob, os
here = os.path.dirname(os.path.abspath(__file__))
chunks = sorted(glob.glob(os.path.join(here, "_chunks", "*.part*")))
names = sorted({os.path.basename(c).split(".part")[0] for c in chunks})
for n in names:
    parts = sorted(c for c in chunks if os.path.basename(c).startswith(n + ".part"))
    out = os.path.join(here, n)
    with open(out, "wb") as f:
        for p in parts:
            f.write(open(p, "rb").read())
    print(f"{n}: {len(parts)} parts -> {os.path.getsize(out) / 1e6:.1f} MB")
print("done; data/_chunks/ can be deleted")
