# Greedy-OPM experiments: transport, shallow water 2D, Navier-Stokes

Comparison of six methods on three transport-dominated problems with a
**temporal train / validation / test split** (`exp_split.py`): the first
window of timesteps trains the models, the next window (near future) tunes
the Greedy-OPM hyperparameters, and the final window (far future) is held out
for the reported errors — a genuine forward-in-time out-of-distribution test.

## Methods
| Label | What it is |
|---|---|
| POD | leading-r linear projection (reference floor) |
| Greedy-QM | Schwerdtner–Peherstorfer greedy mode selection, quadratic monomials (`qm/quadmani.py`) |
| Geelen-QM | leading-r POD modes + group-sparse quadratic column selection (`qm/geelen_qm.py`, FISTA) |
| Greedy-Fixed-OPM (p=2) | greedy modes, **all** orthonormal quadratic features (`GreedyFixedOPM`) |
| Greedy-OPM (p=2) | greedy modes **+ feature selection + Lipschitz budget**, orthonormal quadratic (`GreedyOPM`) |
| Greedy-OPM (p=[2,3]) | same, orthonormal quadratic **+ cubic** |

## Budget protocol (pure Algorithm 4; `exp_methods.fit_opm_tuned`)
One **Lipschitz budget `L` per experiment**, chosen by `budget_study.py` as
the smallest `L` whose tuned validation error is within 2% of the best over
all budgets: **transport `L=1e5`, SWE `L=3`, NSE `L=1e3`** (set as
`CFG["budget"]` in each compare script). Fit once per mode-selection variant
(`greedy` pool `s=100`; `leading` = leading-r modes), then sweep the full
`(gamma, gamma_sp)` grid on the **validation window** in reduced space
(closed-form re-solves; no refits). The active set is the paper's group
soft-threshold; **`mu` is never tuned** — for each grid point it is set to
the smallest value whose certificate satisfies `L̂_Γ(δ) ≤ L` (the fixed point
of Algorithm 4's outer `mu`-loop, closed form from the cached per-feature
gradient bounds; surrogate lifting throughout, no exact polish). Selection:
minimize validation error; within `gamma_sp_tol`, prefer the best training
error, then the fewest active features; prefer the leading-modes variant
within `variant_tol`. Every reported model carries the certificate
`L̂_Γ(0.1) ≤ L`.

## Problem setups
- **Transport** (`transport/transport_compare.py`) — periodic
  advection–diffusion pulse (exact spectral solution), `NX=1024`, `NT=1200`,
  `t∈[0,2]` (the pulse crosses the domain once per unit time), `nu=1e-4`.
  Split 50/25/25: training covers one full traversal; the future pulse
  revisits known positions with never-seen (decayed/widened) profiles.
- **Shallow water 2D** (`shallow_water/shallow_water_compare.py`) — spectral
  SWE2d solver, height field `h` (`d=108²=11664`), broad moderate bump
  (`ic_sharpness=1.5`, `bump_amp=0.25`), `t∈[0,20]`, 2001 snapshots, split
  50/25/25. Coherent sloshing: slow n-width decay, reconstructible future.
- **Navier–Stokes** (`nse/`) — Kármán vortex street at Re=30 (flow over a
  cylinder), 2000 velocity snapshots (`d=41682`), fit on the **full data**
  (`nse_compare_full.py`; `nse_compare.py` is the lossless compressed
  variant), sensor outputs `y=Cv`. Split 70/15/15 so the training window
  reaches the saturated shedding limit cycle (with 50% training the future
  has a 40% out-of-span floor and NO method can reconstruct it).

Run (each phase < ~35 s for the sandbox limit; sweeps are resumable):
```
python experiments/budget_study.py                         # L-sweep + L* selection
python experiments/transport/transport_compare.py sweep    # repeat until "SWEEP DONE"
python experiments/transport/transport_compare.py curves|study|active|field
python experiments/shallow_water/shallow_water_compare.py sweep|curves|study|active|field
python experiments/nse/prep_nse.py                         # once
python experiments/nse/nse_compare_full.py sweep|curves|active|timeseries
python experiments/nse/nse_greedyopm_field.py              # test-snapshot contour
python experiments/stability_study.py mu|certificate|all
python experiments/runtime_compare.py; python experiments/runtime_compare_full.py
```

## Results (relative reconstruction error, train / test, (active features, certificate))

Transport (state, train/test; budget L=1e5):
| r | POD | Greedy-QM | Geelen-QM (a) | OPM p2 (a, cert) | OPM p=[2,3] (a, cert) |
|--|--|--|--|--|--|
| 4 | 0.80/0.77 | 0.48/0.46 | 0.60/0.54 (7) | 0.44/0.45 (10, 795) | 0.53/0.51 (13, 322) |
| 8 | 0.60/0.54 | 0.02/2.3 | 0.24/0.56 (17) | 0.45/0.42 (10, 339) | 0.33/0.34 (145, 1e5) |
| 9 | 0.55/0.49 | 0.005/1.5e2 | 0.18/0.35 (20) | 0.48/0.46 (10, 282) | 0.22/0.23 (51, 1e5) |
| 10 | 0.50/0.44 | ~0/1.9e2 | 0.14/1.3 (20) | 0.50/0.44 (3, 78) | 0.22/0.23 (62, 1e5) |

Shallow water (height h, train/test; budget L=3; featured r=8):
| r | POD | Greedy-QM | Geelen-QM (a) | OPM p2 (a, cert) | OPM p=[2,3] (a, cert) |
|--|--|--|--|--|--|
| 3 | 0.43/0.55 | 0.32/0.58 | 0.32/0.58 (6) | 0.30/0.46 (5, ≤3) | 0.40/0.38 (5, ≤3) |
| 4 | 0.31/0.38 | 0.18/0.48 | 0.19/0.47 (9) | 0.27/0.37 (10, ≤3) | 0.27/0.37 (30, ≤3) |
| 8 | 0.14/0.32 | 0.015/8 | 0.076/0.33 (17) | 0.12/0.318 (5, ≤3) | 0.119/0.314 (17, ≤3) |
| 10 | 0.10/0.286 | 0.0037/54 | 0.072/0.288 (15) | 0.093/0.285 (7, ≤3) | 0.090/0.284 (4, ≤3) |

Navier-Stokes FULL data (state, train/test; budget L=1e3):
| r | POD | Greedy-QM | Geelen-QM (a) | OPM p2 (a, cert) | OPM p=[2,3] (a, cert) |
|--|--|--|--|--|--|
| 4 | 0.43/0.36 | 0.28/0.34 | 0.28/0.34 (10) | 0.24/0.19 (10, 42) | 0.105/0.131 (30, 207) |
| 8 | 0.26/0.23 | 0.08/0.28 | 0.18/0.16 (14) | 0.099/0.125 (35, 315) | 0.038/0.090 (84, 993) |
| 10 | 0.22/0.19 | 0.042/0.34 | 0.17/0.16 (11) | 0.055/0.155 (53, 796) | 0.042/0.115 (71, 719) |

NSE sensor outputs at r=10 (train/test): OPM p23 5.6e-3/1.4e-2, Geelen
2.7e-2/2.7e-2, Greedy-QM 5.6e-3/5.8e-2, POD 3.4e-2/2.7e-2.

## Takeaways
1. **Out-of-distribution testing exposes the unregularized methods**:
   Greedy-QM and Fixed-OPM interpolate training data at higher r but blow up
   on the future window (test errors up to 10²).
2. **Every reported Greedy-OPM model is certified** (`L̂_Γ(0.1) ≤ L`) with
   one budget per experiment fixed a priori from the budget study — nothing
   is tuned on the test window and `mu` is not tuned at all.
3. **The budget is a regularizer, not just a guarantee** (`budget_study.py`):
   on transport and NSE the validation-best budget is *finite* (1e5 / 1e3);
   on SWE the very tight L=3 is optimal — heavily damped liftings transfer
   best to the future wave field. Feature counts grow with L: looser budgets
   license denser certified dictionaries.
4. **Results under the pure Algorithm-4 pipeline**: transport — OPM p23 is
   the only method improving at large r (test 0.23 at r≥9, below any
   baseline at any r) though at r=7 Geelen-QM (0.236) wins locally; SWE — at
   every r≥2 an OPM variant beats POD on both windows (split across variants
   at r=7) while no baseline does; NSE — OPM p23 dominates all methods on
   both windows at every r (state test 0.090 at r=8, ~1.8x below Geelen-QM).
5. **Stability study** (`stability_study.py`): at `mu=0` the full-feature
   manifold's test error explodes (transport: 7.8e3); the surrogate targets
   the certificate (drives it monotonically to 1, guaranteeing termination
   of the budget loop for any `L>1`); the optional exact penalty
   (`polish_exact`, not used in the main pipeline) minimizes the true
   Jacobian norm (tighter actual stability, looser certificate).
