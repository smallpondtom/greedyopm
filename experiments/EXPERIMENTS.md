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
| Greedy-OPM (p=2) | greedy modes + **budget-constrained knapsack feature admission**, orthonormal quadratic (`GreedyOPM`) |
| Greedy-OPM (p=[2,3]) | same, orthonormal quadratic **+ cubic** |
| Greedy-OPM (p=[2,3,4]) | same, quadratic **+ cubic + quartic** (greedy pool s=24; only r with n >= C(r+4,4)) |

## Algorithm (budget-constrained selection; no penalty, no shrinkage)
Feature k has closed-form **gain** `G_k = (n||r_k|| - gamma_sp)_+^2/(n+gamma)`
and closed-form **Lipschitz cost** `lambda_k = (n||r_k|| - gamma_sp)_+ *
L_psi_k(delta) / (n+gamma)` (sup-norm gradient bound on the delta-dilated
cloud). Under a budget `L`, features that pass the `gamma_sp` eligibility
threshold are admitted greedily by gain density `G_k/lambda_k` while the
cumulative cost stays within the slack `sqrt(L^2-1)` (knapsack, 1/2-approx
guarantee) — both **in-loop** (each candidate mode is scored with the best
feature set it can afford) and in the exact final selection. Admitted
columns keep their unshrunk least-squares magnitudes; the certificate
`L_hat_Gamma(delta) = sqrt(1+(sum lambda_k)^2) <= L` holds BY CONSTRUCTION.
`L=inf` recovers the plain group-sparse algorithm exactly.

## Tuning protocol (`exp_methods.fit_opm_tuned`)
One **Lipschitz budget `L` per experiment**, chosen by `budget_study.py`
(full fit+tune per L; smallest L within 2% of the best validation error):
**transport `L=100`, SWE `L=3`, NSE `L=316`** (`CFG["budget"]`). Per r: fit
two variants (`greedy` pool s=100; `leading` = leading-r modes), sweep the
`(gamma, gamma_sp)` grid on the **validation window** in reduced space
(cheap re-admissions; every grid point certified), select by min val within
`gamma_sp_tol` → best train → fewest features; prefer `leading` within
`variant_tol`. In-loop gradient probes subsample `n_grad_samples=150`
points; the final certificate pass uses the full dilated cloud.

## Problem setups
- **Transport** (`transport/transport_compare.py`) — periodic
  advection–diffusion pulse (exact spectral solution), `NX=1024`, `NT=1200`,
  `t∈[0,2]`, `nu=1e-4`. Split 50/25/25: training covers one full traversal;
  the future pulse revisits known positions with never-seen profiles.
- **Shallow water 2D** (`shallow_water/shallow_water_compare.py`) — spectral
  SWE2d solver, height field `h` (`d=108²=11664`), broad moderate bump,
  `t∈[0,20]`, 2001 snapshots, split 50/25/25. Coherent sloshing: slow
  n-width decay, reconstructible future.
- **Navier–Stokes** (`nse/`) — Kármán vortex street at Re=30, 2000 velocity
  snapshots (`d=41682`), fit on the **full data** (`nse_compare_full.py`;
  `nse_compare.py` = lossless compressed variant), sensor outputs `y=Cv`.
  Split 70/15/15 so training reaches the saturated shedding limit cycle.

Run (each phase < ~40 s for the sandbox limit; sweeps are resumable):
```
python experiments/budget_study.py                         # L-sweep + L* selection (staged)
python experiments/transport/transport_compare.py sweep    # repeat until "SWEEP DONE"
python experiments/transport/transport_compare.py curves|study|active|field
python experiments/shallow_water/shallow_water_compare.py sweep|curves|study|active|field
python experiments/nse/prep_nse.py                         # once
python experiments/nse/nse_compare_full.py sweep|curves|active|timeseries
python experiments/nse/nse_greedyopm_field.py              # 2 calls: fit stage, then plot
python experiments/stability_study.py budget|certificate|plots|all
python experiments/runtime_compare.py; python experiments/runtime_compare_full.py
```

## Results (relative error, train / test, (admitted features, certificate))

Transport (state; budget L=100):
| r | POD | Greedy-QM | Geelen-QM (a) | OPM p2 (a, cert) | OPM p=[2,3] (a, cert) |
|--|--|--|--|--|--|
| 4 | 0.80/0.77 | 0.48/0.46 | 0.60/0.54 (7) | 0.43/0.39 (10, 85) | 0.23/0.20 (15, 87) |
| 6 | 0.70/0.65 | 0.21/0.43 | 0.40/0.36 (13) | 0.23/0.22 (18, 100) | 0.056/0.089 (24, 79) |
| 8 | 0.60/0.54 | 0.02/2.3 | 0.24/0.56 (17) | 0.18/0.15 (15, 97) | 0.088/0.139 (23, 93) |
| 10 | 0.50/0.44 | 0.001/1.9e2 | 0.14/1.3 (20) | 0.081/0.086 (16, 99) | 0.033/0.123 (21, 98) |

Shallow water (height h; budget L=3; featured r=8):
| r | POD | Greedy-QM | Geelen-QM (a) | OPM p2 (a, cert) | OPM p=[2,3] (a, cert) |
|--|--|--|--|--|--|
| 3 | 0.43/0.545 | 0.32/0.58 | 0.32/0.58 (6) | 0.31/0.46 (2, ≤3) | 0.31/0.456 (3, ≤3) |
| 4 | 0.31/0.384 | 0.18/0.48 | 0.19/0.47 (9) | 0.26/0.363 (5, ≤3) | 0.26/0.362 (4, ≤3) |
| 8 | 0.136/0.323 | 0.015/8 | 0.076/0.329 (17) | 0.138/0.322 (3, ≤3) | 0.136/0.317 (3, ≤3) |
| 10 | 0.100/0.286 | 0.004/54 | 0.072/0.288 (15) | 0.096/0.288 (1, ≤3) | 0.096/0.286 (4, ≤3) |

Navier-Stokes FULL data (state; budget L=316):
| r | POD | Greedy-QM | Geelen-QM (a) | OPM p2 (a, cert) | OPM p=[2,3] (a, cert) |
|--|--|--|--|--|--|
| 4 | 0.43/0.36 | 0.28/0.34 | 0.28/0.34 (10) | 0.24/0.19 (10, 42) | 0.105/0.131 (30, 207) |
| 8 | 0.26/0.23 | 0.08/0.28 | 0.18/0.16 (14) | 0.112/0.131 (30, 199) | 0.055/0.090 (56, 316) |
| 10 | 0.22/0.19 | 0.042/0.34 | 0.17/0.16 (11) | 0.066/0.107 (34, 316) | 0.041/0.097 (58, 316) |

NSE sensor outputs at r=10 (train/test): OPM p23 5.1e-3/1.3e-2, OPM p2
8.8e-3/1.8e-2, Geelen 2.7e-2/2.7e-2, Greedy-QM 5.6e-3/5.8e-2. Field
snapshot (t=11.1, r=10): rel. err 1.9e-2.

## Takeaways
1. **The budget replaces the Jacobian penalty and wins on every problem.**
   The knapsack admission excludes low-value/high-cost features instead of
   shrinking all columns: transport p23 test 0.089 at r=6 (2.7x below the
   best baseline at ANY r); NSE p23 dominates all methods on both windows at
   every r; SWE best-OPM matches or beats POD's test error at every r
   (beats at 7 of 9; ties within 0.4% at r=7,9) with certs <= 3.
2. **Certificates hold by construction** (`L_hat <= L`, verified against
   measured pointwise Jacobian norms in `stability_study.py`) — no outer
   loop, no re-solve, and the fit stays a pure closed-form pipeline.
3. **The budget is a regularizer** (`budget_study.py`, full fit+tune per L):
   test error is U-shaped in L on transport (min at L=100: 0.054/0.093 with
   24/77 features at r=6) and NSE (min at 316); SWE is optimal at the very
   tight L=3. Admitted features grow monotonically with L.
4. **OOD testing still exposes the unregularized methods**: Greedy-QM and
   Fixed-OPM interpolate training data and blow up on the future window
   (test errors up to 10^2).
5. **Cost**: in-loop gradient passes make Greedy-OPM ~3x Greedy-QM at
   r=10 on the full NSE data (27 s vs 8 s incl. all tuning; p23 85 s),
   still d-independent in the selection loop.
6. **Degree enrichment to p=4** (Bensalah-style) pays off only on NSE and
   only at small r: p234 state test 0.101 at r=4 (vs p23's 0.131 — near
   the r=10 accuracy of the other variants). At larger r, and on
   transport/SWE, quartics don't help: their large gradient bounds crowd
   the budget (transport r=6: 0.150 vs p23's 0.089; SWE: identical to
   p23). Heavy p234 entries are fitted via `stage_opm.py` (resumable
   greedy loop, `fit(J_init=..., max_new_iters=...)`).
