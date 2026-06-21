# Greedy-OPM experiments: linear transport & shallow water 2D

Comparison of six methods on two transport-dominated problems, with an
even/odd snapshot split (even = train, odd = test).

## Methods
| Label | What it is |
|---|---|
| POD | leading-r linear projection (reference floor) |
| Greedy-QM | Schwerdtner–Peherstorfer greedy mode selection, quadratic monomials (`qm/quadmani.py`) |
| Geelen-QM | leading-r POD modes + group-sparse quadratic column selection (`qm/geelen_qm.py`, FISTA) |
| Greedy-Fixed-OPM (p=2) | greedy modes, **all** orthonormal quadratic features (`GreedyFixedOPM`) |
| Greedy-OPM (p=2) | greedy modes **+ feature selection**, orthonormal quadratic (`GreedyOPM`) |
| Greedy-OPM (p=[2,3]) | greedy modes + feature selection, orthonormal quadratic **+ cubic** |

## Files added
```
experiments/
  exp_methods.py                     # method registry + resumable r-sweep (checkpoint + time budget)
  exp_plots.py                       # error-vs-r curves + field/error panels
  data/shallow_water.py             # faithful spectral SWE2d solver (h, phi)
  qm/geelen_qm.py                   # Geelen group-sparse quadratic manifold
  transport/transport_compare.py    # transport driver (phases: sweep|curves|field|all)
  shallow_water/shallow_water_compare.py
  */plots/*.png|pdf                 # outputs + checkpoint JSONs
```
Run (each phase < ~35s to fit the sandbox limit; sweep is resumable):
```
python experiments/transport/transport_compare.py sweep   # repeat until "SWEEP DONE"
python experiments/transport/transport_compare.py curves
python experiments/transport/transport_compare.py field
# same for experiments/shallow_water/shallow_water_compare.py
```

## Problem setups
- **Transport** — advecting Gaussian pulse, `nx=512`, `nt=800` (400 train / 400 test),
  width `1e-3`, speed `5`, `t∈[0,0.2]`. Candidate pool `s=50`, `r=1..10`.
- **Shallow water 2D** — faithful Fourier pseudo-spectral + RK4 solver of the documented
  Hamiltonian SWE (height `h`, velocity potential `φ`) on periodic `[-π,π]²`, bump IC
  `h₀ = 1 + 0.33 e^{-α(x²+y²)}`, `α=4`, `N=64`. Hamiltonian conserved to ~1e-13.
  We model the **height field** `h` (`d = N² = 4096`); 801 snapshots (401 train / 400 test).
  Candidate pool `s=40`, `r=1..10`.

Feature-selection threshold for Greedy-OPM uses an adaptive `gamma_sp` retaining
features whose gain exceeds `τ=1e-6` of the largest (near-lossless but still sparse).

## Results (relative reconstruction error, **test** split)

Transport:
| r | POD | Greedy-QM | Geelen-QM | Fixed-OPM p2 | OPM p2 | OPM p=[2,3] |
|--|--|--|--|--|--|--|
| 4 | 8.0e-1 | 5.2e-1 | 5.6e-1 | 3.9e-1 | 3.9e-1 | **4.1e-2** |
| 6 | 7.0e-1 | 2.2e-1 | 2.5e-1 | 1.0e-1 | 1.0e-1 | **3.7e-4** |
| 8 | 6.1e-1 | 3.4e-2 | 1.3e-1 | 6.1e-3 | 6.1e-3 | **8.0e-5** |
| 10 | 5.1e-1 | 6.4e-4 | 9.0e-2 | 1.2e-4 | 2.5e-4 | 1.7e-4 |

Shallow water (height h):
| r | POD | Greedy-QM | Geelen-QM | Fixed-OPM p2 | OPM p2 | OPM p=[2,3] |
|--|--|--|--|--|--|--|
| 4 | 5.7e-1 | 3.6e-1 | 3.8e-1 | 2.4e-1 | 2.4e-1 | **7.6e-2** |
| 6 | 4.4e-1 | 1.5e-1 | 1.7e-1 | 1.1e-1 | 1.1e-1 | **1.1e-3** |
| 8 | 3.2e-1 | 7.1e-2 | 1.1e-1 | 4.5e-2 | 4.5e-2 | **6.4e-4** |
| 10 | 2.5e-1 | 2.5e-2 | 6.8e-2 | 7.4e-3 | 7.4e-3 | **4.6e-4** |

Train and test curves are nearly identical (no overfitting).

## Takeaways
1. **Greedy mode selection helps**: Greedy-QM < Geelen-QM (which uses leading-r modes), the gap widening with r.
2. **Orthonormal features beat monomials**: Greedy-Fixed-OPM (p=2) < Greedy-QM at equal degree — the QM Gram matrix is increasingly ill-conditioned.
3. **Feature selection is ~free**: Greedy-OPM (p=2) tracks the all-features Fixed-OPM while using fewer features (e.g. transport r=10: 49/55).
4. **Degree enrichment is decisive**: Greedy-OPM (p=[2,3]) is 1–3 orders of magnitude better at low-to-mid r on both problems.

## Implementation note (recommend a fix)
`GreedyFixedOPM` caps polynomial degrees so the feature count stays ≤ `n`
(`_effective_degrees`), but **`GreedyOPM` has no such guard**. With p=[2,3] and
only ~100 SWE training snapshots, the degree-≤3 feature count exceeded `n`,
violating unisolvence (`n ≥ #features`); the empirical Arnoldi basis became
rank-deficient and reconstructions blew up to ~1e29. I worked around it here by
using denser snapshots (401 train > 286 max features). **Recommend** adding the
same degree/feature-count guard (or a clear error) to `GreedyOPM.fit`.

---

# Third experiment: transient Navier–Stokes (Kármán vortex, Re=30)

Heiland/Benner operator-inference dataset in `Dev/Research/transient_ns_pbenner`.
2000 velocity snapshots over `t∈[0,12]` (`d = 41682` interior DOFs), with a
`6×d` sensor matrix `C` (`cmat`) and the output `y(t) = C v(t) ∈ ℝ⁶` — exactly
the `cmat` / output used in the source `runme.py` (read from the index JSON;
mass matrix `M` also available). No mesh is shipped, so we report the **sensor
outputs** (as requested) rather than spatial fields.

### Data pipeline (`experiments/nse/`)
1. `compile_nse.py` — reads `simulation_nse/sc_30_0-12_2000.json`, loads the 2000
   `*_vel.npy` snapshots restricted to `invinds`, **in time order**, and writes one
   compiled file: **`transient_ns_pbenner/nse_snapshots.npz`** (`V` (d×2000), `trange`,
   `C` (6×d)).
2. `prep_nse.py` — even/odd split; lossless **method-of-snapshots POD** of the
   training set (`K=408`, captures ≥1−8e-7 of train energy via a Gram-matrix
   eig), giving reduced coordinates `A=Q^T(V−mean)`, `C_red=C Q`, `y_ref=C mean`,
   true outputs `Y=C V`, and the (tiny) truncation floors. This makes the
   `d=41682` problem cheap (every method's internal SVD is now `408×n`).
3. `nse_compare.py` — resumable sweep + plots (`sweep|curves|timeseries|all`).

POD decays slowly here (only ~96% energy in 10 modes), i.e. a genuinely hard,
transport-dominated case. `r=1..10` (matching `runme.py`'s `rv=10`), pool `s=25`.

### Results — sensor-output error ‖y−ŷ‖/‖y‖ (test)
| r | POD | Greedy-QM | Geelen-QM | Fixed-OPM p2 | OPM p2 | OPM p=[2,3] |
|--|--|--|--|--|--|--|
| 4 | 5.4e-2 | 4.3e-2 | 5.0e-2 | 3.0e-2 | 3.0e-2 | 1.3e-2 |
| 6 | 4.1e-2 | 2.6e-2 | 3.3e-2 | 1.6e-2 | 1.6e-2 | **2.0e-3** |
| 8 | 4.0e-2 | 1.1e-2 | 2.9e-2 | 8.4e-3 | 8.4e-3 | **2.0e-4** |
| 10 | 3.4e-2 | 4.7e-3 | 1.6e-2 | 3.3e-3 | 3.3e-3 | **1.2e-4** |

Same ordering as the other two problems (Greedy-QM > Geelen-QM; orthonormal OPM >
monomial QM; feature selection ≈ lossless; cubic enrichment decisive — ~40× lower
output error than Greedy-QM at r=8). Train ≈ test throughout.

### Plots
`nse_state_error_vs_r.png`, `nse_output_error_vs_r.png`,
`nse_output_timeseries.png` (the six `y(t)=Cx` traces, ground truth vs every
method, full trajectory at r=10).

Note: errors here are Euclidean (consistent with the other experiments and with
what the methods optimize). The dataset also provides the FEM mass matrix `M` if
you'd prefer the physical `‖·‖_M` norm used in `runme.py`.

