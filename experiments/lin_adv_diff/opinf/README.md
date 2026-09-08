# OpInf on learned manifolds: standard vs geometric manifold-Galerkin

Implements and tests the **geometric manifold-Galerkin (GMG) Operator
Inference** of the working note ``paper/.../notes/opinf_opm.tex`` on the
periodic linear advection-diffusion problem, against the **standard (plain
latent) OpInf** baseline, on all six learned manifolds (POD, Greedy-QM,
Greedy-CM, Geelen-QM, Greedy-OPM p=2/p=3).  The core machinery lives in the
package module ``src/greedyopm/opinf.py``; this directory holds the
experiment.

## The two reduced models (note, Secs. 2-3)

For a manifold ``Gamma(shat) = s_ref + V shat + Vbar p(shat)`` and a linear
full-order model, the exact reduced dynamics of an on-manifold state are affine
in the feature triple ``(1, shat, p(shat))``:

* **Standard OpInf** evolves the encoder coordinates directly,
  ``shat' = c + A shat + N p(shat)``, with the operators fit non-intrusively
  (FBCT4 fourth-order derivative estimates, order-wise Tikhonov).  The fit is
  excellent -- and the simulated ROM still tends to diverge: the latent ODE
  discards the manifold metric, integration drift leaves the certified collar,
  and the polynomial features blow up.  **This failure is deliberate and
  reported**: it is the empirical case for the geometric model.

* **GMG OpInf** keeps the metric by solving the tangent condition
  ``J_Gamma shat' = f`` in least squares:

      shat' = M(shat)^{-1} ( a(shat) + J_p(shat)^T b(shat) ),
      M(shat) = I + J_p^T (Vbar^T Vbar) J_p,

  with ``a = V^T f`` and ``b = Vbar^T f`` learned exactly like the plain
  operators.  ``M`` is SPD with ``cond(M) <= L^2`` wherever the certificate
  holds, so the manifold's Lipschitz budget conditions the reduced dynamics --
  where features steepen, ``M^{-1}`` contracts the velocity and resists the
  drift that destroys the plain model.

## The two regimes (``OPINF_REGIME``, see ``problem_opinf.py``)

* ``gentle`` (default) -- the regime where the reduced dynamics are the
  question.  Softer physics (``nu = 2e-3``, pulse width ``8e-3``), nine
  training speeds on ``[1, 5]``, window ``T = 0.35``; validation speeds
  ``{2.25, 2.75, 4.25}`` (ROM budget selection), interpolation test speed
  ``c = 3.3``.  ROM certificate collar ``delta = 1`` (the reduced trajectory
  visits a dilated neighborhood of the fit data).
* ``manifold`` -- the reconstruction experiment's settings verbatim
  (``nu = 1e-4``, near-delta pulse, speeds ``{1,4,7,10}``, test ``c = 7.719``;
  ``delta = 0.1`` as in that experiment): a deliberately brutal stress case in
  which every ROM-stable manifold at ``r <= 10`` carries ~50% representation
  error.  It demonstrates *stability*, not accuracy.

In both regimes: one operator set per training speed (dense trajectories, so
FBCT4 velocities are resolved at the fastest speed), entrywise
natural-cubic-spline interpolation of the operators at a query speed, RK4 on
the same grid, relative Frobenius errors of the decoded trajectory.
Extrapolated speeds are not evaluated.

## Budget selection for the ROM (note, Sec. 6)

Reconstruction-selected budgets are loose: past the certified collar a loose
budget "returns the plain model's divergence", and empirically the GMG-OPM ROM
diverges at those budgets (archived in
``plots/opinf_manifold_results_reconL.json``).  The budget of each Greedy-OPM
variant is therefore selected by minimizing the **reduced-model** error on the
held-out *validation* speeds (spline operators; the test speed is never
touched), **per reduced dimension**: ROM stability is a per-``r`` property.
The selection sweeps ``(L_ref, L)`` pairs -- the manifold fit at a reference
budget, recalibrated by the closed-form ``set_budget`` -- and the deployed
model is constructed exactly as selected (``_build_mrom``).

## Files

* ``../../../../src/greedyopm/opinf.py`` -- FBCT4 derivatives, operator
  regression, plain + GMG rollouts, spline operator interpolation (the
  package module; ``from greedyopm import opinf``).
* ``problem_opinf.py``     -- trajectory data for both regimes.
* ``opinf_methods.py``     -- per-method manifold/feature adapters + ``ManifoldROM``.
* ``budget_select_rom.py`` -- reduced-model budget selection (global + per-r).
* ``run_opinf.py``         -- driver; phases ``budget | sweep | plots | report | all``.
* ``plots_opinf.py``       -- journal-quality figures.

## Reproduce

```bash
OPINF_REGIME=gentle   python run_opinf.py budget   # ROM budgets (validation speeds)
OPINF_REGIME=gentle   python run_opinf.py sweep    # 6 methods x r=1..10 x {standard, GMG}
OPINF_REGIME=gentle   python run_opinf.py plots    # figures + report
OPINF_REGIME=manifold python run_opinf.py all      # harsh stress case
```

Outputs are prefixed ``plots/opinf_<regime>_*``.

## Results: gentle regime (accuracy)

``opinf_gentle_error_vs_r_{plain,gmg}.*``, ``opinf_gentle_field.*``,
``opinf_gentle_report.json``.  Headlines (interpolation-test relative error):

1. **Standard OpInf is fragile on every nonlinear manifold.**  The monomial
   manifolds (Greedy-QM/CM, Geelen-QM) diverge at 5-6 of 10 reduced
   dimensions; even the budgeted OPM variants diverge at 2-3 of 10.  Only POD
   is stable everywhere.
2. **Greedy-OPM + GMG + the ROM-selected budget never diverges** (0/10 for
   both variants), while every monomial manifold under GMG still blows up for
   ``r >= 2``-``3`` (no orthonormal features, no certificate -- the mass
   matrix cannot rescue them).
3. **GMG-OPM beats POD across the pre-asymptotic range**: r=3: 0.153 vs 0.807
   (5x), r=4: 0.233 vs 0.381, r=5: 0.170 vs 0.432 (2.5x), r=6: 0.171 vs
   0.186, r=7: 0.178 vs 0.212, r=9: 0.083 vs 0.135; and stays competitive at
   r=8 (0.080 vs 0.078, p=3) and r=10 (0.037 vs 0.028, p=3).  The decoded
   fields at ``c = 3.3`` show both OPM-GMG ROMs tracking the advecting pulse.
4. **The tight ROM budget also stabilizes the plain model where it holds**:
   the p=3 plain OpInf reaches 0.026 at r=10 -- but with no metric it has no
   safety margin, and diverges at r=6, 7 while GMG stays at ~0.17.

## Results: manifold (harsh) regime (stability stress test)

``opinf_manifold_*``; ``delta = 0.1``.  Headlines:

1. **Standard OpInf diverges for every polynomial manifold** (monomials at 8
   of 10 dimensions; OPM survives only where the tight budget makes it nearly
   linear).  Only POD is stable, at its Kolmogorov-barrier accuracy
   (~0.75-0.92).
2. **The monomial manifolds diverge under GMG too** (7-8 of 10 dimensions).
   Greedy-CM is the sharpest illustration: the best reconstruction floor of
   all (2.1e-04 at r=10) and a catastrophically unstable ROM in every
   configuration -- reconstruction quality does not transfer to dynamical
   viability.
3. **Greedy-OPM + GMG + the ROM-selected budget never diverges** (0/10 both
   variants); r=10 interp error **0.542 (p=2) / 0.591 (p=3) vs 0.922 for
   POD** -- at the tight-budget manifolds' own reconstruction floors
   (~0.45-0.52), i.e. the dynamics are as good as the representation allows,
   and the representation is the binding constraint in this regime.
4. **At the loose reconstruction budgets the GMG-OPM ROM diverges for
   ``r >= 4``** (``opinf_manifold_results_reconL.json``) -- the budget
   conditions the mass matrix, and carrying dynamics demands a tighter ``L``
   than reconstruction: neither the certificate nor GMG suffices alone.

## Affine-in-parameter variant (``OPINF_VARIANT=affine``, the default)

Two learning upgrades over the original pipeline (kept as
``OPINF_VARIANT=legacy``), applied to the gentle regime:

1. **Exact velocities**: the FOM solution is spectral, so the regression
   targets are the exact time derivatives ``u_t = -c u_x + nu u_xx``
   (``problem_opinf.exact_velocity``, FFT differentiation) -- no FBCT4
   truncation error (~0.25% relative) in the operator fits.
2. **Pooled affine-in-parameter operators**: the FOM operator is affine in
   the wave speed, so the *intrusive* reduced operators satisfy
   ``O(c) = O0 + tau O1`` exactly (``tau`` the centered/scaled speed).  One
   regression pools all nine training speeds with per-snapshot design
   ``[d; tau d]`` (``greedyopm.opinf.infer_operators_affine``), replacing
   the nine per-speed fits and the entrywise spline interpolation.  The
   per-speed *learned* operators are themselves far from affine in ``c``
   (each fit projects the closure error along its own trajectory) -- which
   is what made the splines noisy; the pooled fit removes that entirely.

Outputs carry the ``opinf_gentle_affine_*`` prefix.  Headlines
(interpolation-test relative error):

1. **The stability window opens up**: with the operator noise gone, the
   ROM-selected budgets jump from ``L* ~ 1.5-43`` to ``L* ~ 100-125`` --
   loose enough to use the manifolds' real reconstruction floors.
2. **GMG + Greedy-OPM (p=3) is now monotone and 8-50x below POD at every
   r >= 2**: 0.113 (r=3, vs 0.788), 0.0141 (r=5, vs 0.425), 0.00695 (r=6),
   0.00337 (r=7), 0.00177 (r=8), **0.00144 (r=10, vs 0.0281 POD)**.
   p=2 reaches 0.00127 at r=10.  Both variants: 0/10 diverged.
3. **Standard OpInf remains fragile** even with exact velocities and affine
   operators: the OPM variants diverge at 3-4 of 10 dimensions under the
   plain model, and the monomial manifolds under GMG at 8 of 10 -- the
   metric + certificate combination is still what carries stability.
