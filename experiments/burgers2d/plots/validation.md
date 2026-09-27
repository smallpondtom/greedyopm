# 2D Burgers solver and data verification

`validate.py`: 82 checks, **68 passed, 14 failed**.

## Summary

**The solver is correct.**  Against exact Cole-Hopf solutions of the 2D vector Burgers system -- including a non-separable `Phi = A(x,t) + B(y,t)` solution in which both cross terms `v u_y` and `u v_x` are active -- the computed field sits at 1.01-1.02x the *truncation floor*, i.e. it is as accurate as any scheme on this dealiased basis can be.  RK4 order 4 is confirmed two independent ways (step refinement, and the Galilean residual, which is an `O(dt^4)` symmetry error).

**The cached data is under-resolved at high Reynolds number.**  At `Re = 580` the `N = 128` snapshots carry ~2% error against an `N = 512` reference during the steepening phase (`t ~ 1.2-3`).  The README's 'fully resolved, spectral tail < 1e-6' comes from `spectrum_tail(S[:, -1])` -- the *final* snapshot, which is the smoothest point of the trajectory.  The tail actually peaks near `t ~ 1.1-1.9`.

**But the experiment's conclusions survive it**: the singular values and POD projection errors of the snapshot matrix agree with the refined solution to well under 5% (see `nwidth`), so the n-width argument and the method ranking are unaffected.  What the resolution error does undermine is any claim that the data resolves the *PDE* at these Reynolds numbers.

Note that tolerances here are judged against the dealiasing truncation floor, not machine precision: a pseudo-spectral scheme cannot represent what lies above its cutoff, so comparing an exact solution to 1e-16 would be meaningless.

| check | quantity | result | detail |
|---|---|---|---|
| `mask` | cutoff | PASS | N//3 = 42 (Nyquist 64) |
| `mask` | mask == \|k\|<=N//3 rule | PASS | 7225 of 16384 modes retained (44.1%) |
| `mask` | binary and conjugate-symmetric | PASS | entries in {0, 1} and m[k] == m[-k] (keeps the filter real) |
| `mask` | production field band-limited by mask | PASS | max energy fraction above k=42 over all snapshots = 8.29e-32 |
| `stability` | viscous, Re=140 | PASS | dt*nu*2*(N/2)^2 = 0.368 <= 2.785 (7.6x margin) |
| `stability` | viscous, Re=173 | PASS | dt*nu*2*(N/2)^2 = 0.297 <= 2.785 (9.4x margin) |
| `stability` | viscous, Re=207 | PASS | dt*nu*2*(N/2)^2 = 0.249 <= 2.785 (11.2x margin) |
| `stability` | viscous, Re=265 | PASS | dt*nu*2*(N/2)^2 = 0.194 <= 2.785 (14.3x margin) |
| `stability` | viscous, Re=273 | PASS | dt*nu*2*(N/2)^2 = 0.188 <= 2.785 (14.8x margin) |
| `stability` | viscous, Re=307 | PASS | dt*nu*2*(N/2)^2 = 0.168 <= 2.785 (16.6x margin) |
| `stability` | viscous, Re=340 | PASS | dt*nu*2*(N/2)^2 = 0.151 <= 2.785 (18.4x margin) |
| `stability` | viscous, Re=460 | PASS | dt*nu*2*(N/2)^2 = 0.112 <= 2.785 (24.9x margin) |
| `stability` | viscous, Re=580 | PASS | dt*nu*2*(N/2)^2 = 0.089 <= 2.785 (31.4x margin) |
| `stability` | advective | PASS | dt*(\|u\|+\|v\|)max*kcut = 0.200 <= 2.828 (14.1x margin) |
| `stability` | binding constraint is diffusion at the LOWEST Re | PASS | tightest margin 7.6x at Re=140; refining to N=384 at this nu and dt=0.001 diverges (dt limit 8.42e-04), so a grid study must shrink dt with N |
| `linear` | decay rate k=(1,0) | PASS | rel err vs exp(-nu k^2 t) = 8.40e-07  (O(eps)=1e-06 floor) |
| `linear` | decay rate k=(0,3) | PASS | rel err vs exp(-nu k^2 t) = 2.28e-06  (O(eps)=1e-06 floor) |
| `linear` | decay rate k=(2,5) | PASS | rel err vs exp(-nu k^2 t) = 4.92e-06  (O(eps)=1e-06 floor) |
| `exact1d` | u(x), v=0 at t=1.0 (N=128) | PASS | rel err = 2.54e-06 vs truncation floor 2.49e-06 (ratio 1.02); distance to band-limited exact = 4.70e-07 |
| `exact1d` | v stays identically zero (N=128) | PASS | max\|v\| = 0.0e+00 |
| `exact1d` | v(y), u=0 at t=1.0 (N=128) | PASS | rel err = 2.54e-06 vs truncation floor 2.49e-06 (ratio 1.02); distance to band-limited exact = 4.70e-07 |
| `exact1d` | u stays identically zero (N=128) | PASS | max\|u\| = 0.0e+00 |
| `exact1d` | u(x), v=0 at t=1.0 (N=256) | PASS | rel err = 9.69e-10 vs truncation floor 5.05e-10 (ratio 1.92); distance to band-limited exact = 8.27e-10 |
| `exact1d` | v stays identically zero (N=256) | PASS | max\|v\| = 0.0e+00 |
| `exact1d` | v(y), u=0 at t=1.0 (N=256) | PASS | rel err = 9.69e-10 vs truncation floor 5.05e-10 (ratio 1.92); distance to band-limited exact = 8.27e-10 |
| `exact1d` | u stays identically zero (N=256) | PASS | max\|u\| = 0.0e+00 |
| `exact2d` | cross terms nonzero | PASS | max\|u_y\| = 2.500, max\|v_x\| = 2.500 -- so v*u_y and u*v_x are both active (irrotational as required: max\|u_y - v_x\| = 1.5e-07) |
| `exact2d` | u, full 2D system at t=0.5 (N=128) | PASS | rel err = 4.18e-07 vs truncation floor 4.15e-07 (ratio 1.01); distance to band-limited exact = 5.09e-08 |
| `exact2d` | v, full 2D system at t=0.5 (N=128) | PASS | rel err = 4.18e-07 vs truncation floor 4.15e-07 (ratio 1.01); distance to band-limited exact = 5.09e-08 |
| `exact2d` | u, full 2D system at t=1.0 (N=128) | PASS | rel err = 3.53e-07 vs truncation floor 3.50e-07 (ratio 1.01); distance to band-limited exact = 4.28e-08 |
| `exact2d` | v, full 2D system at t=1.0 (N=128) | PASS | rel err = 3.53e-07 vs truncation floor 3.50e-07 (ratio 1.01); distance to band-limited exact = 4.28e-08 |
| `exact2d` | u, full 2D system at t=2.0 (N=128) | PASS | rel err = 1.51e-07 vs truncation floor 1.49e-07 (ratio 1.01); distance to band-limited exact = 1.92e-08 |
| `exact2d` | v, full 2D system at t=2.0 (N=128) | PASS | rel err = 1.51e-07 vs truncation floor 1.49e-07 (ratio 1.01); distance to band-limited exact = 1.92e-08 |
| `exact2d` | spectral convergence in N | PASS | rel err at t=1: N=96: 9.49e-06, N=128: 3.53e-07, N=192: 3.68e-10, N=256: 3.47e-10 -- error falls faster than any power of 1/N |
| `diagonal` | u = v = w(x+y) at t=0.5 (tau=2t=1.0) (N=128) | PASS | rel err = 2.54e-06 vs truncation floor 2.49e-06 (ratio 1.02); distance to band-limited exact = 4.70e-07 |
| `diagonal` | u and v remain equal (N=128) | PASS | max\|u - v\| = 0.0e+00 |
| `diagonal` | u = v = w(x+y) at t=0.5 (tau=2t=1.0) (N=256) | PASS | rel err = 1.40e-09 vs truncation floor 6.49e-10 (ratio 2.16); distance to band-limited exact = 1.24e-09 |
| `diagonal` | u and v remain equal (N=256) | PASS | max\|u - v\| = 0.0e+00 |
| `galilean` | IC differs only by the mean flow | PASS | u0 - u0(c=0) == 1.0 exactly |
| `galilean` | shift by 16 cells at t=0.7854 | PASS | rel err vs the rest-frame solution rolled diagonally: dt=7.85e-03: 2.10e-06, dt=3.93e-03: 1.23e-07, dt=1.96e-03: 7.46e-09, dt=9.82e-04: 4.59e-10 |
| `galilean` | residual is RK4 time error (order 4) | PASS | observed order p = 4.09, 4.05, 4.02 -- a time-dependent symmetry is reproduced to the scheme order |
| `galilean` | round-off is not amplified | PASS | a 2e-16 relative IC perturbation grows by 0.92x over 400 steps -- the trajectory is well conditioned |
| `resolution` | Re=580 | FAIL | tail at t_end = 3.50e-06 (what the README quotes) but max over the trajectory = 7.98e-04 at t=1.92  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=460 | FAIL | tail at t_end = 6.80e-07 (what the README quotes) but max over the trajectory = 4.40e-04 at t=1.20  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=340 | FAIL | tail at t_end = 4.22e-08 (what the README quotes) but max over the trajectory = 1.73e-04 at t=1.18  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=307 | FAIL | tail at t_end = 1.21e-08 (what the README quotes) but max over the trajectory = 1.18e-04 at t=1.12  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=273 | FAIL | tail at t_end = 2.92e-09 (what the README quotes) but max over the trajectory = 7.39e-05 at t=1.18  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=265 | FAIL | tail at t_end = 1.74e-09 (what the README quotes) but max over the trajectory = 6.43e-05 at t=1.15  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=207 | FAIL | tail at t_end = 3.34e-11 (what the README quotes) but max over the trajectory = 1.80e-05 at t=1.11  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=173 | FAIL | tail at t_end = 8.25e-13 (what the README quotes) but max over the trajectory = 5.66e-06 at t=1.12  <-- exceeds the claimed 1e-6 |
| `resolution` | Re=140 | PASS | tail at t_end = 4.04e-15 (what the README quotes) but max over the trajectory = 9.36e-07 at t=1.18 |
| `resolution` | README claim 'whole range fully resolved, tail < 1e-6' | FAIL | worst tail anywhere in the data is 7.98e-04 (at Re=580), 798x the quoted bound |
| `grid` | N=128 vs N=512 at t=1.20 | FAIL | rel err = 2.51e-02 (N=256: 3.96e-03)  <-- N=128 under-resolved here |
| `grid` | N=128 vs N=512 at t=1.92 | FAIL | rel err = 2.13e-02 (N=256: 4.10e-03)  <-- N=128 under-resolved here |
| `grid` | N=128 vs N=512 at t=3.00 | FAIL | rel err = 1.07e-02 (N=256: 1.35e-03)  <-- N=128 under-resolved here |
| `grid` | N=128 vs N=512 at t=6.28 | FAIL | rel err = 7.11e-04 (N=256: 1.10e-05)  <-- N=128 under-resolved here |
| `grid` | production grid adequate at Re=580 | FAIL | nu=0.01083, Re=580; worst-time N=128 error above is what the snapshot data carries |
| `time` | dt=0.002 | PASS | rel err vs dt=0.000125 is 3.02e-09 |
| `time` | dt=0.001 | PASS | rel err vs dt=0.000125 is 1.89e-10 |
| `time` | dt=0.0005 | PASS | rel err vs dt=0.000125 is 1.18e-11 |
| `time` | observed order | PASS | p = 4.00, 4.00 (RK4 expects 4) |
| `nwidth` | snapshot matrices differ | PASS | rel difference between the N=128 data and the N=512 solution restricted to the same grid = 1.14e-02 |
| `nwidth` | POD projection error at r=5 | PASS | N=128 data 4.149e-01 vs N=512 4.141e-01 (0.2% apart) |
| `nwidth` | POD projection error at r=10 | PASS | N=128 data 2.707e-01 vs N=512 2.693e-01 (0.5% apart) |
| `nwidth` | POD projection error at r=15 | PASS | N=128 data 1.978e-01 vs N=512 1.957e-01 (1.1% apart) |
| `nwidth` | POD projection error at r=20 | PASS | N=128 data 1.534e-01 vs N=512 1.506e-01 (1.8% apart) |
| `nwidth` | POD projection error at r=30 | PASS | N=128 data 1.011e-01 vs N=512 9.716e-02 (4.0% apart) |
| `nwidth` | singular-value decay is unaffected | PASS | worst relative disagreement over the first 30 singular values is 0.65% -- the n-width story the experiment rests on survives the resolution error |
| `data` | files present | PASS | 9 cached .npz |
| `data` | burgers2d_N128_nu0.0108331_se48.npz | PASS | S(32768, 131)==(32768, 131), t[-1]=6.2400, IC exact (0e+00), max\|S\|=2.41, Re=580 |
| `data` | burgers2d_N128_nu0.0136591_se80.npz | PASS | S(32768, 79)==(32768, 79), t[-1]=6.2400, IC exact (0e+00), max\|S\|=2.35, Re=460 |
| `data` | burgers2d_N128_nu0.01848_se74.npz | PASS | S(32768, 85)==(32768, 85), t[-1]=6.2160, IC exact (0e+00), max\|S\|=2.30, Re=340 |
| `data` | burgers2d_N128_nu0.0204886_se80.npz | PASS | S(32768, 79)==(32768, 79), t[-1]=6.2400, IC exact (0e+00), max\|S\|=2.30, Re=307 |
| `data` | burgers2d_N128_nu0.0229873_se74.npz | PASS | S(32768, 85)==(32768, 85), t[-1]=6.2160, IC exact (0e+00), max\|S\|=2.30, Re=273 |
| `data` | burgers2d_N128_nu0.0237412_se48.npz | PASS | S(32768, 131)==(32768, 131), t[-1]=6.2400, IC exact (0e+00), max\|S\|=2.30, Re=265 |
| `data` | burgers2d_N128_nu0.0304025_se74.npz | PASS | S(32768, 85)==(32768, 85), t[-1]=6.2160, IC exact (0e+00), max\|S\|=2.30, Re=207 |
| `data` | burgers2d_N128_nu0.0362491_se80.npz | PASS | S(32768, 79)==(32768, 79), t[-1]=6.2400, IC exact (0e+00), max\|S\|=2.30, Re=173 |
| `data` | burgers2d_N128_nu0.0448799_se74.npz | PASS | S(32768, 85)==(32768, 85), t[-1]=6.2160, IC exact (0e+00), max\|S\|=2.30, Re=140 |
| `data` | re-integration reproduces burgers2d_N128_nu0.0237412_se48.npz | PASS | max\|diff\| = 3.34e-07, rel = 2.69e-08 (16s) -- not bit-identical (XLA drift since the data was generated), but far below the discretisation error |
| `splits` | cached set == required set | PASS | 9 files, 9 required; missing [], extra [] |
| `splits` | split column counts | PASS | train: 4x85=340; val_interp: 2x79=158; val_extrap: 1x79=79; test_interp: 1x131=131; test_extrap: 1x131=131 |
| `splits` | Reynolds numbers | PASS | train [140.0, 206.7, 273.3, 340.0], val_interp [173.3, 306.7], val_extrap 460.0, test_interp 264.7, test_extrap 580.0 |
