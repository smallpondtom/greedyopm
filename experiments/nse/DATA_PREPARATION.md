# NSE Data Preparation — Method-of-Snapshots Compression

This document explains, with full mathematical detail, how the large transient
Navier–Stokes (Kármán vortex street, Re = 30) dataset is prepared for the
method-comparison experiment. The goal is to make a $d = 41{,}682$-dimensional
state problem cheap enough to fit **6 methods × 10 reduced dimensions**
repeatedly, *without ever working in the full state dimension and without losing
accuracy*.

The pipeline is:

| step | file | output |
|---|---|---|
| 1. assemble raw snapshots | [`compile_nse.py`](compile_nse.py) | `data/nse_snapshots.npz` |
| 2. **compress (this doc)** | [`prep_nse.py`](prep_nse.py) | `data/nse_compressed.npz` |
| 3. recover DOF coordinates | [`recover_dof_coords.py`](recover_dof_coords.py) | `coords_stored.npy` |
| 4. compare methods | [`nse_compare.py`](nse_compare.py) | `plots/` |

---

## 1. The data and why the raw form is expensive

`compile_nse.py` assembles the state matrix and sensor (output) matrix

$$
V \in \mathbb{R}^{d\times n},\qquad C \in \mathbb{R}^{q\times d},\qquad
y(t) = C\,v(t),
$$

with the concrete sizes for this dataset

$$
d = 41{,}682 \ \text{interior velocity DOFs}, \qquad
n = 2000 \ \text{snapshots}, \qquad
q = 6 \ \text{sensor outputs}.
$$

Raw `V` is stored as float32 (~333 MB). The comparison experiment fits POD,
Greedy-QM, Geelen-QM, Greedy-Fixed-OPM, and two Greedy-OPM variants, each across
$r = 1,\dots,10$. Several of these perform repeated SVD-like work whose cost
scales with the **state dimension** $d$. Carrying a 41,682-row matrix through
all of those fits is the bottleneck — it is the linear algebra, not the physics,
that is expensive.

---

## 2. Train / test split and centering

Even columns become training, odd columns become test
([`prep_nse.py:39`](prep_nse.py#L39)). The **training** mean is subtracted from
both sets, so the test set never uses its own statistics:

$$
s_{\mathrm{ref}} = \frac{1}{n_{\mathrm{tr}}}\sum_{j} (V_{\mathrm{tr}})_{:,j},
\qquad
X_{\mathrm{tr}} = V_{\mathrm{tr}} - s_{\mathrm{ref}}\,\mathbf 1^\top \in \mathbb{R}^{d\times 1000},
\qquad
X_{\mathrm{te}} = V_{\mathrm{te}} - s_{\mathrm{ref}}\,\mathbf 1^\top .
$$

Here $n_{\mathrm{tr}} = n_{\mathrm{te}} = 1000$.

---

## 3. Key observation: an orthonormal change of coordinates is free

The training columns span a subspace of dimension at most
$n_{\mathrm{tr}} = 1000 \ll d$. Let $U \in \mathbb{R}^{d\times K}$ be an
orthonormal basis of that span, with thin SVD

$$
X_{\mathrm{tr}} = U\,\Sigma\,W^\top,\qquad
\Sigma = \operatorname{diag}(\sigma_1,\dots,\sigma_K),\quad
W \in \mathbb{R}^{1000\times K},\quad
U^\top U = I_K .
$$

Every method compared here builds its reduced basis **inside** $\operatorname{span}(U)$
and reconstructs the state through linear-plus-nonlinear maps of the projected
coordinates. Because $U$ is orthonormal it is an **isometry**: for any vectors
$x, x'$ in the training span and their coordinates $a = U^\top x$, $a' = U^\top x'$,

$$
\langle x, x'\rangle = \langle Ua, Ua'\rangle = \langle a, a'\rangle,
\qquad \|x\| = \|a\| .
$$

Norms, inner products, and singular values are therefore **identical** whether
evaluated on the full vector $x \in \mathbb{R}^{d}$ or on its coordinate vector
$a \in \mathbb{R}^{K}$. Consequently, running any of these methods on the
coordinate matrix

$$
A = U^\top X \in \mathbb{R}^{K\times n}
$$

reproduces *exactly* the same reconstruction errors and the same selected modes
(up to the fixed rotation $U$) as running on the full $X$ — but in a
$K$-dimensional space rather than a $41{,}682$-dimensional one.

For this dataset the Re = 30 limit cycle is genuinely low rank:

$$
\boxed{K = 408}\qquad(\text{after the energy truncation of §5}),
$$

so the working dimension shrinks by roughly **100×**.

---

## 4. Method of snapshots: coordinates without ever forming $U$

The basis $U$ is a $d \times K$ object and is **never materialized**. Instead we
use the small Gram matrix ([`prep_nse.py:48`](prep_nse.py#L48)):

$$
G = X_{\mathrm{tr}}^\top X_{\mathrm{tr}} \in \mathbb{R}^{1000\times1000},
\qquad
G = X_{\mathrm{tr}}^\top X_{\mathrm{tr}}
  = W\Sigma\,(U^\top U)\,\Sigma W^\top
  = W\,\Sigma^2\,W^\top .
$$

So a single $1000\times1000$ symmetric eigenproblem yields the right singular
vectors $W$ and the singular values $\sigma_i = \sqrt{\lambda_i}$
([`prep_nse.py:49-52`](prep_nse.py#L49-L52)). The coordinates then follow
algebraically.

**Training coordinates** are exact, since $U^\top U = I$:

$$
\boxed{\,A_{\mathrm{tr}} = U^\top X_{\mathrm{tr}} = U^\top U\,\Sigma W^\top
      = \Sigma W^\top = \operatorname{diag}(\sigma)\,W^\top\,}
\qquad\text{([`prep_nse.py:64`](prep_nse.py#L64))}
$$

**Test coordinates** use $U = X_{\mathrm{tr}} W \Sigma^{-1}$ (from the SVD), hence
$U^\top = \Sigma^{-1} W^\top X_{\mathrm{tr}}^\top$, giving

$$
\boxed{\,A_{\mathrm{te}} = U^\top X_{\mathrm{te}}
      = \Sigma^{-1} W^\top \bigl(X_{\mathrm{tr}}^\top X_{\mathrm{te}}\bigr)\,}
\qquad\text{([`prep_nse.py:65`](prep_nse.py#L65))}
$$

The cross-Gram $X_{\mathrm{tr}}^\top X_{\mathrm{te}} \in \mathbb{R}^{1000\times1000}$
is the only other $d$-touching product. In total the *only* operations that ever
see the full dimension $d$ are the two Gram products
$X_{\mathrm{tr}}^\top X_{\mathrm{tr}}$ and $X_{\mathrm{tr}}^\top X_{\mathrm{te}}$,
each computed once. The resulting coordinate matrices are

$$
A_{\mathrm{tr}}, A_{\mathrm{te}} \in \mathbb{R}^{408\times1000}\quad(\approx 6.5\ \text{MB total}).
$$

---

## 5. Lossless representation and the two error floors

### Training floor

In exact arithmetic with no truncation, $K = \operatorname{rank}(X_{\mathrm{tr}})
= n_{\mathrm{tr}}$ and $\operatorname{span}(U)$ contains every training column
exactly, so $\|A_{\mathrm{tr}}\|_F = \|X_{\mathrm{tr}}\|_F$ and the floor is zero.

In practice we drop POD modes whose singular value falls below
$\sigma_1\sqrt{\texttt{ENERGY\_TOL}}$ with $\texttt{ENERGY\_TOL} = 10^{-13}$
([`prep_nse.py:54`](prep_nse.py#L54)), which yields $K = 408$ and a tiny residual
tail:

$$
\mathrm{floor}_{\mathrm{tr}}
  = \sqrt{\|X_{\mathrm{tr}}\|_F^2 - \|A_{\mathrm{tr}}\|_F^2}
  = \Bigl(\textstyle\sum_{i>K}\sigma_i^2\Bigr)^{1/2},
\qquad
\frac{\mathrm{floor}_{\mathrm{tr}}}{\|X_{\mathrm{tr}}\|_F} = 7.8\times10^{-7}.
$$

### Test floor

Test snapshots may have a component **outside** the training span. By the
Pythagorean theorem (the projection $UU^\top X_{\mathrm{te}}$ is orthogonal to the
residual):

$$
\mathrm{floor}_{\mathrm{te}}
  = \sqrt{\|X_{\mathrm{te}}\|_F^2 - \|A_{\mathrm{te}}\|_F^2}
  = \bigl\|X_{\mathrm{te}} - UU^\top X_{\mathrm{te}}\bigr\|_F,
\qquad
\frac{\mathrm{floor}_{\mathrm{te}}}{\|X_{\mathrm{te}}\|_F} = 8.0\times10^{-6}.
$$

This is the **irreducible** error that no training-span method can beat. It is
computed once here and stored, so it is never mis-attributed to any individual
method.

Both floors use the identity $\|A\|_F^2 = \|U^\top X\|_F^2 = $ (energy of $X$
captured inside $\operatorname{span}(U)$); the difference from $\|X\|_F^2$ is the
energy outside the kept basis.

---

## 6. Folding the floor back into the reported full-state error

A method returns a coordinate reconstruction $\hat A$, implicitly representing the
full-state reconstruction $\hat X = U\hat A$. Decompose the true centered state as
$X = UA + R$ with $R = X - UU^\top X \perp \operatorname{span}(U)$. Since
$U(A-\hat A) \in \operatorname{span}(U)$ is orthogonal to $R$:

$$
\bigl\|X - U\hat A\bigr\|_F^2
  = \underbrace{\bigl\|A - \hat A\bigr\|_F^2}_{\text{evaluated in }K\text{ dim}}
  + \underbrace{\|R\|_F^2}_{=\ \mathrm{floor}^2}.
$$

This is exactly the reported relative full-state error
([`nse_compare.py:63`](nse_compare.py#L63)):

$$
\text{state\_rel}
  = \frac{\sqrt{\|A-\hat A\|_F^2 + \mathrm{floor}^2}}{\|X\|_F}
  = \frac{\|X - U\hat A\|_F}{\|X\|_F},
$$

with $\|X\|_F$ stored as `nrm_train` / `nrm_test`. The full
$d = 41{,}682$-dimensional error is thus reported faithfully while every fit and
every reconstruction stay in $K = 408$ dimensions.

---

## 7. Sensor outputs — also computed without $U$

Because $v = s_{\mathrm{ref}} + U a$, the linear output map reduces to
([`prep_nse.py:75-78`](prep_nse.py#L75-L78)):

$$
y = C v
  = \underbrace{C\,s_{\mathrm{ref}}}_{y_{\mathrm{ref}}\in\mathbb{R}^{6}}
  + \underbrace{C\,U}_{C_{\mathrm{red}}\in\mathbb{R}^{6\times K}}\,a,
\qquad
C_{\mathrm{red}} = C U = (C X_{\mathrm{tr}})\,W\,\Sigma^{-1}.
$$

Again $U$ never appears explicitly: $C X_{\mathrm{tr}} \in \mathbb{R}^{6\times1000}$
is tiny. The ground-truth outputs $Y = C V$ are stored directly
($Y_{\mathrm{tr}}, Y_{\mathrm{te}} \in \mathbb{R}^{6\times1000}$), so the output
relative error

$$
\text{out\_rel}
  = \frac{\bigl\|Y - (y_{\mathrm{ref}}\mathbf 1^\top + C_{\mathrm{red}}\hat A)\bigr\|_F}{\|Y\|_F}
$$

needs **no floor correction**: $Y$ already carries the full-state information,
including the out-of-span component, since $y_{\mathrm{ref}} + C_{\mathrm{red}}\hat A
= C(s_{\mathrm{ref}} + U\hat A) = C\hat v$ and the difference $Y - \hat y =
C(X - U\hat A)$ includes $C R$ automatically.

---

## 8. What the downstream code ever sees

After preparation, [`nse_compare.py`](nse_compare.py) loads only the compressed
arrays:

| array | shape | role |
|---|---|---|
| `A_train`, `A_test` | $408\times1000$ | POD coordinates (the compressed state) |
| `C_red` | $6\times408$ | reduced output map $CU$ |
| `y_ref` | $6$ | output of the mean state $Cs_{\mathrm{ref}}$ |
| `Y_train`, `Y_test` | $6\times1000$ | ground-truth outputs $CV$ |
| `sigma` | $408$ | singular values (diagnostics) |
| `floor_train`, `floor_test` | scalar | irreducible error floors |
| `nrm_train`, `nrm_test` | scalar | $\|X\|_F$ for relative errors |
| `K` | scalar | $408$ |

The $41{,}682$-dimensional state is **absent from the inner loop entirely**.

### Cost summary

| | raw | compressed |
|---|---|---|
| working state dim | $41{,}682$ | **$408$** |
| training matrix | $41{,}682\times1000$ | $408\times1000$ |
| on-disk size | $\approx 333$ MB | $\approx 6.5$ MB |
| per-method linear-algebra cost | $\mathcal O(d\,n^2)$ | $\mathcal O(K\,n^2)$, $\approx 100\times$ less |

---

## 9. Practical notes

- **Centering** uses the *training* mean for both sets, so the test split never
  peeks at its own statistics.
- **`ENERGY_TOL` $= 10^{-13}$** is what turns the idealized "floor_train $= 0$"
  into the actual $\sim 8\times10^{-7}$ — negligible, but nonzero; lowering it
  keeps more modes (larger $K$, smaller floor), raising it keeps fewer.
- **Reproducing the numbers:** the concrete values above
  ($K = 408$, $\text{floor}_{\mathrm{tr,rel}} = 7.8\times10^{-7}$,
  $\text{floor}_{\mathrm{te,rel}} = 8.0\times10^{-6}$,
  $\|X_{\mathrm{tr}}\|_F = 898.59$, $\|X_{\mathrm{te}}\|_F = 898.48$) come from the
  saved `data/nse_compressed.npz`.
