# GRAIN M4-C.3.1 Relational Error Budget

Date: 2026-07-31  
Status: frozen `M4_C31_GO_RELATIONAL_BUDGET`

## 1. Scope correction from M4-C.3

M4-C.3 preserved exact physical reconstruction, gauge invariance, finite
intervention agreement, planted-edge localization, and null calibration. Its
author-wise Frobenius error budget reached only `.7264` against a frozen `.75`
gate, with `4/8` repetition passes.

That result remains `M4_C3_NO_GO_ERROR_ATTRIBUTION`. The failed implication is:

\[
\text{author-wise absolute loop error}
\;\not\Rightarrow\;
\text{author-relation geometry loss}.
\]

Absolute error can be common to every author and leave all relations
unchanged. Conversely, a small error can reorder many nearly tied author
distances.

## 2. Common-mode quotient

For each oracle/discovered edge subset
\(S\subseteq\{A,R,D\}\), stack the physical loop:

\[
X_S=
\begin{bmatrix}
\operatorname{vec}(L_{1,S})\\
\vdots\\
\operatorname{vec}(L_{n,S})
\end{bmatrix}.
\]

Remove the common author mode:

\[
H_n=I_n-\frac1n\mathbf1\mathbf1^\top,
\qquad
\widetilde X_S=H_nX_S.
\]

The primary coordinate is the upper-triangle author-distance vector:

\[
d_S=
\operatorname{vech}
\left[
\|\widetilde X_{S,u}-\widetilde X_{S,v}\|_2
\right]_{u,v}.
\]

This quotients common translation. Midrank Spearman geometry also quotients
positive scale and monotone distance transformation.

## 3. Exact rank-loss coordinate

Let:

\[
z_S=
\frac{
\operatorname{midrank}(d_S)
-\overline{\operatorname{midrank}(d_S)}
}{
\left\|
\operatorname{midrank}(d_S)
-\overline{\operatorname{midrank}(d_S)}
\right\|_2
}.
\]

Then the registered loop loss is exactly:

\[
\ell(S)
=1-\rho_S(d_O,d_S)
=\frac12\|z_O-z_S\|_2^2.
\]

M4-C.3.1 does not replace the old gate. It gives that gate its own error
coordinates.

## 4. Möbius/Harsanyi relation budget

Single-edge perturbations are:

\[
h_e=d_e-d_O.
\]

Pair interactions are:

\[
h_{ef}=d_{ef}-d_e-d_f+d_O.
\]

Without using the complete DDD outcome, the second-order prediction is:

\[
\widehat d^{(2)}
=
d_O+\sum_eh_e+\sum_{e<f}h_{ef}.
\]

The irreducible third-order residual is:

\[
h_{ARD}=d_{ARD}-\widehat d^{(2)}.
\]

The primary question is whether \(\widehat d^{(2)}\) predicts the actual
relation loss, or whether a stable three-edge interaction remains.

## 5. Continuous secondary geometry

To reduce discontinuity from rank ties, define:

\[
K_S=
\frac{
\widetilde X_S\widetilde X_S^\top
}{
\operatorname{tr}
(\widetilde X_S\widetilde X_S^\top)
}.
\]

Normalized Gram distortion and CKA are secondary diagnostics. They cannot
replace the registered distance-rank endpoint.

## 6. Frozen gates

- Spearman-loss identity error at most `1e-12`;
- complete Möbius reconstruction error at most `1e-10`;
- common translation, positive scale, and shared orthogonal invariance error
  at most `1e-10`;
- Shapley efficiency error at most `1e-10`;
- pooled second-order predicted-versus-actual loss Spearman at least `.80`;
- repetition/world cluster-bootstrap 95% lower bound at least `.65`;
- minimum leave-one-repetition-out Spearman at least `.75`;
- `MAE / IQR(actual loss)` at most `.25`;
- planted-edge localization at least `.85`;
- attribution-margin bootstrap lower bound above `0`; and
- null false attribution at most `.05`.

If predictive gates fail but controls pass and third-order residual exceeds
the one-edge-fault reference by at least `.05`, the result is
`RELATIONAL_THIRD_ORDER_INTERACTION`, not a successful second-order budget.

## 7. Frozen hashes

```text
a758a86d64985dd0bd498c6be7317af6db6c0e0d58d190dc64514bff401bb315  configs/m4_relational_error_budget.json
913bc3ba0c62c9e96f966b1cb869f47a5c8b359b28105c5ed8d3755f4aa49e16  scripts/run_suica_m4_relational_error_budget.py
598d6e73fe25ce6b4e74d7e81baa2e6af81d89e2d07e50e4102b7d03a82bf785  suica_core/m4_relational_error_budget.py
930698a3d517e1b43964f692f02d45ce733cc303ad9968861990d8e43d4bd503  tests/test_m4_relational_error_budget.py
```

## 8. Boundary

M4-C.3.1 can coordinate finite synthetic composition loss in the same
author-relation space as the loop gate. It cannot alter M4-C.2 or M4-C.3,
establish causal edge importance, identify personality, validate natural text,
or authorize M4-D.

## 9. Frozen result

All registered gates passed:

- pooled second-order predicted-versus-actual loss Spearman: `.9868`;
- repetition/world cluster-bootstrap 95% lower bound: `.9609`;
- minimum leave-one-repetition-out Spearman: `.9821`;
- normalized loss MAE: `.0547`;
- maximum Spearman identity error: \(3.12\times10^{-16}\);
- maximum Möbius reconstruction error: \(6.94\times10^{-18}\);
- maximum Shapley efficiency error: \(1.11\times10^{-16}\);
- similarity-invariance error: `0`;
- planted-edge localization: `1.000`;
- attribution-margin lower bound: `.0249`; and
- null false attribution: `0`.

The decision is:

```text
M4_C31_GO_RELATIONAL_BUDGET
```

Mean actual relation loss was `.3126`; the second-order budget predicted
`.3036`. Mean Shapley allocation was:

- creation: `.2501` (about 80.0%);
- response: `.0336` (about 10.7%); and
- choice: `.0289` (about 9.2%).

The median third-order relative norm was `.0600`, but single-edge and pairwise
terms still predicted total relation loss with strong held-out stability.
Creation is therefore the primary symmetric loss allocation, not a proven
causal bottleneck. A separately frozen intervention is required before
changing the creation estimator.
