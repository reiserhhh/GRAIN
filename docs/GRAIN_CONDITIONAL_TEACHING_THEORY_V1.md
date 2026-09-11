# GRAIN: conditional teaching versus compatible composition

2026-09-11. NEXT86 executed; [report](../reports/grain_vector_mechanisms/CONDITIONAL_TEACHING_REPORT_V1.md).
This develops the same-material query/GRAIN interface, not new construct names.

## 1. Exact conditional teaching

For value and directional observations Y, source covariance K, frozen weights W
and lambda, use H=WKW+lambda I and b=WY. When adding observations J to I:

```text
T = H_II^-1 H_IJ
S = H_JJ - H_JI T
gamma = S^-1 (b_J - H_JI theta_I)
theta_(I+J) = [theta_I - T gamma ; gamma]
beta_(I+J) = W_(I+J) theta_(I+J).
```

This block elimination is exactly the joint solve. Its update changes old as
well as new coefficients. A raw full-fit context term is not the effect of
adding context observations. Final prediction is order-invariant; intermediate
innovations depend on the declared order. Empirical order here is V -> VB -> VBC.
No outcome-selected rearrangement was used.

Function and derivative observations can share a Gaussian covariance model,
with differentiated kernels and distinct observation scales. This is standard
methodology, not a newly claimed theorem; see [Rasmussen & Williams (2006),
section9.4](https://gaussianprocess.org/gpml/chapters/RW9.pdf). Our weights are
fixed regularization units, not estimated human measurement-noise variances.

## 2. Recovering a native component is not the same target as correct composition

Let p_N,p_C be native evidence vectors, r_N,r_C their native endpoint readers.
Use m=(r_N+r_C)/2 and d=r_N-r_C; bars/deltas denote query means/differences.
For the local query contrast, exactly

```text
t = p_N^T r_N - p_C^T r_C = delta_p^T m + pbar^T d.
that = delta_phat^T m0 + pbarhat^T dhat.
```

m0 is the source-fixed common reader. Source-compiled evidence phat is not p.
Consequently

```text
that - t = e_transport + e_common + e_differential
e_transport = (phat_N-p_N)^T r_N - (phat_C-p_C)^T r_C
e_common = delta_phat^T (m0-m)
e_differential = pbarhat^T (dhat-d).
```

The same identity holds for each anchored GRAIN action by replacing p with its
key/value/total direction vector. All pairwise cross terms matter in squared
error. Mean, variance and vector norm by themselves do not determine total error.

NEXT86 gives a concrete witness: moving V -> VBC lowers normalized differential
scalar error .184407 -> .176292 but weakens the negative cross term -.417847
-> -.394788. Total error rises .166481 -> .181425. A native-d oracle retains
.399920 other error. For K+V the cross term reinforces error instead of cancelling.

This is more specific than the old observation that vector loss and functional
loss differ. Here the **correct native component is not generally the correct
component for an approximate upstream representation**. Scalar V's gain is
operationally useful, while its interpretation includes compensation, not just
more faithful recovery of the original reader.

## 3. An effective reader in the compiled program

At a source material, assemble five compiled evidence rows: scalar common
evidence and four common total-action vectors, with fixed row units u_j. Let
Rhat_j=pbarhat_j/u_j. For the corresponding source native contrast t_j define

```text
b_j = delta_phat_j^T m0 + pbarhat_j^T d0
c_native_j = Rhat_j (d_native-d0)
c_compatible_j = (t_j - b_j) / u_j
               = c_native_j - (e_transport_j+e_common_j)/u_j.
```

Learn c_compatible(x) on sources only and use the already-verified lift

```text
dhat = db + Rhat^+ [chat - Rhat(db-d0)].
```

When the five coordinates match at a material, the compiled scalar/anchored
contrasts match there. The frame-null complement remains db. This is an
algebraic construction condition, not proof of off-source prediction. The
resulting reader is program-relative; it need not equal the native neural d.

The four anchored actions are **coordinate values**, not automatically the four
derivatives of the scalar-coordinate target. The application remains a coherent
paired function with separate anchored and diagonal derivatives. No native
reader curvature is silently replaced by an anchored first derivative.

There is also a population-level statement. Let Z be all available material
inputs and X=phi(Z) the restricted features used by the coordinate regressor.
With fixed source row units, define Cstar=(T-b(Z))/u. Under squared loss,

```text
h_compatible(X) = E[Cstar | X]
               = E[Cnative | X] - E[e_other/u | X].
```

This is the usual conditional-expectation projection: expand the squared loss
around E[Cstar|X]; its cross term is zero after conditioning. Native-coordinate
teaching has the same optimum only when the conditional other-error mean is0
(for the relevant output). More accurate native targets cannot by themselves
remove a systematic omitted correction. This applies a standard projection
identity to our composed program; the small empirical panel does not identify
the population conditional expectations. T remains a fixed-model local target,
not an independently validated human trait.

## 4. Research consequence

NEXT87 uses matched original-source-only teachers because native local scalar
and action targets are verified there; clamped reader caches alone do not supply
matching clamped local evidence/action targets. Do not borrow original-state
rulers for clamped target construction. Existing V/VB/VBC results stay intact.

The test now develops compatibility of evidence and readout, not an alpha grid
or another norm-accuracy contest. If successful, it supplies a more appropriate
local operator for the same-text GRAIN/query system. External psychological or
relational interpretation still follows independent evidence and coded behavior.
