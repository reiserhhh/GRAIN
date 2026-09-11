# Observable teaching and source output directions

2026-09-11. NEXT83 computed; NEXT84 derived and planned. This is development of
the same-material GRAIN/query interface, not a replacement psychometric scale.

## 1. Identical input and function, a different source loss

Keep NEXT82's joint material coordinates and Gaussian value/derivative kernel.
In kernel-whitened coordinates A and output basis U=row(Y), the source residual
field is `A Theta U^T`, with penalty `lambda ||Theta||_F^2`. This is a change of
coordinates preserving the old RKHS penalty, not a new data whitening rule.

For residual e_i and source fixed ruler r_i, the functional loss adds
`(r_i^T e_i)^2`, normalized by each source group's total target projection energy.
The ordinary vector term and regularizer remain. Scalar and four action rulers
are taught jointly, with no individual-target division or recipient selection.
Both scores and derivatives come from the same learned field.

Vector-valued regularization and kernel representations are established methods:
[Micchelli & Pontil (2005)](https://pubmed.ncbi.nlm.nih.gov/15563752/) and
[Alvarez et al. (2012), sections3-4](https://arxiv.org/html/1106.6251v2).
Our source-conditioned probes and finite objective are an application choice.
No claim of newly inventing vector-valued kernels or a GP posterior is made.

## 2. What the experiment establishes

The source function can prioritize these observations without breaking calculus:
same control reproduced, source functional loss reduced, finite differences pass.
Recipient scalar error improves only .58%, and K+V amplitude does not improve.
Therefore the particular restricted objective is usable but not a general repair.
More loss-weight tuning is not the next mechanism.

The exact output decomposition finds .215128 squared reader error outside S,
about42.8% of total. That term is fixed for all functions constrained to d0+S.
Yet for a ruler p,

```text
(p^T e)^2 = (p^T e_in)^2 + (p^T e_out)^2
            + 2(p^T e_in)(p^T e_out).
```

Even when e_in is orthogonal to e_out, their scalar effects need not be.
NEXT83 shows negative scalar cross-energy and positive K+V cross-energy.
This resolves the apparent contradiction between slightly improved vector
error and worse evidence-projection error. It does not partition personality
variance or establish a unique psychological cause.

## 3. A missing representer direction in the restricted candidate

NEXT83 explicitly chose S=row(Y) to compare both objectives in the same finite
family. This is sufficient for its vector-only control. It need not be sufficient
for the unrestricted functional objective, because rulers supply output geometry.

Let B be the coefficient matrix in whitened observation coordinates. Consider

```text
J(B) = ||W(AB-Y)||_F^2 + sum_i,l c_il [r_il^T((AB)_i-Y_i)]^2
       + lambda ||B||_F^2, lambda>0, c_il>=0.
S_star = span(rows(Y), all source rulers r_il).
```

Decompose B=B_parallel+B_perp relative to S_star. Y and every r_il have zero
perpendicular component. Functional terms are unchanged by B_perp; vector loss
and regularizer gain exactly

```text
||W A B_perp||_F^2 + lambda ||B_perp||_F^2 >= 0.
```

Thus the unique optimum has B_perp=0. S_star is a sufficient output subspace for
this objective; no energy-selected rank is needed. It may contain directions
outside row(Y). This proof assumes the stated scalar input kernel, fixed rulers
and isotropic coefficient penalty. It is not a universal claim for all kernels.

In2D, use y=(1,0), r=(1,1) and
`J(b)=||b-y||^2+(r^T(b-y))^2+||b||^2`.
The constrained solution in span(y) is(2/3,0), J=2/3; the full solution is
(5/8,1/8), J=5/8. At the constrained solution, the missing-coordinate derivative
is -2/3. A new independent mathematical test verifies these numbers. This is
a source-space witness, not a newly discovered human trait or a recipient fit.

## 4. Consequence for the mainline

Source evidence is already available but was used only to weight errors. The
next concrete development is to also let its directions enter the possible
reader outputs. This tests the link between what the reader is taught to do
and which vector directions it is allowed to express, without new neural calls.

Keep the earlier GRAIN field and all contrary measurements. Apply the expanded
source function to the same current H16 inputs without current native readers;
only then assess functional benefit. H16 still comes from the fixed model, and
the anchored component/diagonal scalar/full-L distinctions remain intact.
See [NEXT84 plan](GRAIN_SOURCE_OBSERVABLE_SPAN_PLAN_V1.md).
