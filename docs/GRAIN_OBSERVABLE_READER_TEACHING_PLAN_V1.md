# NEXT83: teach the reader's source-observable effects

2026-09-11. PLANNED, not executed. NEXT82's GRAIN field improves the fixed local
assembled score error, yet Euclidean vector fidelity and scalar/directional
fidelity order readers differently. Develop the functional objective directly.
No bandwidth/rank/feature grid, new native capture or human labels.

## 1. Fixed data and hypothesis

Retain source4 ALL, the same28 value/derivative observations and the NEXT82
GRAIN input map. New8 remains reused discovery application, never fit material.
Do not claim sourceLOO with an all4 clamp. Keep d0 and common reader m fixed.

Question: does source teaching in observable directions improve new8 scalar
and GRAIN-response reading beyond matched all-coordinate teaching?

Read source original native evidence only from already frozen source packets.
For each source author use common evidence pbar and four common native K+V
directions bbar_f. Keep those five rulers fixed for all seven observations
belonging to that source (original/clamped values, four BODY, one context).
They define anchored functional probes, not derivatives of moving rulers. No
new native evidence on clamped states is required or claimed.

## 2. Finite parameterization and matched controls

Let K be NEXT82's unweighted28x28 observation Gram matrix and Y its28xh residual
targets. Obtain an orthonormal output basis U from row(Y), using only a
machine-relative SVD rank tolerance; retain all numerical rank, no energy cutoff.
Whiten the positive observation eigenvalues to form A=V sqrt(D), so a reader
coefficient Theta represents source residual values/derivatives `A Theta U^T`.
At a new point `a(x)=k_x V D^(-1/2)` and `dhat(x)=d0+a(x)Theta U^T`.
Derivative rows use exactly the same map differentiated in x.

The source-only output basis limits this candidate explicitly. Orthogonal target
residuals and numerical-rank reconstruction errors are reported, not hidden by
rank truncation. If the all-coordinate control fails to reproduce NEXT82 within
the declared numerical tolerance, fix the coordinate implementation before fit
comparison; do not change the input/kernel to rescue outcomes.

Control objective:

```text
J0 = sum_g sum_i_in_g w_g^2 ||A_i Theta U^T - Y_i||^2
     + lambda ||Theta||_F^2.
```

Use the same w_g and lambda as the NEXT82 GRAIN field. Positive-mode coefficient
whitening preserves its RKHS penalty, not an arbitrary new Ridge geometry.

For e_i=A_i Theta U^T-Y_i, construct two additional losses:

```text
Ls = sum_g [sum_i_in_g (pbar_author(i)^T e_i)^2]
            / [sum_i_in_g (pbar_author(i)^T Y_i)^2]
La = sum_g [sum_i_in_g,f (bbar_author(i),f^T e_i)^2]
            / [sum_i_in_g,f (bbar_author(i),f^T Y_i)^2].
J1 = J0 + Ls + La.
```

Use the same source total-energy normalization rule as NEXT82; exactly zero
denominators are flagged and that loss group omitted, not infinitesimally
amplified. Do not divide each tiny individual observation by its own target.
Two fixed fits only: J0 versus J1. There is no loss-weight sweep. J1 has additional
task emphasis, not additional independent authors; report all source losses.

The normal equations in vec(Theta) are positive definite under fixed lambda.
Maximum size is28x28=784 coefficients. Direct solve or a predeclared converged
linear solver is permitted; report residual and iterations, never early-stop
using recipient scores. Work through scalar-projection design rows and check
the normal-equation solution independently on synthetic data.

## 3. Application and matched evaluation

Predict both fields on new8 using H16/source packages only. Freeze before opening
current native readers/targets. Keep all six NEXT82 comparators, all four
selector/value compositions, both queries and their contrast.

Report reader vector error; fixed-native-evidence scalar and K/V projections;
assembled scalar/action error, author-centered error, r and sign; all four
factor effects. Check anchored and diagonal derivatives separately against
matching finite differences as in NEXT82. No whole-L/psychological-validity claim.

Also perform one source-output-range account using all numerical rank of U:
for current target t=d_native-d0, split any reader error into
`e_in=P_U(dhat-d_native)` and `e_out=-(I-P_U)t`. Publish vector norm identity
and each functional projection's two squares plus signed cross term. This is
post-freeze explanation only, never an oracle predictor or new8 construction.
It distinguishes missing output directions from errors inside teachable space.

## 4. Decision and cost

If observable teaching improves the matched scalar/actions, retain it as a usable
local functional learner, even if full-vector distance changes little. If it
only changes source fits, report whether missing output support or within-span
generalization dominates the measured projections. Use this to choose one new
mechanism, not more penalty tuning. Retain contrary query/factor outcomes.

At most2fits,90 cumulative numerical CPU seconds,0neural/API/human/MEPS calls.
One immutable prepare/model/predict/evaluate sequence and exact cached replay.
No new jobs on Windows/Linux or old long-run restart. Publish only cohort tables.
Suggested module `observable_reader_teaching.py`, runner/test/publisher, private
`logic_observable_reader_teaching_v1/`, and an aggregate report/theory note.
Do not alter NEXT82 frozen implementations, datasets, scorer or lockbox. Overall
goal stays ACTIVE; this is a finite mechanism-development card.

## 5. Mathematical precedents

- [Solak et al. (2002)](https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html): jointly using values and derivatives.
- [Micchelli & Pontil (2005)](https://pubmed.ncbi.nlm.nih.gov/15563752/): Hilbert-space-valued regression and regularization.
- [Alvarez, Rosasco & Lawrence (2012)](https://arxiv.org/abs/1106.6251): vector-valued kernels and output coupling.

Our source-functional loss and matched finite design are experimental choices,
not claims that these papers establish GRAIN personality validity. The point
is to change what is taught within a consistent vector-function family.
