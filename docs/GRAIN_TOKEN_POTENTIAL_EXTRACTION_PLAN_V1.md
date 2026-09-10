# NEXT60: source-trained token potential with compatible score and gradient

2026-09-11. Planned, not executed. NEXT59 found native action in the old pooled
feature kernel. Change input processing, not just the head on that same phi.
Question: can nonlinear token-local source learning retain GRAIN allocation
responses while producing one explicit whole scalar without a current suffix?

## 1. Materials, roles and budget

Same source4/new8, two queries, four fixed directions, exactH16/layout/contrast.
Reuse NEXT57 inputs and NEXT39 H16 BODY gradients verified in NEXT59. Each source
author is evaluated using the other3; new8 uses all4. No new human labels, MEPS,
lockbox opening, native/API calls or old queue. Source gradients are teaching
data; current gradient/L/tau are evaluator-only after prediction freeze.

Two fixed compatible functions,10 query/reference bases,20 small linear solves,
180s CPU construction/application/replay budget. Cache-only one replay. If the
first complete source-reference unit projects beyond the cap, stop at that
unit with artifacts; no silent reduced-outcome or new sample selection.

## 2. Shared source descriptor and two arms

Construct source-only query/reference rank8 U from stacked sqrt(T_u/n_source)
H16 BODY gradients using randomized SVD, seed60,oversamples10,power5. This
minimizes equal-author density-field reconstruction, not input variance. Fix
column signs deterministically. Reuse existing implementation if its contracts
match. No source/application rank selection.

Project sourceH into U, fit mean and per-coordinateSD with author weights
1/(n_source*T_u). Floors1e-8->1. z=(H U-mu_z)/s_z, lastz uses the same map.
Use p=(P1,P2,P3) Legendre values at BODY ordinal midpoints in[-1,1].
Landmarks:16 deterministic evenly spaced BODY indices per source author on
descriptorv=[z,p];48/64 centers for3/4-source references, no outcome-ranked centers.
Use bandwidth equal to median positive Euclidean pairwise landmark distance;
undefined/zero bandwidth is a preflight stop, not filled using application data.
Kernel k(v,l)=exp(-||v-l||²/(2 bandwidth²)).

Both models have an unpenalized intercept b, linear BODY mean term a^T mean(z),
linear last-token term beta^T z_last, plus the same fixed kernel-center weights c.

1. **mean_then_kernel control**: sum_m c_m k([mean(z),mean(p)],l_m).
2. **kernel_then_mean candidate**: mean_t sum_m c_m k([z_t,p_t],l_m).

No neural optimizer, softmax reweighting, post-hoc gain or gradient-oracle inputs
at application. Query-specific source U/centers/coefficients are allowed fixed
assets. The two models have the same parameter count and targets, but different
information before pooling. GRAIN enters through measurement directions, not as
the target response encoded into the features.

## 3. Fit values and token-gradient field in one linear system

For each source author measure scalarL and projected coordinate gradient density
K_u=T_u*(g_u @ U)*s_z. This is the derivative w.r.t. source-normalized z,
not just gU; derive the normalization explicitly and test it.

Loss for each arm:

    mean_authors (Lhat-L)^2 / sL^2
    + mean_authors mean_token_and_8coords (Khat-K)^2 / sK^2
    + .01 * (||a||^2 + ||beta||^2 + ||c||^2).

sL=max(sourceSD(L),1e-6);sK=max(sqrt(equal-author mean squared K),1e-6).
Khat=T_u*partial Lhat/partial z_t is the analytic derivative of the SAME scalar.
Build bounded normal equations or augmented least squares; do not materialize a
token-by-hidden-by-center tensor. Derive score, BODY gradient, four GRAIN
responses, and query contrast from that same frozen program. Both arms use the
same source normalization and penalty. Record data rank/condition and source
value/gradient fit errors, not only application r.

The candidate can produce nonconstant token gradients. Its total score still
includes an underconstrained last-token readout; do not reuse current L as an
intercept or current gradient as a correction. Gradient-field compatibility
does not automatically establish complete value fidelity.

## 4. Smallest useful evaluation and safeguards

Before opening current targets, export allmodels and24x2 predictions from pure
current H16/GRAIN/layout inputs. A separate predictor must succeed without access
to native gradient/score targets; target-denial probes and checksum replay verify.

Report source4/new8 separately, both queries/all4 factors:
- WholeL r/RMSE/bias/SD and SSE/sourceconstant.
- FullBODY gradient error, GRAINtau r/RMS/SSEzero/signs, query contrast.
- Native allocation component vs candidate allocation response, computed using
  the same original D_allocation (no refit or gain). Keep signed cross terms.
- Oracle source-U projected gradient response is a labeled evaluation-only
  capacity bound; it is not the candidate's prediction.
- Compare new control/candidate; retain NEXT57's original full-score/response
  numbers as historical different-input comparators, not a matched ablation.

Promotion means candidate improves matched-control action error and retains
SSEzero<1 and whole-scoreSSE/sourceconstant<1 for both queries. If only local
action improves, keep the mechanism and do not relabel it full inference.
No mandatory psychological r>.5; these are model-fidelity utility comparisons.
No retuning on source4/new8 or human Y after outcomes.

Tests: exact gradients/finite differences for both kernels and scalings, a
zero-mean allocation witness, target-independent inference, identical source
weights despite different token counts, source exclusion, compatible scalar
integration on a synthetic line. Record all outputs in a private versioned
packet and publish only aggregate report/theory/tables. Then choose ONE next
mainline question from the actual result, not a center/bandwidth/rank sweep.

## 5. Literature and distinctions

Uses nonlinear element maps before aggregation from
[Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf), and the scalar/gradient
compatibility principle illustrated by
[GDML](https://arxiv.org/pdf/1611.04678) and
[Sobolev Training](https://arxiv.org/abs/1706.04859).
This finite scalar Gaussian-kernel expansion is not GDML's exact Hessian-kernel
estimator, a universal set-function guarantee, or psychological energy physics.

NEXT34's linear token coefficient predictor did not guarantee a scalar and
used a different target/basis. NEXT57 guaranteed a scalar but erased allocation
before reading. NEXT60 combines compatible scalar learning with nonlinear
token-addressed inputs; old failed families stay closed, not silently restarted.
H16 still requires the original prefix encoder. A no-suffix candidate is not
yet raw-text inference without neural representation.
