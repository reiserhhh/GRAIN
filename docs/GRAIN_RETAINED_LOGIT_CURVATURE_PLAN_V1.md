# NEXT24: retained-state curvature on the binary log-odds scale

2026-09-10. Cache-only follow-up to NEXT21-23 and the native same-material
interface. Freeze before new statistics. No neural calls, questionnaire reads,
new participants, factor selection or changes to the release scorer.

## Decision and inherited material

Is the measured T/F retained-state nonlinear response just the final binary
probability link, or does the internal log-odds field change along the path?
Use the existing four reference authors, both M0/Mu body contexts, and every
saved 3- or 5-node path in `logic_joint_carrier_readout_v1`. Verify its node
freeze and the old displacement cache. Preserve integration errors and the
failed/negative results of older experiments. No new nodes.

For p(t)=F(M,R0+t*d;x0,L), let u(t)=logit(p(t)), s(t)=<grad p(t),d>,
v(t)=grad p(t)/(p(t)*(1-p(t))). Require 0<p<1 without clipping.
With a=u(0), b=<v(0),d>, compute the exact decomposition:

    p(1)-p(0)-s(0)
      = [sigmoid(a+b)-p(0)-s(0)] + [p(1)-sigmoid(a+b)]
      = link_curvature + internal_remainder.

Report the original probability change, tangent, both signed components,
logit change/tangent/remainder, probability and logit quadrature errors,
and relative change of the entire logit gradient field. These are numerical
function accounts, not independent variances or personality effect sizes.
Average both body paths per author before any author-level summary.

## One finite derivative-enhanced candidate

If there is a non-negligible internal remainder relative to the saved numerical
errors, freeze one separate implementation amendment before evaluation. Candidate:

    delta_u(d) = <mean_reference v(0), d> + 0.5 * sum_r <d_r,K_r*d_r>
    K_r = k_parallel[r]*e_r*e_r.T + k_perp[r]*(I-e_r*e_r.T)
    e_r = R0_r / norm(R0_r).

Fit the 272 curvature scalars (136 role-aligned retained slots x two components)
from reference gradient differences v(t)-v(0), against t*d. Equal weight per
reference path, equal weight per nonzero saved node within path; fixed shrinkage
of .01 times that component's design energy. Two body contexts have separately
fitted functions, no parameter grid. This is a constrained quadratic surrogate,
not a measured full Hessian or proof that RMSNorm causes the response.

Compare origin-linear and quadratic delta-logit functions on the old20 evaluation
authors. Compare retained probability allocations with the old frozen/support
linear readers. Any probability conversion using an actual evaluation origin
must be labeled oracle-conditioned; also report the fully source-origin version.
Freeze functions and predicted delta-logits before loading evaluation endpoints.
No endpoint or human-label selection of candidate, coefficient or penalty.
The existing panel is already exposed exploratory evidence, not a new lockbox.

## Outputs and acceptance

- Pure NumPy algebra with exact-decomposition, logistic-linear, nonlinear,
  gradient-change, shape/finite-value, zero-displacement and surrogate tests.
- Private rows/models/predictions plus source/code hashes, public aggregates only.
- Deterministic rerun of diagnostic and predictions; retain any failure.
- Report whether this single candidate changes the NEXT26 interface. If not,
  locate the missed input or geometry and move on, not a curvature search grid.
- Never describe conditional block16 R inputs as a complete raw-text runtime.

## Primary methodological precedents

- Zhang and Sennrich (2019), [RMSNorm](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html):
  radial normalization motivates, but does not validate, radial/tangential blocks.
- Fuhrlaender and Schoeps (2023), [Hermite least squares](https://link.springer.com/article/10.1007/s11081-023-09795-y):
  derivative observations can constrain quadratic surrogates without new expensive
  function calls. Their optimizer and convergence theorem are not used here.
- Bouhlel and Martins, [gradient-enhanced kriging](https://arxiv.org/abs/1708.02663):
  high-dimensional surrogate complexity needs explicit reduction. We use neither
  their kernel nor their hyperparameter optimization.
