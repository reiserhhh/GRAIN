# S19: recover the operative response covector, then test sharing

2026-09-08. After S18, the next question is no longer whether one more moment
reconstructs the same statistical reader. It is whether the final decoder's
operative response can be recovered as a shared mathematical direction across
authors. A reader-null vector retained nearly all of the measured pairing effect.

## Mathematical basis and targets

For each fixed author's context define F_u(h), the frozen suffix score after
replacing only its final-token layer17 concatenated head vector. All other
positions and residual paths remain native. With actual pairing vector d_u,

\[
E_u=F_u(h_u)-F_u(h_u-d_u)
=\bar g_u^T d_u,\qquad
\bar g_u=\int_0^1\nabla_hF_u(h_u-t d_u)\,dt.
\]

This is the fundamental theorem along the internal intervention path, not a new
assumption of linearity. It adapts the completeness principle of
[Sundararajan, Taly & Yan, ICML2017](https://proceedings.mlr.press/v70/sundararajan17a.html)
to the declared head site. Do not interpret it as word attribution or human
causation. The target is S18's pairing effect E, not S9's body-collapse contrast C.

[Constantine & Gleich, Computing active subspaces with Monte Carlo](https://arxiv.org/abs/1408.0545)
motivate spectral analysis of uncentered gradient second moments. Here the
functions F_u are context-conditioned, and sampling is a small author panel.
Use the algebra and finite sensitivity analysis; do not import their global
approximation/error guarantees without the input-domain assumptions.

Estimate M=mean_u(bar_g_u bar_g_u^T) via a thin SVD, and report its singular
spectrum. Do not center away the shared mean gradient. A few leading directions
would be a candidate shared functional subspace, not newly named personality
factors. The coordinate basis is the frozen native head basis; no claim of a
uniquely identified causal metric.

## Bounded implementation

Use precisely the24 S18 authors and view0. Load the S18 source hashes and replay
native head/pairing/score endpoints. No questionnaires or new outcome selection.
Implement a leaf-vector hook before layer17 W_O; freeze all model weights and
differentiate only with respect to the1536-dimensional inserted head vector.
Reuse `capture_head`, token offsets and the fixed native N score alphabet.

Measure t=0,.5,1; Simpson integral. If scalar completeness error exceeds
max(1e-5,.05*abs(E_u)), add t=.25,.75 once and report both quadrature values.
Maximum24capture forwards +120forward/backward evaluations =144forwards and
120backwards.15-minute cap, one neural process. Preserve completed author units.
Start with8 authors as a numerical/resource checkpoint, not a human-outcome gate.
If supported by cost and useful effects, finish the fixed24; never expand cohort
automatically or continue a failing gradient implementation for hours.

Save each gradient node, integral, d,h, endpoint replays and quadrature error
privately. A later phase can reuse these gradients instead of repeating them.
Test a synthetic linear suffix and quadratic suffix first, including hook cleanup,
finite values, score replay, scalar completeness and no parameter gradients.

## Analyses with no new calls

1. Compare measured E with bar_g_u^T d_u: both absolute error and correlation.
2. Leave each author out; use the other23 authors' mean integrated gradient
   to predict that author's effect from d_u. No held-out gradient or effect fit
   is used in this prediction. Report r,MSE/full-effect-energy,bias and sign.
3. Build training-only uncentered active subspaces at fixed ranks1,2,4. Report
   how well they capture the held-out gradient and its contraction with d_u.
   This is a subspace diagnostic using a held-out gradient, not a deployable
   no-backprop predictor. Keep it separate from the mean-gradient prediction.
4. Compare native t=0 versus path-integrated gradients and their directions.
   Check whether path curvature or cross-author orientation drives the error.
5. For S18's two selected/null decompositions, contract the actual gradient with
   each component; explain the observed near-additive effect split. Do not
   quietly replace beta with a gradient and retain beta's human-validity label.

## What follows from each answer

If an author-out shared gradient predicts effects, freeze it and test a cheap
functional readout on the larger source pool, with source outcomes joined only
after all scores freeze. If shared orientation exists but gain varies, study
a bounded context-conditioned gain model. If orientations vary substantially,
use a small response subspace or kernel operator conditioned on the query/context.
Choose between these from the observed gradient spectrum and held-out errors,
not a questionnaire-selected ranking.

Files to add: `grain_research/vector_mechanisms/head_gradient.py`,
`scripts/run_grain_functional_gradient.py`, `tests/test_grain_head_gradient.py`.
Output `results/grain_vector_mechanisms/functional_gradient_s19_r1/`.
Report `reports/grain_vector_mechanisms/S19_FUNCTIONAL_GRADIENT_REPORT.md`.
Current status: planned, not collected. Preserve T/F, exact N reference, release
scorer, protected bridge/MEPS/lockbox and all previously frozen code. Local commits
only; no cloud text upload, Claude invocation, LoRA or old queues.
