# NEXT52: remove current blocks17-21 from the selected component path

2026-09-10. Planned, not executed. NEXT51 endpoint-selection family is complete.
One source-only state-transport comparison, not another query/head/readout grid.

## Question and target

Can a small, differentiable numerical map supply the current K/V/dK inputs of
the retained source-anchor/source-suffix component, without executing blocks17–21
or reading their current-author cache? Same12 authors/two queries/four GRAIN
directions. Current H16 remains a neural-prefix dependency; do not call it raw-text
LLM independence. Endpoint target is NEXT51 kappa, not human Y or the whole L.

Use unchanged NEXT51 same-query Qbar/rbar, actual attention22 norm/projections,
fixed positions/mask. Fit only the source authors' **primal** H16->pre22 states;
no endpoint, gradient, factor-action, questionnaire or new8 target loss.

## Fixed family and derivative consistency

For each allowed source set (four leave-one-source-out sets plus all4), pool
both queries with equal author weight, equal query weight within author and
equal token weight within material. All token rows are used; do not conflate
their count with independent participants. Compute source-weighted mean mu and
rank32 input PCA basis U using deterministic weighted randomized SVD (seed52,
3 power iterations). Source-only scales, no whitening of outputs.

    p_t = (x_t-mu) U
    c_t = (1/t) sum_(j<=t) p_j
    s_t = (t-1)/(T-1).

Exactly three arms:

1. Identity transport: X22hat=X16, dX22hat=D. No fit.
2. Pointwise residual map: X22hat_t=x_t+phi_t B,
   phi_t=[p_t,1,s_t,s_t^2].
3. Causal-context residual map: same formula with
   phi_t=[p_t,c_t,1,s_t,s_t^2]. No future-token mean or new feature search.

Fit B to X22-X16 by source-weighted Ridge: normalized total row weights sum1,
alpha=.01 on source-standardized nonconstant feature columns; intercept unpenalized.
All constants/layout are fixed during the directional derivative. Their derivative
is zero; p and c use the same U/scales and causal cumulative operator:

    dX22hat_f,t = D_f,t + D phi(X16)[D_f]_t B.

Use the existing `native_material_reader.direction` to build the original
relative-RMS BODY directions, round to the original float32 direction units,
and zero non-BODY positions. Verify against original source direction semantics.
A pointwise map cannot transmit a BODY-only perturbation to an initially fixed
final-query tangent; the causal-context map can. Do not assume that extra ability
necessarily improves the selected K-channel endpoint.

## Execution and forbidden dependencies

- New `source_state_transport.py`, `run_grain_source_state_transport.py`, tests.
  Verify exact token/hidden shapes, masks and source exclusions. Unit tests:
  finite differences of the **same map**, source-only centering/scales, no future
  leakage, identity limit, source-group weighting, BODY-to-query transmission.
- Source inputs: NEXT49 `private_inputs/*` H16 and source4 `private_capture/*`.
  Fit5 source references x2 maps=10 small Ridge solves. No new neural capture.
- Current prediction may read only H16/grain/positions, fixed source maps,
  Qbar/rbar and attention22 weights. Do not read new8 pre22 states, tangents,
  K/V/dK, endpoint gradients or actions. Freeze candidate states/tangents/actions
  before evaluation opens those files; preserve source4 exclusions for every map.
- Apply actual RMSNorm **and its differential** to the approximated state and
  direction. Then construct K/V/dK and run the retained full key-vector + rbar
  program, not two separately fitted primal/tangent predictors.
- Evaluate once against existing NEXT49/51 targets. Tables: primal state error,
  full tangent error, endpoint kappa fidelity, contrasts and signs, all cohorts/
  queries/factors. Keep native-input compiled program as a labeled reference,
  not an eligible current-data-free candidate. Distinguish better primal fit
  from actual direction/endpoint preservation.

## Decision, resources and deliverables

Preserve a map if its new8 endpoint SSE/zero<1 in each query; show improvement
or degradation against identity separately. Report amplitude even when r is high.
If neither fitted map is useful, use the state/tangent/endpoint split to identify
which dependency remains; do not add ranks/alphas/queries in this packet. The
source-only component still counts as useful; failure of transport approximation
does not negate NEXT51. A new mechanism question must justify the next experiment.

0 neural/backward/API/Y calls,10 source-only Ridge fits maximum,180s combined
numeric construction/prediction plus one cached numerical replay. CPU only;
no dependency downloads, no old queue restart, no large training. If rank32
is unsupported by source rank, fail with evidence rather than silently tune it.

Private: `results/grain_vector_mechanisms/logic_source_state_transport_v1/`.
Public report: `SOURCE_STATE_TRANSPORT_REPORT_V1.md`, aggregate state/tangent/
endpoint tables, fixed dependency formula, replay/unit checks, ledger update.
No raw text/identities/matrices in tracked output; no release/scorer/lockbox/MEPS
changes. Local commits only. The scope is a runtime dependency removal test,
not evidence that all personality inference is linear.

Method motivation: [FitNets](https://arxiv.org/abs/1412.6550) uses intermediate
representations; [Sobolev Training](https://arxiv.org/abs/1706.04859) distinguishes
matching a function from also matching derivatives. This packet fits primal
states only and tests its analytically coupled derivative, not a claimed
implementation of the full Sobolev training method.
