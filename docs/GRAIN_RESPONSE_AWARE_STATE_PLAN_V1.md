# NEXT53: teach the same source map its GRAIN-direction response

2026-09-10. Planned, not executed. NEXT52 primal-only family complete, no retained
numeric replacement. This packet tests objective alignment, not a rank/alpha grid.

## Question and immutable inputs

Does adding source-only directional targets preserve useful response while fitting
the same intermediate states? Keep the two NEXT52 pointwise/causal map families,
every source fold's U32/mu/feature centering/scales, alpha=.01, both queries, same12
authors, same four BODY directions and fixed Qbar/rbar/attention22 program.
Load existing maps for their feature assets; refit only B. No PCA or query selection.
Compare two new joint maps with cached NEXT52 identity/primal maps/native reference.
Do not fit a separate incompatible tangent function or new8 endpoint calibration.

## Source-only objective, fully specified

For each reference S (four leave-one-source-out plus all4), source row weights w
sum1 with equal people, equal queries/person, equal tokens/material. Let Z be the
existing standardized primal design and Z_f its exact direction design. Define:

    Y_X = X22-X16
    Y_Df = native_d22_f-D_f
    R_X = sum_t w_t ||Y_X,t||^2 / H
    R_D = sum_f,t w_t ||Y_Df,t||^2 / (4H)

Here sums include all source materials; apply causal features separately per
material. R_X/R_D are fixed source-only normalizers, never new8 estimates. Fail
explicitly if either <=1e-12. Fit one B per map by the single convex objective:

    L(B) = sum_t w_t ||Z_t B-Y_X,t||^2/(H R_X)
         + sum_f,t w_t ||Z_f,t B-Y_Df,t||^2/(4H R_D)
         + .01 ||P B||_F^2.

P has0 for the original intercept column and1 elsewhere. The constant layout
columns have zero directional design. This gives a stacked weighted Ridge solve
with the same B for primal and tangent; no derivative-only auxiliary predictor.
The second loss is GRAIN-direction weighted Jacobian matching, not full Jacobian
recovery. Analytic tests: stacked normal equations, same-function FD, source
exclusion, fixed asset equality, zero nullspace update, finite source normalizers.

## Execution and decision

1. New `response_aware_state.py`, runner/tests. Freeze code+plan before fitting.
2. Read only source4 NEXT49 primal/tangent targets. Ten small solves maximum,
   no new native calls. Current new8 prediction still reads only H16/GRAIN/layout
   and fixed assets; deny current pre22 states/tangents/gradients/actions/Y.
3. Freeze factorized candidate Xhat/dhat/endpoint packets before evaluation.
4. Compare source-LOO and new8 state/tangent/endpoint errors, amplitudes, signs,
   query contrasts and all factors. Native-input reference remains ineligible.
   Include source training losses to distinguish fitting constraints from transfer.
5. Component replacement retained only if new8 SSE/zero<1 separately both queries.
   Mechanism development success: a joint map reduces new8 directional error
   versus its same-family primal map in both queries, with all endpoint costs shown.
   Report these two decisions separately; a tradeoff is not a replacement pass.

If directional fitting helps, connect the resulting function at its demonstrated
scope. If it does not, close this fixed-basis family and inspect response-direction
coverage rather than sweep lambda/rank/alpha on the same new8 outcomes. No automatic
larger training, extra heads, new query grid or return to old benchmark queues.

Resources: CPU only, ten solves,180s combined preparation/fitting/prediction plus
one cached numerical replay,0 neural/backward/API/Y calls. H16 remains an original
prefix dependency; learning source intermediate targets is not human-label training.
Private packet `results/grain_vector_mechanisms/logic_response_aware_state_v1/`;
public `RESPONSE_AWARE_STATE_REPORT_V1.md`, aggregate state/tangent/endpoint tables,
dependency formula and replay audit. Preserve scorer/release/lockbox/MEPS; local only.

Motivation: [Sobolev Training](https://arxiv.org/abs/1706.04859) combines values and
derivatives; [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html)
connects derivative matching with perturbed-input distillation. Here the perturbation
distribution is the already-fixed GRAIN directions, rather than claimed isotropic
noise or a complete implementation of either paper.
