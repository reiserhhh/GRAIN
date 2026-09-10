# NEXT66: teach the paired context response in the same scalar

2026-09-11. PLANNED, not executed. NEXT65 restored raw context geometry and a
local source allocation response, but not complete new-author inference.
Its post-outcome source-only arithmetic found a small paired response channel:
half-difference target energy is .1905%/.4836% of total projected field energy.
Change that learning priority once, not inputs or a kernel/hyperparameter grid.

## One candidate and frozen controls

Use EXACTLY NEXT65 source4 original/shifted views, full-context Gaussian kernel,
BODY U8, 64 landmarks/Hermite features, source normalization, 1169 coefficients,
alpha=.01 and the original score/density scales. Original plus shifted per-author
views have equal weight. Original source4 context centers remain unchanged.
Reuse all four frozen previous models; only two new fits, one per fixed query.
No new text, LLM/API call, humanY/MEPS, source LOO, donor/dose/rank/alpha search.

Let k_u0,k_u1 be the compatible projected BODY densities on the same BODY under
its original and existing intervened last state. Define

    d_u = (k_u1-k_u0)/2
    s_delta^2 = (1/4) sum_u mean_elements(d_u^2)
    J66(theta) = J65(theta)
              + (1/4) sum_u mean_elements((dhat_u(theta)-d_u)^2)/s_delta^2.

All J65 terms, including original endpoint scores and field teaching, remain.
The added channel has weight ONE and zero-change error ONE. This is intentional
relative response emphasis, not a change of physical units or an estimate of
personality-information importance. Relative to old field normalization it is
approximately494.18/186.24 extra weight on the half-difference coordinate.
Report the realized source scale and its ratio. Do not clip to obtain a result.
If source delta energy is nonfinite or <=1e-12 times original field scale^2,
stop that query as unsupported instead of dividing by a floor and continuing.

Same-scalar derivatives only. No separately predicted GRAIN action output.
Build delta design directly from paired derivative design blocks; the additive
prefix cancels exactly because BODY is identical and its derivative does not
depend on last. Score feature differences are not substituted for field changes.

## Execution and cost

1. Freeze plan and module/runner/tests before fitting; inherit input hashes and
   the source-only path restrictions of NEXT65.
2. Reuse frozen J65 Gram/RHS and independently validate unchanged base terms.
   Add the PSD delta Gram/RHS, solve once/query, no iterative search.
3. Predict the same32 pure-input packets, save/freeze before native targets are
   joined. Application remains suffix-free, H16 still requires the old encoder.
4. Reuse all old control rows and publish every query/factor/cohort result.
5. Replay cached predictions/tables, source objective and derivative checks;
   document outcome/next mechanism and commit locally. No release/lockbox edits.

Hard budget120s cumulative computeCPU, two new fits maximum, sequential. After
first query estimate remaining work plus30s reserve; stop safely if budget would
be exceeded. No long-running model process. Budget excludes wall-time tests/docs.

## Required outputs and decisions

Source: paired average/difference energies and losses for all models; delta SSE
relative to zero; full and allocation GRAIN finite-change r/RMSE/SSE, actual and
predicted change RMS; original objective and added-channel penalty separately.
New8: full scalar r/RMSE/SSE against source constant; all four GRAIN actions and
allocation actions r/RMSE/SSE against zero; all query contrasts and factor cells.
Do not join or optimize human questionnaire labels.

Mechanism decision: compare source delta SSE with NEXT65; a source-only gain is
response learning at source, not transferable inference. Complete candidate
promotion requires both queries' new8 action RMSE lower than matched augmented
additive, action SSE/zero<1, and scalar SSE/sourceconstant<1. Also compare NEXT65
and original-source model explicitly; no local winning cell substitutes for this.
Existing new8 is exposed development data, not a new blind human validation set.

If response learning improves source but not new8, inspect the same function's
source-contrast support before collecting a deliberately broader source context
panel. If even source delta remains at zero quality, account for target/design
overlap and penalty cost in this fixed dictionary; no automatic alpha rescue.
This named next question is not permission for an indefinite diagnostic series.

## Minimal implementation / tests

- New `grain_research/vector_mechanisms/paired_response_objective.py`, runner
  `scripts/run_grain_paired_response_objective.py`, focused tests and publisher.
- Private `results/grain_vector_mechanisms/logic_paired_response_objective_v1/`;
  public aggregate tables and `PAIRED_RESPONSE_OBJECTIVE_REPORT_V1.md`.
- Test equal-author weighting, pair exchange invariance, cancellation of the
  additive delta prefix, source-only delta scale, PSD/update and independent
  objective reconstruction, scalar/BODY/last compatibility and finite differences.
- Synthetic small-delta example must show why endpoint-average and relative
  response objectives differ, without claiming guaranteed new-author improvement.

Mathematical rationale and primary literature:
[NEXT65 theory](GRAIN_FULL_CONTEXT_GEOMETRY_THEORY_V1.md), especially §§3–5.
