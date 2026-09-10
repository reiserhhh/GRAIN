# NEXT75: response-informed coordinates reduce amplitude error

2026-09-11. Finite pass64, overall logic goal ACTIVE. Primary `9defd3d`,
aggregation-only compatibility amendment `d7a1136`. No neural/API/human-label/MEPS
calls. All figures below are fixed-model computations on cached author materials.

## Result and decision

**Changing the source coordinate system changes how accurately GRAIN directions
are read, without increasing the downstream feature width.** On the existing
eight source-disjoint authors, the new scalar selector reduces the native-query
N-minus-confirmation response error from1.265729 to .373022 relative to the old
joint model. Response RMS falls from .121471 to .077174 (native target .059927).
Its correlation falls from .945037 to .881432: amplitude and ordering are distinct
properties. Retain this local development, not a universal winning predictor.

The source account also identifies why the earlier derivative fit stalled:
only13.16%/13.27% of its residual target energy lies in the declared numerical
derivative-design range. Fitting that accessible part without the joint value
constraint would make the source value error enormous. Both representation and
coupled fitting matter; neither is fixed by simply adding more weight to a loss.

The direct source-query teaching target is still badly approximated in amplitude.
The lower error to the native-query target is therefore not evidence that all
intermediate mechanisms have been reconstructed. The next card develops the
remaining native value dependency, not another selector/rank/alpha sweep.

## Fixed design and source account

- Same source4 leave-one-author-out and new8/all4; both neuroticism and confirmation
  seeking queries, all four `whole:F01-F04` directions, both own/pooled readouts.
- Source derivative capacity from ten old sufficient-statistic sets; Gram
  pseudoinverse cutoff1e-10 times its largest eigenvalue. Diagnostic minimizers
  are never used as new8 predictors.
- Source-standardized full1536-coordinate H16, value/directional cross moments
  for both queries, rank32 SVD. Same67-column causal-context/position design,
  one normalized value-plus-derivative objective, alpha=.01, ten new solves.
- All source references precede application. Current native K/dK/Q and action
  targets are denied in the application process. H16, current V and fixed g remain
  separate declared inputs. The new eight authors were repeatedly examined in
  previous experiments; they are source-disjoint, not an untouched replication.

All4-source results; losses are residual-target normalized:

| query | old derivative numerical rank / width | accessible residual energy | minimum derivative loss | value loss at derivative minimum | old joint d loss | new joint d loss | old / new joint value loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Neuroticism |64/67|.131600|.868400|37048.62|1.000642|.962501|.582980 / .526707|
| Confirmation seeking |64/67|.132708|.867292|31237.16|1.000577|.963165|.574724 / .526380|

The all4 Gram retained condition number is1.271071e8. Minimum derivative
coefficient norms664.91/602.54 compare with old joint norms2.064/2.068. Across
LOO references, accessible fractions are .1462-.2246. These are finite-design
calculations at a declared numerical tolerance, not limits on all textual signal.

All five response-informed cross matrices have rank48; rank32 retains .999729-
.999753 of **cross-moment** squared singular mass. This is not99.97% recovery of
the derivative target, hidden state, or personality information. The new all4
derivative residual losses remain .9625/.9632 despite that concentrated cross
moment. The selection-related coordinates and the full residual are different
objects.

## New8 selector and responses

Equal-author averages, source-query/native-key selector target:

| query | arm | attention KL | centered logit MSE | derivative SSE / zero |
|---|---|---:|---:|---:|
| N | identity |.290431|1.179495|7.196952|
| N | old joint |.217831|.714139|7.311336|
| N | response coordinates |.186394|.687386|7.246516|
| Confirmation | identity |.292344|1.149259|7.214869|
| Confirmation | old joint |.193510|.688982|7.335363|
| Confirmation | response coordinates |.199128|.671221|7.269358|

Both primal selectors improve over identity, but confirmation KL worsens relative
to old joint and derivative error remains above identity. Source4-LOO has the same
qualitative qualifications; no failed query or readout is excluded.

Own-readout N-minus-confirmation contrast,32 factor actions from8 authors:

| arm | r to source-query target | SSE/zero to source-query target | response RMS | r to native-query target | SSE/zero to native-query target |
|---|---:|---:|---:|---:|---:|
| identity |.953586|24.227114|.102489|.893283|.856562|
| old logit-only |.962602|29.951061|.112333|.947977|.949533|
| old joint |.962547|35.944281|.121471|.945037|1.265729|
| response coordinates |.934566|12.233658|.077174|.881432|.373022|
| original source-query/native-key |1.000000|0|.017467|.939345|.535449|

Native-query target RMS=.059927, source-query target=.017467. Thus the new
response remains4.42x the direct teaching target despite moving closer to native
query amplitude. Treat the native-query gain as a useful composite effect, not
proof of faithful source-query mechanism recovery or improved questionnaire r.

Per-query native-target action SSE/zero: N old joint2.379117 -> new .717866;
confirmation6.697798 ->3.332438. Relative to identity1.056120/2.227144, N improves
but confirmation worsens. This is why the full reader is not promoted. The complete
query/factor/mode tables retain these differences rather than selecting a contrast.

## Reproduction and artifacts

- Audit PASS:10 new solves,10 capacity accounts,24 primary query packets and
  their numeric replay, four exact CSV replays. Published aggregate tables:7.
-16 central differences of the same scalar: max error3.847373e-11; original
  teacher replay6.245005e-17; old comparator maximum deviation7.105427e-15.
-36 focused tests pass; implementation, compatibility wrapper, publisher and
  their tests pass Ruff. This is a focused-suite result, not a full-repo claim.
- Successful numeric stages including replay total8.173818CPU seconds. The
  first aggregation attempt failed before writing tables and its CPU time was not
  captured. Imports/provenance preflight are also outside numeric stage timing.
  Do not describe8.173818 as total all-attempt execution cost.
- The failure was pandas3 StringDtype classification in the frozen comparator.
  A wrapper sets `future.infer_string=False`; primary numeric code, source fits
  and predictions are unchanged. Original empty analysis folder and amendment
  metadata are retained. No model rerun was needed for the fix.

Private output: `results/grain_vector_mechanisms/logic_source_response_coordinates_v1/`.
Public: [seven aggregate tables and provenance](source_response_coordinates_v1/).

Use the existing parent venv from the independent repository. Primary script
`scripts/run_grain_source_response_coordinates.py` stages: `prepare`, `fit`,
`predict`, then `predict --folder replay`. Evaluate through
`scripts/evaluate_grain_source_response_coordinates_compat.py`, once normally and
once with `--folder replay`; finally primary `--stage audit`, followed by
`scripts/report_grain_source_response_coordinates.py`. Existing primary outputs
are frozen; a repetition needs a fresh output path, not an overwrite.

[Theory](../../docs/GRAIN_SOURCE_RESPONSE_COORDINATES_THEORY_V1.md),
[next value-channel card](../../docs/GRAIN_SOURCE_VALUE_CHANNEL_PLAN_V1.md).
