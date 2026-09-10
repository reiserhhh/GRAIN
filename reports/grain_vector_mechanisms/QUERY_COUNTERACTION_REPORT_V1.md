# NEXT46: the MLP24 counteraction mostly carries an existing difference

2026-09-10. Completed finite mechanism packet; primary implementation frozen at
`4db9699`. Same12 authors, two queries, four GRAIN directions, unchanged own/pooled
shared readers. These are local model-response contrasts, not questionnaire scores.

## Main finding

The negative MLP24 update mostly arises from applying the two-query average local
Jacobian to an already different incoming direction. The operator-difference term
is smaller and, in aggregate, partially offsets that negative projection on final T.

| Cohort/mode | Carried RMS | Operator RMS | Local contrast RMS | Carried share of local contrast | Carried projection on final T | Operator projection on final T | Total on final T |
|---|---:|---:|---:|---:|---:|---:|---:|
| new8 own | .090579 | .024635 | .095267 | .918570 | -.313540 | +.067912 | -.245629 |
| new8 pooled | .090066 | .024652 | .094889 | .916719 | -.313741 | +.066621 | -.247120 |
| source4 own | .084357 | .016109 | .076845 | 1.080572 | -.387425 | +.167956 | -.219469 |
| source4 pooled | .083885 | .015981 | .076415 | 1.080668 | -.400094 | +.166364 | -.233731 |

Signed shares use dot(component,target)/dot(target,target), not independent
variance fractions. The local-contrast target differs from final T, so the same
component can have positive local share and positive final-T projection while
partially offsetting the carried component's negative final-T projection.

All4 new8 factors have negative carried projections on final T:
F01 -.439427, F02 -.246137, F03 -.197715, F04 -.387909 (own).
The operator term is -.002709,+.105595,+.039041,+.101691. Do not describe every
factor as positive operator correction. Source4 F03 has positive carried and
operator projections; source4 F02 has a small positive total. Aggregate cohort
agreement is not universal factor-level agreement.

The new8 own local cross-energy ratio is +.029122 and covariance +.000167;
source4 own is -.249032 and -.000734. All signed/cancellation terms remain in
the [summary](query_counteraction_tables/summary.csv) and
[all-factor table](query_counteraction_tables/factors.csv).

## Computation and verification

For each query q, J_q is the full RMSNorm/SwiGLU derivative at actual preMLP24 x_q.
The2x2 exchange applies both J_N/J_C to both d_N/d_C. Product-rule terms remain
separate until summation. With bars denoting pair means:

    J_N d_N - J_C d_C = Jbar (d_N-d_C) + (J_N-J_C) dbar.

This is an exact symmetric accounting identity, not a unique causal decomposition.
Current x and incoming d still require native forward contexts/transport. Jbar
is within-author across queries, not a constant across authors.

- Collected24 primal prefixes plus4 fixed F01 boundary JVPs; stopped after
  attention24. Runtime hooks forbid MLP24, all later blocks and the output head.
  Actual block24 weights exported, not block25. Native14.102889s plus3.492165s
  model load, within the120s native/28-call cap. No scoring/backwards/fitting/API.
- Boundary input maximum relative/absolute error7.181e-8/1.690e-7. The two
  query checks for source4_a00 and new8_a00 all pass the original tolerance.
- Numerical crossing.955888s; replay.952978s (compute/save stage, excluding
  verification and weight loading). No native target opened in either stage;
  four target-denial probes pass with zero additional denied accesses.
- Predictions frozen before evaluation opens NEXT42 MLP24 increments. These
  increments were observed in earlier studies, so this is staged mechanism
  construction, not a previously unseen target test. Prefix preparation copies
  only initial+updates[:15]; target updates[:,15] is not a construction field.
- All96 diagonal full vectors pass relative<=1e-4 OR absolute<=1e-5. Maximum
  relative2.210e-6/absolute6.656e-6. No scalar-only parity shortcut.
-12 paired NPZ array replays exact;3 CSV and1 JSON byte replays exact.
  Independent regrouping of the four exchange cells differs by3.331e-16;
  independently recomputed signed metrics differ by4.441e-16.
-96 focused tests pass,20 existing Python3.14/PyTorch/SWIG warnings. Ruff and
  diff checks pass; not a full-suite claim. No active native job remains.

Private artifacts: `results/grain_vector_mechanisms/logic_query_counteraction_v1/`:
manifest, private inputs/prefix/capture/weights, capture freeze/status,
predictions and predictions_replay, analysis and analysis_replay, replay audit.
Only aggregate tables and this report are published in the repository.

## Decision and relationship to the mainline

Retain the counteraction as a functional part of the same-material GRAIN/query
interface. Do not delete it as noise or rename it a defense mechanism.
The next useful simplification is a source-fixed, query-common local reader:
can old-author MLP24 Jacobians preserve this negative response in new authors?
This tests a concrete remaining dependency, not another leaderboard search.

The user's second connection, psychodynamic/transactional auxiliary interpretation,
continues under the existing [same-material interpretation protocol](../../docs/GRAIN_SAME_MATERIAL_INTERPRETATION_THEORY_V1.md).
Numeric factor/path evidence and relation-event interpretation share material
addresses; they do not become the same measurement. No participant text was
sent to an LLM and no new clinical coding result is claimed in this packet.

Read [the mechanism derivation](../../docs/GRAIN_QUERY_COUNTERACTION_THEORY_V1.md)
and [NEXT47 plan](../../docs/GRAIN_SOURCE_COMMON_COUNTERACTION_PLAN_V1.md).

Reproduction from this commit, using the existing private source artifacts:

```bash
PY="../project persona/.venv/bin/python"
$PY scripts/run_grain_query_counteraction.py prepare
$PY scripts/run_grain_query_counteraction.py capture
$PY scripts/run_grain_query_counteraction.py predict --name predictions
$PY scripts/run_grain_query_counteraction.py evaluate --name analysis --predictions predictions
```

The existing output is frozen and will refuse accidental repeated capture.
Use a fresh `--output` directory for an authorized reproduction; no private
data or model weights are bundled in the public report.
