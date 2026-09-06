# Direct scoring: behavioral ordering and output-code dependence

2026-09-07, round34. Complete frozen90-case panel,180 real forwards,
independently reconstructed. No natural personality labels or model training.

## Decision

The pinned Qwen1.5B observer often orders obvious invented behaviors correctly
under the original wording, but its numeric/letter answers strongly depend on
answer coding and listing order. **Format compliance is not stable measurement
semantics.** Keep the fixed direct control and report its source performance;
do not rescue prompts, reverse axes, discard traits or select a better variant.
This panel passed numerical execution, not code-invariant personality assessment.

## Complete results

Each task has two invented low/high behavior pairs. Primary is the original
direct wording. Counter reverses numeric-scale meaning for Big5, with scores
properly recoded to the original trait direction. For MBTI it reverses preference
and answer listing, leaving positive E/N/F/J coding unchanged. These are different
manipulations; their effect magnitudes are not directly comparable.

| Task | Primary positive pairs /2 | Counter positive pairs /2 | Mean absolute recoded score change | No-evidence primary -> counter |
|---|---:|---:|---:|---|
| Extraversion |2|0|.5951|.0812 -> .9447|
| Neuroticism |1|0|.1838|.5980 -> .7842|
| Agreeableness |2|0|.8000|.1797 -> .9039|
| Conscientiousness |2|0|.5834|.0779 -> .9460|
| Openness |2|0|.8710|.3227 -> .8000|
| EI, positive E |2|2|.4135|.9399 -> .0474|
| SN, positive N |2|2|.5589|.7311 -> .0474|
| TF, positive F |2|1|.7054|.8808 -> .0203|
| JP, positive J |2|2|.4003|.9579 -> .0059|

Primary ordering:17/18 positive pairs (Big5 9/10, MBTI 8/8). Counter:7/18
(Big5 0/10, MBTI 7/8). Every Big5 counter difference is negative, not merely zero.
Mean Big5 high-minus-low changes from+.5942 to-.5875 after correct recoding.
This is consistent with conventional-number response habits despite the inversion
instruction, but does not identify their internal circuit.

MBTI ordering mostly survives while absolute scores shift by.4003-.7054 per axis
on average. Ranking signal and stable score meaning are different. No-evidence
values are model/prompt priors, not validated neutral-person scores or targets
that ought to equal.5. The contrast changes order in both question and return
instruction; it cannot isolate those contributions or prove an attention mechanism.

Reported unrestricted first-token compliance is90/90. Minimum candidate mass
is.9924065, median.9998002. Low answer-alphabet mass therefore cannot explain
the large recoded changes in THIS panel. This is not full-response JSON parsing.
No population p-values, equivalence tests or prevalence claims are appropriate
for two authored examples per task and one pinned observer.

## Mathematical implication

For a coding/listing convention e, the actual native score is

```
s_e(x) = v_e^T softmax_Ae(logits_theta(x, question_e))
```

A code-invariant measurement would approximately preserve s_e(x) after explicit
value recoding when only the response convention changes. This observer does not
do so on these examples. Its score composes evidence processing, question
interpretation, response preferences and decoding, rather than author evidence
alone. This is a standard invariance/measurement distinction, not a new theorem.

This does NOT establish absent personality information inside hidden states, or
that generic vector heads capture the native decoder's computation. Conversely,
primary anchor ordering does not establish natural-text prediction. Full source
scores, context effects, readout controls and source-frozen bridge remain required.

## Evidence and limits

Round35's bounded post-hoc [logit interaction diagnostic](VM05_ENCODING_INTERACTION_REPORT.md)
finds that the change is not merely a constant candidate-bias shift on these
examples. It does not apply a calibration or invalidate repeatability under a
fixed prompt. Cross-convention invariance and fixed-protocol repeatability are
distinct requirements; the latter passed the original double-forward check.

- Packet vm05_direct_semantics_real_r1, identity
  cae61a997fb36bd5d94605b9117ebfd9a089169c537f248963e2c6fdcc09fe9b.
  All90 cases repeated exactly by the collector;180 forwards in85.625s.
- Independent audit bound every case/text/token/alphabet to the earlier Mac
  token preflight and checked model/qualification/code/archive/output hashes.
  Shifted-exponential reconstruction of scores, entropy, mass,36 contrasts and
  nine task tables differs by at most4.11e-15 from native calculations. Full
  vocabulary logits were not saved, so unrestricted argmax cannot be independently
  recomputed from this packet; original native repeated forwards are separate evidence.
- Audit vm05_direct_semantics_independent_audit_r2, identity
  196b21250296c304b1f64085c81ffd96aad954b3ceb57d63ac26aaf66ad4fb07.
  Retained r1 accidentally bound macOS metadata sidecars as outputs; r2 explicitly
  names only the two scientific CSVs. Quantitative results are unchanged.
- Script audit_grain_direct_semantics.py; grid/arithmetic/recoding mutation tests.
  The scoped direct/history/I/O suite is rerun at closeout. Synthetic fixtures
  are not additional real-model replications.

Execution followed evidence-preserving history I/O recovery and zero-forward
revalidation. Semantic inputs and gates were not weakened to bypass FAILED_STOP.
See GRAIN_VM05_HISTORY_STATUS_IO_RECOVERY_V1.md. No bridge/MEPS scale values,
paid APIs, extra training or released-scorer changes.
