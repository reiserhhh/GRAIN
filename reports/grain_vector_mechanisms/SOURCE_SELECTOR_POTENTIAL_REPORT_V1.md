# NEXT74: a learned selector, with a still-miscalibrated GRAIN response

2026-09-11. Finite pass63 of the ACTIVE logic/mechanism goal. Primary code/test
freeze `201a037`. All fixed comparisons completed; no new neural, API, human-label
or MEPS calls. This is numerical function construction on cached real-author
materials, not a new personality prediction benchmark.

## Finding and mainline decision

**A source-trained scalar function improves which tokens a query selects.**
On the eight source-disjoint, previously exposed authors, its logit-only version
reduces attention KL by25.62% for neuroticism and33.56% for confirmation seeking.
The same function also yields analytical GRAIN derivatives; those derivatives
retain substantial ordering but overstate response amplitude. Keep the selector
construction as progress. Do not replace the original local key-channel reader
with this complete response candidate, and do not tune on new8 to repair it.

This locates the next development task: learn an input coordinate system that
resolves *response variation*, rather than assuming directions of large H16
variance are also sufficient for a query's derivative. A fixed joint derivative
loss by itself did not solve that problem in this feature space.

## Exact experiment

- Source4 leave-one-author-out and disjoint new8/all4, both queries, all four
  frozen `whole:F01-F04` GRAIN directions and both own/pooled readout modes.
- H16 full token states, exact BODY positions, masks, RoPE and grouped heads.
  Only source native K/dK teach the function. Each split uses the same permitted
  source-mean Q for every one of its training targets and predictions.
- Identity H16 -> fixed attention22 RMSNorm/K/RoPE is the base. The learned
  correction uses the frozen source rank32 causal-context feature map. Per-sequence
  constant logit gauge and its derivative are centered together; intercept pinned0.
- Two fixed objectives, logit-only and normalized logit+GRAIN-derivative; alpha=.01,
  equal source authors/tokens within each query. Five references x2queries x2losses
  =20 solves. No new8 model selection or hyperparameter search.
- Application explicitly retains current native V and fixed g in separate packets.
  Numeric current K/dK, current Q and native action targets are denied during
  application; provenance hash checks occur before the guard. H16 and V still
  require neural extraction. No claim of fully eliminating the LLM forward pass.

## Selector fidelity

Each entry is an equal-author mean, not a token-level independent-sample test.
Targets here are source-Q/native-key logits, distinct from native-Q actions below.

| new8 query | arm | attention KL | centered-logit MSE | dlogit SSE / zero |
|---|---|---:|---:|---:|
| Neuroticism | identity | .290431 | 1.179495 | 7.196952 |
| Neuroticism | logit_only | .216025 | .700637 | 7.402913 |
| Neuroticism | joint | .217831 | .714139 | 7.311336 |
| Confirmation seeking | identity | .292344 | 1.149259 | 7.214869 |
| Confirmation seeking | logit_only | .194220 | .675864 | 7.429087 |
| Confirmation seeking | joint | .193510 | .688982 | 7.335363 |

The source4-LOO direction agrees: logit-only KL .292432->.222102 and
.290691->.205682. This is local source-disjoint transfer of the scalar selection
rule, not evidence from an untouched population. The new8 tangent error does not
improve even though the primal logit MSE falls40.60%/41.19%.

## Response fidelity and amplitude

Own readout, new8, N-minus-confirmation contrast;32 paired factor actions from
8 authors. These correlations mix the declared author/factor action panel and
are **not** a correlation with human personality labels.

| arm | r to source-Q key target | response RMS | r to native-Q key target | SSE/zero to native-Q |
|---|---:|---:|---:|---:|
| identity | .953586 | .102489 | .893283 | .856562 |
| logit_only | .962602 | .112333 | .947977 | .949533 |
| joint | .962547 | .121471 | .945037 | 1.265729 |
| original source-Q/native-key | 1.000000 | .017467 | .939345 | .535449 |

Native-Q target RMS=.059927. Source-Q target RMS=.017467. The learned logit-only
contrast RMS is6.43x its direct source-Q teaching target and1.87x the native-Q
target. A high r therefore does not make this a calibrated substitute.

Separate native-Q query correlations rise from identity .800414/.425185 to
logit-only .932900/.790547 (joint .933489/.795766), while their action errors
increase. All query/factor/mode/target conditions, including this negative result,
are retained in the accompanying tables; neither query was chosen after scoring.

## What the source objective says

All4-source normalized residual logit loss falls to .569972/.561799 (logit-only).
Joint gives .582980/.574724; its derivative residual loss is1.000642/1.000577.
Thus this feature family learns the logit correction but almost no reduction of
the derivative residual, already on its sources. That identifies a *development
target*, not yet whether limited derivative span, regularization geometry or
value/derivative conflict dominates. NEXT75 distinguishes them on source data
before constructing a response-informed feature map. No arbitrary shrinkage of
the output derivative or new8-based calibration is adopted.

## Verification, outputs and limits

- Audit PASS,20solves,24 query packets,4 analysis tables exactly replayed.
-48 same-function central differences: max error4.7815e-11.
- Original native-key comparator replay: max error6.2450e-17.
-31 focused tests pass; new implementation/publisher pass Ruff and whitespace checks.
- Cumulative measured stage CPU11.527251s, including one application/evaluation
  replay, below the120s cap. Provenance preflight hashing/imports are not timed
  as numeric application; no background job remains.
- Source fit denies new8; application denies native K/dK/capture/actions. No
  unexpected opens. Canonical table inventory excludes filesystem sidecars.
- Frozen H16 perturbations are local numerical interventions, not edited natural
  utterances or clinical states. Neither attention weights nor their effects are
  automatically human personality importance.

Private reproducible outputs: `results/grain_vector_mechanisms/logic_source_selector_potential_v1/`.
Public aggregates: [source_selector_potential_v1](source_selector_potential_v1/).
Runtime: `../project\ persona/.venv/bin/python scripts/run_grain_source_selector_potential.py`
with stages `prepare`, `fit`, `predict`, `evaluate`, `predict --folder replay`,
`evaluate --folder replay`, `audit`, in separate processes and a fresh output
directory for any repetition. Do not overwrite the frozen primary run.

Theory: [scalar selector and response geometry](../../docs/GRAIN_SOURCE_SELECTOR_POTENTIAL_THEORY_V1.md).
Next: [source response-informed coordinates](../../docs/GRAIN_SOURCE_RESPONSE_COORDINATES_PLAN_V1.md).
