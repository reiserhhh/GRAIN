# NEXT40: source-fixed reading of GRAIN attention responses

2026-09-10. Finite pass29 complete; whole-system logic goal remains ACTIVE.
Implementation and main analysis were committed in `72817e8` before outcomes.

## 1. Finding and decision

A fixed direction compiled from four source authors can read the last-attention
GRAIN response of eight source-disjoint authors without their gradients, scores
or output cotangents. Same-query source means recover local action r=.809 for
neuroticism and .955 for confirmation-seeking. The complete value-plus-key
program substantially reduces squared error versus retaining value alone.

Keep this mechanism and move to complete upstream transport. Do not tune source
means on these eight authors or substitute these correlations for human trait
accuracy. The target is the normalized BODY-input derivative of the fixed
digit-logit contrast, not the original probability score or block16 derivative.

## 2. Fixed comparison

All rows below use the same8 authors x4 declared GRAIN directions (32 actions).
`SSE/zero` is sum squared action error divided by sum squared native action;
it is not conventional mean-centered R-squared. Centered r removes each
factor's across-author mean before pooling. These are descriptive repeated-unit
comparisons, not 32-independent-person correlations or new blind significance tests.

| Query | Source g | Channel | Action r | Factor-centered r | SSE/zero | Predicted RMS | Native RMS |
|---|---|---|---:|---:|---:|---:|---:|
| N | same query | full | .8090 | .7983 | .3833 | .09019 | .09134 |
| N | same query | value | .7684 | .7330 | .7651 | .01576 | .09134 |
| N | pooled queries | full | .9192 | .9152 | .1570 | .08006 | .09134 |
| N | pooled queries | value | .8763 | .8529 | .7574 | .01384 | .09134 |
| Confirmation | same query | full | .9546 | .9560 | .0985 | .07975 | .09332 |
| Confirmation | same query | value | .8962 | .8998 | .7504 | .01420 | .09332 |
| Confirmation | pooled queries | full | .8692 | .8660 | .2445 | .08195 | .09332 |
| Confirmation | pooled queries | value | .8222 | .8187 | .7621 | .01495 | .09332 |

Same-query/full remains the declared primary program. Pooled/full is a separate
predeclared program, not a selected replacement for N alone. All16 cohort/query/
program/channel rows are retained in [comparison.csv](source_cotangent_tables_v1/comparison.csv).

Old4 author-LOO uses three other sources, whereas new8 uses all four sources.
Old4 full-action r is N .9469/.9021 and confirmation .7467/.7691 (own/pooled).
Do not interpret cohort differences as a controlled improvement in generalization.
Some old4 value-only correlations exceed full correlations despite much worse
squared error/amplitude; the report keeps both metrics rather than ranking by r.

New8 same-query/full per-factor N correlations are .940/.940/.560/.508; pooled
are .930/.960/.848/.939. Confirmation own is .956/.962/.909/.986 and pooled
.929/.907/.696/.950. Full [factor table](source_cotangent_tables_v1/factor_actions.csv)
preserves weaker directions and all omission controls. Factor codes are unchanged.

## 3. Mechanism and remaining inputs

Let A be the attention derivative at the current forward context, D_f the fixed
GRAIN BODY direction and g the downstream cotangent. Compute lambda_f=A D_f
from q/k/v, rotary positions and the complete-context softmax; then read
tau_hat=lambda_f^T gbar with a source-only gbar. This is dual to pulling back
gbar and contracting the full token field. The program explicitly retains both
value transport and key-weight change. The runtime checks both computations.

Current q/k/v, normalized hidden states, positions and GRAIN coordinates are
the only per-unit predictive inputs. Projection weights and source g means are
fixed assets. These forward activations still come from the native encoder;
source g was originally collected through source-model backward passes.
Zero new neural calls in this run does not mean those assets were never neural.

New8 same-query/full field cosine means are .9796/.9775 (N/confirmation), pooled
field relative squared error .0878/.0540. Field and GRAIN-action geometry differ.
The [source geometry table](source_cotangent_tables_v1/source_geometry.csv) shows
N g-error rising .0198->.0357 with pooling while action SSE/zero falls .3833->.1570.
The exact action error is Lambda(gbar-g), whose squared norm weights g error by
Lambda^T Lambda. This is a useful local error explanation, not a fitted rescue.

The four forward response vectors contain6144 numbers total. This suggests a
compact measurement interface, but the current verification runtime also creates
full fields. No field-free speedup benchmark or new personality factor is claimed.

## 4. Execution and replay

-24 explicit units: old4/new8 x2 queries.48 source programs,96 field conditions,
 384 action conditions. These counts are not sample size or independent tests.
-Old4 source means exclude the evaluated author; new8 never enter construction.
 No optimizer, label read, calibration, rank search or new model forward/backward.
-Restricted scoring process receives split forward-only inputs. Both attempts
 to read native target packets/job metadata are denied; no unexpected denials.
 All predictions freeze before evaluator reads current gradients/cotangents.
-Numerical runtime25.566s; fresh restricted replay25.588s, excluding preparation
 and loading. Both satisfy the120s numerical cap. No live job remains.
-24 prediction NPZ packets and6 analysis CSVs replay byte-for-byte. The error
 identity closes within6.11e-16; forward/native action parity within3.35e-7.
-54 scoped tests pass in2.24s, including source exclusion, strict schemas,
 attention finite differences and JVP/pullback duality. No full-suite claim.

Audit the already frozen run from the independent repository, with its private caches:

```bash
"../project persona/.venv/bin/python" scripts/audit_grain_source_cotangent.py
"../project persona/.venv/bin/python" -m pytest -q tests/test_grain_source_cotangent.py
```

For an additional complete numerical reproduction, use a fresh name throughout:

```bash
"../project persona/.venv/bin/python" scripts/run_grain_source_cotangent.py prepare --name logic_source_cotangent_replication_v1
"../project persona/.venv/bin/python" scripts/run_grain_source_cotangent.py predict --name logic_source_cotangent_replication_v1
"../project persona/.venv/bin/python" scripts/run_grain_source_cotangent.py replay --name logic_source_cotangent_replication_v1
"../project persona/.venv/bin/python" scripts/run_grain_source_cotangent.py evaluate --name logic_source_cotangent_replication_v1
"../project persona/.venv/bin/python" scripts/run_grain_source_cotangent.py evaluate --name logic_source_cotangent_replication_v1 --analysis-name analysis_replay
```

Do not overwrite an existing destination. The audit script above checks the
original frozen run, not arbitrary names. No parameters change in a reproduction.

Private artifacts: `results/grain_vector_mechanisms/logic_source_cotangent_reader_v1/`:
manifest, source-only models, forward inputs, prediction/replay freezes, raw
per-unit analysis and replay audit. Public tables above are aggregate only.
Code: `grain_research/vector_mechanisms/source_cotangent.py`,
`scripts/run_grain_source_cotangent.py`, `scripts/replay_grain_source_cotangent.py`,
`scripts/audit_grain_source_cotangent.py`. No raw participant material is published.

## 5. Next decision

[NEXT41](../../docs/GRAIN_COMPLETE_TRANSPORT_JVP_PLAN_V1.md) carries block16 GRAIN
directions through the actual remaining context, including final-query and direct
residual routes, to a source-fixed downstream reader. It does not simply compare
this .955 with NEXT37's different-layer target. Current local reading is retained
while its full upstream usefulness is tested.

[Theory and formulas](../../docs/GRAIN_SOURCE_COTANGENT_READER_THEORY_V1.md),
[pre-outcome plan](../../docs/GRAIN_SOURCE_COTANGENT_READER_PLAN_V1.md).
No release scorer, old lockbox, clinical interpretation or human-label model changed.
