# NEXT47: a source-fixed reader preserves the local counteraction on new authors

2026-09-10. Completed; primary implementation frozen in `d9b9081`. Same12
authors/two queries/four frozen GRAIN directions/two shared-reader modes as
NEXT46. Source4 evaluations use leave-one-author-out; new8 use all4 source
authors. The new8 materials have been inspected before; this is source-disjoint
mechanism transfer, not a new blind personality benchmark.

## What was gained

The MLP24 adjustment no longer needs each new author's current preMLP24 state
or its current gate/norm coefficients at the local reader step. The same two
fixed1536-dimensional vectors (one per reader mode) serve all8 new authors
and both queries. Incoming GRAIN tangents d still require native upstream
computation; no complete raw-text LLM replacement is claimed.

| Cohort/mode | Contrast r | SSE/zero | Predicted RMS | Native RMS | Same signs | Projection on final T |
|---|---:|---:|---:|---:|---:|---:|
| new8 own | .955274 | .091333 | .084474 | .095267 | 28/32 | -.304183 |
| new8 pooled | .954853 | .092402 | .083911 | .094889 | 28/32 | -.303845 |
| source4 own | .980468 | .121568 | .097434 | .076845 | 15/16 | -.474176 |
| source4 pooled | .980548 | .121727 | .096938 | .076415 | 15/16 | -.488534 |

The predeclared local retention rule passes in both new8 modes: negative
final-T projection retained and contrast SSE/zero<1. This is not exact amplitude
recovery: new8 own retains88.67% native contrast RMS; source4 overshoots. Native
final-T projections remain -.245629/-.247120 for new8 and -.219469/-.233731
for source4. Projection coefficients are signed action accounts, not personality
information percentages.

The current-author pair-average carried term remains an explanatory comparator:
new8 own r=.965867,SSE/zero=.066869. It is somewhat closer than the fixed-source
version but requires the current x/J that this packet removes.

Per-query new8 own actions: N r=.987321,SSE/zero=.052770; confirmation
r=.975147,SSE/zero=.054740. Full per-query/cohort/mode values are retained in
[actions](source_common_counteraction_tables/actions_by_query.csv).

All new8 factor projections stay negative. Own-mode per-factor contrast r is
.989343/.934548/.614982/.936716 for F01–04; SSE/zero is
.048847/.332232/.639807/.146375. Thus aggregate fidelity does not imply all
factor-specific actions are equally easy. Factors are directions within authors,
not32 independent people. No significance or new questionnaire r is reported.

## Remaining error now has an address

Let c be the previous current pair-average carried contribution and o its
operator-difference contribution. Native n differs from c+o only by measured
floating-point parity. For frozen-source prediction p:

    p-n = (p-c) - o + (c+o-n).

| Cohort/mode | Source replacement error RMS | Missing operator RMS | Native parity RMS | Total error RMS |
|---|---:|---:|---:|---:|
| new8 own | .013454 | .024635 | 1.84e-7 | .028791 |
| new8 pooled | .013467 | .024652 | 1.82e-7 | .028844 |
| source4 own | .013857 | .016109 | 1.35e-7 | .026793 |
| source4 pooled | .013842 | .015981 | 1.35e-7 | .026661 |

The missing query-operator contrast is the larger component on new8. Errors
are correlated: their cross-energy/native-target-energy ratio is+.004520 for
new8 own and+.045110 for source4 own. Keep covariance rather than adding RMS
values or calling the errors independent. The tiny numerical parity remainder
is retained explicitly, not assigned to a psychological mechanism.

## Implementation / checks

Source compilation: v*=mean_s,q J24(x_s,q)^T w*. Prediction: action_q=v*^T d_q,
contrast=v*^T(d_N-d_C). The full paired B_s N_s operator is averaged, not the
product of independently averaged gate/norm terms. Current prediction reads
only v,d; construction never reads new8 x or output targets.

-8 source x records only;12 author slots and all source exclusions verified.
  Actual block24 weights pinned to NEXT46 export. Current d/w copied as a
  two-field schema; no target or current x included.
-192 actions/96 contrasts frozen before target evaluation. Four denial probes,
  zero extra denied accesses;0 neural/scoring/backward/API/fit/human-label reads.
- Numeric4.069140s, replay4.065182s, each under60s, including weight loading,
  source compilation, independent direct-Jacobian identity checks and saving.
- Compiled-vector vs direct-source-operator scalar error<=1.222e-15.
 12 exact paired-array replays;6 exact CSV replays. New8 vectors are bytewise
 equal across all8 authors. Independent metric error2.221e-16, error identity
 <=1.111e-16 after CSV rereading.
-101 focused tests pass,20 existing Python3.14/PyTorch/SWIG warnings. Ruff and
  diff whitespace checks pass; no full-suite claim. No running model job.

See [aggregate comparisons](source_common_counteraction_tables/summary.csv),
[all factors](source_common_counteraction_tables/factors.csv),
[error account](source_common_counteraction_tables/error_accounting.csv).
Private lineage/models/predictions/targets stay under
`results/grain_vector_mechanisms/logic_source_common_counteraction_v1/`.
Source and release scoring, lockbox, MEPS boundaries and unrelated edits unchanged.

## Decision

Retain this useful fixed local reader. NEXT48 asks once whether the smaller
query contrast can also be compiled from sources into a fixed query-difference
vector. That is a dependency question, not a search for another decimal of r.
After that packet, stop expanding MLP24 alternatives and return to the remaining
upstream input/transport dependency. See [theory](../../docs/GRAIN_SOURCE_COMMON_COUNTERACTION_THEORY_V1.md)
and [NEXT48](../../docs/GRAIN_SOURCE_QUERY_CORRECTION_PLAN_V1.md).

Reproduce with the existing private source artifacts, using a fresh `--output`
if the default packet is already frozen:

```bash
PY="../project persona/.venv/bin/python"
$PY scripts/run_grain_source_common_counteraction.py prepare
$PY scripts/run_grain_source_common_counteraction.py compile --name compiled
$PY scripts/run_grain_source_common_counteraction.py evaluate --name analysis --compiled compiled
```

No new native capture or credential is needed. Do not publish private model arrays.
