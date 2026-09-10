# NEXT48: the query-difference correction also has a reusable source component

2026-09-10. Completed after primary code freeze `247fe90`; source design was
specified in NEXT48 before this experiment. Same12 authors, two queries
(neuroticism and confirmation seeking), four frozen GRAIN directions and two
reader modes. Source4 leave-one-author-out; new8 use all4 sources. These are
previously inspected materials, not a new blind personality benchmark.

## Finding and usable result

The source-fixed query correction tracks the current local query-operator
contrast on new8: r=.886208 own/.885192 pooled. Both correction SSE/zero
ratios are below1, satisfying the fixed candidate-retention rule. Added to the
unchanged common reader, it improves the total local contrast as well.

| Cohort/mode | Common contrast r | Corrected contrast r | Common SSE/zero | Corrected SSE/zero | Corrected signs |
|---|---:|---:|---:|---:|---:|
| new8 own | .955274 | .976535 | .091333 | .060734 | 32/32 |
| new8 pooled | .954853 | .976360 | .092402 | .061466 | 32/32 |
| source4 own | .980468 | .992293 | .121568 | .091009 | 15/16 |
| source4 pooled | .980548 | .992219 | .121727 | .091598 | 15/16 |

These r values compare a mathematical reader with actual model-local effects,
not predicted personality with human questionnaires. New8 own total MSE falls
by33.5% relative to the previous common reader. Amplitude is not exact:
corrected/native RMS=.081478/.095267; source4 .097374/.076845 overshoots.
Final-T signed projection changes from -.304183 to -.216430 (native -.245629).
These projections are signed response accounting, not information fractions.

| Cohort/mode | Correction vs actual operator r | SSE/zero | Correction RMS | Actual operator RMS | Same signs |
|---|---:|---:|---:|---:|---:|
| new8 own | .886208 | .240669 | .025547 | .024635 | 30/32 |
| new8 pooled | .885192 | .241317 | .025435 | .024652 | 30/32 |
| source4 own | .769220 | .434771 | .015018 | .016109 | 14/16 |
| source4 pooled | .767218 | .438605 | .014908 | .015981 | 14/16 |

All factor results are retained. New8 own F01–04 operator correlations are
.808/.972/.982/.888, with SSE/zero .417/.161/.039/.312. Source4 F02 is an
important contrary detail: r=.734/.731 but SSE/zero=1.012/1.016 (own/pooled),
slightly worse than predicting zero. The aggregate result does not validate
every individual factor. Four directions per author are not independent people.

## What is actually computed

For the permitted source set S and fixed shared readout w*, compile:

    v_q = mean_(s in S) J24(x_s,q)^T w*
    v_common = (v_N+v_C)/2; v_difference = v_N-v_C
    p = v_common^T(d_N-d_C)
    qhat = v_difference^T(d_N+d_C)/2
    predicted_contrast = p+qhat = v_N^T d_N-v_C^T d_C.

The common vector equals NEXT47's frozen reader within5.205e-18. All8 new
authors use exactly the same query/mode readers. Construction uses only source4
states and actual block24 weights; current application uses only v and d.
It does not read new8 x, gates, target increments, gradients or human labels.
The upstream d remains a native transported GRAIN direction, not a new
standalone text encoder. Source transfer is empirical; the symmetric algebra
and adjoint identity are standard mathematics, not new psychological theorems.

## Error accounting prevents a cancellation-only interpretation

Let c,o,n be current carried/operator/native contrast, and p,qhat be frozen
source common/correction estimates. The exact accounting is:

    e_old = p-n
    e_new = (p-c) + (qhat-o) + (c+o-n)
    e_new^2-e_old^2 = 2*e_old*qhat + qhat^2.

| Cohort/mode | Source error RMS | Query error RMS | Parity RMS | Old total error RMS | New total error RMS |
|---|---:|---:|---:|---:|---:|
| new8 own | .013454 | .012086 | 1.84e-7 | .028791 | .023478 |
| new8 pooled | .013467 | .012110 | 1.82e-7 | .028844 | .023525 |
| source4 own | .013857 | .010622 | 1.35e-7 | .026793 | .023182 |
| source4 pooled | .013842 | .010584 | 1.35e-7 | .026661 | .023127 |

New8 own mean squared-error change=-.000277713: cross term=-.000930379,
correction-square term=+.000652666. Source/query errors have positive covariance
.000109851; no independence assumption or RMS addition. The operator-target
test itself succeeds, so the result is not supported solely by final-error
cancellation. The total improvement and the recovered query term remain
different claims, each measured separately.

## Reproducibility and resources

-192 actions/96 contrasts frozen before evaluation; four blocked-target probes,
  no unexpected denied access.0 native/scoring/backward/API/fit/human-label calls.
- Primary numerical run4.077576s; exact replay4.094619s, both below60s.
-12 prediction arrays and8 CSVs replay exactly;8 new-author readers identical.
  Independent metric difference<=1.222e-15, source direct/compiled identity
  <=8.327e-16, reread error identity<=1.111e-16 and MSE-change identity<=1.453e-16.
-105 scoped tests pass in4.53s;20 existing PyTorch/Python3.14/SWIG warnings.
  No full-suite claim. AppleDouble metadata was excluded by manifest addressing
  after an audit glob encountered it; unsafe pickle loading was never enabled.
- Private states, readers, targets, predictions and access/replay audit remain
  in `results/grain_vector_mechanisms/logic_source_query_correction_v1/`.
  Only the six aggregate CSVs are published here. No live native experiment,
  release/scorer/lockbox/MEPS change or remote push.

Aggregate [total comparison](source_query_correction_tables/summary.csv),
[operator comparison](source_query_correction_tables/operator_summary.csv),
[factor detail](source_query_correction_tables/factors.csv),
[operator factors](source_query_correction_tables/operator_factors.csv),
[per-query actions](source_query_correction_tables/actions_by_query.csv),
[error accounting](source_query_correction_tables/error_accounting.csv).

## Decision and next action

Retain the fixed common-plus-query reader. Close the MLP24-only refinement
family as specified before this result. NEXT49 turns to attention22's full
query/key/value input-transport channels, where the carried final-query tangent
may matter. It does not reopen another alpha/layer/source search for r=1.
See [theory](../../docs/GRAIN_SOURCE_QUERY_CORRECTION_THEORY_V1.md),
[full-chain integration map](../../docs/GRAIN_UPSTREAM_MECHANISM_INTEGRATION_V1.md)
and [NEXT49 plan](../../docs/GRAIN_UPSTREAM_ATTENTION_CHANNELS_PLAN_V1.md).

Reproduce with existing private inputs, using fresh output paths instead of
overwriting the completed packet:

```bash
PY="../project persona/.venv/bin/python"
$PY scripts/run_grain_source_query_correction.py prepare
$PY scripts/run_grain_source_query_correction.py predict --name predictions
$PY scripts/run_grain_source_query_correction.py evaluate --name analysis --predictions predictions
```
