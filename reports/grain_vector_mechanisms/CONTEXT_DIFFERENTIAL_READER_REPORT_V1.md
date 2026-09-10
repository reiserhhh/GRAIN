# Context-conditioned differential reading: a source-range bottleneck

2026-09-11. NEXT79 COMPLETE. Primary implementation `627decc`. Overall logic
goal ACTIVE. **Retain the range finding; do not promote this kernel reader.**

## Finding and decision

The one source-normalized context smoother slightly improves direct reader
vector fit, but does not recover native new8 opposing response modulation.
For new8, **78.60% of its differential-reader squared error is orthogonal to
the affine span of the four available source reader differences**. Changing
normalized weights alone cannot remove that component. This is a limit of this
source-output family, not a limit of context-dependent reading in general.

The practical next step is to teach how source readers change with context,
thereby adding output directions, rather than run a bandwidth/weight grid.
[NEXT80](../../docs/GRAIN_SOURCE_READER_MOTION_PLAN_V1.md) specifies a bounded
source-only intervention and one compatible numerical reader field.

## Fixed experiment

- Same two queries: neuroticism and confirmation-seeking (not conscientiousness).
- Source4 leave-one-author-out uses three sources each; new8 uses all four.
  These are reused discovery panels, not twelve new blinded participants.
- Five packages; no coefficient solves, new neural calls, APIs or human labels.
- Paired H16 last-position mean sets Gaussian weights; bandwidth squared is the
  source median pairwise squared distance. No whitening or parameter sweep.
- Shared source mean reader stays fixed; only the query difference is weighted.
- All four NEXT77 selector/value functions, both endpoint modes, both queries,
  query contrast, key/value/total and all four GRAIN directions are retained.
- Native readers and targets are opened only after candidate prediction freeze.

## Quantitative results

Representative previously developed `response_coordinates__source_value` path:

| Cohort / observable | Constant reader | Context reader |
|---|---:|---:|
| new8 scalar N-C correlation | .577035 | .596467 |
| new8 scalar SSE / zero-predictor SSE | .193610 | .196983 |
| new8 scalar author-centered MSE | .030494 | .029350 |
| new8 K+V N-C correlation | .677814 | .679080 |
| new8 K+V SSE / zero-predictor SSE | 7.288066 | 7.311582 |
| new8 differential-reader modulation projection | +.329503 | +.330431 |
| source4 scalar N-C correlation | -.778636 | -.778650 |
| source4 K+V SSE / zero-predictor SSE | .554387 | .549948 |

Native new8 K+V differential-reader projection remains **-.128743**, not the
positive sign of either candidate. These are signed covariance projections,
not causal variance percentages. The scalar ranking gain and centered-error
gain coexist with worse raw scalar error; neither licenses a full promotion.
The identity-selector paths lose some scalar correlation (.686->.664 and
.690->.671); all four paths remain in the tables, not just the above example.

Holding upstream vectors at their native values isolates differential reading:
new8 differential scalar r .474821->.502678, SSE/zero .111139->.107554,
but differential K+V r .172628->.138315, SSE/zero 2.161903->2.299762.
Thus the mixed result is not solely an upstream vector-fitting error.

| Source-range account | source4 LOO | new8 all4 |
|---|---:|---:|
| Source differential affine rank | 2 | 3 |
| Constant differential vector SSE | .245236 | .509475 |
| Conditional differential vector SSE | .241837 | .503916 |
| Orthogonal lower bound, either weighting | .142268 | .396060 |
| Bound / conditional vector SSE | 58.83% | 78.60% |
| Mean effective source count | 2.993 / 3 | 3.955 / 4 |

Weights are nearly uniform at the fixed source distance scale, but even arbitrary
normalized signed weights cannot escape the same affine span. The orthogonal
bound **does not bound projected scalar/action error**: projected components can
cancel, and the observable weights vector directions unequally. It motivates
source-range expansion; it does not prove that expansion alone will fix actions.

## Reproduction and scope

Audit PASS: 12 application and 12 private evaluation packets; all eight CSVs
replay exactly. Baseline arrays agree with NEXT77 within 4.22e-15; 288 NEXT78
baseline metrics within 4.44e-15. All 24 BODY/context checks have D[last]=0;
128 full-scalar finite differences agree within 4.20e-10. Seven new tests plus
22 related tests pass (29 total); Ruff/format/diff checks pass.

Canonical numeric-stage CPU, including cached replay: **10.238926 seconds** of90.
No failed runtime stage. Source-only package construction and application use
explicit data-open allowlists and denial probes; these are trusted-code audit
hooks, not a hostile-code security boundary. Provenance hashing precedes them.
An additional post-audit least-squares projection check, independent of the SVD
projector helper, reproduces bounds .142267886165/.396060140134 and fractions
.588280597/.785964333. This diagnostic does not construct prediction coefficients;
it is outside the canonical stage timing, as are tests and publication checks.

```bash
PY='../project persona/.venv/bin/python'
"$PY" scripts/run_grain_context_differential_reader.py --stage prepare
"$PY" scripts/run_grain_context_differential_reader.py --stage models
"$PY" scripts/run_grain_context_differential_reader.py --stage predict
"$PY" scripts/run_grain_context_differential_reader.py --stage evaluate
"$PY" scripts/run_grain_context_differential_reader.py --stage predict --folder replay
"$PY" scripts/run_grain_context_differential_reader.py --stage evaluate --folder replay
"$PY" scripts/run_grain_context_differential_reader.py --stage audit
"$PY" scripts/report_grain_context_differential_reader.py
```

Existing canonical outputs are immutable. To reproduce the entire pipeline into
a fresh private directory, supply the same `--output` to each numerical stage;
do not remove the originals. The publisher exports the audited canonical result.

Private root: `results/grain_vector_mechanisms/logic_context_differential_reader_v1/`.
Public aggregates: [context_differential_reader_v1](context_differential_reader_v1/).
H16 remains model-derived. Scores here are a fixed local endpoint component, not
full model L, questionnaire correlation or new human personality accuracy.
The GRAIN interface remains the same-material BODY direction, not score addition.
