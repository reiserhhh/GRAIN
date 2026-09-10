# Source reader motion supplies new directions

2026-09-11. NEXT80 COMPLETE, finite logic pass69. Primary `2514a40`, numerical
accounting correction `41e4d96`. Overall research goal ACTIVE. **Retain directional
reader motion as a mechanism and a small development gain, not a whole-scorer
replacement.**

## Finding and decision

Source-context perturbations reveal differential-reader changes outside the
space spanned by the original source readers. In the all-four-source reference,
**68.88% of slope energy lies outside that old space**. Adding those columns
expands the differential output range from rank3 to rank7. The new8 orthogonal
vector-error lower bound drops from .396060 to .334639 (15.51%).

The resulting fixed local-linear field makes small useful numerical progress:
with native evidence held fixed, differential K+V error falls **5.30% relative to
NEXT79's context kernel**. The earlier constant reader is still slightly better
on this particular projected error. The assembled response/source function's
query-contrast correlation moves .596467 to .600150, while its response error
also decreases modestly. Opposing native modulation is not recovered.

The next useful distinction is **finite extrapolation versus omitted non-last
material dependence**, not another weight or bandwidth sweep. The prescribed
[NEXT81 experiment](../../docs/GRAIN_READER_MATERIAL_DEPENDENCE_PLAN_V1.md) fixes
the complete query-specific last vector and tests source material and GRAIN
directional changes in the reader itself.

## What was run

- Same four source authors, source4 leave-one-out and new8 all-four application.
  Both are reused discovery panels, not fresh blind human validation.
- Two fixed queries: neuroticism and confirmation-seeking. The latter is not
  conscientiousness. No human labels, MEPS or API calls used.
- Five reference packages, 16 source/reference pairs, not16 independent people.
- Move only both queries' H16 last rows by plus/minus .01 source-distance-scale
  toward that reference's mean context. BODY and query-specific offsets remain.
- Run the frozen suffix17..end and differentiate original L with respect to the
  block22 post-attention last state. This is not a block16 or block17 reader.
- Teach one directional secant column per source/reference; all four previously
  fixed selector/value functions and both modes remain unchanged.
- Application reads source packages and H16 inputs, not the current author's
  native readers, Q/K/V, labels or evaluation targets. Source4LOO excludes the
  recipient even when constructing the intervention direction.

## Quantitative comparison

Representative previously developed `response_coordinates__source_value` path:

| Cohort / observable | Constant | Context kernel | Source motion |
|---|---:|---:|---:|
| new8 scalar N-C r | .577035 | .596467 | .600150 |
| new8 scalar SSE / zero SSE | .193610 | .196983 | .194852 |
| new8 scalar author-centered MSE | .030494 | .029350 | .028941 |
| new8 K+V N-C r | .677814 | .679080 | .677279 |
| new8 K+V SSE / zero SSE | 7.288066 | 7.311582 | 7.210446 |
| new8 differential-reader modulation projection | +.329503 | +.330431 | +.325624 |
| source4 scalar N-C r | -.778636 | -.778650 | -.846457 |
| source4 scalar SSE / zero SSE | .212643 | .210845 | .203827 |
| source4 K+V N-C r | .684494 | .687037 | .692359 |
| source4 K+V SSE / zero SSE | .554387 | .549948 | .541118 |

Native new8 K+V reader projection is **-.128743**. Projections are signed
covariance allocations, not causal shares. Correlation and error can move in
opposite directions; source4 scalar ordering remains negative. All four paths
are published, including worse scalar errors for the identity-value paths.

Holding upstream evidence native isolates the differential reader:

| Cohort / observable | Constant | Context kernel | Source motion |
|---|---:|---:|---:|
| new8 differential vector error / native energy | .229089 | .226589 | .223379 |
| new8 differential scalar r | .474821 | .502678 | .507991 |
| new8 differential scalar SSE / zero SSE | .111139 | .107554 | .106185 |
| new8 differential K+V r | .172628 | .138315 | .163043 |
| new8 differential K+V SSE / zero SSE | 2.161903 | 2.299762 | 2.177830 |
| source4 differential vector error / native energy | .267650 | .263941 | .259830 |
| source4 differential K+V SSE / zero SSE | .075118 | .074363 | .070273 |

### Output range and extrapolation

| Cohort / reader | Affine rank | Vector SSE | Orthogonal lower bound | In-range error |
|---|---:|---:|---:|---:|
| source4 constant | 2 | .245236 | .142268 | .102968 |
| source4 kernel | 2 | .241837 | .142268 | .099569 |
| source4 motion | 5 | .238070 | .131277 | .106793 |
| new8 constant | 3 | .509475 | .396060 | .113415 |
| new8 kernel | 3 | .503916 | .396060 | .107856 |
| new8 motion | 7 | .496778 | .334639 | .162138 |

New capacity is not automatically learned correctly: expanding the range lowers
the orthogonal bound but increases the in-range component. Actual new8 vector
SSE improves only1.42% over the kernel. No vector bound is a projected-action
bound or a percentage of human personality information.

Source all4 slope RMS norm is .017411; energy outside the old range is .688785.
For source3LOO, pooled over12 source/reference units, these are .017223/.702359.
The latter is not the unweighted mean of per-reference fractions (.711329).
Projected displacement along the taught directions averages about59.42 teaching
steps for new8 (.5942 bandwidth units), and57.21 for source4. Small local slope
error therefore does not certify accurate extrapolation to application states.

`source_motion_summary.csv` also projects slopes through native source evidence
and all four fixed program vectors. Program vectors are the pre-existing
author-LOO NEXT77 predictions, not new ref-S-specific fits; keep this distinction
when interpreting the source/reference aggregation.

## Numerical verification and cost

Audit PASS. 76 source native reverse calls, **66.235962 seconds**, observer loading
3.932008 seconds. Eight unperturbed readers reproduce the earlier capture exactly.
The one prescribed half-step check differs by .0150545, below its .20 numerical
gate. No neural application/capture was performed on new8 in this card.

Five source constructions, zero coefficient solves. 12 application packets and
12 private evaluation packets, nine aggregate CSVs replay exactly. 64 scalar
finite differences agree within4.734422e-10;24 BODY/last zero checks pass. Old
constant/kernel predictions reproduce exactly; old metrics within1.42e-14.
**38 focused tests pass**; Ruff and formatting checks pass. This is not a
full-repository test claim.

Successful canonical numeric stages including replay: **11.172378 CPU seconds**.
Native time, loading, tests, hashing preflight and publication are separate.
The first evaluation failed an old-metric equality gate by1.07e-8, exclusively
because float32 dot-product reduction differed from the old square/sum reduction.
No model, prediction, source direction, perturbation or target changed. The
single-line accounting fix was committed separately; original failed outputs
are preserved and both hashes recorded in the manifest/publication. Failed-stage
CPU was not captured: unknown and excluded, not zero. See the
[runtime note](../../docs/GRAIN_SOURCE_READER_MOTION_RUNTIME_NOTE_V1.md).

## Reproduction and handoff

```bash
PY='../project persona/.venv/bin/python'
"$PY" scripts/run_grain_source_reader_motion.py --stage prepare
"$PY" scripts/run_grain_source_reader_motion.py --stage capture
"$PY" scripts/run_grain_source_reader_motion.py --stage models
"$PY" scripts/run_grain_source_reader_motion.py --stage predict
"$PY" scripts/run_grain_source_reader_motion.py --stage evaluate
"$PY" scripts/run_grain_source_reader_motion.py --stage predict --folder replay
"$PY" scripts/run_grain_source_reader_motion.py --stage evaluate --folder replay
"$PY" scripts/run_grain_source_reader_motion.py --stage audit
"$PY" scripts/report_grain_source_reader_motion.py
```

Do not rerun native capture merely to fill a table. The canonical cache is
immutable; a fresh reproduction must use a separate `--output` consistently and
the frozen source/model dependencies. The publisher exports canonical aggregates.

Private root: `results/grain_vector_mechanisms/logic_source_reader_motion_v1/`.
Public: [nine tables and provenance](source_reader_motion_v1/).
Theory: [conditional reader motion](../../docs/GRAIN_SOURCE_READER_MOTION_THEORY_V1.md).
H16 remains neural-derived; the numbers above concern a local endpoint component,
not full L or questionnaire prediction. Same-material GRAIN directions remain
the interface. The old release/scorer/lockbox and unrelated work are unchanged.
