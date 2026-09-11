# NEXT87: teaching the reader for the evidence it actually receives

2026-09-11. **PROGRESS: retain a local scalar-compatible interface; expand source
teaching for action reading, not another four-source parameter sweep.**
Source4ALL, reused source-disjoint new8 discovery authors, two fixed queries
(neuroticism / confirmation seeking). This is a fixed-model local attention
component, not final LLM scores or human questionnaire accuracy. H16 remains
neural-derived. No new neural/API calls or human labels were used.

## Matched experiment and result

Two original-four source value fits have identical Gaussian kernel, old numeric
lambda .003201, old observation weights45.417633, input units and five evidence
row units. Each solves20 coefficients. One teaches native-reader coordinates;
the other teaches coordinates that reproduce source native local observations
using the **actual all4 compiled evidence**. SourceLOO evidence, clamped readers,
full output logits and original local targets were not substituted for each other.

Primary response-coordinate selector/source-value arm, query-specific contrast:

| Reader | Scalar r | Scalar SSE/zero | Centered MSE | K+V r | K+V SSE/zero |
|---|---:|---:|---:|---:|---:|
| Earlier constant | .577035 | .193610 | .030494 | .677814 | 7.288066 |
| Earlier GRAIN field | .618477 | .179712 | .028278 | .678197 | 7.188125 |
| NEXT86 fixed V, eight value positions | .651589 | .165198 | .027335 | .679144 | 7.235659 |
| NEXT86 moving V, eight value positions | .651727 | .166481 | .027367 | .678550 | 7.274616 |
| Matched original4 native coordinates | .649764 | .165529 | .027589 | .679580 | 7.309327 |
| Matched original4 compatible coordinates | .640219 | **.088286** | **.026929** | .676183 | 7.286511 |

The matched scalar error falls **46.6644%**; centered error falls only **2.3931%**
and ranking r decreases .0095. Prediction RMS rises .419781->.523557 versus actual
.654353. Thus this is predominantly better magnitude/offset compatibility, not
a comparable gain in author ranking. Error is also46.56% below old fixedV, but
that is an unmatched source-position comparison, not the primary control.

All four selector/value compositions improve contrast raw error. Query-specific
outcomes differ: in the primary arm N error .068951->.050143, confirmation
.099081->.129525. Shared-reader mode is unchanged by construction; the common
reader is fixed. All20 readers, all queries/modes/arms and contrary cells remain.

Anchored K+V error changes only .31% and remains worse than earlier GRAIN.
F01/F04 errors increase .9931->1.0291 /93.9081->94.5085; F02/F03 decrease
2.5939->2.5046 /5.2585->5.1099. No action-reading promotion.

## Mechanism identified

Normalized scalar error has the exact signed decomposition:

| Teacher | Evidence + common-reader error | Differential-reader error | Their signed cross term | Total |
|---|---:|---:|---:|---:|
| Native coordinates | .399920 | .187260 | -.421651 | .165529 |
| Compatible coordinates | .399920 | .271761 | -.583395 | .088286 |

The effective reader becomes **less native-like**, yet reads approximate evidence
better. Full differential-vector error/native energy rises .226076->.228804.
This experimentally develops NEXT86's target-mismatch finding: accurate native
components are not necessarily the best composition with imperfect components.
Compensation is an observed computational use, not recovered psychological variance.

Both source fits interpolate their20 targets to SSE/zero below2.10e-11; the
regularized kernel condition is8.638858. Current compatible-target RMS relative
to source is1.24x for scalar and2.96/5.45/15.40/15.05x for four actions. Predicted
action-coordinate amplitudes remain small; F02-F04 coordinate rankings are
negative. The source fit is solvable; broader teaching, not a tighter solver or
another lambda, is the next useful test. Scale shift alone is not a unique cause.

## Execution and reproducibility

- Pre-fit implementation `b50eef1`; numerical-promotion repair `389b24b` before
  any solve or new8 evaluation. Exact kernel/target/units/tolerances preserved.
- First teacher attempt failed parity because cached float32 readers were
  subtracted before promotion (max3.72529e-9). Float64-before-subtraction gives
  **zero** discrepancy from frozen source targets. Aborted preparation retained
  under `logic_compiled_compatible_reader_v1_aborted_precision/`; regression test
  added. No model fitted or recipient target read in this attempt.
- Canonical run:8.172084 numerical CPU seconds including full exact replay;
  excludes hashes/tests/imports/publication. Failed-stage CPU was not frozen and
  is unknown, not silently counted as zero. Two source solves only.
- Canonical local compiled query calls8source+16application+16replay=40;
  aborted source adds8, total48. Cached native source query reconstructions8
  canonical+8aborted; these are not neural calls. No live job remains.
-34 focused tests/Ruff pass, two pre-existing SWIG deprecation warnings.
 128anchored FD max4.923e-10,128diagonal FD max7.282e-10. Ten CSVs and all cached
  arrays replay exactly. All18 prior core-table prefixes preserved.
- Trusted-code stage guards deny current native targets in construction/application;
  native evaluation arrays open only after prediction freeze. Private arrays,
  texts and author identifiers are not published. No old lockbox access.

Module: `grain_research/vector_mechanisms/compiled_compatible_reader.py`.
Runner: `scripts/run_grain_compiled_compatible_reader.py --stage STAGE`, stages
`prepare`, `teachers`, `models`, `predict`, `evaluate`, `replay`, `replay-evaluate`,
`audit`. Existing completed directory is intentionally not overwritten.
Publisher: `scripts/report_grain_compiled_compatible_reader.py`.
Private root: `results/grain_vector_mechanisms/logic_compiled_compatible_reader_v1/`.
Public [ten aggregate tables and provenance](compiled_compatible_reader_v1/publication.json).

Read [theoretical consequences](../../docs/GRAIN_COMPILED_COMPATIBLE_READER_THEORY_V1.md)
and [NEXT88 source-diversity plan](../../docs/GRAIN_SOURCE_DIVERSITY_TEACHING_PLAN_V1.md).

The distinction between an intermediate prediction loss and its downstream use
has a precedent in [Donti, Amos & Kolter,2017](https://papers.neurips.cc/paper_files/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf).
Their task is stochastic optimization, not personality inference; our five-row
compatible target is a local linear construction, not their end-to-end optimizer.
