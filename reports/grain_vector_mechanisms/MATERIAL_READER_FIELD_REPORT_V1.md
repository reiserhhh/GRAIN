# Material-conditioned reader fields: NEXT82

2026-09-11. COMPLETE, finite logic pass71. Implementation `7fdf436`, canonical
pre-fit revision `f0a0494`. Overall research goal remains ACTIVE.

## Decision

The same-material GRAIN connection is now an executable reader field, not just
a measured intervention. With the fixed response-coordinate/source-value
composition, adding GRAIN cross-moments gives local score-contrast error
0.179712 versus 0.193610 for a constant reader, a **7.18% reduction**. Against
the previous source-motion reader, the reduction is **7.77%**. The new context
field instead recovers the strongest ordering (r=0.683999), but worse raw error.

Keep this bounded functional gain. Do not declare one reader universally best:
vector distance, scalar reading and GRAIN directional response rank them
differently. Develop source-observable teaching next, not a bandwidth sweep.

All figures below concern a fixed model's local component on eight reused
discovery authors, not questionnaire prediction or the complete model output.
N denotes neuroticism; the other query is confirmation-seeking, not Big Five C.

## What ran

- Source4 ALL supplied eight reader values, sixteen BODY directional derivatives
  and four context derivatives. Original/clamped values alternate in the first
  eight rows. The all4 clamp precludes a legitimate source leave-one-out claim.
- Three fixed Gaussian value/derivative kernel solves: paired last context;
  context plus BODY mean; context plus BODY mean and four GRAIN cross-moments.
  Original-source distances set units. No label, bandwidth or rank selection.
- New8 application used current H16 and source packages, not current native
  readers or targets. Those were opened only for frozen evaluation. H16 remains
  neural-derived; no new neural inference, human labels, API or MEPS use occurred.
- All six readers and all four old selector/value compositions are published.
  Input addition also changes the prescribed block-balanced kernel geometry;
  this is a comparison of input contracts, not a pure causal GRAIN ablation.

## Source fitting

Residual SSE divided by the corresponding source target energy:

| Input contract | Value (8) | BODY derivative (16) | Context derivative (4) | Kernel rank |
|---|---:|---:|---:|---:|
| Context | 0.403314 | 1.000000 | 0.000003829 | 9 |
| Context + BODY mean | 0.000063640 | 0.001996307 | 0.000000464 | 28 |
| Context + GRAIN moments | 0.000198919 | 0.000038664 | 0.000002457 | 28 |

Context-only has sixteen exactly zero BODY operator rows. The new material
inputs remove that structural restriction. These are training descriptions,
not independent recovery of 99.996% of personality information.

## Recipient reading and composition

The first two columns compare differential reader vectors and their projection
onto **fixed native common evidence**. The last three compare the assembled
`response_coordinates__source_value`, query-specific N-minus-confirmation scalar
with its native local-component target.

| Reader | Vector SSE / native energy | Native-evidence scalar r | Assembled scalar r | Assembled SSE / zero | Author-centered MSE |
|---|---:|---:|---:|---:|---:|
| Constant | 0.229089 | 0.474821 | 0.577035 | 0.193610 | 0.030494 |
| Old context kernel | 0.226589 | 0.502678 | 0.596467 | 0.196983 | 0.029350 |
| Old source motion | 0.223379 | 0.507991 | 0.600150 | 0.194852 | 0.028941 |
| New context field | 0.227928 | 0.581816 | 0.683999 | 0.222332 | 0.024125 |
| New BODY mean field | 0.223320 | 0.181225 | 0.588101 | 0.185570 | 0.027820 |
| New GRAIN field | 0.226100 | 0.378854 | 0.618477 | 0.179712 | 0.028278 |

The mean field has the smallest vector error but a poor native-evidence scalar
ordering. The GRAIN field's centered vector SSE is smallest (0.365000, versus
constant 0.372668); that does not make it best on every functional projection.
All three new nearest-source kernel means are moderate, 0.6385-0.6582. They
have not simply vanished to the constant intercept, but this distance statistic
alone does not establish adequate teaching support in relevant directions.

For the same assembled function, anchored K+V contrast results are:

| Reader | K+V r | K+V SSE / zero | Author-centered MSE |
|---|---:|---:|---:|
| Constant | 0.677814 | 7.288066 | 0.095039 |
| Old context kernel | 0.679080 | 7.311582 | 0.095311 |
| Old source motion | 0.677279 | 7.210446 | 0.093990 |
| New context field | 0.685747 | 7.409714 | 0.096252 |
| New BODY mean field | 0.683622 | 7.134352 | 0.093087 |
| New GRAIN field | 0.678197 | 7.188125 | 0.093616 |

Directional amplitude remains the clear open component: the modest improvement
does not remove the excess error. Pooled directions are eight authors by four
directions, not 32 independent authors. All individual factors and negative
cells remain in `factor_actions.csv`.

## One scalar, two correctly distinguished derivatives

`Fhat(H;A)=rhat(A)^T phat(H)` has anchored partial `rhat(A)^T Dphat(H)`.
Its diagonal `Fhat(H;H)` adds `Drhat(H)^T phat(H)`. Both were checked against
their own finite differences. The extra term is not scored against the old
native first-order K/V component; doing so would change the estimand silently.

For the GRAIN field and the displayed composition, the four contrast correction
RMS values are 0.022537, 0.015840, 0.020441 and 0.017818. They are nonzero
properties of the learned function, not newly captured native reader derivatives
on new8. `diagonal_actions.csv` retains signed cross-energy as well as magnitudes.

## Theory update and next decision

Reader error `e` enters a scalar as `e^T p`, so its loss is `e^T p p^T e`, not
`||e||^2`. The source vector-fit objective therefore need not prioritize the
directions that matter for scalar or GRAIN-response reading. This identity
explains why the rankings can differ; it does not by itself identify the sole
cause of observed transfer error.

NEXT83 will retain the source-only GRAIN input/kernel and teach a reader through
source functional projections, with a matched vector-only control in the same
coefficient family. It must preserve a single field and its analytic derivatives,
and report both vector and functional outcomes. No alpha/rank grid or new8 fit.
See [theory](../../docs/GRAIN_MATERIAL_READER_FIELD_THEORY_V1.md) and
[NEXT83 plan](../../docs/GRAIN_OBSERVABLE_READER_TEACHING_PLAN_V1.md).

## Verification and provenance

- 52 focused tests pass; Ruff and formatting pass. This is not a full-suite run.
- Three source solves, eight application packets, nine public tables. Exact
  cached replay of all prediction/evaluation arrays and CSVs passes.
- 192 anchored plus 192 diagonal finite differences: maximum absolute errors
  5.3894e-10 and 4.7281e-10. Old comparator reproduction is exact.
- Canonical staged numerical CPU, including prepare and cached replay:
  **9.652626 seconds**. Later preflight hashing, tests, publication, audit and
  exploratory terminal summaries are outside that timer, not claimed free.
- First focused test run had a test-only `DataFrame.query` column-access collision.
  The prepare-only directory from before that fix is preserved as
  `logic_material_reader_field_v1_preflight_before_test_fix/`; no fit/prediction
  was made there. After test repair and 20 passing focused tests, canonical
  preparation began at `f0a0494`. No empirical outcome-driven repair occurred.
- No old scorer/lockbox change, private text/reader publication or push.

Public cohort tables and hashes: [publication](material_reader_field_v1/publication.json).
Private reproducibility stages are in
`results/grain_vector_mechanisms/logic_material_reader_field_v1/`.
On a fresh output directory, use `scripts/run_grain_material_reader_field.py`
stages `prepare`, `models`, `predict`, `evaluate`, then `predict --folder replay`,
`evaluate --folder replay`, `audit`. Existing frozen stage directories are not
overwritten. `scripts/report_grain_material_reader_field.py` publishes only the
audited canonical cohort summaries.

Method precedent: [Solak et al. (2002)](https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html)
combine function and derivative observations for nonlinear-system modelling.
Here the tested Gaussian derivative kernel is used as a fixed regularized
vector regression, without a Gaussian-process posterior-uncertainty claim.
