# Raw-native and frozen-representation construction comparison

2026-09-13. C01/C10/C17 available-cache tranche of the ACTIVE all-plan goal.
This specification is fixed before running this new comparison; the development
fold has appeared in earlier experiments, so it is not a new untouched test.

## Question and data

Does a query-conditioned native readout add criterion-relevant information to
the current frozen E5 or E5/GRAIN construction under the SAME source-author
budget? This completes a missing comparison, not a hyperparameter search branch.

Use ALL 213 currently token-verified raw-native PANDORA authors, membership
determined by archive availability, not scores. They have the same original
36 comments in three 12-comment views as the construction material. Official
folds 2/3/4: 126 training authors; fold 1: 44 source-validation authors; fold 0:
43 held evaluation authors. Refit selected statistical heads on all 170 source
authors. Big5-only roles exclusively; no bridge, MEPS, Essays fitting or lockbox.

The frozen E5 comment vectors cap each comment at 512 tokens; native Qwen reads
12 comments jointly using its own tokenizer/prompt. Equal available material is
NOT equal internal context, model size, pretraining or token access. The archived
prompt identity audit is retained. No new neural calls, queues or downloads.

## Fixed family

| Arm | Input | Source-label fitting |
|---|---|---|
| native_raw | Mean of three native five-trait score vectors | None |
| native_calibrated | Native five scores, source StandardScaler + Ridge | Yes |
| semantic | E5 mean, source PCA(64) + StandardScaler + Ridge | Yes |
| semantic_native | Semantic PCA(64) plus native five | Yes |
| additive | Semantic PCA(64), GRAIN mean/std(48), IPIP affinity(30) | Yes |
| additive_native | Additive plus native five | Yes |

All learned arms: alpha in {1,10,100,1000}, deterministic ascending tie break,
source-validation mean of ALL five Pearson correlations. Final source refit
re-estimates PCA/scaling; held rows never enter fitting or parameter selection.
No trait dropping, sign orientation, output clipping or per-trait arm selection.
The older 504-source models are NOT comparison arms under this 170-source budget.

## Outputs and formulas

Native scoring is a conditional digit expectation, not a clinical percentile:

\[
n_{uj}=\frac13\sum_{v=1}^3\sum_{d=0}^9
\frac d9\operatorname{softmax}(\ell_{uvj})_d.
\]

For a fitted linear head and source-frozen transforms, export a raw-block affine
map. For semantic input h, GRAIN g, affinity s and native n:

\[
\hat y=b+hW_h+gW_g+sW_s+nW_n.
\]

Omitted blocks have no coefficient. This is a numerical reader decomposition,
not causal mediation or separation into independent psychological constructs.
Cov(y, prediction) equals the SUM of block covariances; report signed block
covariances divided by sd(y)sd(prediction), preserving negative contributions.
GRAIN std pooling is nonlinear before this reader; do not claim a fully linear
raw-text model. IPIP affinity is a fixed projection of E5, not a new information
source. The native readout still requires its LLM unless separately replaced.

Report per-trait Pearson/Fisher95CI, Spearman, MAE, RMSE and prediction SD; mean
five-trait metrics; all 20 source-validation fits; five final selected models;
source norms and raw-block weights; individual/batch score replay; label-free
native view dispersion as descriptive material sensitivity, not test-retest.

Five fixed contrasts, 1,000 paired AUTHOR resamples, conditional on fitted heads:
semantic_native-semantic; additive_native-additive; native_calibrated-native_raw;
semantic-native_raw; additive-semantic. Report delta mean-r, percentile95CI and
draw failures. No multiplicity-adjusted success declaration or population-wide
equivalence from these developmental intervals. Compare whole vectors on the
same 43 authors only.

## Decision and stop

If adding native readouts helps, retain them as a distinct supervised-reference
channel and identify the responsible traits/block contributions. If not, retain
raw native as a reference without inventing a new blending/alpha sweep. In either
case return to condition/path construction objects and historical reconciliation.

Store private arrays, authors, labels, readouts and checkpoints under ignored
`results/grain_vector_mechanisms/all_plan_closure_v1/native_construction_r1/`.
Public report contains aggregate values and protocol. Independently replay cache
joins, source transforms, scores and metrics; run synthetic leakage/affine tests.
This does not close missing 411 native PANDORA authors, native Essays, full official
CV, independent interpretation or the complete two-context Study B.
