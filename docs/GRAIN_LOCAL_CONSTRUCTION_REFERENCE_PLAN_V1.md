# Local Construction Reference: PANDORA and Essays

2026-09-13. C01/C17 construction tranche, before new fits/results.

## Scope and preserved requirements

Study B (V3 B1-B7; V4 section4) requires local construction cases, not only
prediction tables. This tranche builds its standard lexical structural reference
in both contexts: frozen scoring, local norms, material uncertainty and numerical
explanation. Semantic, native-model, GRAIN conditional/common-evidence arms and
independent psychological/behavioral interpretation remain separate requirements.
Do not mark all Study B complete from these reference tools.

## Object cards fixed now

| Rule | PANDORA local object | Essays local object |
|---|---|---|
| B1 target | Relative word-rate common-variation profile in36 natural comments | Relative profile in one prompted essay |
| Psychological hypothesis | Anonymous coordinates may relate to self-reported Big5; not assumed identical | Same question, with binary outcomes and one occasion |
| B2 conditions | English Big5-only Reddit authors; observed community/time; opportunity set unknown | English original essays; prompted writing; no repeated time axis |
| B3 evidence | All alphabetic tokens>=2chars as denominator; source-selected vocabulary; no forced slices | Same rate rule; whole document as scoring unit |
| B4 model | MINRES/promax5 and matched PCA5/varimax | Same algorithm/budget, locally fitted vocabulary and parameters |
| B5 reference | Source504/held120, official fold0; reuse exact prior source transforms | Source2213/held254, essay_fold9; existing split, not new lockbox |
| B6 evaluation | All5factors x all5Big5 correlations; conditional material bootstrap | Same; binary labels also get fixed-direction AUC |
| B7 relation | Recipe transfers; coefficients and norms not automatically equal | Matching factor numbers do not imply matching constructs |

Inherit BLT/MEM-style vocabulary structure and psych scoring, not the whole GRAIN
or exact BLT reproduction. Five factors is a fixed reference budget, not a
dimensionality discovery or five new personality traits. Numerical contributions
do not replace psychological naming. Residuals and other GRAIN routes remain.

## Fixed computation

Export fitted transforms as JSON/NumPy-only scorers. Freeze source mean/SD and
empirical score reference before scoring held authors singly/in batches. Include
exact word contributions, intercept and all omitted display tails. No outcome
labels for structure, sign, vocabulary or model selection.

PANDORA uses the previous624-author/36-comment material and exact source index
verification. Essays uses all2467 original unique IDs. Outcomes are read after
the four artifacts are saved; labels0..1 versus0/1 remain distinct. Prior exposure
is developmental reanalysis, not first-time blind confirmation.

Material resampling:200draws, seed923; whole comments in PANDORA, consecutive
pairs of approximate sentence spans in Essays. Intervals are conditional on the
frozen scorer and exchangeable blocks, NOT trait CI or test-retest. Fewer than2
blocks has no interval. No alphabetic tokens means unscored, not average author.

Report Fisher-z approximate correlation CI, two-sided p, Spearman and BH over
all100 domain/method/factor/trait connections. Report log material length links.
No sign reversal or renaming selected from held criterion results.

## Acceptance and stopping

Run both cases, four exports, exact native/JSON replay, single/batch invariance,
source-only references, complete100-row criterion map and raw-address arithmetic.
Two new local fits only; R timeout180seconds each. No rank/rotation grid, downloads,
neural calls, MEPS/bridge fit or lockbox. Retain inadmissibility/warnings and missing
Study B arms; code producing scores does not alone establish psychological validity.

## Deliverables

`grain_research/local_measure.py`, construction/scoring/audit scripts and tests;
ignored `results/grain_vector_mechanisms/all_plan_closure_v1/local_construction_r1/`
with frozen JSON scorers, author manifest, private scores/intervals/explanations,
criterion map and audits. Track `LOCAL_CONSTRUCTION_REFERENCE_REPORT.md` and ledger.
