# GRAIN factor / query-path connection: all-route pilot V1

Date: 2026-09-10. Scope frozen before this packet's outcomes.

## Question and identity of the objects

GRAIN supplies independently defined, reference-scored author coordinates.
Query-path analysis may add psychological meaning to those coordinates; it
does not replace them with query scores. Combining predictors is a separate
engineering hypothesis, not a prerequisite for interpretability.

Use the existing E0 frozen factor bundles: whole document, native comments,
and fixed-128 cross-comment slices, four factors each. Namespace their codes
by operator; do not assign personality names. Their actual implementation is
word TF-IDF (5000 features), SVD (24 dimensions), author means/dispersion,
factor posterior scores, and frozen reference normalization. This is a
specific reproducible GRAIN factor family, not all GRAIN dynamic mechanisms
and not a new embedding-based factor discovery run.

Source: S9/M29's same 277 authors, three 12-comment views each; 213 Big5-only
and 64 MBTI-only authors. All five Big5 and four MBTI queries are included.
No bridge or MEPS authors enter fitting. Historical E0 membership must be
checked against source IDs. Existing source labels were previously exposed:
these are exploratory source-CV comparisons, not a reopened lockbox.

## Four executable work packages

| ID | Question / calculation | Acceptance artifact |
|---|---|---|
| GFP-01 | Which of 12 frozen factors align with nine query functions? Correlate author-averaged factors with total, query mean, configuration, final-state terms, and native scores; inspect three-view consistency. | Full 540-cell table, primary 108-cell BH correction, factor consistency and material support tables |
| GFP-02 | Do factor-aligned activation directions change native model judgments? Fit native-factor-to-P/Q directions on source authors; replay actual suffix under signed interventions on 12 disjoint authors, T/F and neuroticism. | 24 units x 16 signed suffix edits, paired actual effects, author-permuted norm-matched controls, replay checks |
| GFP-03 | Which factors connect with actual criteria, rather than only LLM judgments? Compute all 12 x 9 source questionnaire associations. | 108-cell signed matrix, Fisher intervals, BH adjustment, anonymous candidate cards |
| GFP-04 | Are the two routes additive in prediction? Compare GRAIN12, path3, joint15, joint plus36 interactions; retain frozen query function and native judgments as descriptive references. | Author-level nested-CV predictions, paired bootstrap changes, all-nine comparison, one label-permutation fit per arm/query |

### Equations

Let g_u be the frozen GRAIN coordinates averaged over the three views,
and q-specific path components be p_uq=(mean, configuration, final).

Association: C_jq = corr(g_uj, f_q(P_u,Q_u)). External connection uses
corr(g_uj, Y_uq), not f as a substitute for Y.

Vector alignment uses only the four native factors:

    X = source-standardized native GRAIN scores
    H_q = vec(source mean P_q, source mean Q_q)
    D_q = (X'X + 100 I)^(-1) X' (H_q - mean(H_q))
    H'_uq = H_uq +/- 0.25 D_q[j]
    effect_uqj = (F_q(H'_plus) - F_q(H'_minus)) / 2

The rest of the original full hidden state is unchanged. Null directions
use one seeded source-author permutation of X, with each full P/Q direction
norm matched to its corresponding real direction. This retains activation
geometry better than an isotropic random null. One null is a diagnostic,
not a permutation significance test. We report actual suffix outputs and
fixed affine-function predicted effects separately. This is an intervention
on a GRAIN-aligned *model direction*, not on a person's personality.

Prediction: Y_hat = a + b_g'g + b_p'p (+ g' W p). Outer five author folds;
inner three folds choose Ridge alpha from 1,10,100,1000 by MSE. Scalers and
interaction formation/standardization stay inside fits. The same splits
are used for all arms. Binary axes use continuous scores and AUC, with no
held-out threshold tuning. Report Pearson, Spearman, MAE/RMSE, paired
participant-bootstrap delta r/AUC (1000 draws). Negative control permutes
Y at the author level and repeats the full nested search once per arm.

## Scientific decisions, not a long accuracy search

- Association route has potential if a factor connects coherently to query
  components and to real criteria. Report magnitude and coverage; no forced
  renaming and no requirement that all factors map to a known scale.
- Intervention route has potential if effects are directional across held
  authors and exceed comparable null effects; report failures equally.
- Joint prediction has potential if participant-paired gains are positive.
  An uncertain or negative gain does not invalidate the interpretation route.
- Report each operator's text coverage and factor support. Whole/native/fixed
  are distinct observation designs; do not silently pretend they see identical
  token budgets. Material is the same upstream 12-comment view.
- No new long training, no layer/trait selection from held labels, no cloud
  uploads, no lockbox opening, no automatic continuation into endless tuning.

## Execution and artifacts

Scripts: `scripts/run_grain_factor_path_routes.py` (CPU packet) and
`scripts/run_grain_factor_path_interventions.py` (bounded native replay).
Math: `grain_research/vector_mechanisms/factor_path_routes.py`.
Tests: `tests/test_grain_factor_path_routes.py`.
Private output: `results/grain_vector_mechanisms/factor_path_routes_v1/`.
Public report: `reports/grain_vector_mechanisms/FACTOR_PATH_ALL_ROUTES_REPORT.md`.
Budget: CPU two threads, one complete pass; neural <=408 calls, <=1800 seconds
per bounded invocation, resumable per-unit packets, no automatic retraining.

## Methodological precedents

- [Kim et al. (2018), TCAV](https://proceedings.mlr.press/v80/kim18d.html):
  use concept-aligned directions to ask about model sensitivity. Here the
  coordinates pre-exist independently; finite native replay replaces a
  directional-derivative-only claim.
- [Koh et al. (2020), Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html):
  concept interfaces motivate the joint/interpretable head comparison. We
  do not claim that anonymous GRAIN factors are already validated concepts.
- [Geiger et al. (2025), Causal Abstraction](https://www.jmlr.org/papers/v26/23-0058.html):
  keep low-level intervention effects distinct from a high-level causal
  interpretation. Our pilot measures a candidate alignment, not full causal
  equivalence or mediation of human personality.

MEPS small traits and GRAIN dynamics remain separately tracked follow-ups;
this packet closes all four *routes* on the declared available factor family.

## Execution amendment: identity namespace and pooled-CV accounting

The first pass compared raw historical usernames with new hashed author IDs.
Recomputing the exact cohort pseudonym mapping finds one historical discovery
author in the 277-person source. Therefore source_r2 excludes that author from
GFP01/03/04 and is the final questionnaire/association comparison (276 people).
The inclusive source_r1 is retained, not silently overwritten; its direct-string
`reference_author_overlap=0` field is superseded by this corrected identity audit.

GFP02's native_r1 directions use the original 277-person source, without labels.
All12 native evaluation authors are disjoint from both that source and the180
historical factor authors under the corrected mapping. A source author having
helped develop the unsupervised factor basis does not leak any of these12 held
native responses. Do not rerun native calls merely to remove that one source
case; report the different source counts explicitly.

Add diagnostic accounting without changing the predeclared predictions: pooled
OOF correlations, within-fold-centered correlation, mean valid within-fold AUC,
minority counts, and MSE improvement over each fold's training-mean predictor.
Small, imbalanced samples can yield negative pooled correlation even for the
training-mean null. Preserve that fact, the original metrics and one full
label-permutation fit; do not flip negative scores, tune fold assignments or
claim a single permutation is a calibrated significance test.
