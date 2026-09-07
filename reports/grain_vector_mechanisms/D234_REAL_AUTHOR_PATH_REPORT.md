# D2-D4 real-author pathway findings

2026-09-08. Discovery-first mainline:64 new source authors,128 text views,
1156 neural forwards,1422.01seconds. All planned observations completed.

## What was discovered

**A real-author evidence -> intermediate vector -> query judgment route is now
measured, and its response can be approximated on new authors.** Query-selected
four-comment subsets retain author ordering; one query-end vector transmits part
of an evidence effect; an8-coordinate operator predicts unseen authors' vector-
intervention responses. GRAIN neighborhoods connect to particular response
channels, rather than one uniform response profile.

The external personality anchor is D1: native Big5 mean r=.1993 on213 source
authors, and native TF AUC=.6600 on2328. The present r values describe mechanisms
and response prediction, not additional questionnaire-prediction scores.

## D2: evidence and vector transport

40 MBTI-only authors were queried for TF and24 Big5-only authors for neuroticism.
All were outside D1 and the frozen GRAIN reference population. Two existing
12-comment views per author were used in the cue-mask arm. Four comments were
selected by absolute E5 similarity contrast to the fixed two trait descriptions.
The comparator retained four disjoint comments with matched total token length.

| Author-mean association with full-text query score | TF (40 authors) | Neuroticism (24 authors) |
|---|---:|---:|
| Selected four comments |.6104|.8193|
| Matched four other comments |.5147|.6800|
| Eight comments after removing selected evidence |.7542|.6972|
| Eight comments after removing control evidence |.7698|.8302|

Query selection is useful for retaining author ordering in a short evidence set.
Absolute-score preservation and removal size give a more mixed picture: selected
minus control absolute removal effect is -.01266 for TF (author-bootstrap95%
interval[-.03994,.01183]) and+.002935 for neuroticism[-.00431,.00919]. Selected-only
absolute-score error is slightly larger than the control in both tasks. Ranking
retention, numerical calibration, and removal sensitivity are different functions;
this result favors pursuing query-guided evidence organization rather than a
claim that the selected comments are the unique indispensable evidence.

At block16, replace the original final-query vector with the corresponding vector
from the removed-evidence input, leaving the original recipient input in place:

    h(alpha) = h_original + alpha * (h_removed - h_original).

| Full-dose patch versus natural evidence-change response, author means | TF | Neuroticism |
|---|---:|---:|
| Selected-evidence removal: r |.3108|.4592|
| Control-evidence removal: r |.4209|.5136|
| Selected: patch RMS / natural RMS |.02126 / .07869|.01206 / .02157|
| Control: patch RMS / natural RMS |.02068 / .10140|.01422 / .02061|

Both evidence conditions reveal a measurable vector route. The query-end position
carries a portion of the natural response, and other paths carry the remainder.
No discarded-residual/noise assumption is needed to obtain this result.

The two-dose curvature contrast is `M(1)-2*M(.5)`. Its RMS is about9.2-9.3% of
full-dose effect RMS for TF and6.5-6.9% for neuroticism, after averaging views within
author. This supports trying a compact linear approximation at the observed scale.

## D4: a compact operator predicts held-author responses

The discovery/held split was fixed before inference:20/20 TF authors and12/12
neuroticism authors. Fit an uncentered training-vector SVD basis U and ridge1
readout b, preserving zero input -> zero response:

    predicted_score_change = b^T U^T (alpha * delta_h).

| Task | Readout | Held authors / interventions | Response r | MSE / zero-change MSE |
|---|---|---:|---:|---:|
| TF |rank2|20 /160|.3101|.9128|
| TF |rank8|20 /160|.5314|.7401|
| TF |norm-only|20 /160|-.1402|1.0561|
| Neuroticism |rank2|12 /96|.0647|1.0511|
| Neuroticism |rank8|12 /96|.5584|.7986|
| Neuroticism |norm-only|12 /96|.1322|1.0720|

The8-coordinate operators reduce response MSE by25.99% and20.14% relative to no
change. Vector orientation contributes beyond displacement magnitude. The unit of
holdout is the author; multiple views, donors and doses are repeated observations,
not160/96 independent people. Both fixed ranks are retained; the8 coordinates are
an approximation basis, not eight newly named personality factors.

## D3: GRAIN group links are response-specific

Three nearest discovery-author neighbors predicted held-author response profiles.
Geometry was independently frozen:48 GRAIN whitened coordinates,16 identified
landmark distances, or E5 mean vectors. Response normalization used discovery
authors only. All seven response coordinates and all three geometries are saved.

Local leads worth developing:

| Held-author response | GRAIN channel | r | MSE / training-mean baseline |
|---|---|---:|---:|
| TF selected-only score change (20 held authors) |whitened48|.4549|.7581|
| TF full score (20 held authors) |identified landmarks16|.3358|.9122|
| Neuroticism selected-vector patch change (12 held authors) |whitened48|.5838|.6700|

These are exploratory leads from the fixed coordinate table. Aggregating all
response coordinates does not give a uniform neighborhood predictor: GRAIN
whitened overall MSE ratios are1.2222 for TF and1.0429 for neuroticism; the E5
ratios are1.3332 and1.3144. Some individual coordinates also have negative r.
The complete table remains attached in the private analysis output rather than
being replaced by the favorable rows above.

The theoretical consequence is constructive: **group similarity needs to specify
which response is being compared.** This points to a query/response-conditioned
GRAIN operator, not a single nearest-person geometry for all behaviors.

## Integrated mathematical result

For a fixed recipient continuation g and evidence-induced displacement d:

    natural_score_change = M(1) + R,
    M(alpha) = integral_0^alpha grad g(h + lambda*d)^T d dlambda,
    M(alpha) approximately b_q^T U_q^T (alpha*d).

The first equality is exact bookkeeping, the integral follows from differentiability,
and the low-rank approximation is supported here by held-author intervention
prediction. D3 motivates extending b_q to a conditional gain `b_q(z_author)`.
See `docs/GRAIN_QUERY_PATH_MATHEMATICS_V1.md` for assumptions and the separate
omitted-direction, recipient-gain, and curvature terms.

## Next development, not another defense series

Use all64 D2 authors as development data for a conditional response model:

    M_hat = [b_q + W_q phi(z_author)]^T U_q^T (alpha*d).

Compare it with the existing shared operator on a small NEW source-author packet,
keeping model choice inside development authors. This tests whether GRAIN's
response-specific neighborhood connection acts through a varying gain. It joins
the already useful vector operator to the group mechanism rather than reopening
generic prompt or code-invariance controls. The active goal remains open for this
connection and the final integrated theory record.

## Artifacts and execution

- Input/observations: `results/grain_vector_mechanisms/d2_real_author_path_r1/`.
- Analysis: `results/grain_vector_mechanisms/d234_real_author_analysis_r2/`.
- Reproduction: `scripts/run_grain_real_author_path.py` (`prepare`, then `run`),
  followed by `scripts/analyze_grain_real_author_path.py` with input/output paths.
- Saved tables: `d2_evidence_vector_summary.csv`, `d3_neighborhood_summary.csv`,
  `d4_operator_summary.csv`; rank2/rank8 operators are saved as NPZ files.
- All texts, identities, per-author predictions and vectors remain gitignored.
  No psychological label values, bridge outcomes, MEPS questionnaires or APIs used.
- Basic repeat/self-patch checks completed;9 new focused tests pass. An initial
  analysis attempt encountered pandas read-only array behavior; `.to_numpy(copy=True)`
  fixed it. The second analysis completed using unchanged neural outputs.
