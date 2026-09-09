# NEXT05/04b: extract a defined configuration observation

2026-09-10. Frozen before the new comparisons. This follows natural transport
and query-residual discovery, rather than restarting general student tuning.

## Mechanism question

The fixed query-position configuration component c_q and its GRAIN-unrepresented
part R_q are now explicit targets. Can simpler input geometry compute these
objects on new authors and actual material removals? Fit their values, not human
labels, then test whether fitted values follow the native deletion response.
Do not mistake a high source-level correlation for a correct deletion derivative.

## Data and fixed alternatives

- Same276 source authors, three12-comment views:63 T/F and213 neuroticism.
- Same12 source-disjoint authors, full/E1/E2. No new selection of people or edits.
- Same frozen historical GRAIN12 factor scores. E1 has full P/Q and native
  responses; E2 only native/encoder observations, so no invented configuration
  target for E2.
- Cache-only: exact E5 comment vectors and query poles, verified token signatures
  against the stored source comments; Qwen token embedding table plus local
  tokenizer, without executing Transformer layers. No APIs or neural training.

Four fixed input programs, no hyperparameter search:

1. `e5_mean`: average of cached comment vectors.
2. `e5_query`: mean plus prior fixed absolute-pole-relevance pooling (temperature20).
3. `e5_distribution`: mean plus256-dimensional Gaussian random-feature bag mean;
   frequency seed20260910; bandwidth is a median of training-only comment pairs.
4. `token_lookup`: average token embedding plus cosine-to-fixed-question weighted
   average (temperature8). This is a new B0-style baseline, NOT the old released
   B0 scoring function. No contextual layers, attention heads or test-fitted norms.

E5 is itself a Transformer encoder; cached evaluation is not proof of a
Transformer-free end-to-end system. Only token_lookup has no contextual layers
at scoring. B4 caches use other query/material designs and are not silently
substituted. A later B4 collection needs a separate scientific decision.

Each block is train-centered and trace-normalized; equal-weight concatenation;
Ridge objective average squared error +.01||w||². No feature-wise whitening,
PCA, label-directed selection or chosen winning seed. This fixed regularization
is a bounded mechanism comparison, not an optimal-regressor claim.

## Separate reference fitting from new-author evaluation

Five author folds. In each fold, fit the GRAIN between-author component map
using training authors only (samealpha100 as NEXT01). Form configuration targets
and residual targets with that map. Fit two-output input heads on training views:

    c_q(T), R_q(T) = c_q(T) - [b_q + beta_q'G(T)].

Predict held-author views. Report raw c, directly predicted R, and the deterministic
alternative predicted_c - frozen_GRAIN_part. Source diagnostics report author
means and within-author variation separately. Fold means are not recalibrated
using test authors. This avoids defining CV residuals with a map trained on their
own target values. Fit final heads on all declared source authors only, then
freeze before reading the12-author native response outcomes.

Held test: full-position and full-to-E1 finite-change fidelity to c/R; full-to-E1/E2
change association with native F; compare the measured target and historical
GRAIN control on matching people. Author-paired bootstrap1,999; no threshold
search, post-test sign reversal, or selecting one result while hiding the others.

## Same-material multiquery extension (NEXT04b)

Independently fit source GRAIN parts for all five Big5 queries on the213-person
panel and all four MBTI queries on the63-person panel. Derive author-level and
within-author configuration residual correlations, plus paired GRAIN-part and
total-configuration correlations. The two populations stay separate; there is
no same-person nine-query covariance in these data. These relations are among
model-derived readouts, not external questionnaire relations. No criterion access.

## Prior work and inference boundary

- [Szabo et al. (2015)](https://proceedings.mlr.press/v38/szabo15.html): bags sampled
  from distributions and source-only distribution regression. Finite bag features
  are not assumed to identify an entire psychological distribution.
- [Rahimi & Recht (2007)](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html):
  random Fourier features approximate shift-invariant kernel inner products.
- [Ilse et al. (2018)](https://proceedings.mlr.press/v80/ilse18a.html): instance
  aggregation can be learned instead of an unweighted bag average. Our fixed
  semantic weighting is only a cheap comparison, not their learned attention
  model and not evidence of a causal human attention mechanism.

Previous S5/S9 studied other teachers, source populations, and full-suffix
configuration interventions. Here c_q is the frozen affine **query-position**
configuration component; do not equate the two merely because both say
configuration. This targets the specific residual recovered in the last loop.

## Finish and branch rule

Run all four fixed alternatives once; repair execution bugs, not disappointing
effects. If level works but deletion fails, locate the sampling/aggregation
derivative or readout-conditioning gap. If a target transfers, retain it as a
scoreable local computation and investigate external relevance separately. Do not
claim every context mechanism solved, or expand another accuracy grid.

Private packet: `results/grain_vector_mechanisms/logic_configuration_extraction_v1/`.
Public aggregate report, source code, tests, updated logic edges and goal ledger.
