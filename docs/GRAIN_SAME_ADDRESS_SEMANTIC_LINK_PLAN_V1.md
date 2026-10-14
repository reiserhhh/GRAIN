# C14 Same-Address Semantic Link V1

2026-09-13. Frozen before this tranche's results. Bounded development experiment,
not a new lockbox or an exact SIMPA replication. Source: all-plan C14, RQ-C,
V5 G2/G3; motivated by NEXT91 interaction and within-cell results.

## Decision

Can an existing GRAIN input representation carry query-specific semantic
affinity on unseen authors, and does a quadratic map add anything beyond a
linear map and observed opportunity proxies? This informs the source-learned
GRAIN/evidence interface, not personality predictive accuracy.

## Material and Target

- Reuse 624 Big5-only authors, 36 comments each, and 29,075 rule-defined
  first-person candidate sentences. No MEPS, bridge, new downloads or inference.
- Verify frozen source files, sentence offsets, IPIP keys and cached top-1
  matches. Reuse the external-discovery TF-IDF/SVD24 runtime with zero current
  author overlap. Sentence coordinates are a unit adapter, not the native
  author geometry score or the four H16 diagnostic factors in NEXT91.
- With normalized cached sentence vector h and item vectors q, for every one
  of 30 IPIP facets compute both mean and max positive-minus-negative cosine:
  d_mean = mean(h.q_positive) - mean(h.q_negative);
  d_max = max(h.q_positive) - max(h.q_negative).
  These 60 outputs are affinity contrasts, NOT entailment, questionnaire
  answers, habitual traits, or proof of ownership. No-match is not a low trait.
- Retain original threshold .6 detection flags separately. Join existing
  80 source-only model/NLI judgments by exact sentence index/address. Do not
  use these small non-human judgments to fit or tune the reader.

## Models and Split

- Equal total training weight per author. Folds 2/3/4 fit, fold1 tunes alpha
  in [1,10,100,1000]; then refit folds1/2/3/4 and evaluate fold0 once. Fold0
  is a historically explored development holdout, not fresh confirmation.
- Four variants: opportunity only; GRAIN linear; opportunity + GRAIN linear;
  opportunity + all degree-2 GRAIN terms. Coordinate/scaler fitting occurs
  on training authors only. Do not remove topic selection from the raw arm.
- Opportunity proxies: log sentence/comment length, punctuation, numeric,
  quote, first-person and sentence-density features plus subreddit categories
  supported by >=10 source authors. These are observed proxies, not known
  causal opportunity sets. Names/categories are fitted within each split.
- Ridge, equal source-variance-normalized target loss. No alpha search using
  holdout. One fixed-alpha (100) source-row permutation negative control;
  this is a pipeline sanity check, not a null significance test.
- Save frozen fit packages and replay predictions after reloading. A single
  short smoke run verifies mechanics before the full cached CPU experiment.

## Outputs and Acceptance

Output: ignored `results/grain_vector_mechanisms/all_plan_closure_v1/semantic_link_r1/`.
Save provenance, model bundles, original addresses, coordinates/targets,
source validation loss, all 60 target results, author-cluster paired MSE CIs,
held author-aggregate agreement, annotation join, and errors if any. Publish
cohort summaries only in the accompanying report.

Main evidence: held-author agreement with each affinity function; incremental
paired MSE relative to opportunity and linear baselines. Report negative
increments and all facets, not only leaders. Bootstrap unit is author (500
draws, seed 913); intervals condition on fitted models and are not selection-
adjusted confirmatory tests. Split-half author affinity is descriptive.

Bound: local cached CPU, 20 minutes maximum, no neural training. Stop after
the declared comparison; no higher polynomial grid based on holdout results.
Successful delivery means a reproducible source-fitted interface and a scoped
decision. It does not automatically complete C14's semantic validity question.

## Predecessors

[SIMPA, author description](https://takelab.fer.hr/publications/simpa/): use
statement/item affinity as an inspectable intermediate, retaining the
distinction between trait-constrained evidence and generic similarity.
[Hinton et al. 2015](https://arxiv.org/abs/1503.02531): transferring a teacher
function to a smaller reader is an established pattern. Here continuous
affinities use squared-error regression, not their softmax-temperature loss;
teacher imitation is not an independently identified human trait mechanism.
