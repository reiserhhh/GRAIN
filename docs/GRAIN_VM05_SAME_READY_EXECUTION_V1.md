# VM05 same-ready comparison: material and estimator contract

2026-09-06. Implementation of the previously required same-ready comparison,
not a new claim based on partial neural results. Existing profile source heads
are complete; the missing work is genuinely refitting sparse/neural comparators
and their increments on identical eligible authors. No released scorer changes.

## Population and observation rules

Reuse the immutable `vm05_profile_material_r1` outputs and the same7,266 common
author bundles. Preserve `raw_ready` versus `all_arms_ready` as separate cohorts.
Raw comparisons use the former; raw/cue/placebo contrasts use the latter.
The already observed counts are144/1566/62 and143/1495/61 for Big5-only/MBTI-only/
bridge respectively. They are support-selected populations, not all PANDORA.

Before any comparator fitting, seal the eligible author indices, official axis
folds, reference overlap and outcome-blind purge/count feasibility. Include a
strict branch excluding the99 historical reference members, without recomputing
the frozen geometry. Do not exclude them from a model only after prediction and
call that a refit. Do not change support using a candidate's validation result.

Every method consumes the same36 comments and the same eligible authors. A
profile support failure must not become an imputed valid profile. Source folds
remain per task. Bridge provenance can be checked but no bridge outcome values
are read or used to tune anything. For bridge prediction, both fitted heads
must share the appropriate frozen eligibility rule.

## Exact feature objects

Let g be the16 sorted frozen landmark distances (the primary formal V7 profile,
not all GRAIN4W theory), h the existing48-dimensional intermediate representation,
b the BGE36-comment mean/std and q the Qwen block10 token-weighted body mean/std.
The existing canonical-distance and canonical-shape experiments remain explicitly
named sensitivity results; their higher score is not grounds for replacing g.

The original comparators remain TF-IDF, BGE, E5 and Qwen pooled body. Primary
increments use concatenation [g,q], with [g,b] as an architecture-control
comparison. Also compare [g,q] against q alone: adding a strong q to weak g is not
proof that g itself contributes unique information. Scaler/PCA and statistical
head fit on source-inner data only, using the already frozen parameter grid.
No source/bridge result chooses blend weights, a new threshold or a new layer.

Default dense PCA is min(16,n_inner_train//10,p); full-column/no-PCA remains the
predeclared sensitivity, not a winner-selection pool. Joint PCA may spend most
of its capacity on the high-dimensional block. Report that as a property of this
readout, not loss of all information or a failure of GRAIN's complete theory.
No block-specific weighting or extra nonlinear head is silently introduced.
Qwen04/16/final, equal-view pooling and post-question states remain separately
labeled original sensitivities, not substitutes for the primary body10 contrast.

## Acceptance and boundaries

The label-free registry binds actual masks/statuses and every input hash, not
just claimed counts. Its independent audit must reconstruct masks and nested
training membership from saved source evidence. A synthetic fixture must catch
swapped authors, wrong mask types, wrong group/fold and changed profile arrays.

Fit outputs must retain source-only labels, per-fold models and selections,
OOF predictions, metrics and calibrated-probability status. Replay each saved
head; compare paired predictions only when author IDs, labels and official
folds match exactly. Restricted full-source OOF may be descriptive but must be
labeled a different estimator from ready-only retraining. No bridge tuning.

Do not start incomplete neural comparisons from whichever shards happen to be
finished. The full24-shard observer contract remains binding. Do not repeat
already completed profile fits when their identities, cohort, parameters and
saved-head replay match. Reuse them as explicit parents rather than hidden refits.

This registry does not complete source head comparison, source controls/nulls,
same-ready neural fitting, bridge evaluation or the final mechanism claim. It
prevents mismatched populations from being treated as a representation effect.
