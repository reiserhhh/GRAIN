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

## Executable comparator entrypoint

`scripts/run_grain_same_ready_comparison.py` runs one condition in an owned,
time/RSS-bounded CPU subprocess. It never modifies the existing source runner.
`--preflight-only` checks exact feature/cohort inputs without converting labels
or fitting a model. Without this flag it checks target support in all47 outer/
inner/full-source partitions, then uses the unchanged inner selection grid,
five official outer folds and a source-only final head. Retain-thread is explicit;
this entrypoint does not claim the pending discussion controls are unnecessary.

Outputs live in a fresh `results/.../<condition>/fit/` directory: manifest,
source_label_audit, private_target_support, selection, five private_fold models,
private_full_source_head, private_oof_predictions, metrics_by_trait/metrics_by_fold,
saved_head_replay, summary and source_snapshot. The parent directory holds
execution.log and supervisor_status. Summary binds output checksums; saved heads
must reproduce predictions within1e-10 before COMPLETE. Partial runs are preserved,
not overwritten or automatically counted as completed. No outer/bridge calibration.

Example, from the independent GRAIN repository (substitute the appropriate local
Python; this preflight does not run a new experiment):

```bash
python scripts/run_grain_same_ready_comparison.py \
  --views results/grain_vector_mechanisms/vm05_qualification_r2/common_views.json \
  --registry-dir results/grain_vector_mechanisms/vm05_same_ready_registry_r1 \
  --registry-audit-dir results/grain_vector_mechanisms/vm05_same_ready_registry_audit_r1 \
  --profile-dir results/grain_vector_mechanisms/vm05_profile_material_r1 \
  --preflight-dir results/grain_vector_mechanisms/vm05_preflight_r1 \
  --parent-dir results/grain_vector_mechanisms/vm05_full_source_cohort_r1 \
  --prepared-root '../project persona/data_sets/prepared/pandora_official' \
  --cohort raw_ready --task big5 --feature word_char --arm raw \
  --output-dir results/grain_vector_mechanisms/new_ready_big5_preflight \
  --preflight-only --max-seconds 600
```

Actual fitting uses a different fresh output and omits `--preflight-only`, only
when the single-Mac heavy-fit queue is available. Dense observers additionally
require `--vector-dir .../vm05_full_vectors_r1`: all72 stage-shards must be complete
and checksum-verified first. `grain_qwen_block_10` and `grain_bge_large` mean plain
concatenation, with explicit block boundaries, not tuned blending. Body layer
sensitivities retain their exact feature names. The subsequent cached-readout
extension accepts`--readout body_equal_view`, `body_variance_parts`,
`decision_behavior`, `decision_disposition` and `decision_task_delta` for Qwen
blocks, including their GRAIN concatenations. Default`body_token` keeps the
original pooling path. `logit_margins` is a four-scalar diagnostic accepted only
under the single block10 identifier, since it is site-independent. The readout
identity is explicit in both manifest and summary; it is not selected by outer
labels. See GRAIN_VM05_READOUT_SENSITIVITIES_V1.md. This closes the cache-to-head
implementation gap, not the unrun full-source/same-ready comparisons, genuine
same-current history-effect collection or direct LLM Big5/MBTI scoring.

Existing non-reference-excluded sorted profiles are reused, not refitted. The
`sorted_distances` feature here only permits a genuinely changed reference-free
cohort; if exclusion changes no eligible author, reuse its original profile parent.
Synthetic tests verify column alignment, source isolation, support failures,
fold-local scaling/PCA, exact agreement with the old head, complete-queue gating,
packet mutation rejection and full Big5/MBTI output replay. Passing tests does not
establish a positive neural increment on the actual corpus.
