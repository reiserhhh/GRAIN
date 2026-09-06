# Complete cached readout comparison

2026-09-07, round35. Analysis implementation during the running, already frozen
315-case family. This is not a new outcome-blind preregistration or permission to
choose a winner. No additional representations, prompts, layers or source fits.

## Question and estimands

With the same author/comment budget, can information about permitted source
labels be read differently from pre-question body pooling, post-question decision
states and the contrast between the two declared questions?

All315 secondary heads are paired with their already fitted body_token head at
the same site/task/text arm and exact same authors, targets and official folds.
There are60 reused body heads. Each task uses its own source-exclusive folds;
MBTI axes are not forced into a common author fold. Report all375 head scores,
trait-level metrics and both pooled and mean-official-fold estimands.

-315 candidate-minus-matched-body contrasts, each with two estimands.
-120 explicitly exploratory contrasts: decision_disposition minus
 decision_behavior and decision_task_delta minus decision_behavior, at all60
 site/task/arm groups, again with both estimands.
-Total870 paired rows. Four-margin features occur only once at the block10 ID;
 this is an identifier and does not make margins block10-specific.
-1,000 seed731 author-within-official-fold bootstrap resamples, shared across
 all heads in a comparison group. Reject undefined observed scores and insufficient
 resample support; report undefined draws. This is conditional on fitted OOF
 predictions, not refit or cluster-of-discussion sampling inference.
-Intervals are pointwise, not simultaneous/familywise confirmation. Do not count
 one or several unadjusted positive intervals as a confirmed mechanism, select
 bridge layers, or infer a causal circuit from source-label readability.

Two question states concern social initiation, not nine trait-specific questions.
All statistical heads were source-trained; none is the native LLM personality
answer. Retain-thread source/discussion controls and full nulls remain separate.

## Entry and safeguards

`scripts/analyze_grain_cached_readouts.py` accepts only a complete315-case
manifest/status, absent child and exited owner. It refuses a partial family before
OOF access or output creation. Acquire existing CPU leases nonblockingly; do not
compete with another fitter. Bind original input/code hashes, all source cohorts,
each accepted fit, complete saved-fold replay and previous audited body heads.

The legacy body-head summaries lack the newer packet-level outputs_sha256 field.
Verify those using the prior primary/layer audit's explicit artifact hashes,
including all five fold models and replay files, rather than inventing missing
metadata or accepting unchecked legacy files. The new readout packets use their
own existing complete-packet checksums.

From the GRAIN repository, after complete secondary execution and CPU release:

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/analyze_grain_cached_readouts.py \
  --batch-dir "$R/vm05_secondary_cached_readouts_r1" \
  --views "$R/vm05_qualification_r2/common_views.json" \
  --preflight-dir "$R/vm05_preflight_r1" \
  --parent-dir "$R/vm05_full_source_cohort_r1" \
  --vector-dir "$R/vm05_full_vectors_r1" \
  --ready-dir "$R/vm05_raw_ready_primary_r1" \
  --sensitivity-dir "$R/vm05_neural_sensitivities_r1" \
  --primary-dir "$R/vm05_full_neural_source_heads_r1" \
  --resource-dir "$R/vm05_strong_source_controls_r1" \
  --paired-dir "$R/vm05_raw_ready_paired_comparison_r1" \
  --primary-audit-dir "$R/vm05_primary_neural_comparison_full_r1" \
  --layer-audit-dir "$R/vm05_fixed_layer_comparison_full_r1" \
  --output-dir "$R/vm05_secondary_readout_comparison_full_r1"
```

Outputs: observer_scores.csv, metrics_by_trait.csv, paired_readouts.csv,
manifest.json, summary.json, source_snapshot.zip. Reuse existing source OOF values
only; no source-label table load, bridge values, MEPS questionnaire data, new LLM
training, neural forwards or calibration. All outputs are private/gitignored.
The current mainline is unchanged and whole-program completion remains false.

## Decision after complete results

Assess all contrasts, mask consistency, effect magnitudes and fold signs before
choosing a new hypothesis. A post-question advantage would motivate a specific
functional intervention, not establish semantic reasoning by itself. A null
contrast would constrain this fixed head and readout scope, not all possible
reasoning. Keep the independent history, native answer, stronger null and frozen
bridge comparisons required by the original mechanism plan.
