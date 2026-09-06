# VM05 complete-history source readouts

2026-09-07. Implements the readout step of the existing history-counterfactual
protocol. Instrument tests are complete; real source fitting has NOT started.
The original collector, reference rule and live secondary runner are unchanged.

## Question and estimands

For author u and view v, hold the current excerpt's tokens and positions fixed.
Let m_uv(H) be its mean hidden state under a preceding history H. The primary
context channel uses the already registered length/time-optimized donor:

```
d_uv = m_uv(H_own) - m_uv(H_reference)
x_u = concat(mean_v(d_uv), std_v(d_uv, ddof=0))
```

There are three views and1536 coordinates, giving3072 columns. The second block
is the dispersion of three VIEW-MEAN contrasts. It is not the variance of token
effects: those require within-token covariance, which this cache does not retain.
No question can influence current-token states before it in the causal decoder.
The two question conditions only differ at later decision sites.

The eight existing channels are current_own, current_context, current_visibility,
decision_own_behavior, decision_own_disposition, decision_context_behavior,
decision_context_disposition and decision_task_interaction. The final channel is
the difference between two own-minus-reference decision contrasts. These two
questions concern social initiation, NOT nine trait-specific neural states.
All four declared non-input sites remain accessible; block10 is primary.

This is a model-side, donor-relative context response. It does not subtract all
topic/situation variation or identify pure personality. Median donor subreddit
Jaccard is0. The separately planned overlap-restricted donor control remains open.

## Cohorts and evidence budget

The complete strict registry has7,108 authors. Source heads never use its bridge
rows. Its99 fixed MBTI-only reference authors are excluded from every prediction
target and are not chosen using source labels or outer-fold outcomes.

| Cohort | Big5-only | MBTI-only, each axis | Use |
|---|---:|---:|---|
| history_full |620|6,239|Original baseline and history channels|
| history_ready |143|1,459|Same-author GRAIN, history and combination|

These counts and all official-fold memberships were recomputed without label
values in `vm05_history_readout_cohorts_r1/cohorts.json`. Its parent identities
and code version are in summary.json; they precede the additional donor-type
guard and remain evidence for that recorded code, not a silently relabeled rerun.
The original complete primary cohort624/6294 and raw-ready144/1539 are DIFFERENT.
Refit baselines on the relevant history cohort. Do not subtract older scores or
join unequal OOF samples and call the result paired improvement.

History channels use an additional fixed reference dictionary. Raw TF-IDF and
pooled BGE/E5/Qwen baselines do not consume substituted donor texts. Their common
target budget is36 own comments, but their reference-information budgets differ.
Report that distinction; a history gain alone would not isolate readout capacity
without a comparable reference-input control. The16-column sorted-distance g is
one frozen GRAIN observer, not the complete4W methodology. Joint PCA and linear
heads test restricted readability; they do not exhaust nonlinear cross-block use.

## Safety and fitting

`scripts/run_grain_history_source_readout.py` runs ONE bounded condition. It does
not launch a Cartesian search, modify neural weights or choose the next condition
from outer outcomes. The parent history plan governs subsequent execution.

Before any label conversion it verifies the strict registry, reference/target
separation, source-specific official folds, qualification, technical replay,
immutable collector identity and ALL128 complete history shards. Original
embedding baselines also verify all72 original observer shards. Every requested
author's cached array checksum and exact identity are rechecked when loaded.
Missing shards cause refusal, not a smaller improvised benchmark.

Source-only Big5/MBTI use the existing official5 outer / seed731 inner3 author
folds. Scaler, PCA and statistical hyperparameters are fitted inside training
folds. Saved heads must reconstruct all five OOF blocks within1e-10. Report
pooled and mean-fold comparisons on identical authors, shared author bootstrap
draws and both proper scores and discrimination for MBTI. Such intervals are
conditional on saved OOF predictions, not selection-aware refit uncertainty.
The runner's single-model bootstrap is not a paired incremental comparison.

`--preflight-only` validates feature construction but stops before source labels
and fitting. A successful preflight is not a personality or prediction result.
The actual incomplete-cache probe correctly refused with104 missing shards and
no target-support or model files (`vm05_history_readout_gate_probe_r1`). Synthetic
tests cover full-cache refusal, source/donor/bridge roles, altered folds/caches,
view-contrast algebra, no-label preflight and both task types' saved-head replay.

## Reproducible invocation

Run from the independent GRAIN repository after all history caches are verified
AND the current CPU owner has exited. Do not compete with the315-case secondary
queue or resume paused strong controls concurrently. This one-case CLI itself
does not coordinate those separate queues. It uses one child, default1h/8GiB,
absolute2h runtime. A fresh output directory is required; failures are preserved.

```bash
PY='/Volumes/mobile3/projects/project persona/.venv/bin/python'
R=results/grain_vector_mechanisms
"$PY" scripts/run_grain_history_source_readout.py \
  --views "$R/vm05_qualification_r2/common_views.json" \
  --preflight-dir "$R/vm05_preflight_r1" \
  --parent-dir "$R/vm05_full_source_cohort_r1" \
  --prepared-root '/Volumes/mobile3/projects/project persona/data_sets/prepared/pandora_official' \
  --registry-dir "$R/vm05_same_ready_registry_r1" \
  --registry-audit-dir "$R/vm05_same_ready_registry_audit_r1" \
  --profile-dir "$R/vm05_profile_material_r1" \
  --material-dir "$R/vm05_source_history_material_r2" \
  --audit-dir "$R/vm05_source_history_audit_r2" \
  --qualification-dir "$R/vm05_history_real_qualification_r1" \
  --technical-parent "$R/vm05_history_production_replay_r1" \
  --history-dir "$R/vm05_full_history_vectors_r2" \
  --vector-dir "$R/vm05_full_vectors_r1" \
  --cohort history_full --task big5 --feature history \
  --arm raw --site block_10 --kind current_context \
  --preflight-only --max-seconds 3600 \
  --output-dir "$R/vm05_history_full_big5_context_preflight_r1"
```

Remove preflight-only and use a NEW directory only after the required parents
and resource gates pass. Comparator features are word_char, bge_large, e5_large,
qwen_body and sorted_distances; grain_history adds the3072-column history channel
to g on history_ready only. Each condition stores manifests, source snapshots,
feature commitments, support tables, private OOF, fold/full heads, metric tables,
saved-head replay and checksummed summaries. Bridge outcomes and MEPS scale values
remain closed. No new LLM training, model downloads, paid APIs or release changes.
