# VM05 complete native direct-source comparison

2026-09-07, round37. Analysis-only implementation of the declared native scoring
control. Full128-shard collection is running; no full natural-source accuracy
result has been computed with this entry point.

## Question and boundaries

Compare the frozen native Qwen answer distribution against audited BGE/E5/Qwen10
external statistical heads on exactly the same source authors and three36-comment
arms. Big5-only supports Big5; MBTI-only supports its axes. Reference exclusion
matches the primary comparison. No bridge outcome table is an input.

Native prediction is the unchanged arithmetic mean of three complete12-comment
view scores:expected digit divided by9 for Big5, or fixed-positive preference
probability for MBTI, conditional on the registered answer tokens. No hard-answer
replacement, axis flipping, prompt adjustment or outcome-tuned threshold.

Bodies and populations match, but instructions, pooling and prior training differ.
A native advantage would not isolate reasoning causality; a disadvantage would
not show absence of hidden-state information. Big5 tendency digits and questionnaire
percentiles are not an independently calibrated common psychological scale.
MAE/RMSE remain numerical diagnostics. MBTI candidate-conditioned probabilities
are likewise not automatically calibrated population probabilities.

## Acceptance

`scripts/analyze_grain_direct_source_scores.py` requires actual COMPLETE status,
all128 unique shards, exact author total, no child/queue lock, unchanged registered
material/code/model, original native qualification and complete semantic panel.
Every native shard and saved probability is replayed with the existing verifier
before any source OOF outcomes are read. No partial option exists.

All45 source heads must match the completed audit, including saved-fold replay
and private prediction hashes. Pair exactly those authors and task-specific folds.
No new labels, model, forward, calibration or head selection. Bridge prediction
caches can be numerically verified, but are excluded from source evaluation.

Outputs:60 observer-condition scores,108 trait rows,540 trait/fold rows,120 paired
contrasts,27 native quality rows and private native predictions.90 paired rows
cover native versus three observers x five tasks x three arms x two estimands;
30 cover three native mask contrasts x five tasks x two estimands. Use1,000 shared
within-fold author bootstrap draws, seed731; pointwise conditional on predictions,
not familywise, refit-uncertainty, causal or post-search confirmation intervals.

Pearson/Spearman/MAE/RMSE and AUC/PR-AUC/balanced accuracy/accuracy/Brier/log loss
remain distinct. Low candidate mass and non-candidate greedy tokens are reported,
never used to filter authors. Complete response parse rate was not measured.
Constant Big5 predictions stay explicitly undefined, not removed or scored0;
constant MBTI predictions retain AUC0.5. No winner is automatically promoted.

## Execution

After all128 real shards and final queue_status.json are copied and verified on
Mac, wait for the current CPU fitter to release resources before this bootstrap
report. This entry does not schedule or stop predecessors.

```bash
"/Volumes/mobile3/projects/project persona/.venv/bin/python" \
  scripts/analyze_grain_direct_source_scores.py \
  --queue-dir results/grain_vector_mechanisms/vm05_full_direct_traits_r1 \
  --views results/grain_vector_mechanisms/vm05_qualification_r2/common_views.json \
  --material-dir results/grain_vector_mechanisms/vm05_direct_trait_material_r1 \
  --qualification-dir results/grain_vector_mechanisms/vm05_direct_real_qualification_r1 \
  --semantic-dir results/grain_vector_mechanisms/vm05_direct_semantics_real_r1 \
  --tokenizer-dir results/grain_vector_mechanisms/local_model_metadata/qwen_989aa798 \
  --preflight-dir results/grain_vector_mechanisms/vm05_preflight_r1 \
  --primary-dir results/grain_vector_mechanisms/vm05_full_neural_source_heads_r1 \
  --primary-audit-dir results/grain_vector_mechanisms/vm05_primary_neural_comparison_full_r1 \
  --resource-dir results/grain_vector_mechanisms/vm05_strong_source_controls_r1 \
  --output-dir results/grain_vector_mechanisms/vm05_native_source_comparison_full_r1
```

Qualification:8 new tests and24 combined direct tests pass, with two existing
SWIG deprecation warnings. Independent SciPy/sklearn scalar/CI checks cover both
tasks, role separation, complete-only rejection, trait order, missing views and
constant outputs. The actual2/128 queue was rejected before source outcome access
without creating analysis output.15 task/arm aggregations on the verified first
real shard passed without reading outcomes. Evidence:
vm05_direct_source_analysis_qualification_r1, identity
7a5a40313d2474d223d67f8dc8fc959afb65887adc357c647e0c70fb839b02d9.
This is readiness, not full results. Sparse/same-ready, history, stronger nulls,
bridge and VM07 retain separate requirements.
