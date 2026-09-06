# Bounded full direct-trait queue

2026-09-06, round29. Executes the existing direct-trait control; no new head,
prompt, direction, population or feature-selection rule is introduced.

## Fixed production scope

All7,266 supported common-material authors,128 hash shards, three complete
12-comment views and raw/cue/placebo arms. The registered278,397 logical prompts
reuse only143,045 identical inputs within authors. Every source shard replays its
first native input. All author/task/arm scores, including weak or noncompliant
ones, remain in the result. Bridge texts may be processed without accessing their
outcomes; only Big5-only and MBTI-only sources will train/statistically select heads.

Default invocation8h, single shard2,400s, cache200MB/shard and6GB total. The latter
is a conservative upper bound, not estimated usage. Actual collection uses the
already-qualified pinned CUDA BF16 observer. Never load a second model while
history owns the GPU. CPU/token-only preflight is explicitly separate.

## Acceptance and resume

1. Recheck registered material, tokenizer, source-code fingerprints, supported
   author union and applicable tasks before any child.
2. Real collection requires the existing six-author native direct qualification
   and the complete90-case/180-forward semantic panel. A negative or flat semantic
   result is retained as a limitation, **not** a reason to change prompts or omit
   tasks. This parent requirement means numerical/protocol completeness, not a
   claim that all semantic contrasts must pass.
3. History must be COMPLETE or PAUSED_BUDGET_BETWEEN_SHARDS, its lock absent and
   owner/child dead. Recheck before each real shard. Queue coordination assumes
   the current single supervising agent; these checks are not a distributed GPU
   reservation protocol for unrelated launchers. Do not manually resume history
   while the direct queue is active.
4. A fresh direct output obtains its own exclusive creation lock. At a budget
   boundary it pauses **between shards**, leaving completed data untouched.
5. An identical explicit `--resume-completed` rechecks every completed shard:
   role/task/author order, prompt commitments, actual model signature, every cache
   checksum and score arithmetic, native-forward count, cache size and output set.
   It does not overwrite or automatically retry an incomplete/failed shard.
6. COMPLETE requires all128 shards and the exact7,266-author union. Metadata-only
   preflight completion cannot be used as neural evidence. The immutable manifest
   permits a different invocation wall-time budget on resume, but not different
   material, model, code, shard count, per-shard or total cache caps.

Entry: `scripts/run_grain_direct_traits_queue.py`; independent checks:
`grain_research/vector_mechanisms/direct_queue.py`. The original qualified
collector and history-active files remain unchanged. Native observer precision,
prompt length and template are not optimized away for throughput.

## Execution

Run from `D:/projects/GRAIN-vector-direct-vm05` with argv equivalent to:

```text
python scripts/run_grain_direct_traits_queue.py
  --views results/grain_vector_mechanisms/vm05_qualification_r2/common_views.json
  --material-dir results/grain_vector_mechanisms/vm05_direct_trait_material_r1
  --tokenizer-dir results/grain_vector_mechanisms/local_model_metadata/qwen_989aa798
  --qualification-dir results/grain_vector_mechanisms/vm05_direct_real_qualification_r1
  --semantic-dir results/grain_vector_mechanisms/vm05_direct_semantics_real_r1
  --observer-queue-dir D:/projects/GRAIN-vector-research-vm01/results/grain_vector_mechanisms/vm05_full_vectors_r1
  --history-queue-dir D:/projects/GRAIN-vector-history-vm05/results/grain_vector_mechanisms/vm05_full_history_vectors_r2
  --output-dir results/grain_vector_mechanisms/vm05_full_direct_traits_r1
  --max-seconds 28800 --shard-seconds 2400
```

Lines are wrapped for reading, not separate shell commands. Keep the SSH parent
open for Windows execution, as the earlier detached launch did not survive logout.
Do not launch this real command while history is still running or the semantic
panel is only token-checked. Source evaluation and bridge acceptance remain separate.
