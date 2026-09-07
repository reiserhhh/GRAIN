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

## Status-only operational wrapper (round37)

Use `scripts/run_grain_direct_queue_io_guard.py` with the same arguments for the
next Windows invocation. This applies the existing tested Windows access/sharing
retry only to this queue's queue_status.json. It does not change the frozen
collector/queue, prompt coding, shard manifests, caches, predictions or numerical
gates. Five bounded waits total1.55s for WinError5/32/33; other paths/errors pass
through immediately and permanent failure keeps the original exception.

The wrapper journals beside the output in `<output_name>_io_guard/<UTC>_<pid>`.
It must not prepopulate a fresh output directory, since the original queue's
nonempty-directory rejection remains intact. Fresh and explicit-resume modes,
history-idle checks and the original queue lock stay in force. Each journal keeps
argv, original/wrapper code hashes, source snapshot, any prior status and
append-only retry/exit events. The original queue writer is restored on return
or exception. This is the known history I/O remedy, not a new scientific protocol
or a reason to alter a negative result.

Round37 qualification:13 wrapper/generic tests pass; Mac fresh and explicit-resume
zero-forward runs and a Windows fresh zero-forward run all stop at the original
budget gate. Windows six copied qualification files were independently hashed.
No source/bridge label values or neural forwards were used in these checks.

Actual full collection began11:25 JST on2026-09-07, after history628 naturally
paused65/128 and its owner/child and lock disappeared. Remote wrapper48952,
SSH session23843, retains the frozen128-shard command above. First shard000
completed57 authors/2,151 logical prompts/1,160 native forwards and was copied
to Mac and independently replayed:maximum saved-arithmetic error3.55e-15.
No native full-vocabulary logits were newly reconstructed by this arithmetic
check. Full source prediction accuracy and all128-shard acceptance remain pending.
Do not resume history or launch a second GPU model while this queue is active.

The one-off waiting launcher's events and command metadata contained literal
backslash-n separators. Raw snapshots are retained under
`vm05_direct_launch_round37_r1/remote_metadata_snapshot`; a JSON-decoder-based
derived canonical copy and raw/canonical hashes preserve all323 recorded events.
Only inter-object separators were normalized. The actual queue journals,
manifests, scientific caches and frozen launcher source were not modified.
