# VM05 history observation: production path and qualified limits

## Round29: four shards independently backed up, fifth completed remotely

Latest remote snapshot: queue65840 and child60648 alive,5/128 complete,280 authors,
shard005 started. This is a moving snapshot, not a permanent process identifier.
Mac backup now independently verifies000-003:224 authors,20,160 arrays. New002/003
contain1,670/1,706 unique layouts and1,671/1,707 actual forwards, respectively.
Their archive is277,781,390 bytes, SHA
`8e5b738e2687654d3631d991d8972325e62ceb12a52e5d738209f2e1c49fa1cd`.
The previous000/001 archive remains unchanged. local_backup_status.json records
both archives and remains VERIFIED_PARTIAL_BACKUP; remote004 is not yet part of
this local acceptance. No partial current shard is copied or scored.

The full direct queue has been separately staged but remains token-only. Do not
start its real semantic panel/model until history stops at the accepted budget
boundary. Full history source readout/bridge conclusions remain open.

## Round28 live collection and verified partial backup

Windows queue65840 is alive, completed shards000/001 (112 authors), and currently
collecting shard002: child31392 alive,45/56 authors at the final observed snapshot.
This replaces the old first-child64464 snapshot below; that completed child is
not supposed to remain alive. No restart or active-code update occurred.

Completed000/001 copied to Mac and independently passed `verify_history_shard`
against the full strict registry, material/audit, real qualification, production
replay, source hashes and model identity. Total10,080 feature arrays; unique
layouts1800/1764 and forwards1801/1765. Backup292,691,489 bytes, SHA
`6166d9eeb6f56843666701b1afccb9aaa292c749a6827e26e0fa4b9db6caa4e1`.
Local `vm05_full_history_vectors_r2/local_backup_status.json` is
`VERIFIED_PARTIAL_BACKUP`,2/128, not a complete collection. Do not copy or score
partial shard002. Original whole BGE/E5/Qwen collection remains72/72 complete.

No source or bridge prediction conclusion follows from this checkpoint. Preserve
the8h invocation boundary and explicitly resume identical settings only after
the owner exits and completed shards are verified. Direct semantic code was
staged in a separate root with token-only execution, not a competing GPU model.

2026-09-06, updated round27. **Real history qualification and exact production
replay passed; full-registry collection is the next running stage, not yet complete.**
This extends the material
report without replacing its7,108-person strict registry or reopening outcomes.

## Round27 actual acceptance

- The original BGE/E5/Qwen queue completed all72 shards. Mac backup verification
  checked every manifest/cache hash and the exact7,266-author union per observer.
- `vm05_history_real_qualification_r1`: six source authors,324 layouts,648 actual
  CUDA BF16 forwards in308.81s; peak allocated CUDA memory4,311,216,128 bytes.
  All108 rows of the five error checks were exactly0: native replay, masked-token
  irrelevance, native equivalence, input embedding and causal prefix invariance.
  Identity `292edd8364a589f6c283178982e1cd5aca9147784a3a7cfbadc7f8d588edc8ce`.
- `vm05_history_production_replay_r1`: same324 layouts,152 unique complete inputs,
  153 forwards including the first-input replay,76.91s and11,694,822 cache bytes.
  All2,268 state/logit comparison groups matched exactly. Independent local alias/
  index reconstruction checked7,465,608 scalar coordinates with error0.
  Identity `d61ffdf830db6dc2d5d740006120c7eb49e0b94e5b92acaba0d5dc315f1418c1`.
- Evidence is copied locally; `vm05_history_real_qualification_audit_r1` and
  `vm05_history_production_independent_replay_r1` hold aggregate rechecks. These
  establish a working instrument, not a context-driven personality improvement.
- The technical panel also rules out a trivial all-zero observation: across its
  raw18 views, input-embedding context differences are0 but block10 current
  differences are nonzero in18/18 views (mean relative L2 change.161826).
  Decision task interactions are nonzero. `vm05_history_technical_nontriviality_r1`
  records these descriptive checks; this is not a16% gain in personality signal,
  a cross-layer magnitude ranking or population-level external validity.
- Six-author throughput suggests roughly25.3 hours and13.85GB for7,108 authors.
  This is a small-panel planning estimate, not a runtime or support guarantee.
  The worst56-author x54-layout shard at observed throughput is about1,520s;
  full collection therefore allows2,400s/shard,500MB/shard,24GB total and8h per
  queue invocation. Same128 shards and all predeclared readouts; no score-based
  sampling. See `vm05_history_full_collection_capacity_r1`.
- First detached SSH launch did not remain alive and produced no queue status;
  preserve its empty launch log rather than claiming it ran. The replacement
  keeps an owned SSH parent open: `vm05_full_history_vectors_r2`, queue PID65840.
  Inspect its actual child/run status before resume; never reuse the failed r1
  launch or the historical token-only queue as a completed experiment.

The following sections retain the original design, repaired failures and executed
technical commands for reproducibility. The full collection still precedes
source-head fitting, same-ready increments and source-frozen bridge evaluation.

## What will be measured

For each current excerpt x, layer l, question q and registered history H, retain
both the mean current-token state C_l(x,H,q) and final decision state D_l(x,H,q).
Each author has three disjoint original12-comment views, each partitioned into
11 earlier comments and one current comment. No source text is dropped to make
the experiment appear successful.

The declared contrasts are:

```
current_context = C_l(x,H_own,q_behavior) - C_l(x,H_matched,q_behavior)
current_visibility = C_l(x,H_own,q_behavior) - C_l(x,H_invisible,q_behavior)
decision_context(q) = D_l(x,H_own,q) - D_l(x,H_matched,q)
task_interaction = decision_context(q_disposition) - decision_context(q_behavior)
```

There are eight channels including own-current, own-decision and the contrasts
above. Each channel is an equal-view mean plus standard deviation, not a trait
score. The five recorded sites include input embeddings as a negative control,
blocks4/10/16 and final norm; block10 remains the primary neural site. No layer
was selected using new outcomes.

The current excerpt precedes the question in this causal decoder. Thus current
states can reflect history but cannot reflect a later question. The interaction
belongs at the decision position, not the current-token position. The technical
panel's162 paired-question layouts also have exactly equal current and decision
positions. This removes a position difference in these tested layouts, not all
semantic differences between questions.

Variation across three view-level mean effects is not token-level effect
variance. In particular, Var(C_own - C_matched) at the token level requires the
cross-history covariance; subtracting two separately stored standard deviations
does not recover it. No such covariance was measured by the current cache.

## Implemented acceptance chain

1. `qualify_grain_source_history.py`: fixed six source-only authors,324 layouts,
   planned648 actual Qwen forwards; exact replay, masking and causal controls.
2. `collect_grain_source_history.py --qualification-replay`: production capture
   must reconstruct the qualification states and diagnostic logits exactly:
   2,268 comparison groups across six authors. These include matrix groups, not
   2,268 scalar coordinates. Token-only packets cannot authorize this step.
3. Full collection: all strict-registry authors partitioned into128 fixed shards,
   same donors/masks/layouts. Each author retains90 arrays of shape3x3x1536
   (view x decision/current mean/current std x coordinate). Identical complete
   inputs and identical arrays may be cached/aliased only losslessly.
4. `history_cache.py` independently verifies parent/role/author/shard identities,
   exact output/checksum sets, all54 logical readouts, array shapes and finite
   values, causal/input invariants, forward counts and byte bounds.
5. `run_grain_history_queue.py` uses one owned child, a lock, an eight-hour maximum
   invocation and24GB maximum cache budget. It stops before an unaffordable
   shard. Only completely verified shards may be resumed; partial output is
   preserved and causes a stop, never an overwrite or fake COMPLETE.

Raw diagnostic Yes/No margins concern the existing social-initiation questions.
They are **not direct Big5/MBTI LLM scores**, which remain a separate open control.

## Actual checks and repaired failures

- cb851d0 and b3a82d8 implemented the collector/cache/queue. Five token-only queue
  shards covered280 authors and15,120 layouts (8,656 unique complete layouts),
  then paused correctly. An explicit short resume reverified all five before
  pausing again. This is not all128 shards and has zero neural forwards.
- A real command pointing the production replay at an old token-only parent was
  rejected before model loading or output-directory creation. The original
  six-author layout bytes are unchanged across local loader refactors.
- Exported-source execution on Windows first failed because artifact helpers
  eagerly imported unrelated V8 geometry modules omitted by the history archive.
  d0206a3 made those imports local to the geometry functions and bound the helper
  source in the instrument/collector fingerprints. Three entrypoints now have
  isolated exported-tree import tests; no extra legacy scorer was copied.
- Windows next exposed a real GBK-default decoding error on UTF-8 participant
  text, before any model forward. adb66fd makes the history JSON readers explicit
  UTF-8 and adds an encoding regression test. No text was rewritten or lost.
- Current Mac r4 and Windows r3 technical preflights both pass324 layouts with
  identical instrument code and semantic layout SHA
  `b3d2368a760d54661f547d4bd9c4958c19308646177c3ccc5ee575033765a45b`.
  Their raw JSON byte hashes differ due to platform newline serialization;
  semantic equality was checked separately, not assumed from green statuses.
- The scoped suite passes61 tests, including geometry/readout regressions and
  exported-source imports. Two existing SWIG deprecation warnings remain. These
  tests do not replace actual pinned-CUDA qualification.

The separate Windows root is `D:/projects/GRAIN-vector-history-vm05`. Original
`D:/projects/GRAIN-vector-research-vm01` stays at its frozen running source.
The material archive SHA is
`ed3c29a202b5a9125c7a2cf1e73fccd41a566e0a1c02240061acf8db69dcdebb`;
current source update r3 SHA is
`11256afddc3bcb51231365ad147496bc98bc548c8a439f15b22c0d6052366437`.
Private PANDORA material only; no secrets, MEPS files or questionnaire outcomes
were copied. Staging and cross-platform records are in
`vm05_history_windows_stage_r1`, operational checks in
`vm05_history_collection_operational_audit_r1`.

## Executed technical commands

Wait for the original72-observer-shard queue to be COMPLETE, not merely idle or
time-paused. The real loader enforces this and at least6GB free CUDA memory.
In the separate Windows root, with fresh output names:

```powershell
$py = 'C:/Users/huoke/AppData/Local/Programs/Python/Python311/python.exe'
$R = 'results/grain_vector_mechanisms'
$old = 'D:/projects/GRAIN-vector-research-vm01/results/grain_vector_mechanisms/vm05_full_vectors_r1'
& $py scripts/qualify_grain_source_history.py --views "$R/vm05_qualification_r2/common_views.json" --material-dir "$R/vm05_source_history_material_r2" --audit-dir "$R/vm05_source_history_audit_r2" --qwen-tokenizer "$R/local_model_metadata/qwen_989aa798" --observer-queue-dir $old --output-dir "$R/vm05_history_real_qualification_r1"
& $py scripts/collect_grain_source_history.py --views "$R/vm05_qualification_r2/common_views.json" --material-dir "$R/vm05_source_history_material_r2" --audit-dir "$R/vm05_source_history_audit_r2" --qwen-tokenizer "$R/local_model_metadata/qwen_989aa798" --observer-queue-dir $old --qualification-dir "$R/vm05_history_real_qualification_r1" --qualification-replay --output-dir "$R/vm05_history_production_replay_r1"
```

Do not submit the second command unless the first actually qualifies. Estimate
full runtime/cache from these real six-author observations before starting the
bounded128-shard queue with that production replay as `--technical-parent`.
Only after complete collection fit source-only official-fold heads on the same
supported author/text budgets, followed by same-ready GRAIN increments. Donor
reference dependence, imperfect topic/time match and near-copy exclusions remain
in force. Source-head choices must be frozen before any bridge outcome read.
