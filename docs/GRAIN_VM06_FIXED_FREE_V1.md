# VM06: complete MEPS fixed/free representation comparison

2026-09-06. Freeze before new private neural extraction. This closes a missing
condition-comparison requirement, not the full vector program or personality validity.

## Material and access

Use the audited text-only source and preflight identity
`2a1a13248f7bc74984ad1e4eab1f72aa80fd53993f4f26d146530bfa8121bebb`.
46 participants, 138 direct MEPS answers, 126 chat views. Replay the existing
1,609 lossless, speaker-preserving windows over 1,600 turns. No questionnaire,
mood, or psychological outcome values; mixed Markdown containers are scanned
and hashed only to recover permitted direct-answer sections. Mac only, offline.
No training, new API calls, model downloads, or changes to the released scorer.

## Frozen observer

Cached `intfloat/multilingual-e5-large`, revision
`3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`, float32 MPS, eager attention,
batch one, fixed 512 positions, `query: ` prefix. Masked mean of final states,
including non-padding special/prefix tokens, followed by L2 normalization,
as in the [official model card](https://huggingface.co/intfloat/multilingual-e5-large).
Retokenize each existing span without truncation; verify its length and hash.
This independently encodes windows, NOT a context-conditioned full conversation.

Before reading private material, public English/Japanese probes check finite unit
vectors, exact repeated-input replay, masked-padding ID replacement invariance,
and compact versus fixed-padding error (max component error <= 1e-5).
Record actual model/config/tokenizer hashes and device/software. Stop on technical
failure, source mismatch, >6 GiB sampled MPS allocation, or 2 hours elapsed.
Partial outputs cannot become a complete condition comparison.

## Aggregation and primary comparisons

For normalized window vector e_w and own-text token count c_w (exclude prefix,
special tokens using tokenizer offsets):

`v_token = normalize(sum_w c_w e_w)`.

Sensitivity: form normalized token-weighted vectors per original turn, then
average those turns equally and normalize (`equal_turn`). No vector/label-based
choice among aggregators. Direct answers are each a user turn. Also create an
equal-question mean of the three separately scored answers (`direct_all`).

Keep user-only, assistant-only, and explicitly naive mixed user+assistant pooling
separate. System messages never enter an author vector. Absent assistant text
stays missing, never zeros or invented turns.

Seven primary user-only token-pooling contrasts: Q1-Q2, Q1-Q3, Q2-Q3;
Q1-free, Q2-free, Q3-free; equal-question direct_all-free. All use the common
participants with both views; never use the 44 short-reply RC subset as a substitute.

For cosine matrix S_ij comparing condition A person i with condition B person j:

`gap = mean_i [S_ii - mean_(j!=i) S_ij]`.

Report query-wise same-person percentile (ties count one half), not MBTI AUC;
mean diagonal/off-diagonal cosine and top-one matching are descriptive. Both
query directions are averaged for the percentile. Conditions fixed on each side.
Matched-length sensitivity retains five other target authors nearest to the
true target's log character count, per query and separately per direction;
matching uses no embeddings or labels. It does not control topic or AI assistance.

Use 1,000 participant/node bootstraps (seed731) for gap and percentile, excluding
same original IDs from negative pairs even when sampled twice. A bijective
author-link permutation (1,999 draws, seed1731) gives a one-sided gap p-value.
Apply Holm across all seven primary contrasts; sensitivities are descriptive,
not another search for significant findings. No pair-edge pseudo-replication.

## Role controls and interpretation

Repeat direct_all-free for assistant-only and naive mixed free-chat vectors,
and free user-assistant within-person correspondence; repeat all contrasts with
equal-turn pooling. Report mixed-to-user and mixed-to-assistant cosine plus
actual token shares; character share alone is not causal attribution. For the
three assisted tasks, report direct answer versus associated user/assistant
chat correspondence on available participants only. These may reflect copying,
common subject matter or feedback, not stable individual psychology.

Primary positive evidence means cross-task author correspondence under this
observer. It cannot establish trait, clinical utility, reasoning, language
invariance, true temporal retest, or absence of topic/copying confounds. An AI-only
association is evidence of interaction contamination risk, not an AI personality
label. A null result limits this fixed representation/aggregation on this N,
not every possible personality mechanism.

## Artifacts and audit

Private ignored run folder: manifest, source snapshot, technical checks, per-window
metadata/vector cache, view vectors, source/window lineage, progress/error state,
paired matrices and participant contrasts. Public report contains aggregates only.
Independent audit rechecks cache checksums, window/turn coverage, role isolation,
aggregation, seven primary matrices and permutations. This is scalar/material
and saved-vector replay, not independent re-extraction of all neural activations.

Next after this bounded packet: current-mechanism Essays/X domain inventory and
the ongoing source-only full observer/head work, then frozen bridge evaluation
and VM07. Do not expand RC layer search or open MEPS psychological outcomes.
