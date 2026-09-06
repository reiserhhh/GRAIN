# VM06 domain geometry/function bridge: frozen V1

2026-09-06. This complementary design is fixed while domain context V1 is still
running, before its private effect table is inspected. Same167 frozen cases and
donor assignments; no additional cohort selection or personality outcome access.

## Geometry observers

Use cached multilingual-E5 on all cases; cached English BGE/E5 on the English
Essays/X cases only. Revisions/prefix/native pooling follow existing observer
definitions. No English-encoder Japanese performance claim. Model files and
tokenizer hashes must match the existing preflight/qualification records.

Encode current text and the correct source-context text independently. Context
is not fed into the current embedding. Each complete text is divided into
contiguous character spans that fit512 tokens in all three frozen tokenizers.
Use conservative bisection and retokenize/check every span; preserve every
character, never silently truncate. Use these identical spans across observers.
Aggregate normalized window embeddings with character-count weights, then
normalize. Character weights are shared across models, not variable tokenizer
counts. This is a window-summary operator, not full-context transformer pooling.

MPS FP32, eager attention, one encoder at a time. English BGE uses CLS, E5 uses
masked final-state mean. The complete raw current/context content is shared,
but native encoder pooling differs and is part of the observer package.
Public repeat/padding-identity tests precede each encoder's domain forwards.
Store file identities, full window metadata, vectors and source code snapshots.
Stop on technical mismatch, numerical nonfiniteness, sampled MPS>6GiB, or2h.

## Frozen comparisons

For case i with frozen matched donor d(i), define:

`delta_c_i = cos(E(current_i),E(context_i))
             - cos(E(current_i),E(context_d(i)))`.

Also retain actual/random similarity and within-cell other-author percentile;
never compare different-symbol/language targets as though condition-matched.
Summarize per corpus/language/model, without treating pairs or windows as subjects.
Use1,000 fixed-case bootstrap draws with seed731, conditional on selected material
and donor assignments, not population or newly refitted-estimator confidence.

Join the frozen Qwen primary endpoint
`delta_l_i = tail_logp(actual)-tail_logp(matched)` by exact case ID and material
hash. Report Pearson and Spearman of delta_c versus delta_l. Bootstrap both on
the same cases; report undefined draws explicitly. No threshold-based model
selection or bridge access follows these correlations.

Descriptive length-controlled analysis removes the intercept and
`delta_log_length = log(T_context_i)-log(T_context_d(i))` from each variable,
using ordinary least squares, then correlates their residuals. This is an
in-sample partial association, not a causal decomposition, fitted personality
head or held-out prediction claim. It does not control all topic/lexical factors.

The purpose is to determine whether a standalone semantic-similarity difference
tracks the contextual model's functional benefit. If it does, simple semantic
matching remains an important explanation. If it does not, unmeasured readout,
order and contextual interaction are candidates, not automatically personality
or reasoning mechanisms. No new directions/factors are selected on these cases.

Outputs: private vectors/window lineage, geometry_by_case.csv,
domain_geometry_function.csv, hashes/replay; public companion report. This is
not a claim that different model coordinates are aligned or interchangeable.
