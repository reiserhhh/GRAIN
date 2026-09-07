# S13: execute the relational reader without the decoder suffix

2026-09-08. Bounded implementation experiment after S12's positive KV recovery.

Question: can the frozen S12 representation and affine reader be evaluated without
the model's final scoring computation? This is execution equivalence and measured
cost, not new human-label validation or a claim of a wholly model-free encoder.

## Fixed design

First24 authors in the existing S12 fixed order, all3views. No label access.
Stop the frozen Qwen forward immediately after layer17 V projection. Blocks0-16
execute; layer17 supplies its input normalization and Q/K/V only. Reconstruct
rotated keys and scaled last query, then compute the within-comment KV-pairing
vector explicitly. Do not execute layer17 attention output/MLP, blocks18-27,
final norm, or LM head. Use the same numerical/model/token settings as S12.

Scoped capture hooks and a private early-exit signal preserve the model and allow
a later ordinary full forward. Guards record forbidden modules if reached.
Retain the full model in memory during this test; no memory-saving claim.

Compare all72 view vectors and24 author-level affine predictions to the frozen
S12 cache. Replay both KV/PCA C and query/PLS C as declared candidates, without
selecting coefficients or direction from labels. Feature max-error threshold1e-5;
prediction max-error1e-6. A tiny random CPU Qwen test checks actual execution and
restoration, not just mock code.

For12 predetermined view0 units, alternate full-capture/prefix order by author
index and report paired elapsed-time ratio. Two initial warmup calls (one of each)
are outside timing summaries but inside cost accounting. Total73 prefix calls and
13 full forwards;0backwards;10-minute cap. Timing is local MPS execution including
feature extraction, not a cross-platform speed guarantee or comparison with
language generation.

## Mathematical and prior-work connection

S12 supplies z(text,q) and C_hat=b+beta'z. Since z depends only on a strict
computational prefix and fixed algebra, downstream decoder evaluation is redundant
for this reader. Numerical replay tests that implementation statement.

[Xin et al.,2020, DeeBERT](https://aclanthology.org/2020.acl-main.204/)
establishes early exit as a practical inference pattern. Here the exit is fixed
at a mechanism-defined KV site, not their adaptive BERT classification scheme.
No new classifier or decoder weights are trained in S13.

Outputs: results/grain_vector_mechanisms/prefix_readout_s13_r1/
Report: reports/grain_vector_mechanisms/S13_PREFIX_READOUT_REPORT.md.
Following question: shared trait-query geometry versus author-context-dependent
query variation, holding each author's actual keys/values fixed. Do not resume
ordinary E5 mean grids or protected-label experiments.
