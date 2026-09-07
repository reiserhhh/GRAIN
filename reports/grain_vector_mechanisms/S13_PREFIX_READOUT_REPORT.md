# S13: the relational reader executes without the decoder suffix

2026-09-08. COMPLETE. Fixed24-source-author panel, three views each, no labels read.

## Result

-72/72 prefix feature vectors match S12 exactly: maximum element difference0.
-48/48 frozen reader predictions (KV/PCA C and query/PLS C) replay within1.31e-16.
-Only blocks0-16 complete. Layer17 performs normalization and Q/K/V projections;
 its attention output/MLP, blocks18-27, final normalization and LM head never execute.
 Forbidden-module calls:0.
-Paired12-unit median prefix/full extraction time ratio=.6121. Pooled ratio=.6070,
 approximately39% less local extraction time. This includes feature processing,
 excludes model loading, and is not a universal hardware/runtime guarantee.
-73 prefix calls plus13 full forwards including two warmups;0backwards,83.40seconds.
 The model still has all weights resident, so no memory-saving result is claimed.

The source contrast now has an actual execution path:
text -> pretrained prefix -> within-comment KV vector -> frozen affine reader.
No generated explanation or downstream scoring pass is needed by this reader.
This is a mechanism-informed early exit, not a fully LLM-independent semantic
encoder, and it is not a new personality accuracy experiment. The source
associations remain the full213-author S12 findings.

## Why this advances the mainline

S9 detected useful mean-null configuration response; S10 supplied a KV operation;
S12 located compact response recovery in the joint vector; S13 turns the frozen
mathematical reader into a verified prefix-only computation. Skipping decoder
layers is now implemented rather than inferred from a diagram.

The early-exit pattern has prior art, e.g.
[Xin et al.,2020, DeeBERT](https://aclanthology.org/2020.acl-main.204/).
Our distinction is the mechanism-defined vector/response target and fixed exit,
not inventing the general idea of stopping a transformer early.

Next investigate common trait-query geometry versus author-conditioned query
variation while each author's KV evidence remains fixed. A successful common
query would identify a reusable mathematical read direction; a useful personalized
increment would localize a further context-dependent component.

## Artifacts and reproducibility

Plan: docs/GRAIN_PREFIX_READOUT_S13_PLAN.md.
Extractor: grain_research/vector_mechanisms/prefix.py.
Runner: scripts/run_grain_prefix_readout.py.
Ignored output: results/grain_vector_mechanisms/prefix_readout_s13_r1/
manifest.json, summary.json, replay_checks.csv, paired_runtime.csv,
private_predictions.csv and private_packets/.

The collection ran from24ef87a; subsequent lint-only closure binds local callback
defaults and renames an unused variable, without changing the experiment.
Model/token/source/reader hashes are in the private manifest. Two tiny-Qwen tests
verify actual prefix exit, matching vectors, hook cleanup and later full forward.

No source questionnaire access, new fitting, LoRA, protected lockbox, API or push.
