# NEXT30 numerical extension: explicitly canonicalize the common prefix

2026-09-10, after preparation only, before any new neural calls. Original
plan a632927 is retained. Initial output `logic_native_material_reader_v1`
contains a failed cache audit, not a completed experiment. On the first author,
full-body query prefixes are identical; edited TF/N prefix max difference is
4.57763671875e-5, RMS7.92784296467e-7. This exceeds the original2e-5 max bound.
No criteria or new response derivatives have been read.

Do not increase the max threshold until it passes or discard an author. Instead:

1. Finish all24 cache-prefix comparisons, recording original-bound flags.
   Exact token/address checks remain hard requirements.
2. For all12 full author/query units, replace the entire exact-common token
   prefix at block16 by the existing TF prefix for that *same author/material*.
   No cross-author donor and no average across queries. This gives an exactly
   common represented material by construction, explicitly distinguishing
   mathematical causal-prefix invariance from finite-precision cache equality.
3. Declare the local target F_q at this canonicalized state. Its scalar score
   must remain within2e-6 of the old native scalar before local analysis. Preserve
   the exact prefix intervention size and scalar replay error. This is a minimal
   numerical canonicalization, not a claim that cached float arrays were equal.
4. Differentiate and finite-perturb this declared state only. All subsequent
   query tokens and terminal states stay at their original values. The same
   token-specific reading decomposition and111-call/900-second cap apply.

Use a new output directory `logic_native_material_reader_v1_r2`. All original
authors, roles, queries, directions, epsilon, source references and no-label
constraints stay fixed. Score-replay failure still stops collection. Report
both the original failed absolute criterion and canonicalized-target findings.
