# VM06-RC4: continued prefix access versus retained body state

2026-09-06. Freeze before RC4 private neural forwards. This follows RC1-RC3;
the same 44 replies are already exposed mechanism material, not a new holdout.
No fitting, new layer selection, personality labels, or changes to GRAIN scoring.

## Intervention

Keep the full native input sequence, role boundaries, fixed body origin768,
position IDs, and causal/padding mask. At a fixed cutoff after block
`k in {-1,10,16,22,27}` (zero based), prohibit **all body queries from attending
to any prefix key** in each subsequent block. This includes role-prefix keys,
which themselves may contain assistant information; cutting assistant keys alone
would leave indirect routes. Prefix queries and body-to-body attention are untouched.
Do not remove tokens, modify value vectors, or change any earlier layer.
`k=-1` cuts from the first block; `k=27` makes no change.
Use eager float32 4-D additive masks, no KV cache. Unexpected mask shape/type or
partial hook coverage is a hard stop, never a silent skip.

Run actual and fixed length-matched contexts at all five cutoffs. The primary
score is tail mean logp (tokens2..T): the first token is predicted by a prefix
position and is intentionally unaffected by body-query interventions. Record
all-token scores only as an explicitly different sensitivity summary.

For each k report:

```text
D_k = ell_tail(actual, cut_k) - ell_tail(matched, cut_k)
R_k = ell_tail(actual, full) - ell_tail(actual, cut_k)
Q_k = ell_tail(matched, full) - ell_tail(matched, cut_k)
I_k = D_full - D_k = R_k - Q_k
```

D_k describes the context difference retained in body states when later direct
prefix access is removed. I_k describes differential effect of the remaining
access under this intervention, not an additive share of information.
Neither D_k nor R_k need be monotone, positive, or bounded by the native gap.
Removing attention keys changes the softmax denominator and can create
off-distribution computation; this is access necessity under an intervention,
not a unique circuit, semantic abstraction, or proof of personality.

## Factorial state/access controls

At blocks10/16/22, transfer the entire actual-native body state into the
mismatched-context forward, once with full later access and once cut after that
block. Compare the full-access patch with RC2 at the same site (<=1e-5).
Report both patch effects versus their corresponding matched baseline and their
difference. Do not interpret these non-independent interventions as independent
replications or combine them into a path-additive decomposition.

There are two algebraic endpoints:

- At cut=-1, actual/matched tail logits must agree within1e-5 because current
  tokens/absolute positions match and no prefix-to-body route remains.
- At a matched patch+cut_k, tail logp must replay actual cut_k within1e-5:
  all remaining positionwise and body-attention operations have identical inputs.
  This is an instrumentation identity, not a new empirical discovery.

Also run a block10-to-end mask-cloning no-change sham, and a self body patch
after block16 with cut16, in the matched arm. Both must replay their baselines
exactly. Every altered forward preserves the first token logp exactly; native
actual/matched all/tail scalar values must replay RC1 within1e-5.

## Material, budget, analysis

Same cached Qwen2.5-1.5B-Instruct revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306,
seven file hashes/native chat template match RC1. Mac only, MPS FP32/eager/eval,
fixed execution length1024, no KV and no training. First repeat RC1 six-forward
numerical gate. Then use two fixed nonprivate EN/JA examples for all18 conditions
each. Private material is not loaded before those gates pass.

Per participant: ten native/cut forwards, six actual-state patches, and two
shams =18. Total834 (=42 technical+44*18), cap860, 2400 seconds, sampled MPS
allocation <=9GiB. Native model/material/assignment replay fails closed. Budget
exhaustion preserves partials as incomplete, not a scientific zero effect.

Summarize all fixed cutoffs/sites and paired participant bootstrap1000, seed731.
These are descriptive intervals conditional on these replies/donors, not unseen
personality validation. Do not choose a winning cutoff, add layers, or resample
donors based on results. Retain negative and nonmonotone effects.

Private MEPS uses the existing text-only whitelist, local parsing of mixed answer
containers, no questionnaire fields or new external transport. No bridge outcomes.
No raw text/token IDs/attention weights/full states persisted. Save coded scalar
rows, intervention counts, errors and hashes; snapshot sources/models/materials,
use a fresh ignored output directory and audit independently without new forwards.

## Decision

An intermediate D_k plus an I_k of either sign quantifies retained state versus
continued access in a way RC2 alone cannot. Null differential I_k does not imply
no context use if both R_k and Q_k are large. Full-access patch deviations from
cut-access identities expose state/context interaction but not its psychological
content. The next separate question is readout-sensitive geometry, followed by
the already-planned source-only heads and held-out bridge comparisons. No return
to arbitrary layer search and no claim that this packet finishes VM05-VM07.
