# VM06-RC3: exact-bag context-order contrasts

2026-09-06. Frozen before RC3 private neural forwards. RC1/RC2 established
context compatibility and a depth-dependent body-state intervention profile;
they did not distinguish shared vocabulary from contextual organization.

## Fixed design and estimands

Reuse exactly RC1's 44 complete replies, actual preceding assistant turns and
length-matched donor assignment. No new response choice, fitting, labels or layer
search. The current user-token sequence, native role tokens, attention mask,
position IDs and input lengths are unchanged within each context's intervention.

For each actual and mismatched assistant context, compare:

1. Native order.
2. Permute contiguous eight-token blocks, retaining order inside each block.
3. Permute individual tokens.

Both permutations use two fixed seeds (731, 1731); a context's source code, mode
and seed determine its permutation, not its recipient or observed effect. Both
actual and donor uses of that context share the same internal ordering. Use one
nonidentity group permutation, rotating by one only if a random identity occurs.
Do not search seeds. Repeated token strings can still yield identical sequences;
record the actual changed-token fraction and duplicate order hashes, not fake
independent replications.

Only complete tokens strictly within the rendered assistant-body character span
move. Any boundary-crossing token stays fixed and is counted. A contiguous,
unambiguous body span with >=8 isolated tokens is required; unexpected special
role tokens cause STOP, not silent deletion or selective case exclusion. Do not
decode/re-encode the shuffled IDs. Exact token multiset, prefix length, untouched
role/prefix positions, response IDs and inverse-permutation restoration are checked.

For each participant and score (all-user-token logp primary; tail sensitivity):

    D_native = ell(actual_native) - ell(mismatched_native)
    D_mode,s = ell(actual_mode,s) - ell(mismatched_mode,s)
    I_mode,s = D_native - D_mode,s
    Q_mode,s = [(ell(actual_native) - ell(actual_mode,s))
                + (ell(mismatched_native) - ell(mismatched_mode,s))] / 2

I is a within-person difference-in-differences, Q the common degradation summary.
Report all four seed/mode conditions and their fixed two-seed arithmetic mean
per mode, retaining N=44 rather than doubling it. 1000 paired participant resamples,
seed731, give descriptive intervals conditional on fixed replies/assignments.
No population p-values or selected best corruption condition. No personality claim.

## Instrument and budget

Existing local Qwen2.5-1.5B-Instruct, revision
989aa7980e4cf806f80c7fef2b1adb7bc71aa306; model/template hashes identical to RC1.
MPS FP32, eager, eval, no KV; fixed response position768, execution length1024.
Repeat RC1's six nonprivate numerical/sham forwards before private text access.
Test exact permutation/inverse invariants on synthetic EN/JA native contexts too.
Then two native + eight altered-context forwards per person: 446 total (44*10+6),
hard cap466, 1800 seconds, MPS sampled allocation <=9GiB. Check before each forward.
Native mean/tail values must replay RC1 within1e-5; input body embeddings stay exact.

All inputs stay on Mac through the existing MEPS text-only whitelist. No mood,
questionnaire or external-validity labels; no training or API. The existing loader
scans/hashes mixed answer Markdown containers to extract permitted text only.
No raw text, token IDs, permutations or activations are persisted; only coded
identities, scalar logp/errors, counts and hashes. Fresh immutable run directory,
source/model/material manifests, per-case results and independent scalar audit.

## Interpretation and next decisions

- If I is consistently positive across both seeds, exact token counts alone cannot
  account for the full native advantage under this perturbation. This demonstrates
  order-sensitive context use, not logical reasoning, personality or a unique circuit.
- If D survives order destruction, retain a lexical/topic-compatible contribution;
  do not declare the surviving part "noise" or the lost part "personality".
- Q measures broad degradation, but differencing cannot guarantee cancellation of
  *differential* grammar/position effects. Unigram shuffles may split byte/subword
  sequences and are deliberately out-of-distribution negative controls. Block8
  preserves local sequences more than token shuffling, not grammatical sentences.
- If native/mismatched effects are unstable across seeds, report unresolved rather
  than searching a favorable seed. Continuous prefix access remains a distinct
  intervention question after this packet; no conclusions are borrowed from it.

Activation-patching research cautions that corruption and metrics affect mechanistic
conclusions: [Zhang & Nanda, ICLR 2024](https://arxiv.org/abs/2309.16042).
This packet manipulates observed token organization; it does not establish a
high-level personality causal abstraction.
