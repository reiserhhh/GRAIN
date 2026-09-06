# VM06-RC2: fixed-site response-state interventions

2026-09-06. Frozen before RC2 private neural forwards. Follows the completed
RC1 context experiment, not an independent replication sample. No new training,
questionnaires, personality labels, text exports, external API, or scorer changes.

## Question and mathematical scope

At layer l the full causal model state contains both prefix C_l and current-user
states U_l, along with attention mask M and positions P. The remaining computation
is F_l(C_l, U_l; M, P). RC1's block10 transplant changed only U_l. Its small
effect cannot establish that context is unused or absent from U_l, because later
layers still access the mismatched C_l. The relevant intervention is:

    delta_l = ell_tail(F_l(C_l^m, U_l^a)) - ell_tail(F_l(C_l^m, U_l^m))
    gap = ell_tail(actual context) - ell_tail(matched context)

Here a/m denote actual/mismatched contexts. Tokenwise teacher forcing is unchanged.
Tail means response tokens 2 through T; token 1's predictor is outside the patched
body. No per-person delta/gap ratios: individual denominators can be near zero.
Report raw paired deltas, residual gap and a descriptive ratio of group means only.
Effects can be negative, nonmonotone or exceed the gap; do not clip them into a
"percentage of reasoning" or call them unique causal mediation.

The final RMSNorm and vocabulary projection act separately at each position.
Replacing final-normalized body states must reproduce actual tail-token logp.
Replacing the last decoder block must do likewise after the unchanged final norm.
These are positive instrumentation controls, not discoveries of semantic circuits.

## Fixed material, model, sites

- Reuse exactly RC1's 44 response/actual-context pairs and matched donor assignments.
  Compare the rebuilt metadata manifest byte-equivalent in canonical JSON values.
  No selecting new responses, exclusions or donors based on effects.
- Same pinned local Qwen2.5-1.5B-Instruct (28 blocks), revision
  989aa7980e4cf806f80c7fef2b1adb7bc71aa306, MPS FP32/eager/eval/no KV.
  Same 1024 sequence length, response start 768, masks and position IDs.
- Zero-based block outputs: 4, 10, 16, 22, 27; final_norm. All six reported.
  Each case has two unmodified forwards plus actual-to-mismatched and mismatched
  self-patch at each site: 14 forwards. No learned direction or layer selection.
- All current-user states may be held in memory; persist only scalar errors,
  paired likelihoods, lengths and coded identities. Never save raw text, token IDs
  or activation tensors. Keep the existing MEPS text-only whitelist and disclose
  that mixed answer Markdown bytes are scanned by the source loader.

## Checks and acceptance

First run RC1's six nonprivate numerical checks. Add two nonprivate EN/JA examples
with actual/mismatched prefixes and final_norm patch (six more forwards).
For both: actual tail must replay within 1e-5 nat per token, while token1 stays at
the mismatched prediction. Technical failure stops before private text access.

For all 44 people: all six self-patches must replay exactly; token1 must remain
unchanged in every body-only patch; block27 and final_norm tail errors <=1e-5.
Unmodified actual/matched mean/tail values must replay RC1 within 1e-5; input
embedding means must be equal. Failures remain visible, not dropped as outliers.

Compute 1000 paired participant bootstrap draws, seed731, conditional on these
fixed responses and donor assignments. They are descriptive intervals, not
population-level p-values, psychological validity, or multiplicity-adjusted claims.
For intermediate layers also report the within-person consecutive layer contrasts.
No selected "best layer" is promoted. Last-block/normalization equality is expected.

Budget: 628 forwards for 44 cases (12 technical + 14*44), hard cap 656 for at most
46; 2400 seconds; MPS sampled allocated memory <=9 GiB. Check before each forward.
Fresh output directory, immutable source/model/material hashes, incremental scalar
CSV/status, sealed completion manifest. Do not restart while another RC2 is live.

## Interpretation and next decision

A depth profile can localize *when current-token states become sufficient under
this intervention*, not which human attribute the model inferred. A weak early
profile plus exact late restoration suggests a need to model repeated prefix access
and state composition rather than rely on one pooled embedding. It does not by
itself prove that repeated prefix access is the cause: nonlinear/off-distribution
interchange effects and vocabulary matching remain alternatives. If the instrument
passes, next bounded diagnostic is prefix/order or context-token access, keeping
the reply fixed. Personality linkage remains a separate source/bridge evaluation.

## Methodological precedents

- Zhang & Nanda, *Towards Best Practices of Activation Patching in Language Models:
  Metrics and Methods*: corruption and metric choices affect localization. This
  motivates fixed donors, raw logp changes and explicit site/mask controls.
  https://arxiv.org/abs/2309.16042
- Geiger et al., *Causal Abstractions of Neural Networks*: interchange interventions
  need a specified high-level causal model. We do not claim a personality abstraction
  from generic state replacement or from a recoverable linear probe.
  https://arxiv.org/abs/2106.02997
