# GRAIN navigation round 2: context and functional exchange

2026-09-09. Overall NAV-01--12 goal remains **ACTIVE**. This is a mechanism
development result, not a model-selection leaderboard or completion of route A.

**Key finding:** the four-block contextual program connects to the 46-person
free-chat low-state criterion at r=0.3506
[0.0382, 0.6013], with original
function fidelity r=0.5117. This recovers part of the state
signal missing from static input-token geometry. T/F behaves differently: its
four-block human AUC is 0.5355, below the earlier token-only
semantic/order candidate's .7047. We preserve both paths rather than forcing all
targets into one representation recipe. These are associations of the frozen
candidates, not a significant paired improvement claim. No human labels were
used to construct or select coefficients.

## 1. What changed

The token-only first-order operator from round 1 linked to T/F but lost
extraversion and low-state behavior. This round retains exact **1, 4 or 8 early
contextual blocks**, pulls the two fixed projections back to those states, and
tests fixed local charts. Construction uses only six synthetic reference texts.
There is no fitting on human labels, no MEPS construction and no original lockbox
access. Every method, including both individual charts, was scored before the
external label join. Reported mean-chart results are the fixed common comparator,
not a method selected on real scores; all twelve candidates are in the CSVs.

The complete runtime comparison loads eight extracted blocks and records states
at depths 1/4/8 in a single forward. A depth-1 score depends only on the first
state, but total measured batch cost still includes eight blocks. The original
28-layer model, its language-model head and the full 17-layer P/Q input are not
loaded at scoring time. This is an **extracted-neural-program**, not a claim of
Transformer-free inference. Its retained weight file is
1.216 GB; pretrained knowledge remains essential.

## 2. Explicit computation

For early cut k, let H_k(x,q) be the retained contextual residual sequence.
At synthetic chart j, the two fixed projection derivatives are

```
J[j,a,k] = derivative of z_a with respect to H_k at reference text x_j
z_hat[j,a,k](x) = z_a(x_j)
                 + <J_evidence, A_j(H_k(x)) - A_j(H_k(x_j))>
                 + <J_query, H_k(x)[P] - H_k(x_j)[P]>
                 + <J_endpoint, H_k(x)[Q] - H_k(x_j)[Q]>
f_hat[k](x,q) = b_q + sum_a sum_j omega_j(x) z_hat[j,a,k](x)
```

Evidence correspondence A uses cosine softmax with fixed tau=8 and relative
position penalty=4. P and Q are exact **within-query** token-slot matches.
Weights are either 1/2 or softmax(8 * evidence-mean cosine to each reference).
No manual scene truth is required by the scorer. Chain-rule derivatives and
reference identity are exact; transporting those derivatives to new contexts is
the tested local approximation. Full-state P/Q reconstruction remains open.

Mathematical provenance: gradient-based attribution motivates the derivative
experiment, but this point-Jacobian operator is **not Integrated Gradients** and
does not inherit its path-integral guarantees. See
[Sundararajan et al., 2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
Construction-only gradient route ranking followed by actual interventions follows
the distinction used in
[Syed et al., 2024](https://aclanthology.org/2024.blackboxnlp-1.25/).
Cross-query endpoint-delta transport and the chart comparison are our tested
combinations, not claims that those papers proved this personality mechanism.

## 3. Held controlled scenes

Here r measures agreement with the full frozen function, **not human validity**.
Each row has 24 renderings of only three held scene skeletons. Magnitude errors
are reported alongside correlation; better ordering does not imply score identity.

| query | method | r | mae | native_r |
|---|---|---|---|---|
| TF_cont | context1_mean | -0.1876 | 0.0886 | -0.1187 |
| TF_cont | context4_mean | 0.8943 | 0.0985 | 0.8611 |
| TF_cont | context8_mean | 0.7837 | 0.1818 | 0.6865 |
| adapted_dejected5 | context1_mean | 0.7266 | 0.1956 | 0.6919 |
| adapted_dejected5 | context4_mean | 0.9518 | 0.1922 | 0.9403 |
| adapted_dejected5 | context8_mean | 0.8991 | 0.1742 | 0.9018 |
| extraversion | context1_mean | -0.1546 | 0.1541 | -0.0992 |
| extraversion | context4_mean | 0.4041 | 0.1474 | 0.4485 |
| extraversion | context8_mean | 0.1145 | 0.1418 | 0.1477 |

The relationship between retained depth and fidelity is not monotone. In
particular, four contextual blocks recover much more low-state and T/F ordering
than the one-block candidate, but the scalar levels still have substantial bias.
More context alone is not a substitute for an appropriate approximation to the
remaining nonlinear computation.

## 4. Immediate real-text connection

Same already-exposed cohorts as round 1: 213 Big5-source authors, 64 MBTI-source
authors and 46 counseling participants, with three source views averaged per
author. These are exploratory endpoint checks, not a new full PANDORA benchmark.
MEPS criteria are never used to update paths, gates, references or weights.

| dataset | query | method | n | full_function_r | function_external_r | function_auc |
|---|---|---|---|---|---|---|
| PANDORA_source | TF_cont | context1_mean | 64 | 0.3590 | 0.1928 | 0.5816 |
| PANDORA_source | TF_cont | context4_mean | 64 | 0.4508 | 0.0276 | 0.5355 |
| PANDORA_source | TF_cont | context8_mean | 64 | 0.4101 | -0.0754 | 0.4665 |
| PANDORA_source | extraversion | context1_mean | 213 | 0.0098 | 0.0360 | n/a |
| PANDORA_source | extraversion | context4_mean | 213 | 0.1110 | 0.0901 | n/a |
| PANDORA_source | extraversion | context8_mean | 213 | 0.0443 | 0.0650 | n/a |
| direct_answers | adapted_dejected5 | context1_mean | 46 | -0.1114 | 0.0447 | n/a |
| direct_answers | adapted_dejected5 | context4_mean | 46 | 0.3239 | -0.0358 | n/a |
| direct_answers | adapted_dejected5 | context8_mean | 46 | 0.5734 | -0.0644 | n/a |
| free_chat | adapted_dejected5 | context1_mean | 46 | 0.0361 | 0.0912 | n/a |
| free_chat | adapted_dejected5 | context4_mean | 46 | 0.5117 | 0.3506 | n/a |
| free_chat | adapted_dejected5 | context8_mean | 46 | 0.5571 | 0.3250 | n/a |

Complete intervals and paired native comparisons are in `NAV-10/context_external_r2/metrics.csv`.
Model fidelity and human association remain separate outcomes. Retain useful
local links without presenting every approximation as the same inference program.

## 5. Shared route identity is not shared use

162 actual native-checked interventions apply the source query's **edit delta**
at matched attention-head/MLP coordinates, only at the last input position. This
avoids inventing alignment between query strings with different numbers of P rows.
Within-query deltas on exactly the same site group are the transport control.
There are six held event edits from three scenes, not 162 independent subjects.

| query_a | query_b | candidate_jaccard | endpoint_delta_cka | native_response_r |
|---|---|---|---|---|
| TF_cont | extraversion | 0.1429 | 0.9906 | 0.1140 |
| TF_cont | adapted_dejected5 | 0.3333 | 0.9362 | -0.7319 |
| adapted_dejected5 | extraversion | 0.3333 | 0.9390 | -0.5041 |

| target_query | source_query | native_r | mean_abs_own | mean_abs_cross | sign_agreement |
|---|---|---|---|---|---|
| TF_cont | adapted_dejected5 | -0.4187 | 0.0038 | 0.0032 | 0.8333 |
| TF_cont | extraversion | 0.8879 | 0.0026 | 0.0016 | 0.6667 |
| adapted_dejected5 | TF_cont | 0.1067 | 0.0014 | 0.0012 | 0.5000 |
| adapted_dejected5 | extraversion | 0.2448 | 0.0006 | 0.0009 | 0.3333 |
| extraversion | TF_cont | 0.8407 | 0.0008 | 0.0009 | 1.0000 |
| extraversion | adapted_dejected5 | 0.6021 | 0.0003 | 0.0008 | 0.3333 |

In the second table r compares cross-query versus own-query intervention effects;
it is neither personality correlation nor whole-model recovery. High CKA can
coexist with different signs and very small native effects. Shared sites are a
place to investigate, not a ready-made shared semantic axis. The remaining
NAV-06 real-author GRAIN-group analysis has not been replaced by these synthetic
conditions.

## 6. State opportunity, paired on the same criterion

Both material views are now compared with the **identical** questionnaire endpoint
and participant set. This removes the earlier apples-to-oranges endpoint timing
comparison. It does not randomize materials or turn observational differences into
causal effects. The three overlapping negative-state composites are not independent
replications of each other.

| criterion | free_r | direct_r | delta_r_free_minus_direct | delta_lower | delta_upper |
|---|---|---|---|---|---|
| baseline_adapted_dejected5 | 0.1719 | 0.0784 | 0.0936 | -0.2326 | 0.4288 |
| mid_adapted_dejected5 | 0.3053 | 0.1316 | 0.1737 | -0.1625 | 0.4838 |
| post_adapted_dejected5 | 0.5690 | 0.2590 | 0.3101 | 0.0298 | 0.5764 |
| pre_adapted_dejected5 | 0.1893 | 0.1939 | -0.0046 | -0.3220 | 0.3117 |

| view | before_r | after_r | partial_r_after_given_before | partial_p | score_vs_criterion_change_r |
|---|---|---|---|---|---|
| direct_answers | 0.1939 | 0.1316 | 0.0044 | 0.9772 | -0.1117 |
| free_chat | 0.3053 | 0.5690 | 0.5050 | 0.0004 | 0.3242 |

The free-chat score retains a relationship with the post-session state after
conditioning descriptively on the preceding state. The direct-answer material
does not show the same pattern. The joint view/time difference remains jointly
confounded; it is saved as such, not interpreted as isolated personality change.
The controlled time-marker/event-order arm of NAV-07 remains open.

## 7. Local charts and signed interactions

We retain the full rectangle rather than deleting its non-additive part:

```
I = f(other,assigned) - f(self,assigned)
    - f(other,voluntary) + f(self,voluntary)
```

| query | method | r | mean_abs_target | mean_abs_predicted | sign_agreement |
|---|---|---|---|---|---|
| TF_cont | context1_mean | -0.1019 | 0.0152 | 0.0021 | 0.3333 |
| TF_cont | context4_mean | -0.6141 | 0.0152 | 0.0033 | 0.3333 |
| TF_cont | context8_mean | 0.5040 | 0.0152 | 0.0040 | 0.5000 |
| adapted_dejected5 | context1_mean | -0.1240 | 0.0062 | 0.0003 | 0.6667 |
| adapted_dejected5 | context4_mean | 0.8920 | 0.0062 | 0.0006 | 0.6667 |
| adapted_dejected5 | context8_mean | 0.6383 | 0.0062 | 0.0010 | 0.8333 |
| extraversion | context1_mean | -0.2174 | 0.0033 | 0.0011 | 0.5000 |
| extraversion | context4_mean | -0.4829 | 0.0033 | 0.0010 | 0.3333 |
| extraversion | context8_mean | -0.1862 | 0.0033 | 0.0019 | 0.3333 |

| query | cut | chart_correlation | mean_abs_chart_disagreement | affinity_minus_mean_mae |
|---|---|---|---|---|
| TF_cont | 0 | 0.5869 | 0.0225 | 0.0000 |
| TF_cont | 3 | 0.5303 | 0.0351 | 0.0000 |
| TF_cont | 7 | 0.9329 | 0.1194 | 0.0000 |
| adapted_dejected5 | 0 | 0.8755 | 0.2643 | 0.0003 |
| adapted_dejected5 | 3 | 0.8730 | 0.2240 | 0.0000 |
| adapted_dejected5 | 7 | 0.8972 | 0.1922 | 0.0000 |
| extraversion | 0 | 0.2356 | 0.2906 | 0.0004 |
| extraversion | 3 | -0.1275 | 0.2415 | 0.0000 |
| extraversion | 7 | -0.3607 | 0.2232 | 0.0000 |

The local chart average and observable-affinity gate are both actually tested.
High scalar ordering does not ensure correct mixed interactions; those terms
remain the next operator-development target rather than being relabeled noise.
The first local-atlas packet is complete, not a universal smooth field claim.

## 8. Execution and next scientific decisions

- NAV-06 local exchange complete; GRAIN real-author grouping still open.
- NAV-03 three contextual depths implemented and connected to real text;
  route A effective P/Q state construction still open, not replaced by B.
- NAV-07 paired observational packet complete; controlled time/order arm open.
- NAV-12 first local chart/interaction comparison complete; retain failures.
- Next prioritize actual context-conditioned interaction construction, natural
  real-text edits and query-identity factorial. No real-label parameter grid.
- NAV-08 reference, NAV-09 exact-prefix reuse and NAV-11 raw-text export retain
  their original scope in the active goal register.

The first contextual export (`context_pullback_r1`) failed before any evaluation
because its layer_types metadata still described 28 layers after cropping to 8.
The config helper was corrected, regression-tested and a fresh `r2` was built.
Six failed-export construction calls remain accounted for; no result was erased.

## Artifacts and replay

All result paths below are relative to `results/grain_vector_mechanisms/navigation_v1/`.
The public report contains only aggregates. Private author tokens, questionnaire
joins and pretrained weights remain gitignored locally.

| Packet | Directory | Key output |
|---|---|---|
| Shared mechanism | NAV-06/endpoint_exchange_r1 | shared_specific_edges.csv, cross_vs_own.csv, response_kernel.csv |
| Shallow program | NAV-03/context_pullback_r2 | program.json, context.safetensors, charts/ |
| Controlled check | NAV-10/context_worlds_r2 | metrics.csv, prediction_freeze.json |
| Real check | NAV-10/context_external_r2 | metrics.csv, prediction_freeze.json |
| State opportunity | NAV-07/state_opportunity_r1 | paired_material_comparison.csv, occasion_specificity.csv |
| Local atlas | NAV-12/context_atlas_r1 | conditional_signed_terms.csv, response_transfer.csv, overlap_consistency.csv |

Use the existing parent-project Python environment. Constructors require new run
IDs; neural evaluations are resumable with unchanged source/code hashes.

```bash
python scripts/analyze_grain_navigation_sharing.py
python scripts/run_grain_context_program.py construct --program <fresh-program-dir>
python scripts/run_grain_context_program.py worlds --program <program-dir> --out <fresh-worlds-dir>
python scripts/run_grain_context_program.py external --program <program-dir> --out <fresh-external-dir>
python scripts/analyze_grain_state_opportunity.py --out <fresh-state-dir>
python scripts/analyze_grain_local_route_atlas.py --out <fresh-atlas-dir>
python scripts/summarize_grain_navigation_round2.py
```

Aggregate provenance is recorded in `round2_evidence.json`; code tests cover
gradient finite differences, exact shallow/full early-state replay, chart identity,
different-query-length endpoint exchange, signed factorial accounting, and paired
participant/criterion alignment. Full goal closure requires the remaining ledger
items; this report does not mark them complete.
