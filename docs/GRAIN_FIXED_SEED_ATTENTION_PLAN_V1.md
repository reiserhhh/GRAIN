# NEXT39: common output seed and an actual attention pullback

2026-09-10. NEXT38 found N/confirmation seed cosines .943/.954, versus body
gradient .704/.649; binary probability scaling does not rescue reader transport.
Next develop the conditional transport, not another rank/alpha grid.

## Fixed inputs and targets

- Same old4/new8 authors from NEXT37/38, original complete material and native
  query-last states. Only neuroticism and confirmation_seeking; no new selection.
- Reuse complete block16 states, common-prefix canonicalization, token positions
  and frozen GRAIN scores. No new prefixes or raw corpus extraction.
- Candidate digits are exactly the existing0-9 tokens. Set c=v-mean(v), then
  c=c/||c|| once, with positive ordinal orientation. Target L=c^T z, not the
  original probability F. Its final-hidden seed W_A^T c is identical for both
  queries, every author and every trial. Do not estimate c from outcomes.
- Retain original F/J results as separate comparators. Candidate logit units,
  normalized hidden units and block16 units must not be merged.

## Primary: fixed seed transported through natural query conditions

Evaluate grad_H16 L for12x2 units using cached full states and the frozen suffix.
Compare N/confirmation body-field cosine, norm ratio, mean-reader cosine and
GRAIN directional actions within each author, with source4/new8 separate.
Matching s removes the output-seed difference; remaining differences are
query-conditioned transport, including query state AND layout, not pure semantics.
Do not change prompts to balance length during this packet.

## Development: exact last-attention value/key decomposition

During those same suffix evaluations, capture the last decoder block's
pre-attention normalized hidden states, projected Q/K/V, native RoPE arguments,
and the cotangent at the input of its output projection. No extra model pass.

Use `attention_pullback.py` on the final-position query for each actual head.
Reverse RoPE for key cotangents, sum repeated GQA key/value heads, and apply
the frozen projection transposes. Compare value+key paths with the native
gradient of the last block's normalized input on selected BODY positions.
Those body positions do not generate the final query; direct last-token query
and residual paths are outside this local body comparison. Retain this explicit
scope rather than dropping inconvenient paths from a claimed full Jacobian.

Contract both components with frozen GRAIN-modulated directions in this LAST
NORMALIZED INPUT plane. Do not substitute them for earlier H16 directions.
Report signed value/key contributions and their cancellation, by query/factor.
If native architecture has additional Q/K normalization or unsupported attention,
record the precise unsupported feature and keep the completed fixed-seed packet;
do not silently replace the observer with a simpler model.

## Numerical boundaries and outputs

-24 suffix forwards with backwards; fixed first new-author whole:F01 at+/- .001
  for each query adds4 suffix forwards. Cap28 forwards/24 backwards, native300s
  excluding loading. Forecast after first complete author; persist cumulative
  calls and explicit packet IDs, ignore AppleDouble sidecars.
- Verify old checkpoint, observer dtype/device/template/attention versions.
  In the same call, replay original probability within2e-6 of old canonical score.
  Analytic last-normalized BODY gradient parity target relativeL2<=1e-3 and
  max absolute<=1e-5, allowing documented float32 roundoff. No tolerance search.
  Report fixed-seed finite-difference error in its own logit units.
- No neural parameter updates, humanY, target-fitted donor weights, API, new
  external data or lockbox changes. No auto-expanded runs to fill additional heads.
- Private `logic_fixed_seed_attention_v1/`; aggregate field/action/channel report,
  analytic tests and one cache-only numeric replay. No duplicate neural replay.

## Decision

If fixed-seed response retains query differences, develop context/query-conditioned
transport rather than a common fixed source vector alone. If differences largely
shrink, preserve an output-coding/shared-transport simplification as a candidate.
In either case, successful analytic parity supplies actual value/key terms for a
mechanistic reader. Use those terms to define a next bounded source-fixed program;
do not infer personality meaning from a head name or high field cosine.

This is a finite local packet in an ACTIVE whole-system goal, not its completion.

## Implementation fixed before collection

Pinned local Qwen2 uses12 query heads,2 key/value heads,head dimension128,
28 decoder blocks; analyze block27 normalized input, separate from block16
output. No sliding attention/QK normalization; the final query admits every
position. Native code uses biases in Q/K/V, captured before rotary application.
The numerical reconstruction includes those biases through captured activations.

For each head h in key/value group b(h), with rotary map R_j, aggregate:

    J_value(j) = sum_h a_hj W_V,b(h)^T g_h
    J_key(j) = sum_h a_hj ((v_b(h),j-o_h)^T g_h)
               * W_K,b(h)^T R_j^T R_last q_h / sqrt(d).

The transpose is implemented as diag(cos) - half_turn @ diag(sin), not an
assumed unrotated key gradient. Same-group contributions SUM; they are not
averaged. g comes from native output-projection input cotangents, so includes
downstream effects. This is a decomposed local derivative, not an independently
estimated g or a complete replacement for the model suffix.

Standard sources: [attention](https://arxiv.org/html/1706.03762v7),
[GQA §2.2](https://arxiv.org/html/2305.13245v3),
[RoPE §3](https://arxiv.org/html/2104.09864v5). The local installed Qwen2 source
and its hash, rather than a generic implementation assumption, define execution.
Use separate prepare/collect/analyze stages; replay analysis from cached arrays
without another neural pass. Public outputs contain cohort aggregates only.
