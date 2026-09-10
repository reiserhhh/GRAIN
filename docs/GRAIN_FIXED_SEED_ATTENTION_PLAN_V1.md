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
