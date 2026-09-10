# NEXT49: transported GRAIN directions change evidence selection as well as writing

2026-09-10. Complete. Primary implementation frozen in `0cc293c` before capture
and prediction. Final analysis is **analysis_r2**, with an explicitly recorded
reporting-only correction.12 previously inspected author pairs, two queries,
whole:F01–04 and unchanged own/pooled shared readers; source4 leave-one-out,
new8 all4 source readers. No human labels or new model fitting.

## Main finding

On new8, the N-minus-confirmation contrast at attention22 is predominantly
carried by the key-side selection response. The propagated final-query response
is nonzero and sometimes counteracts selection; it cannot be globally omitted.
The full Q/K/V sum reconstructs96 native attention-increment vectors, relative
error<=5.272e-6 and maximum-coordinate error<=2.035e-6.

| Cohort/mode | Channel | Contrast RMS | Projection on native local contrast | Projection on final T |
|---|---|---:|---:|---:|
| new8 own | query | .021354 | .083093 | +.057603 |
| new8 own | key | .059927 | .903954 | +.251907 |
| new8 own | value | .009386 | .012952 | -.018569 |
| new8 own | sum | .064332 | .999999 | +.290940 |
| source4 own | query | .009530 | .561950 | -.076490 |
| source4 own | key | .005998 | .271327 | -.000128 |
| source4 own | value | .006331 | .166722 | +.041188 |
| source4 own | sum | .011664 | .999999 | -.035430 |

Pooled-mode local projections Q/K/V are .080606/.904960/.014433 for new8 and
.556486/.278364/.165149 for source4. Every cohort/mode/factor is retained in
the linked tables. These are **signed response projections**, not proportions
of personality information, explained human variance or independent causal effects.
Source4 and new8 also have different reader-exclusion contracts and material
distributions; their difference is not identified as a population psychology effect.

## Why query changes still matter

Omitting the query term from the linearized sum gives error/native-energy ratios
.110181/.110776 on new8, .667587/.658558 on source4 (own/pooled). Thus the old
BODY-only restriction is useful only when the final query is actually fixed.

Per-query accounting is especially informative:

| Cohort, own mode | Query | Q projection | K projection | V projection |
|---|---|---:|---:|---:|
| new8 | neuroticism | -.125849 | 1.125420 | .000428 |
| new8 | confirmation seeking | .444047 | .452388 | .103565 |
| source4 | neuroticism | .175235 | .465134 | .359633 |
| source4 | confirmation seeking | .279152 | .451770 | .269078 |

The denominator here is each query's native local action, not the between-query
contrast. For new8 neuroticism, query response counteracts part of the key effect;
for confirmation, both positively project onto the local action, while other
components of their vectors still cancel. Neither statement is a claim that the
model consciously reasons or that the user has a particular clinical trait.

Factor detail changes the picture further. New8 F03 Q/K/V projections onto its
local contrast are -.482240/1.339739/.142501. For F01 the Q projection is -.068574,
whereas F02/F04 are +.157421/+.352330. Do not delete the negative term as noise.
The full contrast second-moment/covariance account retains all cross-terms.

## What was calculated

At the actual pre-attention22 boundary, for each head:

    a = softmax(Q^T K / sqrt(d_head) + fixed_mask)
    o = sum_t a_t V_t
    doQ = sum_t a_t(V_t-o)(dQ^T K_t/sqrt(d_head))
    doK = sum_t a_t(V_t-o)(Q^T dK_t/sqrt(d_head))
    doV = sum_t a_t dV_t.

Actual input RMSNorm derivatives, projection biases, GQA grouping, RoPE, head
scale and output projection are included. Positions/masks are fixed. Residual
identity is excluded because the target is the attention increment. Directions
start only at H16 BODY positions, but blocks17–21 propagate them to all positions,
including the final query. No full model score/backward collection is performed.

The relevant operation is not just moving V: a GRAIN direction changes which
context states receive attention, weighted by how their values differ from the
currently attended average. This provides a concrete same-material connection
between GRAIN directions and query-conditioned inference. The direction and
upstream state still depend on frozen neural encoding; this is not yet an
independent raw-text personality score or a finite-removal intervention.

## Reproduction, correction and resource accounting

-96 prefix JVPs through blocks17–21 only,77.971897s native compute,80.622899s
  capture wall time,3.542885s loading separately. Under120s native cap. Repeated
  primal across four directions is exactly equal at every material. No live job.
- Numerical prediction1.569161s; replay1.581631s, both under60s.192 scalar
  actions/96 contrasts. Four target-denial probes and0 unexpected denials;
 0 scores/reverse-gradient collection/API/training/human-label reads.
-12 prediction arrays,6 corrected CSVs and1 check JSON replay exactly.
  Independent aggregate metric/energy error<=1.444e-15.
- **r1 report bug corrected:** numeric `key`/`query` channel names collided with
  identifier columns, overstating author counts and breaking query grouping.
  r2 uses `author_key`/`query_name`, uniqueness checks and a regression test.
  The primary vectors, primary numerical contrasts and cohort channel statistics
  are unchanged; no native or prediction rerun. Original r1 retained as superseded.
-116 scoped tests pass; Ruff and diff whitespace checks pass.20 existing
  Python3.14/PyTorch/SWIG warnings. No full-suite claim. The original freeze,
  revised-code hash and amendment scope are all stored; no silent freeze rewrite.

Private packet: `results/grain_vector_mechanisms/logic_upstream_attention_channels_v1/`.
Use `analysis_r2/summary.json`, `replay_audit.json`, `analysis_code_amendment.json`,
`capture_freeze.json` and prediction freezes. Publish only the four aggregate
CSVs: [channels](upstream_attention_channels_tables/channel_summary.csv),
[factors](upstream_attention_channels_tables/factor_channels.csv),
[query actions](upstream_attention_channels_tables/query_actions.csv),
[covariance](upstream_attention_channels_tables/covariance.csv).
Private matrices, author records, release/scorer/lockbox and MEPS remain unchanged.

## Next mainline decision

Keep all three channels; prioritize selection rather than a value-only shortcut.
NEXT50 asks whether a source-fixed query anchor can provide a reusable **key
selection function** on current evidence states. This directly tests the user's
query-trigger/navigation idea, not another MLP24 correction. Use existing captures,
0 neural calls, three fixed source-query choices and no fit. Keep runtime
dependencies explicit; after this one family, return to whole-chain integration.

See [mechanism theory](../../docs/GRAIN_UPSTREAM_ATTENTION_CHANNELS_THEORY_V1.md)
and [NEXT50 plan](../../docs/GRAIN_SOURCE_QUERY_SELECTION_PLAN_V1.md).

For a fresh packet with current code and the existing private lineage:

```bash
PY="../project persona/.venv/bin/python"
$PY scripts/run_grain_upstream_attention_channels.py prepare --output /private/fresh_packet
$PY scripts/run_grain_upstream_attention_channels.py capture --output /private/fresh_packet
$PY scripts/run_grain_upstream_attention_channels.py predict --output /private/fresh_packet
$PY scripts/run_grain_upstream_attention_channels.py evaluate --output /private/fresh_packet --name analysis
```

Do not recapture the completed default packet. Its evaluation-only r2 amendment
explicitly verifies original inputs and frozen predictions under the corrected
reporting code. Existing primary runtime remains governed by its original hash.
