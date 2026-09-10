# NEXT49: full Q/K/V channels of transported GRAIN directions at attention22

2026-09-10. Planned, not executed. NEXT48 closes the MLP24-only refinement
family. This packet investigates a remaining upstream dependency instead of
fitting another local residual or increasing a correlation benchmark.

## Question, evidence and fixed scope

Why does attention22 contribute differently to the two-query transported
contrast in source4 and new8? NEXT45 own-mode signed final-T projections are
-.035430 and+.290940. Test whether the difference arrives through query-side
selection, key-side selection, value writing, or their signed combination.
This is a descriptive mechanism decomposition of a previously located stage,
not a preregistered blind discovery of that stage.

Use exactly the existing12 author pairs, neuroticism/confirmation-seeking,
whole:F01–04, source4 leave-one-out/new8 all4, and both frozen w* modes. No
human labels, prompt change, new sources, head-selection weights or fit.

Old `fixed_seed_attention.py` and `source_cotangent.py` are correct for a direct
BODY-only perturbation with the final query held fixed. Propagated tangents
before attention22 may also change the final query. Do not omit that channel
or reinterpret the old restricted decomposition as a full-path derivative.

## Exact computation to implement

For each actual attention head, let Q be the last-position query, K_t/V_t
the allowed source keys/values after actual projections and Q/K RoPE. Define:

    s_t = Q^T K_t / sqrt(d_head), a = softmax(s + fixed_mask)
    o = sum_t a_t V_t
    dsQ_t = dQ^T K_t / sqrt(d_head)
    dsK_t = Q^T dK_t / sqrt(d_head)
    doV = sum_t a_t dV_t
    doQ = sum_t a_t (V_t-o) dsQ_t
    doK = sum_t a_t (V_t-o) dsK_t
    do = doV + doQ + doK.

Use the actual stored scale if different from the displayed standard scale.
Multiply concatenated channel head outputs by the actual output projection;
retain all heads. Include RMSNorm's exact derivative before Q/K/V projections,
projection biases in primal values, GQA grouping, RoPE derivatives for fixed
positions, fixed causal mask and eval/dropout behavior. The residual identity
is excluded: the target is attention increment, not whole residual update.

These are first-order directional responses, not finite deletions, unique
causal paths or three independent psychological factors. A changed attention
query may already contain text evidence, so doQ is not pure conscious intent.

The basis is standard softmax differential and QK/OV composition:
[Vaswani et al.](https://arxiv.org/abs/1706.03762),
[Elhage et al.](https://transformer-circuits.pub/2021/framework/index.html).
[Kamath et al.](https://transformer-circuits.pub/2025/attention-qk/index.html)
show how query/key feature interactions can explain attention selection. Their
feature-attribution method motivates the split, not an existing validation of
our GRAIN directions or the same derivative experiment.

## Minimal implementation and data

1. Add `grain_research/vector_mechanisms/upstream_attention_channels.py` for
   normalized Q/K/V directional responses and exact three-channel accounting.
   Add `tests/test_grain_upstream_attention_channels.py`; tiny actual Qwen2
   parity must include nonzero final-query tangent, GQA, RoPE, causal mask and
   zero tangent. Also verify BODY-only reduction to the old restricted case.
2. Add `scripts/run_grain_upstream_attention_channels.py` with separate
   prepare/capture/predict/evaluate stages and frozen code/input hashes.
   Reuse NEXT42 H16 state, GRAIN directions and BODY positions. Do not treat
   saved last-position-only tangents as a full token sequence.
3. Capture full primal x and full tangent d immediately before attention22:
   execute blocks17–21 only,96 prefix JVPs (24 query materials ×4 directions),
   returning the full boundary. Each JVP provides its primal; avoid an extra
   24-forward collection. Verify repeated primal across the four directions.
   Export actual block22 Q/K/V/output matrices, biases, input norm, RoPE,
   scale and head metadata. Do not substitute MLP24/25 or attention27 assets.
4. Offline prediction emits channel vectors and w*-projected actions for all
   directions/modes. Freeze predictions before opening old native targets.
   Target: NEXT42 `updates[:,10]`, verified against stage name `attention22`.
   All-source w* for new8 and source exclusions must match NEXT44/45.
5. Evaluate full sum against the actual native attention22 increment, then
   query contrasts and signed final-T projections for Q/K/V and their sum.
   Publish every cohort/mode/factor, including negative contributions; head
   arrays remain private and cannot become a post-outcome best-head selector.

Private packet: `results/grain_vector_mechanisms/logic_upstream_attention_channels_v1/`.
Aggregate report: `reports/grain_vector_mechanisms/UPSTREAM_ATTENTION_CHANNELS_REPORT_V1.md`.
Outputs: full-vector parity, channel summary, per-factor channels, covariance/
cross-energy account, source-vs-new comparison, provenance, runtime/stop audit.

## Decisions, resource limits and finish condition

- First prove implementation on synthetic fixtures; use full-vector error
  relative<=1e-4 OR absolute<=1e-5, with the threshold fixed before native targets.
  If it fails, fix an implementation mismatch, not fit a scaling coefficient.
- Numeric native evidence reports channel RMS, signed projection, covariance,
  query contrasts, counts and paired actions. A sum-to-target identity validates
  accounting; the empirical channel distribution supplies the mechanism finding.
- A retained Q term changes the old BODY-only dependency map; a small Q term
  justifies using the simpler restriction at this stage. Strong K/V differences
  instead prioritize selection/writing extraction. No invented personality-r gate.
- Cap96 native prefix JVPs/120s native compute, loading timed separately;
 60s numerical analysis. Preserve terminal partial packets at the cap; do not
  silently extend it. Check wall time and scientific decision value between jobs.
  No score generation, reverse-mode gradient collection, API, training or Y read.
- One fixed attention location only. No queue restart, layer sweep, head pruning
  or word-level psychodynamic interpretation. Preserve release/scorer/lockbox,
  MEPS restrictions and unrelated worktree changes; local commits, no push.

This finite pass finishes when the observed channels and next integration choice
are documented, including a useful partial result if the resource cap is reached.
It does not require r=1 or identification of every neural path.
