# NEXT42: layer accumulation and fixed-depth early reading

2026-09-10. Planned, not executed. NEXT41 achieved source-fixed full H16 action
reading; most action was already carried by the final residual before block27
attention. Next locate useful earlier computation and test a smaller suffix,
not improve near-perfect last readout correlations.

## Decision and fixed candidates

Question: how early can current GRAIN directional response be read by the same
source t, and which signed updates create or adjust it? Reuse all24 units,
same H16 inputs, four directions, N/confirmation and own/pooled source t.
No newly constructed targets, people, source weights or human labels.

Three fixed exits: output of blocks20,23,26. Final reference remains the
post-attention27 residual. For each exit k:

    response_k = E_final D_H x_after_k[D_f]
    action_hat_k = source_t^T response_k.

The program for k20 stops after four remaining blocks17..20, k23 after seven,
k26 after ten. Its input still requires the original blocks0..16. Do not describe
this as four-layer raw-text encoding or equate suffix removal with whole-model
independence. No per-query post-hoc winner becomes the primary result.

## Exact accounting, not a new factor search

Capture the final-position JVP of every attention and MLP update in blocks17..26
and attention27. Sum21 increments plus initial final-token tangent (zero for
these BODY directions). This telescopes to the complete response. Each increment
includes previous transport; it is not an isolated module-ablation effect.

Keep query/layout and all intermediate token positions unchanged. Use the actual
norms, GQA/RoPE, nonlinear MLP and source assets. Numerical body direction and
unit normalization must match NEXT41, not recompute a new factor at each layer.

## Execution packet

1. Test exact residual-update/JVP telescoping in a small real-Qwen2 architecture.
2. Freeze code/design before current H16 target evaluation; no source fitting.
3. At most96 JVPs, same24 units x4 directions. One augmented forward returns all
   exit responses and21 update vectors, avoiding independent reruns per depth.
   No new source backward, model score or new-author gradient evaluation.
4. Target-denied process freezes responses, all four declared reads and signed
   increments. Then compare full reference response with NEXT41 saved response:
   relativeL2<=1e-4 or maxabsolute<=1e-5. If mismatched, inspect actual operator
   order before claiming an early-exit finding; do not tune the tolerance.
5. Evaluate r, factor-centered r, SSE/zero, RMSE, amplitude and signs at every
   exit, separately old4/new8 and query/factor. Report incremental and cumulative
   signed actions, including cancellation; no positive-only importance plot.
6. Demonstrate the declared k20 truncated function reproduces its captured exit
   response on one fixed source unit, all4 directions: at most4 extra JVPs.
   This is a real early-exit implementation check, not a full second model run.

Native cap300s after loading, maximum100 JVP evaluations. Benchmark the first
complete author; preserve resumable unit IDs if forecast exceeds the cap. No
automatic extra model, API, optimizer or device fallback. Source assets are
reused; runtime score access remains forbidden. Targets/labels do not choose K.

## How the result changes the mainline

If fixed early exits retain useful action, retain the earliest empirically useful
candidate as exploratory and next test it on source-disjoint new material before
production selection; keep all fixed exits in the report. This starts removing
computational dependencies, not just narrating layer importance.

If action is formed late, identify the signed stage where it emerges or changes,
then compare a source-only coordinate translator versus necessary contextual
computation in a later packet. A low early correlation is not evidence that the
earlier hidden state contains no information. Do not use questionnaire accuracy
to choose this mechanism experiment's parameters.

Related primary method: [Tuned Lens, §§2-3](https://arxiv.org/html/2303.08112v6)
models residual updates and layer-specific affine translation. Here source t
remains fixed; no translator is trained in NEXT42. We test a directional response,
not vocabulary distributions or human psychological labels.

## Deliverables

Proposed `run_grain_layer_accumulation.py`, supporting module/synthetic tests,
private `logic_layer_accumulation_v1/`, aggregate
`LAYER_ACCUMULATION_REPORT_V1.md`, source/code hashes, timing and complete exit
tables. Update input dependencies and logic ledger. No release, lockbox, scoring
or interpretation-rubric changes; no raw personal data in public outputs.
