# NEXT41: complete upstream transport to a source-fixed final readout

2026-09-10. Planned, not executed. NEXT40 established source-fixed local
attention reading; now change the computational interface, not the source-g
hyperparameters. Preserve the useful local result and all existing human-criterion
and GRAIN measurement tracks.

## 1. Question and mathematical target

Can a source-fixed downstream direction read the full block16 GRAIN action when
the actual context, final-query and residual pathways are propagated forward?
Use the same12 authors, N/confirmation, common digit-logit L and frozen materials.

H is the complete output of block index16. Let X(H) be the complete input to
last block27; N1 is its input norm, o_l its final-position attention output and
r the final-position post-attention residual before norm/MLP:

    r(H) = X_l(H) + W_O o_l(N1(X(H)))
    t = grad_r L
    dot_r_f = D_H r[D_f] = dot_X_l + W_O dot_o_l
    tau_f = t^T dot_r_f
    tau_hat_f = mean_source(t)^T dot_r_f.

D_f is the existing block16 BODY GRAIN direction at the unperturbed input,
padded with zeros elsewhere and held fixed while differentiating. It propagates
to all later positions. Do not restrict later derivatives to the original BODY
mask or discard changes in the final query. The scalar target remains a local
derivative, not a finite intervention, probability F or human label.

## 2. Source construction, before current target evaluation

1. Run a small synthetic native-shaped test of forward JVP support first.
   Check JVP/VJP duality and residual plus attention decomposition; no pretrained
   calls needed. Inspect pinned torch/model implementations rather than assuming
   hooks, rotary positions or an attention kernel support forward AD.
2. At most8 source suffix forwards/backwards (old4 x2queries) capture t directly
   at r. Verify W_O^T t agrees with NEXT39 source g and the fixed L replay. Do
   not invert W_O to recover t or reconstruct t from a current-author gradient.
3. Freeze same-query and two-query pooled means. Old4 uses other3 authors;
   new8 uses all4. No fit, alignment, source reselection or label-dependent weights.
4. Export current full H16, fixed masks/kwargs/GRAIN directions and source means
   to a target-denied inference process. New-author t/g, native gradient, logits,
   scores and questionnaire files are not scoring inputs.

Source capture parity targets retain existing numerical tolerances: canonical
L absolute error<=2e-6; projected source g relativeL2<=1e-4 or absolute<=1e-7.
Any violation is a capture/definition investigation, not permission to change
weights or silently widen the criterion after outcomes.

## 3. Bounded computation

-Evaluate24 author/query units x4 directions = at most96 forward JVPs to r.
 The scoring function stops before the last MLP/final norm/output head. Its
 upstream layers are still the native model; this is not an LLM-free encoder.
-For the first new author and F01, run fixed central differences at epsilon
 .001 for both queries:4 additional ordinary forwards to r. Compare vectors,
 not only a favorable scalar sign. Record relative error and max absolute error;
 use<=.05 relative or<=1e-4 absolute as a coarse float32 implementation check.
 If this fails, retain the result as diagnostic and inspect implementation before
 claiming transport parity. No epsilon search in this pass.
-Budget300s native execution after loading. Count8 source forwards/8 backwards,
 96 JVP evaluations and4 finite-difference forwards separately; a JVP is not
 assumed to cost one normal forward. Benchmark the first complete author and
 persist unit IDs/costs. Pause resumably if the cap will be exceeded; no automatic
 device change, new neural fallback or unbounded queue.
-If forward AD is unsupported, record the exact kernel boundary after the small
 pilot. Propose a specific supported implementation before recollecting, rather
 than replacing forward inference with current-gradient access.

## 4. Frozen evaluation and useful comparisons

After prediction freeze, compare with existing NEXT39 block16 gradient:D_f.
Report own/pooled source programs without choosing one per query on evaluation.

-Action r, factor-centered r, SSE/zero, RMSE, native/predicted amplitudes and
 signed agreement, separately old4/new8 and each query/factor.
-Complete dot_r versus its dot_X_l and W_O dot_o_l components. Retain signed
 cancellation; attention-only is an omission control, not an alternate fitted head.
-Where the evaluator collects actual current t, keep it evaluation-only and
 count any additional neural calls in a separately approved next packet. This
 pass does not need such collection to compare with existing H16 targets.
-Verify repeated current prediction packets from one declared smoke unit;
 freeze all24 before opening native H16 targets. Do not demand a second complete
 expensive model run merely to duplicate a table.

This pass has a mechanistic progression rule, not a personality accuracy gate:
if the complete source-fixed program reads useful new-author H16 action, retain
it and next develop a lighter input/transport realization. If it loses the local
advantage, locate the residual/query route contribution first; do not erase
NEXT40 or restart a source-alpha/rank sweep. Compare only the same target/layer.

## 5. Deliverables and boundaries

Proposed implementation `scripts/run_grain_complete_transport_jvp.py`, supporting
module/tests as needed; private `logic_complete_transport_jvp_v1/` with source
capture and prediction manifests, timing, hashes, per-unit arrays and summary.
Public `COMPLETE_TRANSPORT_JVP_REPORT_V1.md` plus dependency/goal updates.
Freeze actual implementation and any feasibility-driven plan amendment before
native target evaluation. No raw text/IDs in public aggregates.

Zero new labels, APIs, MEPS reads, output generation, lockbox access or scorer
changes. No new optimizer. Existing GRAIN codes and user/condition/material
addresses remain intact. This closes one upstream computational link, not all
human personality inference or the whole-system goal.

## Implementation clarification before native outcomes

The small real-Qwen2 synthetic network passes CPU and MPS forward-AD tests;
native eager kernels are retained. Source t is differentiated only through the
last tail from its captured residual state (8 source backwards), not by inverting
W_O. The last MLP is evaluated at the final position, with source score/g parity
checked against the original full-sequence implementation before prediction.

The planned smoke replay re-contracts saved JVP vectors with fixed source means;
it adds no native JVP beyond the96-call cap. All24 saved contractions can be
replayed numerically at negligible cost. The two prescribed vector finite
differences test the native JVP itself. No second full native run is planned.
Prediction is a separate process with undeclared data reads denied after model
loading; trusted Python/shared-library code may load lazily. Targets are opened
only by the later evaluator. No outputs have been evaluated when this clarification
is committed; capture failure requires an explicit correction, not relaxed parity.

Implementation method: [PyTorch forward-mode JVP](https://docs.pytorch.org/docs/2.11/generated/torch.func.jvp.html).
No new autodiff library or neural training is required.
