# NEXT38: output seed versus suffix transport, cached-only mechanism packet

2026-09-10. Follow NEXT37 source-disjoint application. Decision: locate which
part of the conditional reader should be developed next, not tune8 authors.

## Fixed scope

- Same3 natural queries, old4 construction authors plus new8 source-disjoint
  reader authors. Keep cohorts and original query alphabets separate.
- Read existing candidate logits/probabilities/values, complete-gradient packets,
  model config and only the necessary frozen LM-head rows. No neural forwards,
  backwards, fits, API, questionnaire access or corpus expansion.
- Verify explicit packet addresses and checkpoint hashes. Ignore AppleDouble
  metadata in counts. Do not modify NEXT35-37 prediction freezes or alphabets.
- Numeric wall bound120s after loading; one exact numerical replay. If unavailable
  output weights/schema block an item, record it rather than rebuild a model.

## Algebra before numbers

For each natural query, p=softmax(z), F=v^T p, b=p*(v-F), s=W_A^T b.
Verify sum(b)=0, nativeF replay and a numerical softmax derivative test.
With final normalized hidden x, J_H=(D_H x)^T s. Output seed and suffix transport
are distinct factors of one chain rule, not independent causal contributions.

For binaryT/F only: J=p(1-p)*J_logit_contrast. Remove this known scalar at the
evaluator, retain saturation flags and the original score. Do not calibrate a
predictor with heldp or call this endpoint-free deployment. For multivalue
queries no analogous single scalar is presumed.

## Required outputs

1. Per-query/cohort output seed norms, probability saturation and gradient norms.
2. Within-author3 query-pair cosines at candidate seed, LM-head seed and full
   body gradient, with clearly different coordinate/address contracts. Candidate
   seed comparison only when alphabets match; no padding unrelated labels to
   invent common coordinates.
3. Source-fixed seed-subspace coverage for new authors. A shared digit alphabet
   alone can force low-dimensional output support; do not call that psychology.
4. Binary scalar-removal descriptive error account using source-fixed assets;
   no new ranks, alphas, author exclusions or gate fitting.
5. Explicit remaining ambiguity: one J/s pair leaves many compatible Jacobians.
   Specify the smallest follow-up contrast needed to distinguish answer coding
   from query-semantic suffix routing; do not silently launch it in this packet.

Private `logic_reader_output_seed_v1/`; public aggregate report/theory ledger.
Finish with an actionable context-reader or matched-answer intervention plan.
The whole-system goal remains ACTIVE; no claim of exhausting all mechanisms.

## Numerical detail fixed before NEXT38 computation

- Seed span is an uncentered SVD span with relative rank tolerance1e-10; this is
  descriptive source geometry, not another response fit. Publish rank and
  source-to-new captured energy, with source4 separate from new8.
- Binary comparison uses own-query models only. For component s, define
  g_s=mean_{i in its3 construction authors} p_i(1-p_i). A fixed contrast reader
  is the old component field divided by g_s, with no coefficient refitting.
  Old4 use their held-out component; new8 average the four converted fields.
- The native contrast target is current J/[p(1-p)] and is evaluator-only.
  Also divide both old prediction and target by current gain as a separately
  labeled error-reweighting diagnostic, never independent prediction.
  Keep original probability errors. Flag gain<1e-6; do not divide gain<=1e-12,
  retaining those unsupported rows and explicit counts. No clipping/search.
- Gradients were taken at the accepted canonicalized prefix. Candidate vectors
  come from the original prefix-capture logits; report their score discrepancy.
  Binary target gain uses the saved gradient-pass score. Multivalue candidate
  seeds therefore inherit the previous <=2e-6 scalar replay approximation; do
  not claim exact recovery of their full canonicalized probability vectors.
- Literature rationale: Zhang & Nanda, ICLR2024 / arXiv2309.16042, section4,
  compares probability versus logit-difference effects. This motivates keeping
  both objects, not retroactively replacing the frozen NEXT37 target.
