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
