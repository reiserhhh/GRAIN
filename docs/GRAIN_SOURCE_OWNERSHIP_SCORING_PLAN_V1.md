# Quote-source ownership to measurement scoring

2026-09-13. C03/C11/C16 continuation of the active all-plan goal. Frozen before
classification or new native inference; no new goal, training or MEPS access.

## Decision

The previous600-call narrow_actions_r1 packet found target response r=.856285,
but friend quotes retained92.1% of native self action. Connect an existing
source/event evidence interface to score input selection. Test actual mixed
sources, not an oracle gate keyed by the designed owner label.

## Protocol

1. Reuse all200 authored quote-worlds from narrow_actions_r1. Add200 paraphrased
   quotation declarations:25targets x high/low x self/other x two new wrappers,
   using the first fixed pair only. The ownership detector receives only source
   text and the exact quoted passage, never query IDs, designed ownership or
   numerical personality scores. No changes after viewing its predictions.
2. Use the already pinned `cross-encoder/nli-deberta-v3-base`, revision
   `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`, no fitting. Two fixed hypotheses:
   the quote is presented as the writer's own account; the quote is presented
   as another person's account rather than the writer's. Exactly one entailed
   claim gives self/other; neither or both gives unresolved. Retain probabilities.
   This is an attribution proxy, not posterior-calibrated ownership probability.
3. Join the same detector to all first-person balanced quotes in the existing
  699-comment PANDORA source_event_r4 graph. Reuse frozen raw addresses and
   syntax records. Full-source premise and target quote must fit512tokens; record
   excluded cases without truncation. Keep syntax and NLI opinions side by side,
   not a fabricated natural gold standard. No automatic natural-text deletion.
4. Build300 mixtures:25targets x high/low own evidence x self-first/other-first
   x original/new-wrapper-1/new-wrapper-2. Use one self passage and the opposing
   friend's passage. This makes both source selection and order effects visible.
5. Capture200 new single-quote endpoints and300 mixed native/PQ endpoints using
   unchanged old query coordinates, model, reference functions. Reuse the old
   single-quote endpoints for the original wrapper, with hash/identity checks.
   Exactly500 new bounded local forwards, max1200s per invocation, resumable.
6. Compare unfiltered score, detector-selected score and the designed-self-only
   reference. The last is an evaluation oracle, never input to the detector.
   Select neither -> no observed author score (null), not zero or imputed prior.
   Select both -> preserve both in original order; one -> preserve that excerpt
   exactly. Quoted/context/unresolved material remains in the evidence ledger.

## Outcomes

Report400 synthetic source decisions by wrapper and polarity, wrong/abstention
counts, natural graph routes and unsupported coverage. Report every mixture's
selected sources, observed-score availability, MAE to self-only reference,
directional response and order sensitivity for native and fixed affine readers.
Always report coverage with conditional errors; abstaining on everything is not
success. True-self selection is not inference accuracy about a human trait.

No parameter search, labels, blind threshold optimization, raw data upload,
lockbox opening or replacement of old scores. Failures route the next mechanism
step; do not turn a pretrained classifier's confidence into scientific truth.
This test is a source-bound input operator, not a claim that quoting another
person carries no information about the writer's choice, beliefs or personality.

## Mathematics and literature inheritance

For ordered excerptsT=(t1,...,tn), detector d(ti) in {self,other,unresolved},
`S(T)=[ti:d(ti)=self]`. Return `(f_q(S(T)), coverage, all_source_records)` only
when S is nonempty, otherwise return an unavailable observed score. Keep the
unfiltered f_q(T) separately as a content diagnostic. Actor-source conditioning
changes the observation object; it is not subtractive denoising or multiplication
of a score by the known synthetic owner bit.

Inherit the existing NLI premise/hypothesis interface and CheckList-style paired
tests (R30 in GRAIN_LITERATURE_REGISTER.md). Existing UD event graph separates
grammatical subject, reporting source, quote scope and factuality; this step
adds only quote-source classification. Psychological experiencer, endorsement,
irony, nested quotations and truth of reported events remain distinct questions.

## Artifacts

`results/grain_vector_mechanisms/all_plan_closure_v1/ownership_scoring_r1/`:
frozen manifest/records, NLI journal, source decisions, natural route join,
500 native packets, mixed scores and contrasts, audits and summary.
Public report contains aggregate numbers and authored examples only.
Keep goalACTIVE: this experiment cannot complete all historical plans by itself.
