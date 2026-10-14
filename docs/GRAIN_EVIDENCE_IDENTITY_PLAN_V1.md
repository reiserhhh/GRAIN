# M01/M09: evidence identity before personal functions

2026-09-13. New companion only; prior frozen experiments stay unchanged.

## Question and boundary

Content about help can express intention, an action report, preference, ability,
current experience, or an actual request in the dialogue. These are different
evidence objects. Reporting an action does not verify its occurrence; making a
request in an observed turn is an observable linguistic act, but not evidence
that help was received. Multilabel coordinates must not be forced into one class.

Inherit source-relative factuality from Sauri/Pustejovsky (2012), and reference
segments/functional dependence from Bunt et al. (2020), not their full classifiers.
NLI outputs are model responses, not calibrated human factuality probabilities.

## Frozen packet

1. Twenty-four new public statements: six identities, two source and two held
   paraphrases per identity. Twelve additional attribution/negation/mixture
   challenges and sixteen contextual replies test boundaries. Labels describe
   author-designed stimulus intentions; they are not independent human coding.
2. Ten fixed NLI questions; no prompt grid, fitting, or source-based threshold
   search. All tests retain raw reply and preceding-turn addresses. Compare
   context and reply-only on dialogue cases, including identical `Yes, please.`
   under distinct antecedents. A predecessor turn is a candidate dependency,
   not automatically the true target of every response.
3. A source-only TF-IDF content reference and fixed NLI query coordinates are
   compared. Source mean matrix B has one row per evidence identity. For a
   controlled single-identity utterance mixture pi, E[h] = B' pi. Estimate pi
   by constrained least squares on the simplex; this decoder is NOT applied to
   open natural text. Use 192 simulated authors with 48 independently sampled
   utterances, source/held wording paired, and report per-mode reconstruction.
4. An identity target q'pi is identifiable from B'pi and 1'pi=1 exactly when q
   lies in the row space of [B';1']. Exhibit a content-only null direction and
   preserve conditioning/wording error separately from numerical rank.
5. Prompt/partner mix example: pooled acceptance is sum_o pi_u(o) f_u(o).
   Fixed-opportunity scoring is sum_o pi_ref(o) f_u(o). Compare pooled versus
   standard rates under equal author responses but different opportunity mix.
   No observational rates are converted into causal responses or missing times.
6. Twelve local PANDORA sentences selected by source/offset hash from the frozen
   parser journal, after only length eligibility (30-600 characters). Preserve
   exact raw offsets and existing event relations; do not select by NLI results,
   questionnaire scores or attractive examples. No participant text in report.
   Natural multilabel readings are validation outputs, not a new gold standard.

## Deliverables and stop

New typed-evidence interface, actual local model calls, input/output hashes,
source versus held reconstruction and target null-space analysis, dialogue
contrasts, natural-address journal, tests, concise report and TODO status.
One fixed run; implementation faults may be repaired with disclosed provenance,
but no semantic rescue/parameter grid. Retain failures as evidence of the missing
interface. M01/M09 acceptance remains tied to their parent requirements, not the
mere existence of this packet. M13 explanation artifacts can reuse these types.

## Primary sources

- Sauri & Pustejovsky (2012), *Are You Sure That This Happened? Assessing the
  Factuality Degree of Events in Text*, CL38(2),261-299,
  https://aclanthology.org/J12-2002.pdf. Existing W88; this turn reread source
  perspectives and nested contexts, especially pp.268,275,280-283.
- Bunt et al. (2020), *The ISO Standard for Dialogue Act Annotation, Second
  Edition*, LREC,549-558, https://aclanthology.org/2020.lrec-1.69.pdf.
  Read sections1-3, especially functional versus feedback dependence and repair
  reference segments. No claim of complete ISO annotation or original taxonomy.
