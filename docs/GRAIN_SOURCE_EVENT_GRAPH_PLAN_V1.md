# Source and Event Graph Reference V1

2026-09-13. C03/C16 following full multiview original-address replay.
Frozen before reading new parser results. This adds learned syntactic structure,
not a new personality predictor or another controlled quotation-template grid.

## Question and Prior Mechanism

An event's grammatical participant and the source asserting it differ. A writer
can quote another person's first person, be described by somebody else, want an
action without doing it, or describe an action under an explicit condition.
GRAIN's source-address sidecar can preserve these distinctions without deleting
text or changing frozen scores. Test a standard learned syntax front-end plus
explicit, inspectable graph rules, keeping every unresolved case.

Use Stanza1.11.0 English EWT models (tokenize/mwt/pos/lemma/depparse), pinned
resource metadata and SHA256 inventory after download; CPU2 threads. No language
model fine-tuning, new outcome labels, remote text submission or new LLM calls.
The parser is neural inference; do not describe this step as wholly non-neural.
Dependencies are isolated under ignored results rather than upgrading mainenv.

Prior methods: Qi et al.2020 Stanza; Universal Dependencies ccomp/xcomp/advcl
specification; Sauri & Pustejovsky2012 source-relative event factuality. We use
learned syntactic edges and expose the proposition/source nesting; this is not
a complete De Facto implementation or a new factuality gold standard.

## Material and Budget

Existing authorized624 Big5-only authors. Deterministically select one original
comment per author from the frozen full native source panel using seed916 and
comment ID hash, independent of text scores and labels. Additionally parse the
full original comments containing all80 old source-model annotations. Deduplicate
comments by original source_row; preserve full raw offsets and enclosing context.
Fixed one-comment sampling is for interface/coverage, not an author trait score.
If earlier raw annotations lie outside32-comment GRAIN budget, preserve their
addresses and mark absence from that budget instead of discarding them.

Parser cache is keyed by raw SHA256 plus pinned model/processor contract. Journal
each completed comment so a bounded pause need not restart. Abort on changed
material/offsets or missing parser dependencies; do not substitute a quote regex.
20min bound for corpus pass; preserve a partial result as partial, never full.

## Outputs and Tests

- Word graph: exact original offsets, lemma/UPOS/features/head/deprel; expanded
  multiword-token words retain enclosing source token address and explicit flag.
- Event candidates: verbal/copular predicates; direct/inherited-candidate
  subjects, passive voice, embedding path, reporting/attitude source candidates,
  local negation/modality, conditional/temporal clauses, quote-container support.
- Keep participant, asserting source and scope as separate attributes. Explicit
  singular first person differs from pluralwe/other/unknown. Control inheritance
  is candidate only; with a possible object controller, keep unresolved.
- Quotes use parser punctuation/address nesting; first person in a reported
  quotation is not automatically the document author. Unpaired quotes, ambiguous
  source and detached speech stay unresolved. Do not strip or rewrite clauses.
- `event_id` is comment/sentence/head identity. Multiple grammatical events in a
  sentence do not become independent people or extra questionnaire responses.
- Associate each existing sentence/item pair with overlapping event candidates,
  not with an automatically correct target event. Compare coverage/disagreements
  against old model self-attribution/scope classes; no gold accuracy claim and
  no tuning on these80 examples. Source-only issues remain pending for review.
- Synthetic tests only check graph algebra: actor/source split, ccomp/xcomp,
  negation scope, passive voice, conditions, quote first-person shifting,
  offsets, missing actors. Do not report them as real-language model accuracy.
- Record exact graph->GRAIN unit join and no-presence counts; empty support is
  unknown, not zero psychological evidence. Raw utterances/IDs stay gitignored.

## Completion and Next Decision

Deliver parser/model/source inventories, private cached graphs and event table,
corpus support summary, annotated-pair joins, tests and public report. If usable,
make this the typed provenance front-end for C14 evidence functions: semantic
query relevance and direction still require their own reader. Syntax does not
decide truth, long-term personality, causality or clinical interpretation.
No new scoring formula selected from coverage alone; no all-plan completion.

Sources:
- https://aclanthology.org/2020.acl-demos.14/
- https://stanfordnlp.github.io/stanza/pipeline.html
- https://universaldependencies.org/u/dep/ccomp.html
- https://universaldependencies.org/u/dep/xcomp.html
- https://aclanthology.org/J12-2002.pdf
