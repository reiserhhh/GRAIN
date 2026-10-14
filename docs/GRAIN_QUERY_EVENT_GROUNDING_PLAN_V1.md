# GRAIN Query-to-Event Grounding V1

2026-09-13. Execution continuation of C03/C14/C16, frozen before new inference.

## Question and decision

A sentence may contain several people and actions. Does a fixed query locate a
different event from the sentence's incidental first-person event? How does
restoring source/negation/intention/adverbial context change the model judgment?
This closes a scoped evidence-interface experiment, not C03 or all-plan theory.

## Predecessors

Chen et al. (2024), [Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845/),
motivates fine-grained retrieval with necessary context; Choi et al. (2021),
[Decontextualization](https://aclanthology.org/2021.tacl-1.27/), requires preserved
meaning, not simply deleting surrounding words. We use existing Stanza graphs
and fixed semantic/NLI models, not their propositionizer or an exact replication.
An extracted discontinuous fragment is not a certified atomic proposition.

## Fixed design

- Only the existing 80 source-only sentence/item pairs; no additional pair
  selection, questionnaire arrays, held-out labels, MEPS or bridge fitting.
- Reuse source_event_r4 raw-character parse/graph cache and original comments.
- For each predicate within the original sentence, build (a) local graph words
  plus available subject words, rendered with explicit omissions; (b) an exact
  contiguous raw-text envelope including ancestor-local words, their sources,
  attached adverbial subtrees and any balanced quote containing the predicate.
- The envelope may contain other events and still lacks discourse-wide resolved
  coreference. Preserve that limitation; do not call it a minimal proposition.
- Frozen paraphrase-distilroberta-base-v1 cosine selects top1 separately for
  local and envelope views. No self-only filtering, thresholds or tuned weights.
  Record ties and distinct envelopes; a broad envelope can fail to locate an event.
- Fixed DeBERTa NLI on original sentence, local-selected fragment, that event's
  envelope, and envelope-selected envelope. Deduplicate identical input pairs.
  No silent context truncation; overlength pairs remain explicitly unresolved.
- Keep participant, source, passive role, intention, modality, negation and
  adverbial flags separate from relevance. All events remain in the output.
- Join selected predicate addresses back to the five frozen GRAIN views; this
  shows addressability, not a newly trained personality head.

## Readouts and hard boundary

Report 80-pair support, event counts, selection changes, tie/margin distributions,
source/subject routes, local-to-envelope NLI changes, entailment-score deltas,
and GRAIN address coverage. Old model annotations are comparison strata, NOT
gold truth; no accuracy, new personality correlation or significance claim.
Inspect the seven previously other-person-coded cases without selecting on
new outputs. Exact source spans, deterministic ranking and finite predictions
must pass tests/audit. Skip unresolvable/overlength cases, never impute zero.

Budget: cached parser only; <=600 s semantic encoding and one bounded NLI pass
on at most 320 selected rows (deduplicated), <=20 min overall. No model download,
training, source upload or rescue grid. Stop this tranche after one comparison;
next decision is separate relevance localization from contextual interpretation,
or adopt a materially better grounded proposition/role tool if this fails.

## Deliverables

`grain_research/query_event.py`, `scripts/run_grain_query_event_grounding.py`,
tests; ignored `results/grain_vector_mechanisms/all_plan_closure_v1/query_event_r1/`
with contract, candidate addresses/scores, selected events, NLI cache, joins,
summary and audit; tracked `QUERY_EVENT_GROUNDING_REPORT.md` and ledger update.
