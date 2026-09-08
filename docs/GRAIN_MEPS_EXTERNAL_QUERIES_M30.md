# M30: local counseling-text external connection

2026-09-09. Implementation follows M28; questionnaire inventory completed in M27.
The latest explicit user request authorizes local exploratory text/scale linkage.
No private text API upload, model training, Study 1 merge or old lockbox opening.

## The target is the writer's relationship/perception, not the AI's personality

The administered forms clarify an important referent change. Pre ratings concern
AI in general and the writer's existing relationship; post ratings concern the
AI just encountered and current wishes. The 22 score columns belong to 17 logical
families, but require phase-aware queries. Freeze one query per score occasion
before obtaining new text predictions; do not reuse a generic neuroticism query
as an attachment or anthropomorphism score.

| Family | Pre criterion/query object | Post criterion/query object |
|---|---|---|
| Mind total | Writer attributes mental capacities to AI generally | Writer attributes mental capacities to this AI |
| Agency | Intentions and free choice attributed to AI generally | Same attribution to this AI |
| Experience | Feelings, awareness and own mind attributed to AI generally | Same attribution to this AI |
| Attachment anxiety | Writer seeks reciprocal emotional closeness/reassurance from AI | Writer now wants reciprocal closeness/reassurance from this AI |
| Attachment avoidance | Writer resists disclosure/closeness to AI | Writer now resists disclosure/closeness to this AI |
| Positive trust | Writer sees AI as useful, honest, correct, trustworthy | Not administered |
| Distrust | Writer expects deception/errors/disobedience from AI | Not administered |
| Privacy concern | Writer expects information leakage by AI | Not administered |
| Perceived empathy | Not administered | Writer feels understood and listened to in this exchange |
| Session depth | Not administered | Writer experiences the exchange as valuable, deep, full, substantial, special |
| Session smoothness | Not administered | Writer experiences ease, relaxation, pleasantness, flow and comfort |
| Global evaluation | Not administered | Writer's overall favorable evaluation of the exchange |
| Disclosure levels 1-4 | Not administered | Willingness to discuss ordinary interests, difficult experiences, minor weaknesses, or negative traits/abilities with this AI |
| Disclosure exploratory mean | Not administered | Across-level willingness, kept separate from ordered indicators |

The narrow confirmation-seeking query is retained as an unlabeled reference,
not declared equivalent to any attachment composite. No diagnosis or clinical
classification is produced. Different negative/trust indicators remain separate.

## Material, scoring and comparison

1. Use existing `load_text_material` speaker-explicit importer. Primary view:
   each participant's free-chat user turns in order. Secondary view: the same
   participant's three MEPS final answers, separated as three excerpts. A final
   answer may be AI-assisted; retain that provenance. Assistance dialogue is a
   separately available view, not concatenated into additional author evidence.
2. Run tokenizer-only coverage first. Keep all available material when within
   the frozen observer context. If needed, select evenly spaced contiguous
   excerpt spans using a fixed token budget independent of labels; retain exact
   source offsets and coverage privately. Do not silently truncate, translate,
   summarize or remove personality-relevant choices.
3. Use the pinned local Qwen2 observer, deterministic native digit probabilities
   with expectation on [0,1], not generated JSON or post-hoc score calibration.
   Freeze exact prompt text, candidate tokens, model/version, material hashes,
   phase/object and member set before inference. These queries establish whether
   the smaller constructs are visible before their functional matrices are built.
4. Save and hash all scores before opening the existing private M27 criteria.
   Main associations: each query with its intended pre/post criterion. Retain
   the complete query-by-criterion matrix as secondary shared/specific evidence.
   Each participant contributes one score per declared view, never one sample
   per turn/chunk. No fitting, calibration, scale-based prompt selection or
   per-query sign flips on the 46 participants.
5. Report n, Pearson/Spearman, Fisher and participant-bootstrap intervals, all
   raw p and BH q values. FDR families: intended pre-session scores, intended
   post-session scores, and exploratory off-diagonal links, by material view.
   Paired view contrasts use the same participants. Include response variation,
   text coverage and candidate mass so uninformative queries are visible.
6. If user-only session appraisal is weak, inspect evidence opportunity first:
   subjective appraisal may not be stated in user turns. One separately declared
   role-labeled assistant-context arm can answer whether interaction context
   helps; assistant text never becomes the participant's self-authored evidence.

## Minimal execution and decisions

First freeze the 22 phase-aware queries and both participant-level views. Run one
complete free-chat packet, at most 20 minutes per bounded invocation, checkpoint
per participant/query. Complete named factor coverage before adding a new prompt
family. Then run the direct-answer comparison if it changes the opportunity/
mechanism question. No repeat-until-significant selection.

For a useful association: construct that query's frozen P/Q function using
separate source/synthetic reference material and test whether it preserves the
actual vector response and external link. Do not use the 46 human outcomes to
fit that function. A weak association leaves an explicit distinction between
limited expressed information, wrong referent, and a weak observer; it need not
invalidate other targets or restart all controls.

Deliverables: private material/prediction manifest; public aggregate per-score
results; coverage and equation cards; source/query/phase-aware cross-path atlas.
This is external association with self-reports and session appraisals, not proof
of all personality theory or medical utility.
