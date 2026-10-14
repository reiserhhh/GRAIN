# HM58: occasion relations under a declared finite group action

2026-10-09. Implementation prepared for a finite run after verified HM57.
Freeze this plan/code/materials before acquisition. No result is implied by
the implementation. Not a teacher-fidelity or precision search.

## Question and mathematical object

HM57 shows identical isolated-event/mention distributions with different whole
account observations. A specific occasion digit or chronology convention could
still drive part of the apparent relation contrast. Preserve equality relations
among occasion identifiers, but consistently rename all identifiers in text and
history. Ask which contrast is common to that orbit, and which varies with codes.

Use G=C4 acting by K_j -> K_(1+(j-1+g) mod 4), g=0,1,2,3. This is an exact
action on material/metadata, not an assumed representation on neural states.
All occurrences and explicit event references must be changed consistently.
Actor identity, claim identity, family, polarity, task, body order and query
remain fixed. The equality partition and event/mention counts must be verified.
Token proofs must demonstrate only occasion digits changed at equal lengths;
if tokenizer positions differ, record it and do not silently claim matching.

For each fixed event-relation contrast, let delta_g be its passive readout change.

    delta_bar = (1/4)*sum_g delta_g
    V_code = (1/4)*sum_g (delta_g-delta_bar)^2
    (1/4)*sum_g delta_g^2 = delta_bar^2 + V_code.

This is standard finite-group averaging/Reynolds projection and variance
decomposition. Averaging is invariant by construction, so that invariance is
not an empirical discovery. The empirical question is whether a substantive
contrast remains in delta_bar rather than depending on a favorable code.
There is no tuning of group weights, label selection or intervention strength.

## Finite scope and controls

- All HM56 24 atoms and 192 four-regime accounts, using the aligned content
  query for each (six content families, two phrasings, two polarities, two orders).
  All four C4 elements: 864 readings, 216 exact historical g=0 reuses, at most
  648 new frozen-observer forwards and 900 model seconds. Do not restart HM56.
- Retain all alias/balanced, distinct/alias, other/balanced and reversal contrasts.
- Verify every input history has the same equality/ownership partition under g.
- Synthetic functions: true relation-only readout, code-only readout and mixed
  readout. The orbit method must identify each, not force a positive outcome.
- Re-read per-orbit isolated atoms too. A pooled-summary collision involving
  changed occasion codes cannot be assumed from g=0 without checking it.
- Freeze the entire roster/model/candidate alphabet/plan/code before any new
  forward. Save attempted/completed calls and checkpoints once, no retry storm.

## Declared probability interface, fixed before acquisition

All four isolated occasion codes are available for each content/family/polarity.
An account record uses the atom matching its **actual renamed occasion code**,
not merely the isolated K1 atom at that group's g. Event/mention weighting is
fixed by explicit history, with both probability-mixture and logit-geometric
pools retained. This corrects a possible design shortcut before any new calls.

Report collisions both at each code assignment and after averaging each
probability summary over the full C4 orbit. The linear mixture's orbit average
has a simple algebraic cancellation: every code occurs equally often for each
content type; the same/different-event pair retains identical mention content
multiplicity. This can make its orbit-mean mention mixture identical even when
individual code-conditioned summaries differ. Geometric normalization need not
commute with orbit averaging, so equality there must be checked, not imposed.
Use complete ten-answer distributions, not just equal scalar scores.

The four-element C4 orbit is not all 24 S4 relabelings and does not vary the
K/digit syntax. It tests robustness to the declared orbit, not universal nominal
invariance. Content-aligned query only, not a new six-query whole profile.

Executable stages: `scripts/run_grain_math_event_relation_renaming.py`
prepare/collect/analyze. Alternate raw-binding audit:
`scripts/verify_grain_math_event_relation_renaming.py`.
Synthetic tests: `tests/test_grain_math_event_relation_renaming.py`.

Numerical K labels may suggest chronology even when treated as arbitrary codes.
Failure of native invariance can reflect that interpretation; do not equate
code sensitivity with absence of all event understanding. Conversely, an
orbit-averaged contrast does not prove a semantic circuit: token repetition
patterns also preserve equality. Report both interpretations and only promote
the structural object actually supported by the controlled material.

## Scientific decision, not score optimization

If orbit-averaged same/different-event effects survive and stay separate from
mention weighting, retain event equality/ownership relationships as necessary
observation units and proceed to a source-edge mechanism localization. If the
effect largely varies with particular codes, narrow the relation claim and
test another explicit event-identity representation rather than tune a scorer.
Mixed results preserve both contributions. No mandatory accuracy threshold,
no average personality named from a model readout, no human-validity claim.

Outputs must include immutable input/token proofs, raw candidate logits, orbit
contrasts with signed mean and code variation, known-world tests, replay
verification and a report explaining the next mechanism decision. Do not use
the count of tests or a process exit as a substitute for that answer.

No API, generation, new neural training, private data, publishing, Claude,
automation or old queue. GRAIN author/state/condition interpretation and full
original research scope remain open. The group action is standard mathematics;
OpenAI/math's conditional-coordinate setup motivates tracking the conditioning
object, not a new theorem transferred to the LLM.
