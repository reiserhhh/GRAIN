# GRAIN Fixation Rulebook v1 (実験固定基準書)

> **Historical scope notice (2026-07-30).** This is the V1 local
> instrument-design rulebook, not the constitution of GRAIN as a foundational
> methodology. Its empirical rules remain part of the historical evidence
> record, but they do not require GRAIN to be a scale or every future study to
> use this battery design. All new V8 studies are governed first by
> `docs/SUICA_FOUNDATIONAL_CLOSURE_STANDARD.md` and its layer/edge/refusal
> contract. A local rule applies only after the new study declares the
> relevant observation design and estimand.

Created: 2026-07-05
Purpose: the historical binding rule set for the V1 local
instrument-development program — what was fixed, by what rule, and with which
measured evidence. New GRAIN foundation studies inherit a rule only when their
foundation contract declares the corresponding design and estimand.
This is the bridge between SUICA_RIND_THEORY_BASE_V3 and the planned
experiment flow (fixed-prompt writing / fixed-prompt AI dialogue -> GRAIN
analysis -> factor identification -> new cohort -> external-scale validation).

Every rule cites its evidence. Rules marked PROVISIONAL lack direct evidence
and must be pilot-tested before being relied on.

## A. Rind (condition) fixation

- **A1. Fix 4-8 rind prompts per battery.** [PROVISIONAL for
  experimenter-imposed prompts] Each rind = one distinct theme/situation
  domain. Rationale: C2 precision requires rind fixing (PRED-1: +0.06..+0.18
  reliability, span-matched); >= 4 rinds are needed if any react/signature
  scoring is ever attempted (E6: signatures need >= 5 shared situations) and
  for rind-robustness checks. **Scope caveat (referee sweep 2026-07-05):
  PRED-1 fixes the rind via the subject's own venue selection; the only
  direct test of an experimenter-imposed prompt (Essays, PRED-1b) did not
  show the predicted precision gain (population confound). The
  imposed-prompt version of this rule is therefore a design hypothesis until
  the OP-7 dialogue/writing pilot tests it within one population.**
- **A2. Choose rinds to span the choice-axis space.** Use the E3 content
  classes (gaming/play, politics/opinion, life-planning, aesthetics/technical,
  narrative/media, social-support...) as the sampling frame so the battery
  covers heterogeneous flesh expression. Evidence: E3 axis structure.
- **A3. Include a CHOICE MODULE: present M optional prompts, subject selects
  K to answer (e.g., M=8, K=4).** The selection pattern itself is scored as
  the C1 channel. Rationale: choice is the strongest single signal family
  (AUC 0.84-0.91; axes retest 0.51-0.70); a fully fixed battery would discard
  it. This module reconciles fixed administration with choice measurement.
- **A4. Never subtract condition means statistically.** All centering variants
  are falsified (P2, E1 cross-fit). Rind control is by design only. Norms are
  computed per rind (see D2), scores reported per rind or aggregated over the
  fixed rind set.

## B. Occasion structure

- **B1. >= 2 occasions (sessions), separated in time; never single-sitting.**
  Evidence: Essays (fixed prompt, one sitting) gained nothing over free-rind
  Reddit at matched budget (PRED-1b); X (many occasions, fixed domain)
  performed best. Trait scores = average over occasions; state scores = per
  occasion.
- **B2. Any design comparison must match time-span and occasion counts across
  arms.** Evidence: round-3 audit — an unmatched chronological-prefix arm
  manufactured a spurious +0.15 for tension.
- **B3. Trait claims require the disjoint-occasion test** (score occasion set
  A vs set B, correlate), never same-occasion split-half alone. Evidence:
  same-text repartition inflation (2026-07-04 review, F1).

## C. Text volume (from measured reliability curves)

- **C1. Per rind per occasion: target >= 250-350 words (~1,000+ tokens per
  construct-bearing unit after slicing).** At ~1,000 tokens/half: style
  split-half r ~ 0.4-0.88 by construct (P1 volume curve; PRED-1b table).
- **C2. For first_person-grade trait precision (r ~ 0.86), plan ~4,500 tokens
  per measurement half** (volume curve). Weaker constructs plateau: more text
  does NOT fix tension/novelty as traits (they are state/choice-carried).
- **C3. Slice width 96-128 tokens; conclusions were invariant to this choice**
  (multiverse checks at s96/s128).

## D. Population and norming

- **D1. Reliability and norms are population-relative.** Never transfer
  reliability claims across populations (PRED-1b: homogeneous students vs
  heterogeneous internet users). Every new population gets its own
  reliability estimate.
- **D2. Norms per (population x rind), leave-target-out.** Evidence: LOO
  centering machinery (P0-validated); self-inclusion biases small-cell norms.
- **D3. Structure transport is approximate (0.53-0.62 across corpora on 23
  rates)**: re-estimate factor structure per deployment; do not assume the
  Reddit-derived structure (PRED-3, audit-corrected).

## E. Channel and construct assignment

- **E1. Trait track (C2 style-base, uncentered)**: style/stance constructs —
  first-person usage, directive stance, novelty vocabulary (choice-coupled),
  and future constructs that pass the disjoint-occasion test.
- **E2. State track (per-occasion scores)**: affect/tension constructs.
  Evidence: tension failed every trait test (P1 0.35 SB; person share 2-5%;
  PRED-1 gain was artifact) — consistent with emotion scales measuring state.
  State scores are per-occasion values with their own per-occasion precision
  (see E5 trait-state spectrum results when available).
- **E3. Choice track (C1)**: selection patterns from the A3 choice module or
  naturalistic venue choice. Exclude any choice axis contaminated with
  criterion-related communities (the class-11/MBTI lesson).
- **E4. Signature/react claims (C3) require the STRANGER null** (compare
  same-person profile alignment against cross-person alignment), never
  within-person permutation alone; and thick cells (>= 3 slices per
  person x rind x occasion). Evidence: E4 audit — within-person nulls let
  normative profiles masquerade as individual signatures.

## F. Scoring and validation process

- **F1. Freeze scoring formulas (lexicons, weights, code hash) before any
  external-criterion contact**; changes after contact restart validation.
- **F2. Preregister by git commit** (hypotheses, directions, estimator, FDR
  rule, success rule); confirmation data gets a fixed opening budget.
- **F3. Builder/auditor separation**: the analysis author never signs their
  own verdict; an independent adversarial pass reproduces headline numbers
  before promotion. Evidence: rounds 2-3 each caught a real builder artifact.
- **F4. Mandatory negative controls in every table**: user-shuffle,
  label-permutation, and (for reference models) cross-fitting. Evidence: E1
  self-estimation leak was only diagnosable via these.
- **F5. New-construct discovery**: item selection must span the score range
  with distractor constructs and a defined chance level; coder panels use
  >= 3 model families + human subsample; heldout item banks are single-shot.
- **F6. Leakage hygiene**: personality-vocabulary masking (incl. MBTI type
  codes) at slice level and item level; criterion-community exclusions.

## G. AI-dialogue administration (the clinical/dialogue use case) — PROVISIONAL

- **G1. The fixed system prompt + opening script IS the rind**; version it and
  never vary it within a cohort.
- **G2. Score only the subject's turns**; assistant text is excluded and also
  screened for lexicon-word priming (an assistant that says "worried" can
  inflate the subject's tension rate by echo — measure echo rate in pilots).
- **G3. One session = one occasion** (B1 applies: >= 2 sessions for traits).
- **G4. Register-transfer is UNVALIDATED**: all current evidence is
  monologue/post text (Reddit/X/essays). A dialogue pilot (OP-7) must rerun
  P0-style and PRED-1-style checks on conversational text before any clinical
  claim. Dialogue turns are shorter and reactive; volume rules (C1-C2) likely
  need re-estimation.
- **G5. Japanese deployment requires a JP scorer** built and validated to the
  same rules (current lexicons are English-only; cross-language claims are
  explicitly out of scope until then).

## H. Reporting language

- Stability numbers must be tagged {same-occasion | disjoint-occasion |
  disjoint-rind | cross-population}; only the middle two support trait claims.
- "validated", "invariant", "signature", "projective" may be used only at
  their ledger status.

## I. Technical-chart applicability

- **I1. Scalar support is metadata, not a utility gate.** The M4 boundary
  ecology showed that the same coverage can contain beneficial source
  partition and harmful source rotation; the historical `.80` fallback lost
  to always using `R`.
- **I2. Do not learn a cell-level router unless it wins both unseen-world and
  unseen-repetition tests.** The fixed-coverage 24-component signature failed
  LOWO and LORO gain, harm, and routing gates despite exact replay and null
  controls.
- **I3. When relative utility is not observable, license at study-design
  level.** Inside a prospectively supported design class use the registered
  chart; outside it report `B0/R` sensitivity or refuse. Never tune support
  thresholds on opened downstream outcomes.
