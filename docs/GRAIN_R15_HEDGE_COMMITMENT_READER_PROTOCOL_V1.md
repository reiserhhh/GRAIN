# R15 hedge-commitment local NLI reader: protocol V1 (frozen before any reading)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_15_HISTORY_AND_PRAGMATIC_READER.md` Part D (bounded local NLI
authorized); `SEMINAR_FABLE_13_DECISION.md`.
- **Runner and output.** `scripts/run_grain_r15_hedge_commitment_reader.py`, writing to
  `results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1/`.
- **Materials** (frozen before reading): `materials_frozen.json`, sha 1b07fb53….
- **Question.** Can the existing local reader detect the observable commitment, scope and ownership that the R15
  speaker model needs? It does **not** assign hidden hedge causes. NLI entailment is neither world truth nor a
  speaker's psychological cause.

## Model and budget

- **Model.** `cross-encoder/nli-deberta-v3-base`, local snapshot 6c749ce3425cd33b46d187e45b92bbf96ee12ec7, loaded with
  `local_files_only` and HF_HUB_OFFLINE=1, on CPU with 2 threads. No download, API, generation or training.
- **Cap.** 576 evaluated pairs and at most 576 forward invocations, counting one invocation per premise–hypothesis
  pair submitted, failed attempts included.
- **Batching.** Batches of 16, with a checkpoint to `readings.jsonl` after each batch. Wall-clock cap 10 min.
- **Recorded:** batches, pairs, failures, and all three probabilities (contradiction, entailment, neutral).

## Materials (deterministic templates; no generation)

- **Size.** 96 premises = 3 propositional cores × 4 explicit forms × 2 owners × 2 audience frames × 2 wordings.
  - Cores 1–2 are SOURCE; core 3 is TEST. The split was fixed before reading.
  - Forms: bare assertion, epistemic hedge, deferential hedge, non-committal.
  - Owners: self, or reported other ("My neighbour told me: '…'").
  - Frames: neutral, or disagreeing listener.
  - Wordings: original, or a meaning-preserving paraphrase of the same form, owner and scope.
- **Premise:** `<frame> the speaker says: "<utterance>"`.
- **Six queries**, each instantiating the proposition and the referent "the speaker":
  - own commitment;
  - uncertainty;
  - deference;
  - reported other;
  - no commitment;
  - the bare proposition — **a truth-conflation diagnostic only**, never verification of P.

## Frozen contrasts (P(entailment); all three probabilities reported)

1. **Form effects.** Within core, with owner = self: for each role, the mean over frames and wordings of each form
   minus bare, compared with the paraphrase variation |original − paraphrase| within the same form.
   - Expected-order agreement per role and core, as a descriptive count: bare highest on own commitment; epistemic
     highest on uncertainty; deferential highest on deference; non-committal highest on no commitment.
2. **Ownership.** For each form, reported − self on own commitment (expected < 0) and on reported other (expected
   > 0).
3. **Audience frame.** Disagreeing − neutral per role; descriptive.
4. **Truth conflation.** The truth-diagnostic entailment for self bare, reported bare, and self hedged forms.

## Source-core boundary rule and interpretation

- **Stop point.** After the 384 source pairs, compute the failure flags:
  - **ownership failure:** mean own commitment for reported bare ≥ self bare, or mean reported other for reported ≤
    self;
  - **form failure** per role: the expected form is not the highest.
- **No rescue.** No tuning, prompt change or material change follows. The frozen test-core pairs are then read if
  technically safe (within cap and time), and any failed channel is interpreted with attribution disabled.
- **What this can conclude.** It concerns reader evidence detection only: availability, detection and attribution in
  RAM terms. There is no accuracy threshold, ambiguous forms may stay ambiguous, and nothing here is validated against
  real speakers.
- **Comparison with R14.** Ownership results are compared with the R14 counterfactual reader descriptively. The banks
  are unmatched, so there is no paired gain.
