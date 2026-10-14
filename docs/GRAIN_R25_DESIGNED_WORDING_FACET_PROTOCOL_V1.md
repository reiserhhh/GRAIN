# R25 designed wording-perturbation facet on cached reader outputs: protocol V1

Frozen 2026-09-26 JST, before any facet quantity was computed.
- **Authority.** `NEXT_PACKET_17_APPLICATION_J_WORDING_AND_PROVENANCE.md` Part C; `SEMINAR_FABLE_15_DECISION.md`.
- **Runner and output.** `scripts/run_grain_r25_wording_facet.py`, writing to
  `results/grain_gptpro_foundation_v3/F12/designed_wording_facet_r1/`.
- **Budget.** CPU ≤ 5 min, 0 reader calls.

## Inputs and label

- **Inputs.** `F17/hedge_commitment_reader_r1/readings.jsonl` (sha 962b5874…) and `materials_frozen.json` (sha
  1b07fb53…): 96 texts, 576 readings, one deterministic local NLI reader.
- **Label.** This is a **designed wording-perturbation facet**: two fixed template wordings per cell, crossed with
  core, form, owner and frame. It is **not** natural-language paraphrase in general, not inter-reader variation, and
  not human or personal reliability.
- **Unit.** One text. The six query responses are joint channels of that text.

## Pair audit (before analysis)

- The 48 within-cell pairs are (core, form, owner, frame) × {original, paraphrase}.
- **Mechanical checks per pair**, from the material IDs and premises:
  - identical core, form, owner and frame;
  - the identical frame prefix;
  - the identical owner wrapper ("My neighbour told me: '…'" for other);
  - the identical proposition string p, or its sentence-cased form S for bare-original.
- **Designed-paraphrase assumptions**, recorded per form:
  - **bare:** S vs "Just so you know, p." The paraphrase adds a discourse marker, which could carry
    addressee-directed emphasis.
  - **epistemic:** "I think p, but I'm not sure." vs "Maybe p; I can't be certain." These use different hedge
    lexemes.
  - **deferential:** "You know this better than I do, but I would say p." vs "I'll defer to you on this, but my view is
    that p."
  - **non-committal:** "I'm not taking a position on whether p." vs "I won't commit either way on whether p."
- **Exceptions** are listed. No pair is dropped.

## Identifiable decomposition (balanced, fixed effects; per query, for p_ent and p_con)

- **Source (cores 1–2).** 64 texts, 32 cells; total 63 df.
  - **Between cells (31 df):**
    - core (1), form (3), owner (1), frame (1);
    - form×owner (3), form×frame (3), owner×frame (1);
    - a lumped higher-order between-cell term (18 df), which aliases all remaining core and three- or four-way
      interactions. It is not named.
  - **Within cells (32 df):**
    - wording main effect W (1);
    - W×form (3), the template contrast, because each form has its own paraphrase template;
    - a lumped W×context term (28 df). With one deterministic reading per text, this term is **not** a stochastic
      residual. It aliases wording interactions with core, owner and frame, including reader nonlinearity.
- **Test (core 3; audit only, no calibration).** 32 texts, 16 cells.
  - **Between (15 df):** form, owner, frame, their three two-way interactions, and a lumped three-way term (3 df).
  - **Within (16 df):** W (1), W×form (3), lumped (12).
- **Reported:**
  - sums of squares, SS shares of the total, df and mean squares;
  - the rank of each term's design (an estimability check);
  - the check that the terms sum exactly to the total SS.
- **Not reported:** random-effects variance components, which cannot be estimated without replication beyond two
  fixed wordings.

## Further outputs

- **Within-pair squared differences.** d² = (y_orig − y_para)² per pair and query, with summaries by form.
- **Cross-query structure.** The covariance and correlation of the 48 wording differences across the six queries'
  p_ent.
- **Sensitivity of the named merged functional.** This is the epistemic + non-committal share from
  `F17/hedge_channel_utilization_r1`.
  - The channel is the packet-16 source channel (V1 features, source cores 1–2, δ at N = 16), with owner and frame
    unknown (K3), estimated by weighted simplex least squares.
  - The estimate is made for six collections, one per (core, wording): 16 texts each, true share .5.
  - Reported: the estimates and the original − paraphrase differences. Source collections are in-sample for the
    channel and are labelled so.
  - **Wording-transport variant:** a channel fitted on source **original-only** texts, applied to the paraphrase
    collections of all cores, and conversely.
- **Raw contrast rows.** The 48 × 12 wording-difference rows.

## Rules

- No new readings. No wording, prompt or sample-size grids. No retuning on core 3.
- Natural semantic equivalence and inter-reader variation remain separate gaps.
