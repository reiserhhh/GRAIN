# R23/R25 fixed-core frame/topic/day response variation on the F16 bank: protocol V1 (frozen before reading effects)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_10_DECISION.md` Packet 2. Original demand: P03:23–24, P04:19–20.
- **Runner and output.** `scripts/run_grain_r23_fixed_core_dif.py`, writing to
  `results/grain_gptpro_foundation_v3/F12/fixed_core_dif_r1/`.
- **Scope.** Cached numbers only, zero calls. CPU, 2 workers × 1 BLAS thread, 20-minute cap.
- **What this is not.** These variants are **not verified meaning-preserving paraphrases**. The name is
  "fixed-core frame/topic/day response variation", not paraphrase invariance.

## Inputs

- **Readings.** `R14/same_object_channels_r1/typed_reader_table.csv` (hash-verified against its inventory): 380
  materials.
- **Outputs.** NLI pole `nli_p`, NLI qualification `nli_q`, historical GRAIN qualification `grain_q_hist`,
  transported lexical `lex96_p`/`lex96_q`, and refit lexical `lex80_p`/`lex80_q`.
  - `grainridge_p` is constant, so its core contrast is undefined and it is reported as such.
- **Factors.** Derived from the public generator (`f16_text_transport.build`):
  - frame = frames[role][v % 3] (3 per role, role-disjoint);
  - day = moments[(v // 15) % 4];
  - topic = endings[(v // 3) % 5];
  - status within the qualification class = v % 2.
- **Parsing check.** Frame, day and topic parsed from the text must equal the formula for all 380 materials.
  Otherwise STOP.

## Support, rank and specification (declared)

- **Support audit.** Per role: cross-tabs of core cell (pole × status) against frame, topic and day.
- **Primary specification M_add** (fit separately per role and output):
  y ~ core cell (8) + frame + topic + day, with treatment coding.
  - Report the design rank.
  - If the design is rank-deficient, report the null space and the estimable contrasts. No regularization and no
    arbitrary unique coefficient.
- **Secondary M_int.** Adds key-contrast × frame and key-contrast × topic. The key contrast is pole for pole outputs
  and metadata q for qualification outputs. Estimability is reported the same way.
- **Core contrast.**
  - Pole outputs: mean(y | pole 1, qualified) − mean(y | pole 0, qualified).
  - Qualification outputs: mean(y | q 1) − mean(y | q 0).
  - If |core| < .05, the result is **undefined / not comparable**.
- **Effect summaries.**
  - Frame, topic and day ranges (max − min of main effects).
  - Loading ranges: key-contrast differences across frames and across topics, from M_int.
  - The ratio of each range to |core|. A 10% cut may be shown descriptively; an effect below it is never called
    neutral.
- **Uncertainty.** Material bootstrap within role, B = 1,000, seed 20260926, percentile 95% CIs.

## Transport scope

- Frames are role-disjoint, so coefficient transport for frames is **unsupported**.
- Topics (5) are shared by all roles; days are shared where present (source/pilot Mon–Tue; application Mon–Thu).
- Source-fit topic/day effects are compared with application estimates, which are **labelled descriptive**. They are
  not used to rescue source calibration, and no held-out prediction is claimed.

## Optional paired cached replay (run only with exact support)

- **Scope.** Application authors, c2_xor seed 20260915, mixed panel, archived hard channels, from the same rendering
  and histories as `R14/same_object_channels_r1`.
- **Event and swap.** For each channel and author, t* is the first t ≥ 12 whose material has a matched alternative
  (same role, pole and status; different frame, topic or day) with a **different** hard reading for that channel.
  The material is swapped for the first such alternative in material_id order.
- **Everything else fixed.** Prior, history before t*, all future events and readings, and the truths are unchanged;
  only one reading changes.
- **Effect.** Mean Brier difference over t ≥ t* + 1 (B1, B2, M), with an author bootstrap CI.
- **No support.** If no author qualifies for a channel, report "no support" and omit it. No new bank and no search.

## Not done

- No regression of same-row prediction loss on that row's material: predictions precede ingestion of the current
  reading.
- No generic topic-removal rule.
- Genuine paraphrase coverage stays a separate, open item.
