# GPT Pro V3 F08 end-to-end: sources fitted on reader-read text (2026-09-26)

Worker: Claude Opus 5.5, TASK-CLAUDE-GRAIN-FULL-LOOP-20260926. Requirement rows
R14 (joint observation law end-to-end), R16 (reader accuracy is not target
validity) and R06 (response-dependent missingness in the likelihood).

## Question

The archived F08 downstream packets fitted the source B1/B2/M models on oracle
source observations, and only the application histories passed through the
reader. In deployment, the source cohort is also text and would be read by the
same reader. The question is whether fitting the source on reader-read text
changes application loss.

## Run

- Script: `scripts/run_grain_gptpro_f08_end_to_end.py`. Output:
  `results/grain_gptpro_foundation_v3/F08/end_to_end_r1`.
- Coverage: 2 measured readers (lexical `reader_recovery_r1`, NLI
  `nli_reader_r1`, cached scores only, no new forwards) x worlds c2_xor/c2_iid x
  3 seeds x panels complete/mixed.
- Source authors: 300 per world and seed, rendered on source-role materials
  with a declared new stream. Application authors: 200, rendered with the
  archived downstream streams.
- Estimator `F08_gap_aware_read_source_v1`: unqualified reads are missing, and
  triples come only from contiguous read runs.
- Analysis: `scripts/analyze_grain_gptpro_f08_end_to_end.py`. Decomposition:
  `scripts/analyze_grain_gptpro_f08_source_read_decomposition.py` writes
  `F08/source_read_decomposition_r1`.

## Results

1. **Replay.** Both oracle-source arms reproduce the archived downstream
   application losses exactly (0.0 over 7,200 rows per reader and arm).
2. **Complete panel.** Read-fitted equals oracle-fitted for both readers, with
   0 source read errors.
3. **Lexical, mixed panel.** Effects are small, below .003 Brier in every cell:
   B1 iid +.00033 HIGHER, M iid -.00261 LOWER, M xor +.00090 HIGHER, the rest
   INCONCLUSIVE.
4. **NLI, mixed panel.** Every candidate has higher loss:

   | candidate | iid | xor |
   |---|---|---|
   | B1 | +.1275 [.1229,.1325] | +.1239 [.1178,.1302] |
   | B2 | +.0093 | +.0106 |
   | M | +.0099 | +.0061 |

5. **Mechanism.** The measured NLI reader keeps qualified positives at .5
   (actual) and 0 (negated), and reads quoted negatives as qualified.
   - Read source events have a true positive rate of .27–.30, against .49–.51
     over all events.
   - The B1 source table shift splits into selection (MNAR missingness) .242
     and value errors .109; the full read is .491, with both pushing the same
     way.
   - The lexical shift (.028) is finite-sample thinning: the gap-aware fit
     keeps about 9% of triples.

## Decision and production rule

The source side is a second place where the reader enters the estimand's
estimator. It is not only an application-side channel.

A reader that is harmless or even helpful at application can corrupt the
source model when it is used to read the source cohort. Dropping unqualified
events is not ignorable when missingness depends on the outcome.

Rule: a text-route source model is fitted on controlled or oracle source
observations, or with the reader's measured emission law in the likelihood. It
is never fitted by dropping unqualified reads. `F08_gap_aware_read_source_v1`
is retained as a failed estimator.

Statuses: implementation DONE; protocol DONE for this packet. Science is
SUPPORTED_FINITE for two controlled worlds and two measured readers. No claim
is made about natural text or human validity.

Next executable gap: an MNAR-aware source fit using a measured emission law
estimated on non-source materials.

## Follow-up: MNAR-aware source fit (`F08/mnar_source_r1`)

Protocol `docs/GRAIN_F08_MNAR_SOURCE_PROTOCOL_V1.md`, frozen before the run.
The source likelihood is fitted by EM over the second-order chain, with the
reader's emission law P(missing/0/1 | y). Two laws are used: one measured on
validation-role materials (deployable) and the true source-material law
(oracle emission reference). The application side is unchanged.

- **Replays:** exact (0.0).
- **B1/B2:** the collapse is removed.

  | arm | NLI B1 iid | NLI B1 xor |
  |---|---|---|
  | gap-aware | +.1275 | +.1239 |
  | EM, deployable law | +.0007 | -.0004 |
  | EM, true law | +.0005 | -.0017 |

  B2 is within ±.0008 of the oracle-fitted source. The deployable law differs
  visibly from the true law (P(missing | y=1) .5625 vs .75), yet costs at most
  .0013.
- **M is not recovered:** NLI +.004 to +.006, lexical +.0005 to +.0022. This
  is the declared pooled-E-step approximation, which shrinks author tables. An
  author-level E-step is the next gap.
- **Protocol deviation (retained):** the frozen check "log-likelihood
  non-decreasing" failed in 4 of 24 fit rows, by up to 4.9e-7. The smoothed
  M-step is a MAP step, so EM guarantees ascent of the penalized objective, not
  of the likelihood. An independent reimplementation reproduces every fit
  (<= 1.4e-15), and the penalized objective is non-decreasing up to rounding
  (-3.6e-12). The protocol status is PARTIAL.
- **Loss below the oracle source** (NLI xor) is compensation for the MAR
  application filter, not a better source model.

Refined rule: fit text-route sources with the measured emission law in the
likelihood. Population transition models are then usable; author-structure
models need an author-level E-step.

## Follow-up 2: author-level E-step for M (`F08/author_estep_r1`)

Protocol `docs/GRAIN_F08_AUTHOR_ESTEP_PROTOCOL_V1.md`, frozen before the run.
The MAP-EM monotonicity criterion was corrected and declared in advance.

- **Method:** each source author's second-order table is fitted by its own
  MAP-EM under the same measured emission law, starting from the pooled
  solution. B1/B2 are unchanged.
- **Checks:** all replays are exact. The per-author penalized objective is
  monotone (-4.3e-14).
- **Result:** M improves on the pooled E-step in all 8 contrasts.

  | reader, world | author E-step M - oracle source |
  |---|---|
  | NLI xor | -.0081 (deployable law) / -.0097 (true law) |
  | lexical xor | -.0010 |
  | lexical iid | -.0034 |
  | NLI iid | +.0051 (deployable) / +.0016 (true law), still higher |

- **Mechanism:** the pooled E-step shrank between-author table variation (SD
  .04-.08 against .11-.13 for complete data); the author E-step restores it.
- **Caveat:** individual author tables are noisier than the pooled
  imputations (RMSE .15-.24 against .09-.13) and must not be reported as person
  estimates.
- **Final production rule:** measured emission law in the source likelihood;
  a pooled E-step for population models; an author-level E-step for
  author-structure models; never drop unqualified reads.
