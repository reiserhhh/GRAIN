# R15 hedge-evidence utilization through a source-calibrated reader channel: protocol V1

Frozen 2026-09-26 JST, before any channel quantity was computed.
- **Authority.** `NEXT_PACKET_16_CHANNEL_UTILIZATION_AND_NATURAL_OOM.md` Part A; `SEMINAR_FABLE_14_DECISION.md`.
- **Runner and output.** `scripts/run_grain_r15_hedge_channel_utilization.py`, writing to
  `results/grain_gptpro_foundation_v3/F17/hedge_channel_utilization_r1/`.
- **Budget.** 0 neural calls, CPU ≤ 5 min.

## Inputs (read only; unchanged)

- `F17/hedge_commitment_reader_r1/readings.jsonl`, sha 962b5874…97f11e: 576 NLI pair readings.
- `materials_frozen.json`, sha 1b07fb53…53d12ac: 96 texts.
- **Unit.** One text with its joint response over all queries. Queries are never treated as separate observations,
  and every resampling unit is a text.
- **Split.** Source cores 1–2 (64 texts) fit everything. Test core 3 (32 texts) is held. Test labels are used only
  to build the simulated mixtures and to score or diagnose, never to fit or correct.

## Question and target

- **Question.** When do the detected textual forms yield usable observable-form probabilities for a collection of
  texts, and when does channel or attribution ambiguity prevent it?
- **Target.** The explicit design-form mixture π = (π_bare, π_epistemic, π_deferential, π_noncommittal) of a text
  collection, the merged functional π_epistemic + π_noncommittal, and the owner share π_other. These are properties
  of the materials, not motives, causes or personality.

## Observation operators (declared)

1. **Soft moment channel (primary).** x(text) = (p_ent, p_con) for each query; p_neu = 1 − p_ent − p_con is implied.
   All probabilities are kept in `readings.jsonl`. Three feature sets are fixed:
   - **V1 (primary):** the five speaker-directed queries (own commitment, uncertainty, deference, reported other, no
     commitment), 10 dimensions. Own commitment enters only through its source-calibrated means. Its semantic label
     is not used, consistent with its disabled attribution.
   - **V2:** V1 without own commitment, 8 dimensions.
   - **V3 (diagnostic):** V1 plus the truth diagnostic, 12 dimensions. This shows what the truth-conflation query adds
     to a calibrated channel. It is never a stance or truth reading.
2. **Hard joint code (deliberate coarsening).** Each text's code is the tuple of per-query argmax labels (E/N/C) over
   the V1 queries. The code dictionary is the set of codes seen on source. Every test code outside it is UNSEEN, and
   UNSEEN mass is reported. The channel is P(code | column) = (count + ½) / (n_column + ½·(|dict| + 1)).
3. **No continuous-vector TV.** Raw total variation between empirical sets of unique continuous vectors is not
   computed.

## Channel matrix, covariance and tolerance (all from source)

- **Cells.** c = (form, owner, frame), 16 cells with 4 source texts each (2 cores × 2 wordings).
  - μ_c: the source mean of x.
  - S: the pooled within-cell covariance of x (48 degrees of freedom), with the cross-query covariance kept.
  - D_c = μ_c(core 1) − μ_c(core 2): the source cross-core shift.
- **Tolerance per feature j at mixture size N:**
  δ_j(N) = 2·sqrt(0.75·mean_c D_{c,j}² + S_jj / N).
  - The factor 0.75 is the expected squared distance of a new core's cell mean from the two-core pooled mean,
    relative to the squared distance between two cores, if cores are exchangeable.
  - This assumes shifts are fully correlated across cells, which is conservative for a mixture.
- **Reported descriptively:** the within-cell query correlation matrix and its participation ratio (the effective
  number of non-redundant queries).

## Knowledge scenarios (fixed)

| Scenario | Known | Columns |
|---|---|---|
| K0 | owner and frame; each of the 4 combinations reported | 4 forms |
| K1 | owner (self; other) | 8 = form × frame |
| K2 | frame (neutral; disagreeing) | 8 = form × owner |
| K3 | neither | 16 |

## Analyses

1. **Algebraic identifiability.**
   - For each scenario, A = [M; 1ᵀ], where M holds the columns' μ.
   - For each target functional a (π_f; merged; π_other where owner is unknown), report the relative row-space
     residual ‖(I − A⁺A)a‖/‖a‖, the numerical rank, and the null dimension. The rank tolerance is numpy's default.
2. **Tolerance-aware compatible sets (usable precision).**
   - For the mixture moment m̄, the set is {a·w : w ∈ simplex, |M w − m̄|_j ≤ δ_j(N)}, computed by linear programming
     (min and max).
   - **Population width.** Set m̄ = M w_ref, where w_ref is uniform over the columns. Report the width at N = 100 and
     at N = ∞ (shift term only).
   - **Infeasibility.** If the application m̄ lies outside every compatible point, that is recorded as a channel/test
     incompatibility, not repaired.
3. **The merged epistemic/non-committal grouping is tested, not assumed.**
   - Standardized differences |μ_epi − μ_non|_j / δ_j(∞) per owner and frame.
   - Compatible-set widths of π_epi, π_non and π_epi + π_non.
   - Hard-code overlap.
   - The grouping is recoverable only if the merged width is materially narrower than both parts, where "materially"
     means less than half.
4. **Utilization experiment (finite material resampling).**
   - **Sampling.** Sampled texts are cached test-core readouts, not new people or independent semantic samples.
   - **Targets.** Uniform plus 11 Dirichlet(1,1,1,1) form mixtures, seed 1601.
   - **Owner and frame.** Known in K0 (self, neutral). 50/50 independent of form when unknown: K1 self, K2 neutral,
     K3.
   - **Samples.** For each target, R = 200 replicate collections of N = 100 texts. Each text is drawn by sampling a
     cell from the target, then one of that cell's 2 test texts uniformly.
   - **Estimators of the same target π:**
     - **N1, naive query average:** π̂_f ∝ the mean p_ent of the form's own query. The form-to-query pairs are bare
       → own commitment, epistemic → uncertainty, deferential → deference, non-committal → no commitment.
       Normalized.
     - **N2, naive hard count:** each text is assigned the form whose own query has the largest p_ent.
     - **C1, channel-aware soft:** simplex least squares, weighted by 1/δ_j(N), on the scenario's columns, then
       marginalized to forms. The compatible-set LP interval is computed for the first 10 replicates of each target.
     - **C2, channel-aware hard:** EM (maximum likelihood) on code counts with the source code channel. At most 500
       iterations, with tolerance 1e-10.
     - **π_other** where the owner is unknown: naive = the mean p_ent of reported other; channel-aware = the
       marginal.
   - **Scores.** Mean absolute error per form component, averaged over targets and replicates; merged-functional
     error; LP coverage and mean width.
   - **Source-internal cross-core check.** The same experiment is run on source, fitting the channel on one source
     core and resampling the other core's texts. It is reported before the test result.
   - **Material uncertainty.** The single-core channels (core 1 only; core 2 only) are each applied to the test. The
     spread is reported, not a confidence interval.
   - **Test shift (diagnostic only).** |μ_c(test) − μ_c(source)|_j / δ_j(∞) per cell.
5. **Mapping to the R15 speaker model's utterances**, which are assert, hedge and withhold, with literal-listener
   P(P | u) = .95, .70 and .50 and commitment c(u) = P(P | u) − .5. The model's alphabet is ordered on one axis.
   Declared compatibility requirements for a mapping g from 4 forms to 3 utterances:
   - **R1.** g must merge at least one form pair. Report the pairwise separability of the four forms in each known
     (owner, frame), as max_j |μ_f − μ_g|_j / δ_j(∞). A merged pair that the channel separates means the reader
     carries information the model alphabet cannot represent.
   - **R2.** The model alphabet is one-dimensional. Report the share of standardized between-form variance on the
     first singular axis, for self-owned texts in each frame. A large off-axis share means no order-preserving g
     keeps the detected distinctions.
   - **R3.** In the model, politeness and uncertainty realize the same hedge utterance (packet 14's shared commitment
     profile). If the channel separates deferential from epistemic, the model's alphabet omits a form that the
     reader detects. That is reported as a contract incompatibility, not renamed.
   - **Outcome.** State whether any g satisfies the requirements and under which assumptions; otherwise report the
     contracts as incompatible.

## Rules

- No new readings. No prompt, cutoff or sample-size grids. No rescue of failed channels.
- The feature sets, tolerance formula, scenarios, targets, R and N are fixed here.
- If only coarse sets survive, they are the result.
