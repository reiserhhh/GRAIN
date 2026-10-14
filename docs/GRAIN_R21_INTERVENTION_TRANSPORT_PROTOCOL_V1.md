# R21/R23 level vs intervention-signature transport: protocol V1 (frozen before computation)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_09_DECISION.md` Packet A.
- **Runner.** `scripts/run_grain_r21_intervention_transport.py`, writing to
  `results/grain_gptpro_foundation_v3/F12/intervention_transport_r1/`.
- **Scope.** Zero neural, API or GPU calls. CPU only, 1 thread, 60-minute cap.

## Question

Does an affine score-unit map, calibrated on ordinary readings of one reader, also transport that reader's response to
specified text interventions? Source: P04 p.13, preserving the intervention effects ∂E[T]/∂do(C) after calibration
g; P02 pp.25–27, functional distance over a probe family. The interventions are controlled text edits: not
do-interventions on true personality, and not a global supremum over inputs.

## Inputs (numeric caches only; verified, readers not rerun)

- **Readings.** `R14/counterfactual_reader_r1/results.jsonl`: 40 units, 40 authors (20 PANDORA, 20 Essays). Five
  conditions: orig, neu, own E+ (pos), own E− (neg), other E+ (oth).
  - Q = the Qwen E readout, a candidate-conditioned digit expectation in [0, 1].
  - N = the NLI p_ent − p_contra in [−1, 1].
  - Scores stay in their native units.
- **Verification.**
  - Every Q and N value must equal its OK record in `accounting.jsonl` exactly.
  - The unit keys, corpora and condition IDs must match `input_checkpoint.json`.
  - All 40 units must be complete.
  - STOP on any mismatch.

## Roles and map

- **Two predeclared transfers:** source PANDORA → application Essays, and the reverse. Source and application
  authors are disjoint.
- **Map.** A single OLS affine map Q = a + b·N, fit on the source authors' **orig and neu readings only**, giving
  40 points from 20 authors.
  - Intervention arms never enter the fit.
  - No clipping; mapped values outside [0, 1] are counted.
  - Rank tolerance: if the variance of the source N is below 1e-12, the slope is reported as non-identified and the
    map is intercept-only. No regularization.

## Estimands (application authors)

1. **Level agreement** on orig and neu, reported separately and together. Mapped M = a + b·N against Q:
   bias mean(M − Q), RMSE, Pearson, Spearman, and variance ratio var(M)/var(Q).
2. **Signature.** For c ∈ {pos, neg, oth}: ΔQ(c) = Q(c) − Q(neu) and Δmap(c) = b·(N(c) − N(neu)).
   - Per arm: mean ΔQ, mean Δmap, mean paired mismatch (Δmap − ΔQ), mismatch RMSE and sign agreement.
   - Population signature discrepancy: the Euclidean norm of the 3-vector of mean mismatches.
   - Author signature discrepancy: the mean over authors of the 3-component norm.
   - Ownership contrast: Q(oth) − Q(pos) against b·(N(oth) − N(pos)).
3. **Individual response configuration.** The statistic is S = mean_i ‖Δmap_i − ΔQ_i‖² over the 3-component author
   vectors.
   - Compare the matched S with S under whole-vector author permutations within the application corpus. This
     preserves arm means and within-vector correlation.
   - Also report the correlation of the author-centred stacked deviations, matched vs permuted.
   - A common response surviving the permutation is not an author-specific shared pathway.

## Uncertainty and references

- **Primary intervals.** 2,000 paired author bootstrap draws per direction, resampling application authors with all
  their conditions; conditional on the fitted source map; seed 20260926.
- **Fit-uncertainty companion.** 2,000 paired source+application draws, refitting the same OLS on each source
  resample. Singular draws are reported, not dropped.
- **Null 1, no source correspondence.** 200 seeded source-author block permutations: Q and N are unpaired at the
  author level, with both readings of an author moving together. The level map is refit and the level and signature
  estimands are recomputed on the application authors.
- **Null 2, no author-specific configuration.** 200 application-author whole-signature shuffles after the correct
  fit.
- Neither null is an unrelated psychological-construct control.

## Algebraic checks

- Δmap = map(c) − map(neu) = b·ΔN, to 1e-12.
- Every permutation preserves arm means exactly.
- Roles are disjoint.
- The rank condition is recorded.

## Not included

- No criterion regression. `F12/criterion_preservation_r1` is cited as separate historical evidence with its
  corrections.
- No new labels, no psychological criterion, no feature/head/rank grid.
- No equivalence claim from non-significance, and no accuracy threshold.
