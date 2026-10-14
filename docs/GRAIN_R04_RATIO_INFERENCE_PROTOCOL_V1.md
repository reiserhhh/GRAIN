# R04 output domain of a contrast-normalized measurement: protocol V1 (packet 33 B)

Frozen 2026-09-27 11:13 JST, before any set is computed on r1 data. Authority:
`NEXT_PACKET_33_RATIO_INFERENCE_DOMAIN.md`; `SEMINAR_FABLE_23_DECISION.md`; `SUPERVISOR_CHECK_20260927_1056.md`. Output root
`results/grain_gptpro_foundation_v3/F01/ratio_inference_r1/` (x-mode). r1 and its provenance supplement are read-only.

## Sources and assumptions

- **Sources.** Fieller (1954), JRSSB 16:175–185; von Luxburg & Franz (2007), arXiv:0711.0198; Franz (2007),
  arXiv:0710.2024.
- **Retrieval.** No local copies exist in either repository, and no fresh retrieval was performed. Content was checked
  by the supervisor (1056: Fieller abstract; von Luxburg & Franz full PDF pp.1–4). Standard mathematics is applied; no
  new theorem is claimed.
- **Assumptions.** Per author and query: fixed design X = [1, g, a] with n = 32 and p = 3, so df = 29. Normal,
  homoscedastic, independent event errors within author and query.
  - In the declared generator, events are iid within author. Residuals are Gaussian with a common variance per
    author and query (x = K M e + σ_x ε with ε not scaled by K), so the exact t pivot applies to the
    ratio-of-coefficients target given the fixed design.
  - The two queries' residuals are correlated (they share ε). The joint family is therefore handled by Bonferroni,
    not by independence.
- **Target.** r_q = α_q/β_Gq (the ratio of population regression coefficients), not E(α̂/β̂).

## Set calculation (module `grain_research/ratio_sets.py`)

- **Inversion.** (a − r b)² ≤ t²(V_aa − 2rV_ab + r²V_bb), where V = s²_q (XᵀX)⁻¹ for (α, β_G). Root arithmetic is
  stable (relative tolerance 1e-12), with a linear branch when A = b² − t²V_bb ≈ 0.
- **Output.** A union of intervals: bounded, point, ray, two rays, whole line or empty. Infinities stay explicit. There
  is no grid, clipping, author deletion or invalid → 0.
- **Calibration mapping.** Map to θ through a FIXED affine calibration r = cθ + o. A negative c reverses endpoints and
  remains valid. c = 0 is a separate case: the whole line if o is in the set, else empty.
- **Declared endpoints and families (before outcomes).**
  - E1 / E2: per-query θ-sets for q1 / q2, two-sided α = .05 each (a separate one-readout family).
  - EJ: joint θ-set for the author's language. The family is the 2 queries, Bonferroni α/2 = .025 each; intersect the
    mapped unions. Coverage ≥ .95 under correct fixed calibration and the assumptions, not exactly .95.
  - The two-language, two-query family (α/4) has no monolingual application target here. It is exercised in the
    algebraic fixtures only.
- **Not used.** Stacked m′Ω⁻¹m or exact χ² with an estimated covariance. The source EIV calibration is not treated as a
  Fieller problem.

## Evaluation (one frozen pass; no new seed, world, gain or contrast-strength grid)

- **Data.** All r1 application authors from the provenance supplement: 3 seeds × 400 authors, pairs distinct and
  replica, the original observations and sufficient statistics. Every author is kept in every denominator.
- **Calibrations.**
  - FIXED-KNOWN reference: oracle c = scale, o = off from the calibration bundles (simulator loadings, labelled).
  - PLUG-IN diagnostic: r1 fitted anchor (both pairs) and no-anchor (distinct) calibrations. Coverage is CONDITIONAL
    on the fitted calibration, not unconditional. The negative fitted scale in replica seed 1 maps with reversed
    endpoints; it is not rejected.
- **Metrics per seed, pair, calibration, language and endpoint.**
  - Coverage (θ_true ∈ set; audit only) with Wilson 95% Monte Carlo interval.
  - Set-type proportions.
  - Median and mean finite width.
  - The old r1 delta-interval coverage on identical authors.
  - Pooled over seeds.
- **Optional analytic check** (simulator quantities, labelled). P(per-query set unbounded) = P(|T| ≤ t_crit) for
  noncentral t_29 with δ_u = K_u b_G^L/(σ_res √[(XᵀX)⁻¹]_gg), σ_res² = σ_x²|r_q|² + σ_q². Compare with the observed
  fraction. This is not a K or sensor scan.
- **Independent audit.** Every set is recomputed with an independently coded roots-and-sign-sampling calculation.
  Direct test-inversion membership is checked at the true θ-implied r and at random points.

## Interpretation (declared)

- Unbounded means limited finite-sample information for that author and sensor, not structural non-identification.
- Empty means incompatibility at the chosen test level, not proof of misspecification.
- A negative transport slope alone is not a rejection condition.
- Gaussian-ratio moment failure needs nondegeneracy; there are noise-free and proportional exceptions.
- Packet-32 MSEs remain valid finite-sample statistics, not population guarantees.
- Source-calibration uncertainty remains OPEN; the plug-in results do not close it.

## Bindings

| file | sha256 |
|---|---|
| `grain_research/ratio_sets.py` | `bab04cca83a3e183b9eb3713bb83e0d2d67ed80ee350b44a328b47e502dd80ba` |
| `tests/test_grain_ratio_sets.py` | `0677977f4f81dc98450523fc830e2f8d8ce7a598e3d786093feab1be647508c6` |
| `results/grain_gptpro_foundation_v3/F01/generation_query_provenance_s1/sufficient_statistics.npz` | `0b29fb38e2bd8090ebd6d30b6d6448042f664a242abb292cd5397efe80b433b4` |
| `results/grain_gptpro_foundation_v3/F01/generation_query_provenance_s1/calibration_bundles.json` | `a573a1f784a0607a2dfbd69c26dbf034f65eaf1fa9f34931aba862beed684700` |
| `results/grain_gptpro_foundation_v3/F01/generation_query_provenance_s1/SIMULATOR_ONLY_truth.npz` | `56bfef6b0ae7222cc957926c4d54e2d5dda79f1bee017df24eabc8fdcc1bb26a` |
| `results/grain_gptpro_foundation_v3/F01/generation_query_provenance_s1/results.json` | `be62bcaed58ed80e23b2cc4293f4e7f68de245034a6fee1ba8179aa00ad04ffe` |
| `results/grain_gptpro_foundation_v3/F01/generation_query_r1/target_metrics.csv` | `8596a7a9ef764135ac81968a1ad04e4d6723d5ea2eab05b5af614ed3c540f16b` |
