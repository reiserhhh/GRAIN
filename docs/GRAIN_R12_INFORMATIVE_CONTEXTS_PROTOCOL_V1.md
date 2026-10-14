# R12 informative contexts versus prompt counts: protocol V1 (packet 31 B)

Frozen 2026-09-27 09:11 JST, before any outcome. Authority: `NEXT_PACKET_31_INFORMATIVE_CONTEXTS.md`,
`SEMINAR_FABLE_22_DECISION.md`, `SUPERVISOR_CHECK_20260927_0855.md`. Original clause: P02 pp.29–30 (independent
informative contexts, not text count; information-matrix rank). Declared world, known loadings, simulated numbers only.
Output root `results/grain_gptpro_foundation_v3/F04/informative_contexts_r1/` (x-mode).

## Model

- **Observation.** y_ut = φ_{c(t)}ᵀθ_u + δ_{u,a(c)} + ε_ut, with φ = (1, zA, zB), θ_u ∈ ℝ³.
  - θ ~ N(0, diag(1, .25, .25)).
  - δ ~ N(0, τ²), persistent within an ACTUAL context a.
  - ε ~ N(0, σ²), independent per reading.
- **Target.** The global θ. δ is nuisance FOR THIS target; that is not a claim that author-context variation is
  meaningless.
- **Information.** Per-author reading covariance R = σ²I + τ²·[same actual context]. Then
  I = ΦᵀR⁻¹Φ = Σ_a n_a φ_aφ_aᵀ/(σ² + n_aτ²), checked against the dense 32 × 32 block inverse.
- **Risk.** For q in the row space of I, the known-covariance GLS variance is qᵀI⁺q. Otherwise the target is
  UNIDENTIFIED with a null-space witness; it is never given zero or finite data risk.
- **Declared-prior reference** (separate, labelled prior-driven): (Σθ⁻¹ + I)⁻¹ with Σθ = diag(1, .25, .25).

## Designs (N = 32 readings per author; no grid)

- **D1.** 4 contexts, zA = (√2, √2, −√2, −√2), zB = 0, 8 readings each. Rank 2.
- **D2.** 4 corners (±1, ±1), 8 readings each. Rank 3. Same N, K and Σn(zA² + zB²) = 64 as D1, NOT the same
  directional information.
- **D3.** Each D2 loading duplicated into 2 ACTUAL contexts with independent δ, 4 readings each. Same N and loading
  distribution as D2; K = 8.
- **D2_relabel.** The same observations and actual context identities as D2, with 8 nominal labels. Correct scoring
  must equal D2. Naive independent-label information is reported only as a deliberately wrong analytic reference.
- **Regimes.** σ² = 1; τ² ∈ {0, .25}.
- **Targets.** q_A = e_A, q_B = e_B, q_AB = (e_A + e_B)/√2. Also full-operator rank and unidentified directions.

## Simulation, estimator and scoring

- **Streams.** 3 seeds. For each seed, application draws come from `SeedSequence([20260927, 31, seed, 1])`: θ (400 × 3),
  δ0 (400 × 8), ε0 (400 × 32). Source draws come from `[.., 2]`: 200 authors, the same shapes.
- **Pairing.** The same base draws serve every design and both regimes (y = φᵀθ + √τ²·δ0[slot] + √σ²·ε0[t]). Every
  design keeps iid δ per actual context with variance τ².
- **Source-only noise estimator.** Source panel in the D3 layout (8 contexts × 4 readings, chosen a priori for df).
  - σ̂² is the pooled within-context variance (df = 200·8·3).
  - τ̂²_raw = mean square of per-author OLS context-mean residuals (df = 200·5) − σ̂²/4.
  - Scoring uses τ̂²_used = max(τ̂²_raw, 0), with the raw value and boundary flag retained.
  - No θ or generating variances are passed to the fitter. Noise is frozen before application scoring.
- **Application scoring.** GLS on actual-context means, with the minimum-norm solution for rank-deficient designs.
  Only identifiable targets are scored.
  - Balanced homoscedastic designs make GLS and OLS point estimates coincide; noise estimation affects SE and
    coverage, not point accuracy.
  - The oracle known-covariance SE is reported separately.
- **Metrics per design, target, regime and seed.** Empirical MSE, bias, 95% interval coverage (source-estimated SE and
  oracle SE), squared correlation corr(score, truth)² (labelled), analytic risk, eigenvalues and rank.
- **Paired contrasts.** Application-author bootstrap, 2,000 draws, `SeedSequence([20260927, 31, seed, 9])`, the same
  indices for every design, conditional on the fitted source. Contrasts on MSE:
  - D1 − D2 (target A);
  - D2 − D3 (targets A, B, AB);
  - D2 − D2_relabel (all targets; must be exactly 0).

## Decision reading (declared)

- D1 versus D2 can hurt a narrow target while making another identifiable.
- D2 versus D3: equal at τ² = 0, persistent-context floor at τ² > 0. This is an algebraic identity under the model.
- Relabel: identical with correct covariance. Context count is not independence.
- No optimal K, no general diversity theorem, no real-token or human claim. Null or unidentified results are valid.

## Bindings

| file | sha256 |
|---|---|
| `grain_research/r12_informative_contexts.py` | `561b1b19165333b43403b1c225e52013d8b7e8dabcfdf71a5c60924d9fb596b7` |
| `tests/test_grain_r12_informative_contexts.py` | `24b6f1ab46e840f15ba0cbb742f79171f8cfb2f44c1bdab8510f9509363ef7a1` |
