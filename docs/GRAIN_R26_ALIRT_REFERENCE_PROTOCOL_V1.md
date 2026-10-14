# R26 ALIRT published-method reference: protocol V1 (packet 34)

Frozen before any application outcome. Authority: `NEXT_PACKET_34_ALIRT_REFERENCE.md`; `SUPERVISOR_CHECK_20260927_1157.md`.
Original clause: P01 p.24 (does response-operator assessment add anything beyond an actual language-based scalar adaptive
method?). Output root `results/grain_gptpro_foundation_v3/F06/alirt_reference_r1/` (x-mode). This is a PAPER-EQUATION
IMPLEMENTATION in SciPy, NOT the authors' code and not a benchmark replay.

## Source and retrieval

- **Paper.** Varadarajan et al. (2024), NAACL, https://aclanthology.org/2024.naacl-long.136.pdf. The PDF was retrieved to
  the job tmp by the worker (sha256 prefix bf93a376f14ad4c8), and the text was extracted locally with pdftotext.
- **Read verbatim by the worker:** Algorithm 1, Section 3.1, Appendix D and the Table 1 caption and rows.
- **Code.** The author page links https://github.com/humanlab/alba-irt, which returned HTTP 404 (worker check, packet
  34). Local R 4.5.1 lacks mirt and mirtCAT, and nothing was installed.

## Fidelity table (paper → implementation)

| paper component (locator) | implemented as | adaptation / omission | consequence |
|---|---|---|---|
| Per-item supervised mapping M_j: ridge regression of the psychometric measure Y on item-j response embeddings, trained on D_poly (Alg.1 l.3; §3.1) | per-item ridge (λ = 1) of source Y on item-j numeric response features x_uj, D_poly only | numeric features replace LSA word embeddings (declared mechanism instantiation) | no natural-language evidence |
| Percentile thresholds on Ŷ_poly (Alg.1 l.6–7; §3.1 quartile example) | K = 4 categories 0–3 via the internal quartiles (25 / 50 / 75%) of Ŷ_poly,j per item | literal Alg.1 uses the (100k/K)-th percentile for k = 1..K and min k with threshold > Ŷ, which leaves values ≥ the 100th percentile uncategorized; reconciled to the prose's K − 1 internal thresholds | valid categories for all inputs |
| Polytomous "2PL" with one discrimination per item and K − 1 thresholds, fitted by EM / BFGS in mirt (Alg.1 l.11; §3.1; App. D) | Samejima graded response model, P*(Y ≥ k \| θ) = σ(a_j(θ − β_jk)), β ordered, a_j > 0; marginal ML over θ ~ N(0,1) with 41-node Gauss–Hermite quadrature; L-BFGS-B, fitted on D_tr only | App. D's written category formula σ(α(θ−β_k)) − σ(α(θ−β_{k−1})) is negative for increasing β, and the prose counts K + 1 values; reconciled to the standard P(k) = P*_k − P*_{k+1}, P*_0 = 1, P*_K = 0; not mirt's EM | real fitted parameters, convergence recorded |
| Next item = maximum Fisher information over unadministered items (Alg.1 l.12–13) | Samejima GRM item information Σ_k (P′_k)²/P_k at the current θ̂; no repeats | — | — |
| Update = MAP of θ from all administered responses, start at the D_tr mean latent (Alg.1 l.15–19; §3.1) | bounded scalar MAP, N(0, 1) prior (the D_tr latent mean is 0 by the model's identification), all administered categories used | — | — |
| Scores L̂ (latent) and Ŷb (average of predicted measures of picked items) (§3.1, §3.4, Table 1) | both computed; L̂ bridged to Y units by a D_poly regression of Y on the full-bank L̂ | bridge added for a common endpoint | declared |
| New-condition target | not in the paper (ALIRT emits a scalar) | source bridge: D_poly regression of the operator-model full-bank R̂ on the full-bank L̂ (declared adaptation) | scalar method cannot represent a slope beyond this bridge |
| 9-fold CV (D_poly 4 : D_tr 4 : D_te 1) | fixed role split per seed: D_poly 400, D_tr 400, application 400 | no CV | declared |

## Declared numeric worlds (mechanism instantiation; not natural language)

- **Item bank.** J = 16 items, a non-repeating fixed bank with recorded condition intensity z_j = linspace(−1.5, 1.5, 16)
  and intercepts d_j (fixed).
- **Expression.** e_uj = d_j + a_u + b_u z_j + ε (sd .5). Features x_uj = e_uj·w + η, with w = (1, .6, −.4, .2) and η sd .5
  per feature. Source measure Y_u = a_u + N(0, .2²), observed in D_poly only.
- **W_pos (positive control, scalar-compatible):** b ≡ 0.
- **W_cond (condition response):** b ~ N(0, .5²), independent of a ~ N(0, 1).
- **Targets (audit truth only).** Scalar τ_u = a_u (Y units). New-condition R_u = a_u + b_u z*, with z* = 2.5 outside the
  bank range.

## Arms (identical items, observations, budgets and source information; no repeats)

- **ALIRT.** MFI selection on the source-fitted GRM with MAP updates.
- **FIXED.** A competent source-chosen fixed order: items ranked by fitted GRM information at θ = 0, top B.
- **OPERATOR.** The existing operator-aware idea, fitted on source only.
  - s_uj = ŵᵀx_uj, with ŵ the first principal direction of pooled centered source x; its sign is set by corr with Y in
    D_poly.
  - Model: s_uj − μ_j = α_u + β_u z_j + e. Item means, a noise-corrected prior and σ² come from D_tr full-bank OLS. The Y
    bridge is a D_poly regression of Y on α̂.
  - Selection minimizes the posterior variance of the new-condition target (1, z*).
- **Scorers on every arm's executed history** (separating selection from estimator): ALIRT L̂ (bridged), ALIRT Ŷb
  (scalar only), and the OPERATOR posterior (both targets; 95% interval coverage).

## Design, uncertainty and checks

- **Design.** Budgets B ∈ {3, 6}. 3 seeds, \`SeedSequence([20260927, 34, seed, role])\`. No grid.
- **Uncertainty.** Paired application-author bootstrap, 2,000 draws, on squared error: ALIRT − FIXED, OPERATOR − ALIRT and
  OPERATOR − FIXED, per scorer, target, budget, world and seed.
- **Checks.** Category probabilities are nonnegative and sum to 1; thresholds are ordered; the analytic information
  matches a numeric derivative; the GRM likelihood gradient matches finite differences; the executed history equals the
  scored history; no repeated item; source isolation (no application truth or simulator loadings in fitting or selection).
- **Positive-control recovery.** Full-bank L̂ against a_u, and a GRM refit on data generated from its own fitted
  parameters.
- **Independent check.** Arithmetic recomputation of one arm's RMSE.
- **Interpretation.** Internal full-bank agreement is kept separate from simulation truth and from human external validity.

## Bindings (appended before any application outcome)

| file | sha256 |
|---|---|
| `grain_research/alirt_reference.py` | `f0e45ae91bf28e70938bcd01f9c89235e34a1a4f94a69c3422854c06dd06c46a` |
| `tests/test_grain_alirt_reference.py` | `96dd6ac89ba229586158c52e3729bcadbbcc8b77948fc4d883f6857becdf1e15` |
