# H02 recovery protocol V1, amendment A1 (after the smoke test, before the main run)

Date 2026-09-26 JST. V1 (sha 96a92840…) is unchanged. The main grid has not been run.

## What the smoke test showed (code check, 1 simulation per truth, not a result)

The frozen T4 incremental F-test (Y_t over Y_q predicting Y_b) rejected in 70% of 200 draws at s = 0 and N = 300.
This is not a bug:
- Y_q = .7Θ + e is an imperfect indicator.
- Y_t = .5Θ + e is a second indicator of the same Θ, so it raises the prediction of Y_b.

The zero-increment null is therefore false even without any text-specific factor S. The incremental test cannot
separate "text carries S beyond the construct" from "text reduces questionnaire error". This is the P04 p.20
point that the questionnaire is not ground truth. T4 stays as frozen, and its s = 0 rate is reported as a
diagnostic property, not as a false-positive rate.

## Added T4b (declared now)

- **Indicators.** A fourth indicator Y_o = .6Θ + e is added. It is the other-report criterion that the H02
  contract already requires (`criterion_source`).
- **Test.** A one-factor model implies the tetrad τ = ρ_tb·ρ_qo − ρ_tq·ρ_bo = 0. S (shared only by Y_t and Y_b)
  makes τ > 0.
- **Estimator.** The sample tetrad with a 200-draw person-bootstrap SE; one-sided test at z ≥ 1.645.
- **Outcomes.** Power at s = .3 and false-positive rate at s = 0, for N ∈ {150, 300, 450, 600}, 1,000 simulations
  each.
- **Rule.** The criterion N is the smallest N with T4b power ≥ .80 and a false-positive rate ≤ .07. T4 is kept for
  reporting only.
