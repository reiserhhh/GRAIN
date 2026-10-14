# H02-specific design recovery: protocol V1 (frozen before simulation)

Date 2026-09-26 JST. Authority: seminar-06 decision item 3, "H02-specific recovery ... do not inherit H01 N".
Runner: `scripts/run_grain_h02_recovery.py`. This is a simulation using the actual analysis estimators. It uses
no human data, and human validation stays pending. Output goes to `results/grain_gptpro_foundation_v3/F17/h02_recovery_r1/`.

## Source requirements (read before freezing)

- **P04 p.19.**
  - Rivals: H1 epistemic caution, H2 evaluation anxiety, H3 relationship management, H4 verbal proficiency.
  - Manipulations: C1 evidence certainty, C2 being evaluated, C3 expression affects others, C4 time pressure.
  - Measures: B private belief, X public language, A subsequent action.
  - Estimand: P(B, X, A | H_k, C_j), compared by predictive likelihood or CV/LOO, then a held-out intervention on an
    unseen condition.
- **P04 p.20.** Y_q, Y_b and Y_t are all imperfect indicators of Θ. A text dimension can predict behaviour beyond the
  questionnaire and should not be judged invalid for that.
- **P03 pp.29–31.**
  - Construct Contract: rivals, discriminating interventions, external consequences, invalidating result.
  - Validation matrix: criterion validity, incremental validity (ΔR²), intervention validity (held-out intervention).
  - Hedge paradox: a mean intervention effect and reliable individual response differences are separate questions.
- **P03 p.32.** A 300–600 main sample, with N set by parameter recovery and the expected person×context structure,
  not by a conventional power rule.
- **P05 §3 (task B).** Rival explanations must make different predictions; show reliable individual differences
  rather than naming factors.
- **H02 contract.** Independent behavioural criterion plus blind content coding. X is coded by two blind coders.

## Frozen targets

- **T1 (primary): population-level rival identification.**
  - Each rival is a restricted model for X and A given B and C:
    - H1: X~B, A~B;
    - H2: X~C2, A~B+C2;
    - H3: X~C3, A~B;
    - H4: X~C4+V, A~B, where V is the verbal-proficiency covariate.
  - When B is not observed, B is replaced by C1 as the reduced form.
  - Estimator: 5-fold leave-persons-out CV predictive log score (Gaussian X, logistic A), summed per person.
  - Decision: pick the best model only if its paired per-person log-score difference over the runner-up has z ≥ 1.96.
    Otherwise the result is "undetermined".
  - Outcomes: P(correct and decided), P(wrong and decided), P(undetermined).
- **T2: held-out intervention.** One odd-parity cell (C1=C2=C3=1, C4=0), outside the fitted half fraction, is given
  once per person and never fitted. Outcome: P(the true model has the best log score on the held-out cell).
- **T3: Hedge separation.** Person slopes on the true mechanism's manipulation have SD τ.
  - Mean effect: P(detected) with a one-sample t on person slopes.
  - Individual differences: Spearman–Brown split-half reliability of person slopes.
  - Both are reported, separately.
- **T4: criterion triangulation** (person level).
  - Indicators: Y_q = .7Θ + e; Y_b = .5Θ + s·S + e; Y_t = .5Θ + s·S + e. All are unit variance, and S is a
    text-and-behaviour factor absent from the questionnaire.
  - Outcome: power of the incremental F-test (Y_t over Y_q for Y_b) at s = .3, and its false-positive rate at s = 0.
- **T5: blind-coding precision.** Sampling SD of the ICC(2,1) estimate over n double-coded texts, at true
  single-coder ICC .7 and .6.

## World and design grid (frozen)

- **Trials.** Per person: the 8 even-parity cells of 2^4 (half fraction I = C1C2C3C4) × r reps, plus the 1 held-out
  cell.
  - B = .5·c1 + u_B + e, with c in ±.5 coding.
  - X_true = a_i + mechanism term + e (e ~ N(0,1)).
  - Person intercept a_i: .5Θ_i, or under H4, .5·(.8V_i + .6ξ_i).
  - Mechanism terms at effect d (per-person slope SD τ):
    - H1: −(d + τ_i)·B;
    - H2: +(d + τ_i)·c2;
    - H3: +(d + τ_i)·c3;
    - H4: −(d + τ_i)·c4.
  - A ~ Bernoulli(logit⁻¹(.8·B − 1(H2)·(d + τ_i)·c2)).
  - X_obs is the mean of two coders, each with single-coder reliability .7 relative to X_true.
- **Grid.**
  - Truth ∈ {H1, H2, H3, H4}.
  - d ∈ {.2 pessimistic, .35 primary, .5 optimistic}; τ = .25.
  - N ∈ {150, 300, 450, 600}; r ∈ {1, 2}; B observed ∈ {yes, no}.
  - 100 simulations per cell, seed 20260926 + cell index.
- **Robustness (descriptive).** A compound truth, H1 + H2 at d/2 each, has no single correct answer. The selection
  distribution and P(decided) are reported.
- **Other targets.**
  - T3: τ ∈ {.1, .25} at primary d.
  - T4: N grid, s ∈ {0, .3}, 1,000 simulations.
  - T5: n ∈ {100, 200, 400}, 1,000 simulations.

## Frozen decision rules

- **Design N.** The smallest N in the grid, at r = 2 and B observed, such that under the primary d both of these
  hold for every true H_k:
  - T1 P(correct and decided) ≥ .80;
  - P(wrong and decided) ≤ .05.
- **Pessimistic d.** The same rule is reported at d = .2, without changing the design.
- **B-observation value.** The B-yes vs B-no difference in T1 at the design N.
- **Criterion N.** The smallest N with T4 power ≥ .80. The design N is the larger of the T1 and T4 values.
- **Hedge.** If T3 reliability is < .6 at the design, H02 claims only population-level mechanism identification and no
  individual mechanism assignment.
- **Coding.** The smallest n with ICC SD ≤ .05.
- **Budget and scope.** A finite CPU budget of at most 30 min with at most 2 workers. The results are design inputs
  under a declared linear world, not evidence of validity.
