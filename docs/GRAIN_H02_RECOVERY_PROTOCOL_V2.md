# H02 recovery protocol V2: encompassing-model mechanism-set recovery (frozen before simulation)

Date 2026-09-26 JST. It follows `F17/h02_recovery_r1`: under a compound truth (H1+H2), the V1 single-winner rule
decided H1 in up to 100% of simulations. V2 recovers the H02 identification rule itself. It is authorized under
seminar-06 item 3, "H02-specific recovery", as a non-neural gap that changes a design decision.

- **Runner.** `scripts/run_grain_h02_recovery_v2.py`.
- **Output.** `results/grain_gptpro_foundation_v3/F17/h02_recovery_r2/`.
- **Scope.** Simulation only.

## World

The V1 world is reused unchanged:
- 8 half-fraction cells × r = 2, plus 1 held-out cell;
- B observed;
- two blind coders at single-coder reliability .7;
- τ = .25.

Truths:
- single: H1, H2, H3, H4;
- compound, each mechanism at d/2: H1+H2 and H3+H4.

## Estimator (declared)

- **Encompassing model E:** X ~ B + c2 + c3 + c4 + V; A ~ B + c2.
- **Drop-one models E−k.** Each removes mechanism k's terms:
  - H1: B from X;
  - H2: c2 from X and A;
  - H3: c3 from X;
  - H4: c4 and V from X.
- **Scores.** 5-fold leave-persons-out CV predictive log score, summed per person.
- **Test.** For each k, z_k is the paired per-person z of score(E) − score(E−k).
- **Active set.**
  - Primary: {k : z_k ≥ 2.50}, a one-sided Bonferroni threshold (.025/4).
  - Secondary: z_k ≥ 1.96.
- **The V1 single-winner rule** is scored on the same draws for comparison.

## Outcomes and rule

- **Outcomes** per truth × d ∈ {.2, .35, .5} × N ∈ {150, 300, 450, 600}, 100 simulations each (seed 20260927 + cell):
  - P(active set = true set);
  - P(any false inclusion);
  - P(any false exclusion);
  - for compound truths, the V1 rule's P(decided single).
- **Design N.** The smallest N at d = .35 such that, for all 6 truths:
  - P(exact set) ≥ .80;
  - P(any false inclusion) ≤ .10.
- **Pessimistic d.** Reported at d = .2 without changing the rule.
- **Budget.** At most 15 min CPU, 2 workers. These are design inputs under a declared linear world, not validity
  evidence.
