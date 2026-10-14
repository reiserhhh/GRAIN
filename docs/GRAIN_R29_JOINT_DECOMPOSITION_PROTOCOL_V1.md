# GRAIN R29 joint multi-facet variance decomposition, protocol V1

Frozen 2026-09-26 JST before computation. Declared simulation; no data or neural calls.
Simulation calibrates the decomposition design, not human truth.

- Fully crossed balanced design with 2 replicates per cell:
  - persons P 200, conditions C 4, languages L 2, topics T 3, methods or readers M 2,
    opportunity O 2 (measurer-offered low/high), occasions S 2. Seed 20260926.
- True variance components:
  - P 1.0, C .5, PxC .3, PxS (person state) .4, O .2, PxO .2, L .1, PxL .15, T .3,
    PxT .1, M .2, PxM .1, CxM .05;
  - residual 1.0; all other effects 0.
- Estimator: exact balanced random-effects ANOVA. Effects come from inclusion-exclusion
  of marginal means over all 127 factor subsets, with
  EMS(E) = r * sum_{F >= E} (prod_{k not in F} n_k) sigma^2_F + sigma^2_e, solved from the
  highest order down. Raw estimates are reported; negatives are not truncated.
- CIs: 500 person-bootstrap draws (seed 20260926).
- Contrast: a collapsed analysis over the person x condition x occasion table only
  (averaging L, T, M and O), as separate-world pipelines do. It reports what V_P, V_PxC
  and V_S absorb.
- Rule: the joint model recovers a component when its 95% CI covers the truth. Report the
  bias of every declared non-zero component and the largest absolute estimate among the
  true-zero components.
