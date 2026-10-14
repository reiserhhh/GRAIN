# GRAIN R26/A07 adaptive head-to-head (ALBA-style scalar vs operator-aware), protocol V1

Frozen 2026-09-26 JST before computation. Declared simulation with exact grid posteriors.
No data, no neural calls. ALBA (NAACL 2024) is represented by its IRT-style variant: a
scalar latent with Fisher-information item choice. It is a baseline family, not a
reimplementation of the paper's language models.

- Person: trait tau ~ N(0,1), situation sensitivity kappa ~ N(1, .5^2), nuisance
  nu ~ N(0,1). N = 400 persons, seed 20260926.
- Item bank of 28 context types, repeatable:
  - condition c in {-1.5, -1, -.5, 0, .5, 1, 1.5};
  - natural (discrimination a = 1, artificial 0) or probe (a = 2, artificial 1);
  - nuisance loading n in {.2, 1.0}.
  - P(y = 1) = logistic(a (tau + kappa c) + n nu).
- Budget K = 8 responses.
- Targets:
  - T1 = tau;
  - T2 = tau + kappa c* with c* = 2.0, an unseen situation outside the bank range.
- Policies:
  - F fixed natural, low-nuisance order over c;
  - S scalar CAT: the model assumes kappa = 1 and no nuisance, and picks the maximum
    Fisher information for tau at the current posterior mean, over all 28 items;
  - O operator-aware: an exact posterior over (tau, kappa) with nu integrated out,
    picking the maximum expected posterior-variance reduction of the target;
  - OC operator-aware with cost: utility = gain / prior target variance - .05 artificial
    - .05 n.
- Estimates: posterior means under each policy's own model. S predicts T2 as
  tau_hat + 1 * c*.
- Grids: tau 41 points on [-3,3]; kappa 21 points on [-1,3]; nu 7-point Gauss-Hermite.
- Metrics per policy and target:
  - RMSE against truth;
  - mean share of artificial items;
  - mean nuisance loading;
  - paired person-bootstrap CI (2,000 draws) of RMSE(O) - RMSE(S) and RMSE(OC) - RMSE(S).
- Rule: a policy is better when the paired CI lies below 0. Artificiality and nuisance
  costs are reported beside accuracy, never folded into it after the fact.
