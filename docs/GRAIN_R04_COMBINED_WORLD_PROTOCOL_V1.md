# GRAIN R04 combined generating world (selection, response, state with X->s, reader, nu), protocol V1

Frozen 2026-09-26 JST before computation. Declared linear-Gaussian world with exact
Gaussian conditioning. No data, no neural calls.

- Person: theta ~ N(0,1), condition sensitivity kappa ~ N(1, .4^2), selection
  q_u(c) proportional to exp(.8 theta beta_c), conditions beta = (-1, 0, 1).
- Assessor action a_t ~ Bernoulli(.5), measurer-chosen and recorded.
- Response: x_t = theta + kappa beta_{c_t} + s_t + eps_t, with eps ~ N(0, .5^2).
- State: s_{t+1} = .6 s_t + .3 a_t + xi x_t + w_t, with w ~ N(0, .4^2) and s_0 = 0.
  xi is the X->s edge: the person's own output feeds the state.
- Reader (sensor only): r_t = x_t + e_t, with e ~ N(0, .3^2).
- Worlds: xi = 0 and xi = .25. N = 400, T = 40, seed 20260926.
- Target: Theta^nu = sum_c nu(c) Psi(theta, kappa, c), with
  Psi = theta + kappa beta_c, the response at resting state.
  - nu1 = uniform;
  - nu2 = (.1, .2, .7), a provoking-weighted reference.
- Estimators, all conditioning on the recorded c_t and a_t:
  - E1 naive mean of r_t;
  - E2 Gaussian posterior of (theta, kappa) under a model with only the a->s edge
    (xi = 0 assumed);
  - E3 the same under the full model (true xi).
  - Theta^nu is estimated as the posterior mean of theta + kappa E_nu[beta].
- Metrics per world, estimator and nu: RMSE, correlation, and Spearman between true
  Theta^nu1 and Theta^nu2 (the effect of nu on person ordering).
