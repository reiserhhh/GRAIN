# GRAIN R05 opportunity / selection / response / condition four-way decomposition, protocol V1

Frozen 2026-09-26 JST before computation. Exact finite computation on a declared
population. No data, no neural calls.

- Conditions c = 1..3 with common response effects delta = (-.8, 0, .8) on the logit.
- Person u (N = 2,000, seed 20260926):
  - opportunity pi_uc = logistic(a_u + A_c), with a_u ~ N(0, .8) and A = (.5, 0, -.5);
  - selection q_uc = logistic(b_u + b'_u s_c + B_c), with b_u ~ N(0, .6), b'_u ~ N(0, .8),
    s = (-1, 0, 1) and B = (0, 0, 0);
  - response p_uc = logistic(r_u + k_u delta_c), with r_u ~ N(-.5, .7) and
    k_u ~ N(1, .3).
- Observed published behavior share: S_u = sum_c w_uc p_uc, with
  w_uc = pi_uc q_uc / sum_c pi_uc q_uc.
- Reference person: population means of a, b, b', r, k.
- Four players, each switched from the reference to the person's value:
  - O = (a_u);
  - Q = (b_u, b'_u);
  - R = (r_u, k_u);
  - C = condition effects present (delta), versus absent (delta = 0).
- Exact Shapley decomposition over the 4! orders of S_u - S_ref. Variance attribution
  share_j = cov(phi_j, S - S_ref) / var(S). The shares sum to 1.
- Identification under designs (exact counterexamples, equality of observables to 1e-12):
  - D_full: opportunities logged, selections logged, condition coded. All four players
    identified, with a constructive solve.
  - D_noO: only published texts with coded condition. The per-condition pi*q is
    identified up to scale, so O and Q are not separable. A counterexample shows equal
    observables with different O and Q shares.
  - D_noC: condition not coded, only the pooled published share. A counterexample shows
    equal observables with different R and C attributions.
