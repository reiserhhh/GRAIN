# GRAIN H01 person sample size from parameter recovery, protocol V1

Frozen 2026-09-26 JST before computation. This answers re-audit A: H01's person N must
come from a parameter-recovery simulation with state dynamics and person x context
interaction (P03 p.32), not from a fixed 200 or a coverage formula. Simulation only; no
data.

- Model:
  y_{u,d,k} = a_u + b_u z_k + s_{u,d} + e.
  - Contexts z in {-1.5, -.5, .5, 1.5}, one reading each per day, over D = 30 days.
  - State: AR(1), rho .6, innovation SD .5, stationary start.
  - a ~ N(0,1); person x context slope b ~ N(.5, .5^2); e ~ N(0, .7^2).
  - External criterion: C = .2 z(b) + sqrt(1 - .04) noise, with criterion reliability 1
    (declared optimistic).
- Estimation: per-person OLS slope; slope split-half reliability over odd and even days;
  person-slope variance by method of moments, minus the mean sampling variance, with a
  person-bootstrap CI (200 draws); Pearson test of slope vs criterion at two-sided
  alpha .05.
- Grid: N in {100, 150, 200, 300, 450, 600}, 200 simulations each, seed 20260926.
- Recommended N: the smallest grid N with power >= .80 for the slope-criterion test and
  >= .95 detection of positive person-slope variance. The criterion effect r .2 and its
  reliability are declared assumptions. A lower criterion reliability raises N; the
  analytic attenuation for reliability .8 is reported too.
