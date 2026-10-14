# GRAIN R11 measurement modes under a penalized objective, protocol V1

Frozen 2026-09-26 JST before computation. Declared simulation with an exact grid
posterior. No data, no neural calls. Source: P01 pp.18-22,
a* = argmax I - lambda B - gamma D, with three modes.

- Persons: N = 500, theta ~ N(0,1), T = 20 turns, seed 20260926.
- Environment (naturalistic) turns:
  - The condition is high provocation with probability logistic(2 h_u).
  - The hidden cue h_u = .7 theta + .71 noise is not observed: the policy is endogenous.
  - The environment effect beta_u ~ N(1, .5) varies by person and is unknown to the
    estimator, which assumes 1. Naturalistic data are therefore not on a linked scale.
  - Low provocation has effect 0.
- Probe turns use standardized probes with known effects:
  - P1 mild: effect .5, reactivity .05;
  - P2 strong: effect 1.5, reactivity .20.
  - Reactivity adds to the person's state for all later turns and is not modelled by the
    estimator (naive practice).
- Response: y_t ~ Bernoulli(logistic(theta + effect_t + state_t)).
- Estimator: grid posterior over theta (81 points on [-4,4]), N(0,1) prior, likelihood
  with the recorded condition of every turn.
- Modes:
  - NAT naturalistic only;
  - CAL naturalistic plus fixed P1 probes at turns 5 and 15 for everyone;
  - TRIG at each turn, if the posterior SD is above .45, choose among
    {no probe, P1, P2} by argmax I - lambda B - gamma D, otherwise naturalistic;
  - ADAPT the same choice at every turn, without a trigger.
  - I is the expected posterior-variance reduction of theta. B is 1 per probe
    (lambda = .02). D is the probe's reactivity (gamma = .10). The no-probe option is a
    naturalistic turn with I evaluated at the assumed effect.
- Metrics per mode:
  - RMSE and correlation of the posterior mean;
  - correlation of the error with h_u (endogeneity);
  - probes per person (burden);
  - accumulated reactivity (distortion);
  - realized utility -MSE - .02 probes - .10 distortion;
  - paired person-bootstrap CI (2,000 draws) of each mode's utility minus NAT.

## Amendment A1 (after the first run, before any joint-model computation)

- First-run finding: TRIG and ADAPT chose zero probes. The planner values naturalistic
  turns under its assumed environment effect (1), so probes never beat them after cost,
  and both modes equal NAT up to Monte Carlo noise. This is recorded as a result about
  the objective, not repaired silently.
- A1 adds joint-model variants J-NAT, J-CAL, J-TRIG and J-ADAPT. The estimator and
  planner use a joint grid posterior over (theta, beta_u):
  - theta has 81 points on [-4,4];
  - beta_u has 21 points on [-1,3] with an N(1, .5^2) prior;
  - naturalistic high-provocation turns load beta_u;
  - probes have known effects.
  - I is the expected posterior-variance reduction of theta under the joint model.
    Everything else is unchanged (lambda, gamma, trigger .45, seeds, persons).
- Outputs go to `measurement_modes_joint.json`. The first-run outputs are kept.
