# GRAIN R13 response-freedom optimum from a declared mechanism, protocol V1

Frozen 2026-09-26 JST before computation. Exact Gaussian information on a declared
mechanism; no curve shape is imposed. No data, no neural calls. The empirical part stays
H04.

- Openness levels o in {0, .25, .5, .75, 1} (five response-freedom levels, P03 p.16).
- Per response, the observation is x = e(o) theta + s(o) sigma + p(o) pi + eps, with
  independent standard-normal theta (target), sigma (person selection style, a declared
  nuisance for the target), pi (pragmatic or contextual draw) and eps.
- Mechanism:
  - expression gain e(o) = e0 + e1 o: freedom lets the person express the target;
  - selection loading s(o) = s1 o: freedom lets the person choose topics, which loads
    a stable person selection style that is not the target;
  - pragmatic noise p(o) = p1 o^2: freedom opens interpretation and context;
  - constant error var 1.
- The person-level observable is the mean of R = 8 responses. theta and sigma are fixed
  within the person; pi and eps are redrawn per response.
- Constructs (the nuisance declared per construct):
  - C1 selection-sensitive target, where selection style is part of the target:
    target = theta + .5 sigma, and sigma is not nuisance;
  - C2 selection-free target: target = theta, and sigma is nuisance.
- Parameter sets (declared, not tuned):
  - A: e0 .3, e1 .7, s1 .8, p1 1.5;
  - B (low pragmatic noise): p1 .5;
  - C (strong selection): s1 1.5.
- Outputs per level, set and construct:
  - exact I(target; mean x) = -.5 log(1 - rho^2) for jointly Gaussian variables;
  - the argmax over a fine grid of o in [0,1], not only the 5 levels;
  - variance components of the mean observable (target, selection, pragmatic, error).
