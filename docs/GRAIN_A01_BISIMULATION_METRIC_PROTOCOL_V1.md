# GRAIN A01/R08 approximate bisimulation metric, protocol V1

Frozen 2026-09-26 JST before computation. Exact finite computation on declared laws; no
data, no neural calls. Standard mathematics (Ferns, Panangaden and Precup, bisimulation
metrics for finite MDPs, arXiv:1207.4114), no novelty claimed.

- States: length-`order` windows of the F15 worlds `c2_xor`, `c2_iid`,
  `hidden_order3`, plus declared variants:
  - `c2_iid_eps{e}` with p(w) = .5 + e (2 w[-1] - 1), for e in {.01, .05, .10};
  - `c2_xor_ctx`: a controlled world with contexts a in {0, 1} and
    p_a(w) = p(w) + a (.10 if w[-1] = 1 else -.10), clipped to [.01, .99].
- Reward r(s, a) = p_a(s), the emission probability. Transition: next window
  (s[1:], y) with probability p_a(s) or 1 - p_a(s).
- Metric: the unique fixed point of
  d(s,t) = max_a [ c_R |r(s,a) - r(t,a)| + c_T W1_d(P(s,a), P(t,a)) ],
  with c_R = .1 and c_T = .9. W1 is solved as a transport LP (HiGHS). Iterate to a max
  change below 1e-12.
- Checks:
  1. d(s,t) = 0 exactly when s and t are in the same class of exact bisimulation. The
     classes come from an independent partition refinement on (r, class-level transition
     mass).
  2. Value bound for gamma = c_T: |V*(s) - V*(t)| <= d(s,t) / c_R for every pair. V* is
     the exact discounted value, with the max over contexts for the controlled world.
  3. Short-history test: the maximum d between windows that share their last two
     outcomes. It is 0 when the last-two state is sufficient, and positive with its size
     when it is not.
  4. Aggregation: merge states with d below .01. Report the max value error of the
     aggregated chain against the bound 2 * max within-block d / (c_R (1 - gamma)).
