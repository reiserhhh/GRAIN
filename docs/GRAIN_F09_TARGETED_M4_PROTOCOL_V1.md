# F09 targeted M4 contrast protocol v1 (frozen 2026-09-26 before scoring)

Worker: Claude Opus 5.5. Follows `F09/m4_power_bound_r1`. There, the omnibus
out-of-family check detected M4 at .027, although a test at the same 24-event
budget could reach up to .297/.564/.830 (per-atom-level NP bound, M4 beta
.5/1/1.5).

## Statistic (declared)

- T = max over beta in {.5, 1, 1.5} of log L(M4_beta), minus the maximum over
  the nine in-family atoms of log L(atom).
- L is the binomial likelihood of the observed development (even corners) and
  future (odd corners) counts, 3 trials per corner. These are the same data
  used by the archived check.

## Calibration (exact; no application data)

- For each in-family atom a, enumerate all 4^8 count vectors, compute T, and
  obtain the exact null law of T under a.
- p_a(t) = P_a(T >= t). Composite p = max over a of p_a. Reject when the
  composite p <= .05.
- The test therefore has level <= .05 at every in-family atom, the same
  validity standard as the archived composite check.

## Evaluation

- **Stored authors:** the 900 individual checks in
  `F09/calibration_r1/individual_checks.json`. That is validation in-family
  300, application in-family 450, application M4 150. Report rejection rates
  with Wilson intervals, next to the archived omnibus rates.
- **Exact power:** for each M4 beta, P_{M4_beta}(composite p <= .05), by
  enumeration. Compare it with the NP per-atom upper bound.

## Scope

A targeted test detects the declared M4 alternative only. It is not a general
out-of-family detector. Non-rejection still never certifies family
membership.
