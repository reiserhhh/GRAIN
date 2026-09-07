# S8: directional curvature of the mean-null response

2026-09-08. Continues the active goal and the S6/S7 mechanism, not a new student
grid. The previous goal turn made progress: S5-S7 results, theory and code are
committed. Existing unrelated worktree changes remain untouched.

## Decision question

Does the second-order local formula account for the shared nonlinear response
of N/confirmation, and over what intervention radius? A useful approximation
would justify a query-specific curvature/covariance operator; a poor full-radius
fit would direct work toward a nonlinear radial law or integrated operator.

## Fixed experiment

- Same S1 eight authors (panel0-7, view0), three queries, E1/E2 groups.
- Use permutation repeat0 of S7 for each condition, selected by index, not effect.
  This is a directional-curvature experiment, not a precise full-group trace estimate.
- Center m=Pi H; displacement r=permuted(H)-m, for the selected comments only.
  Score F(a)=s(m+a*r), keeping all other/query states unchanged.
- Measure a=0, +/-.0625, +/-.125, +/-.5, -1; reuse the frozen S7 value at +1.
  The new a=0 score must match the S6 endpoint within1e-5.384 forwards total,
  no gradients, thirty-minute cap with complete-author checkpoints.
- Primary derivatives come from epsilon=.125:
  L=[F(e)-F(-e)]/(2e), Q=[F(e)+F(-e)-2F(0)]/(2e^2).
  Prediction at held radii .5/1 is L*a+Q*a^2. The .0625 estimate and its
  Richardson combination are numerical sensitivity analyses, not selected by
  large-radius performance. Record all estimates and any cancellation issues.
- Even response [F(a)+F(-a)]/2-F(0) tests Q*a^2 without odd terms. Odd response
  [F(a)-F(-a)]/2 tests L*a. Neither symmetry of scores nor zero higher terms is
  assumed. The negative path is a controlled hidden extrapolation, not necessarily
  another valid permutation or a meaningful sentence.

Report actual score ranges; paired author metrics; coefficient stability; local
versus held-radius errors; shared N/confirmation curvature; and all failure modes.
No questionnaire labels, new source authors, remote APIs, or old observer queues.
Keep TF as the established positive-control query, not as a forced same mechanism.

## Mathematical basis and scope

The exact finite-group covariance operator was already checked by enumeration
in `tests/test_grain_orbit_covariance.py`. S8 estimates a directional contraction
r'K_q r/2 rather than forming K_q or claiming to measure its entire spectrum.

[Janizek, Sturmfels & Lee (JMLR2021)](https://www.jmlr.org/papers/v22/20-1223.html)
motivates second derivatives and interactions beyond first attribution. Our
central finite-difference experiment is not their Integrated Hessians algorithm.
[Otto et al. (JMLR2025)](https://jmlr.org/papers/v26/24-1315.html) provides the
functional symmetry perspective; our permutation group is finite and the radial
path is not a Lie-group transformation, so we do not apply their Lie-derivative
theorems to radial collapse. Taylor identities and approximation remainders
are explicitly tested instead of assumed.

Only after the mechanism is quantified will we connect its contribution to the
actual D1 N teacher-minus-student remainder. S6-S8 internal effects do not by
themselves identify which part of D1's psychological association they carry.
