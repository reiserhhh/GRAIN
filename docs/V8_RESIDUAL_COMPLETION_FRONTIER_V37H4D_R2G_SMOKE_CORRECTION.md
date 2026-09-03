# R2G Smoke Correction Before Formal Seeds

The first executable smoke completed only after two software-only seed-width
repairs. Scikit-learn requires 32-bit `random_state` values, while the GRAIN
seed ledger uses `SeedSequence` uint64 values. All sklearn seeds are now
mapped deterministically modulo \(2^{32}\), and regression tests cover both
RFF and the overfit trap.

The completed two-root smoke then showed that direct 128-feature RFF on the
full 32-dimensional noisy score did not detect the registered quadratic
positive control: calibration gains were negative and rank zero was selected.
This was a detector-capability failure before any discovery or confirmation
seed was opened.

R2G therefore adds one bounded nonlinear positive-control path:

1. fit an eight-coordinate PCA map on training scores only;
2. expand those coordinates to degree two;
3. fit Ridge to independent target observations;
4. retain the existing output-rank grid and one-SE selection;
5. keep RFF as the more generic comparator.

No gate, effect size, discovery seed, confirmation seed, or claim boundary
changed. The correction is intended to make the known quadratic world
detectable without allowing `factor = target` or test-set tuning.
