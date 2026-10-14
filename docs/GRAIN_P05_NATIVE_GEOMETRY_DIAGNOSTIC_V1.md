# GRAIN P05 native-geometry diagnostic V1 (seminar-06 item 1)

Frozen 2026-09-26 JST before computation. It uses only existing caches:
P05/native_action_r1 private_jacobians and private_prepared.npz, and
conditional_operator_v1/r2 action_arrays.npz. No new fits, predictions,
hyperparameters or neural calls.

## Finite questions (per query segment, E: D = 7552; N: D = 7680)

- **Q1, leave-one-out source-gradient span coverage.** For each of the 36 source
  gradients J_i: the share of ||J_i||^2 lying in the span of the other 35 (orthogonal
  projection). The span rank uses singular values > 1e-10 x the maximum.
- **Q2, donor-state-change energy in the 36-gradient span.** The share
  ||P y||^2 / ||y||^2 for source donor changes (36) and application donor changes (72).
- **Q3, compiled changes.** The same share for the four compiled application changes
  (E4, J4, E24, J24). J cells lie in the span algebraically, so their share is 1 by
  construction and is not counted as empirical success.
- **Reference.** A random direction's expected share in a 36-dimensional subspace is
  36/D (about .005).
- **Reported:** sample counts, means with 2,000-draw item-bootstrap CIs, quantiles, norm
  summaries, and source/application input support (|z| maxima already recorded).
- **Limits, stated before computing:**
  - Low donor-energy coverage does not imply low action-relevant coverage.
  - High coverage does not imply action preservation.
  - No application Jacobian is measured or inferred.
