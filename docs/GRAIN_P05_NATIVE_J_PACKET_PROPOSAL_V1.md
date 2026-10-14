# Proposal: native J acquisition and matched compression comparison (P05 C1). NOT AUTHORIZED, NOT RUN

Date 2026-09-26 JST. It follows supervisor check 04:35: native J acquisition needs a
separate predeclared packet, and attributing C1 to compression needs a matched
comparison. This file declares the packet so the supervisor can decide. No call is made
until authorization.

## Question

Does an action-weighted fit, L = sum_i ||J_i (dh_i - W de_i)||^2 with J from source
materials only, recover the measured donor action on application pairs? Does it do so
with the 24-coordinate GRAIN input but not with the four population coordinates, all
else matched?

## Budget and materials

- Frozen `conditional_operator_v1/r2`: 36 source pairs and 72 application pairs, cached
  head states, measured donor actions.
- J: the gradient of each query's readout score with respect to the L12.H0 head-state
  input at the LOW source text, for E and N. That is 36 x 2 = 72 backward passes plus
  their 72 forward passes, 144 item calls in total. The cap is 160, enforced before each
  call.
- Same local observer, device, dtype and site as r2. No API, no generation, no training
  of neural weights. Per-item saves; no automatic retry.
- The action evaluation reuses the measured donor actions. A NEW compiled state change
  would need forwards to measure its action; that step is excluded here. Evaluation uses
  the J-linear predicted action, validated against measured compiled and donor actions
  on the same pairs before use.

## Matched comparison (frozen before any J is read)

- Inputs: de_4 (the four frozen coordinates) or de_24 (the frozen GRAIN runtime), the
  same pairs and splits.
- Fits, on source pairs only, one fixed estimator with no alpha or rank search:
  - (a) Euclidean hidden-state reconstruction, the r2 form;
  - (b) J-weighted reconstruction;
  - (c) direct regression to J dh.
- Metric: application-pair correlation and RMSE against the measured donor action. The
  paired bootstrap is over cells.
- Attribution rule: compression loss is supported only if (b) with de_24 beats (b) with
  de_4 and the paired CI excludes 0. A metric effect is supported only if (b) beats (a)
  at matched input.

## Boundaries

- 144 item calls maximum. Nothing runs until the supervisor authorizes.
- It preserves r2 and all prior negative results. No held-out promotion.
