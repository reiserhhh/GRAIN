# HM15: Full-Vector Ordered Action Kernel

2026-10-08. Planned after verified HM14, not executed. Finite zero-forward
mechanism development on existing controlled states, not new text replication.

## Question

Is the remaining action deficit primarily caused by the original rank2
hidden-vector projection, or does a receiver-independent linear response
remain insufficient even when hidden vectors and short-span order are kept?
This is an observation/response discrimination, not a personality accuracy grid.

## Scope and before-fit freeze

1. Verify frozen HM10B3/HM12/HM13/HM14 artifacts and code. Never replay the
   10368/768/252 old model calls or modify frozen primary files.
2. Keep block16, the original three queries, source8/validation4/application4
   family roles, all A/T/order/polarity/opportunity cells and full original
   token positions. Same-word/token alignment is required.
3. Reuse all1404 HM14 endpoints and append384 original projection plus384
   original complement endpoints at block16:2172 query intervention rows.
   Expected roles source1080/validation528/application564; fail if actual
   inherited design disagrees. This is an inherited follow-up, not a fresh trial.
4. Reconstruct actual float32 projection/complement states using frozen
   intervention code and its original arithmetic/cast order. Verify state
   addresses, native logits and receiver references before freezing inputs.
5. Separate source targets from held targets. Source compilation must not
   load held target values. Store explicit reference/target/full-state hashes.

## Fixed observations and functions

Use six span means and property moments1–3, as in HM14. Keep lengths1–4
and exact positions; no degree search. Compare zero action, rank2 ordered
observation and full1536 hidden coordinates. All spans preserved; don't
drop inconvenient complements or automatically remove condition means.

Both nonzero views use a single source-only RMS norm, not coordinatewise
RMS: s^2=sum(w_i||d_i||^2)/sum(w_i). This fixed convention makes a hidden
orthogonal basis change an isometry of the kernel. New rank2 fitting uses
this same rule; old HM14 per-coordinate-RMS results remain separate.
No application renormalization, intercept, alpha search or model selection.

Per query, fixedalpha1/family-balanced source loss and a linear Gram action
kernel. Solve the weighted source system in sample coordinates; store
source feature/reference hashes, source coefficients, scaling and functions
before loading validation/application actions. Upstream states still come
from the frozen transformer; kernel scoring itself makes no model calls.

## Checks and scientific outputs

- Exact identical-input zero; one-token reversal controls; original rank2.
- Moment/parity/reconstruction inherited checks and explicit full-state shape.
- Gram symmetry/PSD and orthogonal hidden-coordinate gauge invariance.
- Independent augmented primal least-squares on manageable synthetic data,
  plus eigen/dual replay on actual source Gram systems, no held refitting.
- Every actual reference action,2172 addresses and all per-query predictions.
- Full versus rank2 paired family MSE/sign/correlation, per intervention,
  A/T/order/token opportunity and source/validation/application separately.
- Projection/complement actual action versus retained-coordinate difference:
  rounded near-zero is not called exact0 or mathematical impossibility.
- Report source support/rank and model residual separately. No fit-derived
  author personality names or external accuracy claims.

Freeze primary code/config/input/functions/output and alternate verification.
Update parent report/checkpoint and next scientific decision. Full-vector
gain sends us toward source-supported action coordinates; similar failure
sends us toward receiver-conditioned covector/interaction tests. Do not
continue a precision/rank/layer/prompt/alpha grid in either branch.

Canonical planned root:
`results/grain_math_response_v1/full_vector_action_kernel_r1`.
Expected new model/API/private-text/neural-training calls:0.
No Claude/automation/commit/push/tag/lockbox/old queues. Natural authors,
personalB_u,psychological interpretation and original18/100/44 remain tracked.

Theory companion `docs/GRAIN_ORDERED_OBSERVATION_AND_RESPONSE_V1.md` uses the
standard source-span representer argument, not a novel optimization theorem
or a PDE uniqueness theorem transferred to text.
