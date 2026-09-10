# Source response learning: a retained component with two distinct repairs

2026-09-10. NEXT53 complete, primary freeze `4dd244b`. Same12 author materials,
two queries, four GRAIN directions; four source people, eight previously reused
evaluation people. This is model-component response fidelity, not questionnaire
prediction accuracy or a new confirmatory cohort.

## 1. Main finding

The source-only numerical transport now retains the selected endpoint component
without current blocks17–21. Joint pointwise/causal N correlations are .900/.899,
confirmation .924/.923; both pass the fixed per-query SSE/zero<1 gate. Current
H16 still requires the original prefix, and the target remains one attention22
K-channel contribution to the final logit derivative, not the entire score.

The matched control separates two useful effects:

1. Loss normalization increases effective regularization, suppressing harmful
   nonconstant state corrections. Value-only normalized maps already pass.
2. Adding source GRAIN responses on top of that matched objective further reduces
   new8 endpoint SSE by14.8–19.1%, with only0.26–0.27% higher state error. Full
   tangent error falls1.09–1.12%. A small geometric change can matter substantially
   to a selected query readout; the full-vector average and the task functional
   are different error measures.

Keep both joint maps and their matched controls at this demonstrated scope. Do
not attribute the whole improvement to derivative supervision, or select a new
alpha/factor/query using these outcomes. Source4 leave-one-out endpoints remain
weak and are reported below.

## 2. Protocol and pre-execution correction

The previous plan normalized value/direction losses while retaining numeric
alpha=.01. Before any NEXT53 fitting/outcomes, inspection identified that this
also changes relative regularization versus NEXT52. The amended plan added a
normalization-matched value-only arm. Twenty rather than ten small solves were
frozen in `4dd244b`; the same180s total CPU budget was retained.

Every source reference reuses its exact NEXT52 U32, means, scales, positional
features and same-query Qbar/rbar. Both source queries, equal author/query/token
weights, four leave-one-source-out references plus all4 for every new8 person.
Only B is solved anew. Source primal states and source directional targets are
used; current pre22 states/directions/gradients/actions and human labels are not.
The two losses use source residual energies R_X and R_D; the derivative always
comes from the same fitted function as the state. Values and tangents are not
independently invented and spliced together in the eligible predictors.

    Psi(X)=X+Z(X)B
    D Psi(X)[D_f]=D_f+DZ(X)[D_f]B
    kappa_hat=A_q(Psi(X),D Psi(X)[D_f];Qbar,rbar).

After all48 numerical packets froze, native targets were opened for evaluation.
Identity and original fixed assets replay exactly. Shared local asset links avoid
copying old large caches; no prior artifact was changed. NEXT52 remains intact.

## 3. New8 endpoint comparison

| Method | N r | N SSE/zero | Confirmation r | Confirmation SSE/zero | Contrast r | Contrast SSE/zero |
|---|---:|---:|---:|---:|---:|---:|
| NEXT52 identity | .877807 | 11.643160 | .893467 | 8.704698 | .826269 | 12.187527 |
| NEXT52 primal pointwise | .662383 | 12.619267 | .828645 | 12.134430 | .500384 | 10.751751 |
| NEXT52 primal causal | .688443 | 12.204460 | .829507 | 13.236393 | .535973 | 10.134311 |
| Matched value pointwise | .888445 | .478022 | .920347 | .736096 | .804659 | .429612 |
| Matched value causal | .892728 | .464702 | .921324 | .728110 | .809358 | .419184 |
| Joint pointwise | .899755 | .386722 | .923676 | .626874 | .821376 | .365982 |
| Joint causal | .899276 | .382068 | .923339 | .618137 | .819914 | .366112 |
| Native-input reference, ineligible | .972986 | .156129 | .978394 | .071926 | .963235 | .208588 |

Joint pointwise N/confirmation RMS=.300748/.151461 versus native .223946/.093901,
so some amplitude inflation remains (1.34x/1.61x). Query contrast RMS=.152977 versus
.145007,25/32 same signs. Causal contrast RMS=.152066, also25/32 same signs.
New8 per-factor contrast r: pointwise .759/.781/.765/.975; causal .756/.780/.763/.974.
All factors, signs, source cohorts and query contrasts remain in aggregate tables.

Source4 joint N/confirmation r=.168/.037 pointwise and .164/.029 causal; SSE/zero
2.300/2.468 and2.278/2.460. These do not pass. Their source maps/readers exclude
one source person whereas new8 uses all4, and target amplitudes differ; this
cohort difference is preserved rather than averaged away or repaired after outcomes.

## 4. What was actually learned

| New8 matched joint/value ratio | Pointwise N / C | Causal N / C |
|---|---:|---:|
| Full state error | 1.002560 / 1.002575 | 1.002661 / 1.002667 |
| Full tangent error | .989142 / .989145 | .988818 / .988821 |
| Endpoint SSE | .809004 / .851620 | .822180 / .848961 |

The source-trained derivative term helps on both evaluation queries under the
same normalization. This intervention also changes the fitted **state**; attributing
the endpoint gain exclusively to the tangent branch would go beyond this comparison.
NEXT54 will separate those two routes using frozen states, not another fit/grid.

Post-result coefficient inspection: normalized-value non-intercept norm is1.78%
(pointwise)/1.67%(causal) of historical-primal norm; joint is2.88%/2.44%. The source
intercept is unchanged within2.811e-12 across all references/objectives. Stronger
regularization removed much of the earlier correction, not the common source offset.
The training objective is therefore part of the mechanism, not a harmless numerical
detail. The original primal maps fit states better but were worse functional substitutes.

All4 source pointwise normalized-value -> joint losses: state .655449->.657172,
direction1.001573->.987032; causal .650880->.652702 and1.001593->.985201. These are
source-residual-energy-normalized training losses, not evaluation explained variance.
Source PCA nullspace and zero pointwise BODY-to-query tangent properties remain.
Successful K-component substitution does not require recovery of every native channel.

## 5. Mechanism and next use

GRAIN supplies a fixed perturbation covariance Sigma_G. The directional loss
tr((Jhat-J)Sigma_G(Jhat-J)^T) teaches which changes should be transported. This
uses the same text interface to connect author configuration with query response;
it is not an addition of two unrelated personality scores. Value/derivative
learning is related to [Sobolev Training](https://arxiv.org/abs/1706.04859) and
[Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html). Our finite
four-direction linear-map experiment is not their full neural-training setup.

[Theory](../../docs/GRAIN_RESPONSE_AWARE_STATE_THEORY_V1.md) records the objective,
regularization change and remaining composed-functional question.
[NEXT54](../../docs/GRAIN_RESPONSE_REPAIR_MECHANISM_PLAN_V1.md) is one cached2x2
state/tangent crossing to explain the improvement, then return to whole-score
integration. No further basis/rank/alpha refinement on these materials.

## 6. Reproducibility

Preparation .300335s,20 solves1.549487s, two primary predictions5.650774+5.648977s:
total13.149572s. One two-objective cached replay11.312481s.0 neural/API/Y calls.
48 prediction packets,18 primary CSVs,7 comparison CSVs replay exactly;180 original
asset arrays and48 identity/design packets unchanged. Per-packet independent
metric error<=3.553e-15. All four source-data denial probes and prediction probes
pass; no unexpected data opens.49 focused tests pass,18 existing torch/Python
deprecation warnings; Ruff passes. No full-suite or new-human-validation claim.

A post-result coefficient inspection initially globbed a macOS AppleDouble file
as NPZ; it was corrected to use the exact frozen model inventory. No unsafe pickle,
model refit, primary edit, experiment restart or result change occurred.

Private packet: `results/grain_vector_mechanisms/logic_response_aware_state_v1/`.
Public [tables and audit](response_aware_state_tables/) include matched comparisons,
source losses and coefficient diagnostics, not private author rows or matrices.
Runner: `scripts/run_grain_response_aware_state.py`; auditor:
`scripts/audit_grain_response_aware_state.py`. Completed caches are not overwritten.
No release/scorer/lockbox/MEPS change or remote push.
