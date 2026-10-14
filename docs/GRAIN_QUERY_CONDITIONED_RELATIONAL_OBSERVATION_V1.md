# Query-conditioned relational observation: finite HM50 finding

2026-10-09. HM50 executed and numerically verified. A controlled reader mechanism,
not a generic binding theorem, human personality validation or full-goal closure.
Preserve frozen HM46-HM50 programs, plans, material and results.

## Objects

For a fixed query q_j about person r, write the lower-layer full state as

```text
B_l(x)       complete body states before the query
C_{j,r,l}(x) complete suffix states after reading body plus fixed query
F_{j,r}(x) = D_l(B_l(x), C_{j,r,l}(x))
```

D_l includes all continuing upper layers and the frozen scalar readout. The
common preceding prompt is held fixed. C includes query/instruction/chat-wrapper
positions, not only the target word. It depends on x even when suffix token IDs
are identical. No claim that every suffix position is necessary.

Let T exchange only two body ownership-name tokens, leaving property content,
token positions and the query unchanged. The four intervention cells are

```text
Y00 = D_l(B_l(x),  C_{j,r,l}(x))
Y10 = D_l(B_l(Tx), C_{j,r,l}(x))
Y01 = D_l(B_l(x),  C_{j,r,l}(Tx))
Y11 = D_l(B_l(Tx), C_{j,r,l}(Tx))
I_BC = Y11 - Y10 - Y01 + Y00
Y11 - Y00 = (Y10 - Y00) + (Y01 - Y00) + I_BC
```

This is standard finite-difference accounting. Retain signed cross terms when
computing squared effects; component RMS values are not additive information.

## Actual finite evidence

All48 receiver conditions on block16: six matched targets, two queried persons,
two ownership directions, two authored behaviour families. Source192 forwards
sealed before application192. Seven patch arms plus48 native full captures.
Only two name tokens differ; suffix IDs and positions within each pair match.
This is a result-guided HM48/HM49 follow-up, not blind new-material replication.

Application complete natural exchange RMS0.761301; body-only RMS0.007115;
suffix-only RMS0.761448, signed projection1.000148, remaining RMSE0.007115.
Suffix effects align with full exchange24/24, including4/4 for each target.
Source shows the same finite distinction: full0.747490, body0.008313,
suffix0.747891 and remaining0.008313. Interaction RMS source0.012012,
application0.007271; it is small here, not assumed zero.

Reverse ownership pairs exchange Y10/Y01 exactly. Consequently body-effect RMS
equals the suffix remainder RMS by design, not independent confirmation.
Full-cut replay and no-op score errors are0; alternate scalar error3.553e-15.
384 actual forwards/111.143 model seconds,44 focused tests, no fitting or search.

## Theory update

At this cut, this controlled target-person reading is largely carried by the
contextualized suffix state. The body supplies the evidence used to form that
state; small body-only action does not make the body irrelevant or information
free. Sufficiency under this intervention is not exclusive storage or necessity.

HM49's shared property geometry had little ownership-exchange action. HM50's
full suffix carries the action. The next object is therefore a person-qualified,
query-conditioned observation operation, rather than an unconditional trait
vector or geometry-only identity score.

The proposed path is

```text
evidence + subject relation -> query-conditioned state -> target reading
```

It remains a model reading of controlled evidence. Psychological validity,
natural author inference and an independent non-LLM implementation are separate
requirements. Do not attach old human scale correlations to this new operation.

## Provenance and next decision

Entity/attribute binding and mean-vector editing have established precedent:
[Feng and Steinhardt2023](https://arxiv.org/html/2310.17191v1). Their frozen-other-
positions storage estimand differs from our continuing-upper-reader intervention.
[Heimersheim and Nanda2024](https://arxiv.org/html/2404.15255v1) supplies the
sufficiency/necessity and joint-mechanism interpretation, not a new theorem here.

OpenAI/math signed-isotypic sector reasoning inspires keeping both shared and
person-oriented components. We have not established its signed-density or
symmetric tensor assumptions for neural states, and use none of its bounds.

Next HM51 is a plan-only source relational transport experiment. It distinguishes
an additive reusable relation change from instance-dependent query integration,
with no layer/head/alpha/rank rescue grid. Exact donor sufficiency from HM50 is
not already evidence that a source-mean operation transports.

Canonical output: results/grain_math_response_v1/relational_cut_r1.
Parent report: Project Persona reports/openai_math_review_20261007/execution/
RELATIONAL_CUT_REPORT_20261009_ZH.md. Main goal remains active and full scope open.
