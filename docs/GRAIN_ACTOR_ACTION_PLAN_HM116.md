# HM116: source-only assignment action on new behavioral instances

2026-10-10. Fixed before new native forwards. Original18/100/44 remain OPEN.
This tests intervention transport/selectivity, not personality accuracy.

## Inventory and the new question

HM60/HM68/HM73 already measure target/owner-conditioned evidence. HM100/HM106
already account for current K/V reading. HM107 already ablated an actual local
joint term; its retained budget stop must not be resumed. HM115 changes behavior
instances at fixed current tokens and matched actor opportunity, but direction
correspondence is not causal transport. Feng2024 establishes generic binding
interventions; neither that nor the previous local ablation establishes that a
source-derived action operates selectively on new behavior instances.

Use only HM115 original behavior, wording0 native physical V to construct D_f(c)
and S_f(c), with the frozen even/odd formulas. No new native logits, application
vectors or psychological labels enter the action. All six source actions, both
conditions, all wordings and every negative/weak family are retained. The source
K/V state still comes from pretrained Qwen; no encoder-free claim is made.

## Fixed interventions and actual observations

Application is all96 HM115 variant2 rows: 2 opportunity wordings x6 families x
2 conditions x2 behavior polarities x2 actor assignments. Original114token
prefix IDs, positions, padding and current A carrier remain unchanged. Append
the existing six target questions for A and their subject-only B substitutions.
Record12 native digit distributions per prefix; no prompt selection/generation.

At layer17 current v_proj rows apply delta=-b*o*X_f(c). X is respectively:

1. baseline: zero (no hook edit);
2. source_odd: D_f(c), at its original source magnitude;
3. source_even: S_f(c), normalized per physical head to D's norm;
4. cyclic_family: D_(f+1 mod6)(c), likewise norm-matched;
5. random: fixed seeded noise, likewise norm-matched.

Keys, current residual, other token values, later weights and direct history
paths are not patched. All arms have the same b*o pattern: its presence alone
cannot establish semantic selectivity. The surgery is off the natural text
manifold and need not remove the new instance's entire odd component. No rank,
strength, layer, prompt or metric search is permitted.

Repeat a zero edit in all four polarity/owner cells of predefined group
wording0/family0/condition-1. 484 calls x12 native endpoints=5808 maximum;
1200 cumulative model seconds including load/failures, no renewal/retry.
Persist pending call before execution and atomically commit outputs after it.
Terminal failures/cap stops retain partial evidence and are not restarted.

Use actual full12-query shape for all arms. Compare baseline/edited before-values
and keys at the same execution shape <=1e-6 and realized value edit <=1e-5.
Log short HM115 versus full-native prefix discrepancies descriptively; do not
force a different numerical shape into the old local state or calibrate actions
from native/application states. No-op native distributions <=1e-6.

## Mathematical action and author integration

For N_p,q^arm(c,b,o), p=A/B, compute all eight signed factorial coefficients.
For arm a, effect=N^baseline-N^a. Define actor-differential and actor-common
effects X=(effect_A-effect_B)/2, G=(effect_A+effect_B)/2. Their bo coefficients
separate a matched assignment action from a common score shift. Keep raw values,
all coefficients, both conditions and all6x6 action/target cells.

For each query, save mean digit/9, grouped digit log odds, candidate mass and
entropy. Do not discard low mass or treat these outputs as human trait truth.
Report all signed cells, diagonal-minus-off-target contrasts, RMS and alignment
with the baseline actor-differential bo table. These are finite descriptive
properties; no row-as-human significance tests or arbitrary psychometric gate.

Positive numerical paired laws P+/-=(1+/-a*b*o)/4, a=.4, equal condition mass
and family mass1/6 give Delta_s=(2a/6)*mean_c(beta_bo). Replay by explicitly
summing the four cells. These are design accounts, not new participant scores.

## Competing outcomes and next decisions

- Directed source action affects actor-differential matched target responses
  more specifically than same-norm even/cyclic/random controls: retain a finite
  transported qualified operation, then connect its conditional author integral.
- Directed action mostly changes both actors/all targets together, or controls
  produce comparable effects: raw correspondence is not sufficient for the
  qualification operation; preserve content and find the missing routing object.
- Effects vary by condition/family: retain a conditional operation rather than
  fitting one universal direction or deleting context. A weak value edit does
  not rule out causal evidence elsewhere or redundant paths.

Source geometry, arithmetic fidelity and native judgement are distinct objects.
Do not manufacture a PASS by shrinking the battery, selecting E/C only, fitting
a conversion or reusing application-native answers to improve the source action.

## Sources read and limitations

Feng/Russell/Steinhardt2024 sections4.1--4.2: mixed Hessian and interchange
interventions distinguish binding from content. We do not reproduce their
Hessian algorithm or claim a uniquely identified binding subspace.
https://arxiv.org/html/2406.19501v1
Heimersheim/Nanda2024: activation interventions are scoped observations; absence
of a local effect is not automatically absence of information.
https://arxiv.org/html/2404.15255v1
OpenAI/math pinned adc7f124, metric/connection 06-matrix.tex71--109: complementary
fixed observations and a range constraint inspire a design, not a language PDE
theorem, causal metric or precision guarantee. Classical finite characters and
paired integrals are known mathematics. The GRAIN contribution must be the
qualified source-action construction and its observed consequences.

Deliver sealed design/source edits, full native raw endpoints, source/control
norms, character/effect/author tables, alternate verifier, tests and Chinese
scientific decision report. No API/private/training/Claude/Git/automation/
lockbox or old queue; preserve all prior failures and weak results.
