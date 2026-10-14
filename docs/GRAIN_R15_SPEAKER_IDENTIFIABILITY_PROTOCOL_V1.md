# R15-M1 speaker-choice identifiability: protocol V1 (frozen before any result)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_14_PRAGMATIC_IDENTIFIABILITY.md` Part B; `SEMINAR_FABLE_12_DECISION.md`.
- **Sources.** P02 pp.12–14 (utterance = π(belief, goal, audience, social cost, context, language, history); seven
  possible causes of one hedge); P01 pp.9–10 (the RAM chain); P03 p.12 (opportunity → selection → realization →
  reader); P05 §7 (nuisance status depends on the construct).
- **Prior evidence.** FU4-b used a discrete 15-point simplex grid in one context with the truth on the grid, and only
  covered politeness versus presentation.
- **Runner and output.** `scripts/run_grain_r15_speaker_identifiability.py`, writing to
  `results/grain_gptpro_foundation_v3/F17/speaker_identifiability_r1/`.
- **Scope.** CPU, 2 workers × 1 BLAS thread, 0 neural calls, cap 15 min. Synthetic model only: no claim that natural
  hedges have these causes.

## Cited support versus our extension

- **Cited: Yoon et al. 2020, Model equations 1–5.** A speaker softmax with an optimality parameter over a weighted
  sum of informational utility ln P_L(s | w), social utility, and presentational utility ln P_L1(φ | w), minus
  utterance cost. Weights were inferred per goal condition. The paper states it does not model the mapping from
  social situations to utility weights.
- **Our extension, labelled as such:**
  - the choice of terms for hedging;
  - commitment defined from the literal listener;
  - the gating of each term by a context variable (the situation-to-weight mapping, which is stipulated here);
  - the belief/evidence law;
  - the behaviour law.
  None of these is supported by the cited paper.

## Model

- **Proposition and utterances.** Proposition P, world w ∈ {T, F}. Utterances U = {assert "P", hedge "maybe P",
  withhold "not sure"}.
- **Literal semantics** [[u]](T, F): assert (.95, .05), hedge (.70, .30), withhold (.50, .50). With listener prior
  .5, the literal listener gives L0(T | u) = (.95, .70, .50).
- **Commitment** c(u) = L0(T | u) − .5 = (.45, .20, 0). This is derived, not assigned.
- **Informational utility** U_inf(u; b) = b ln L0(T | u) + (1 − b) ln L0(F | u), for speaker belief b = P(T).
- **Presentational profile** m(u) = ln L1(cautious | u). L1 infers the speaker type φ ∈ {cautious, direct}, each
  with prior .5, from S1_φ(u | b) = softmax_u(3[U_inf(u; b) − κ_φ c(u)]), with κ = (1.5, 0) and belief prior
  b ∈ {.55, .75, .95} uniform. Computed once and fixed.
- **Partner style** P_style(u | m): neutral uniform (no effect), hedgy (.15, .65, .20), direct (.70, .15, .15).
- **Cost** C(u) = (0, .10, .05). Known.
- **Utility.** U(u) = U_inf(u; b) − [w_face·a + w_eval·s·(1 − b) + w_lc + w_opt·f]·c(u) + w_pres·p·m(u) +
  w_acc·ln P_style(u | m) − C(u).
- **Speaker.** P(u) = softmax_u(λ U(u)).
- **Belief.** logit b = β0 + Δ·e, with Δ = 2 known. The evidence probe changes a knowledge state, not a stable
  parameter.
- **Behaviour** (its own law; not a cause ID): private action consistent with P, P(B = 1) = σ(η·logit b), with
  η = 1 known. It depends on belief only, and is noisy.
- **Parameters** θ = (β0, λ, w_face, w_eval, w_lc, w_opt, w_pres, w_acc).
- **Gauges.** The informational weight is fixed at 1, which fixes the utility scale (nats of literal-listener belief)
  so that λ is estimable. The additive constant cancels in the softmax. Weights are ≥ 0.

| P02 cause | Model term | Context gate (stipulated) | Status |
|---|---|---|---|
| epistemic uncertainty | low β0 (belief) | evidence e changes the knowledge state | implemented |
| social politeness | w_face·a·c(u) | audience disagreement a | implemented; shares the c(u) profile |
| impression management | w_pres·p·m(u) | public/evaluative audience p | implemented; derived profile m(u) |
| low commitment | w_lc·c(u) | none (stable) | implemented; shares c(u) |
| fear of evaluation | w_eval·s·(1 − b)·c(u) | stakes s | implemented; shares c(u), depends on belief |
| interpersonal accommodation | w_acc·ln P_style(u \| m) | partner style m | implemented |
| strategic ambiguity | w_opt·f·c(u) | future accountability f | implemented; shares c(u); B also constrains it |

## Contexts and designs

- **Passive baseline** C0 = (e = 0, a = .5, p = .5, s = .5, f = .5, m = neutral): one fixed natural context in which
  every gate is active.
- **Probe pairs:** E (e = 1); A (a = 0, 1); P (p = 0, 1); S (s = 0, 1); F (f = 0, 1); M (m = hedgy, direct). The
  others stay at baseline.
- **Designs:** passive {C0}; C0 plus each single probe; C0 plus all probes. Each is run text only and joint (text
  plus B).

## Computations

1. **Analytic Jacobian and Fisher information.** Per context, text is multinomial and B is Bernoulli. The analytic
   Jacobian is verified by central finite differences (declared tolerance 1e-6).
   - Rank at 25 declared evaluation points: seed 20260926; β0 ∈ [−1, 2], λ ∈ [1, 3], weights ∈ [.2, 2].
   - A singular value counts as zero when σ/σ_max < 1e-8.
   - Null directions are reported in parameter names at the reference point (β0 = .5, λ = 2, all weights = 1).
   - Local rank is not global identifiability.
2. **Exact equivalence constructions.**
   - E1, commitment aliasing: at C0 the passive law depends on the c(u) terms only through
     κ0 = w_lc + .5 w_face + .5(1 − b) w_eval + .5 w_opt. Build θ′ ≠ θ with the same κ0; its passive text and joint
     laws must be identical, and some probes must separate them.
   - E2, epistemic versus preference in text only: shift β0 by +1, then solve the 2×2 linear system in (κ-type
     coefficient, w_pres) that restores the passive text log-odds. The text-only laws must be identical, and B must
     separate them. If the solution is infeasible (a negative weight), report it as found.
   - E3, across-person mixture versus within-person homogeneity: a 50/50 mixture of (w_face = 2, w_lc = 0) and
     (w_face = 0, w_lc = 2) against a homogeneous population matched on the single-observation passive marginal.
     Then compare two-observation-per-person laws under (A0, A1).
3. **Predictive functionals.** A functional is identified in a design when its gradient lies in the row space of the
   FIM (residual/norm < 1e-6):
   - P(hedge | C0), P(hedge | a = 0), P(hedge | e = 1), P(B = 1 | C0);
   - a κ0 contrast, the separate w_face, and the separate w_opt.
4. **Finite recovery (the matched contrast).**
   - 40 simulated application authors, with θ drawn from the declared box (seed 20260927).
   - Design P: 120 utterances in C0, text only and joint.
   - Design F: 10 utterances in each of 12 contexts (C0, E, A0, A1, P0, P1, S0, S1, F0, F1, Mh, Md), with B per
     utterance.
   - Fitting: MAP with declared weak priors (β0 ~ N(0, 3²); ln λ ~ N(ln 2, 1); weights half-normal with scale 3),
     L-BFGS-B within bounds (β0 ∈ [−4, 4], λ ∈ [.1, 10], weights ∈ [0, 10]), 3 fixed starts, best objective.
   - Laplace SDs from the observed information plus the prior.
   - The truth is read only after all fits: RMSE and 95% coverage per parameter and for identified combinations.
   - No source role is needed, because all measurement laws are declared and frozen, and no fitted quantity crosses
     authors.

## Pre-declared contrasts and falsifiers

- **G1.** If E1 fails (TV > 1e-12 between passive laws), the derived-aliasing claim is false.
- **G2.** If the all-probe joint design is not locally full rank (8) at most evaluation points, the model is not
  identifiable even with probes. That is reported, not redesigned.
- **G3.** If the passive text-only rank exceeds 2 (the degrees of freedom of one 3-category law), the implementation
  is wrong: STOP.
- **G4.** If recovery in Design F does not beat Design P for w_face and w_opt in RMSE, the probe design is
  uninformative at this sample size. That is reported.
- **No search.** No search over utilities, parameters, contexts or seeds.
