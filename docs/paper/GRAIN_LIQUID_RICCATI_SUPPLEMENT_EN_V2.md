# Supplement to the GRAIN Theory Manuscript: Liquid Trajectories, Constrained Riccati-Möbius Flows, and Interpersonal Dynamics (Version 2)

**Status:** companion theoretical and empirical supplement to
[`GRAIN_THEORY_MANUSCRIPT_EN_V1.md`](GRAIN_THEORY_MANUSCRIPT_EN_V1.md) and
[`GRAIN_MECHANISM_SUPPLEMENT_EN_V1.md`](GRAIN_MECHANISM_SUPPLEMENT_EN_V1.md).
This document extends Sections 3, 5, 6, and 8 of Manuscript V1 by integrating
frozen liquid trajectory instruments, exact Riccati-Möbius continuous-time
flows, dynamic multi-time trajectory set inversion, machine-checked Lean 4
theorems in the companion `project-persona` repository, and public
multi-corpus validation across disjoint human cohorts. Every numerical value is
indexed in
[`GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md`](GRAIN_LIQUID_RICCATI_EVIDENCE_PROVENANCE.md).

> **Core non-claim.** GRAIN is an integrative analysis method for developing
> natural-language measurement systems; it is **not a personality scale**, not
> a diagnostic instrument, and not a substitute for construct, population, or
> clinical validation.

## S1. Purpose and position

Manuscript V1 (Equations 1–15) and Mechanism Supplement V1 established why topic
and context cannot be unconditionally subtracted, why condition selection and
conditional response must remain separate, and why individual, relational, and
population outputs belong to distinct inferential types. However, three
structural gaps remained open when extending GRAIN from discrete slice-level
regressions to continuous-time dialogue and multi-occasion language streams:

1. **Continuous-time semigroup defects and long-horizon memory loss.** Standard
   discrete-step ODE approximations used in off-the-shelf liquid neural
   cells (`LTCCell`) violate exact time-subdivision consistency under irregular
   sampling (`dt = 0` or subdivided inter-event delays) and allow fast
   exponential relaxation to erase high-signal opening events over empty gaps.
2. **Equal-rate non-identifiability in static partial identification.** Two-view
   static bounds cannot separate shared latent trajectories from arbitrary
   view-specific initial offsets when single-step dynamics collapse to shared
   decay rates.
3. **Empirical isolation of intrapersonal monologues from dyadic interpersonal
   dynamics.** Cross-sectional or unperturbed single-corpus correlations leave
   open how dynamic interpersonal signatures (conditional response under
   partner actions, evocation, interpersonal plasticity, and narrative dynamic
   viscosity) compare against traditional self-report scales when evaluated on
   strictly disjoint human panels.

Version 2 closes these gaps inside a frozen, source-calibrated framework without
claiming clinical diagnosis or trait universality.

## S2. Inherited methods, exact Riccati-Möbius flows, and Lean 4 formalization

This release promotes `grain_core.riccati_mobius` (`suica_core/riccati_mobius.py`)
alongside the frozen liquid node suite (`grain_core.liquid_*`) and pairs them
with `58` machine-checked Lean 4 theorems (Lean `v4.36.0` + Mathlib `v4.36.0`,
zero `sorryAx`, axioms restricted to `propext`, `Classical.choice`, and
`Quot.sound`) maintained across seven Lean modules in the companion
`project-persona` repository (`RiccatiMobiusFlow.lean`: 7 theorems;
`DynamicSetInversion.lean`: 6 theorems; `PersonalityDynamics.lean`: 7 theorems;
`LiquidResearchLoop.lean`: 14 theorems; plus 24 core GRAIN theorems). Formal
proofs certify deductive validity under explicit assumptions; they never
replace out-of-sample human empirical testing.

| Component / Formal Module | Mathematical Mechanism | Proven Property / Guarantee | Boundary / Limitation |
|---|---|---|---|
| `ConstrainedRiccatiCell` (`RiccatiMobiusFlow.lean`) | Scalar contractive Riccati ODE $\frac{dx}{dt}=\epsilon(u)(\alpha(u)+\beta(u)x+\gamma(u)x^2)$ integrated via normalized $2\times 2$ Möbius matrix exponential | Exact $[-B, B]$ domain invariance, `dt = 0` identity preservation, time-subdivision semigroup composition (`M(dt1+dt2) ~ M(dt2)M(dt1)`), associative Kogge-Stone $O(\log T)$ parallel prefix scan (`<= 1e-12`) | Input-frozen piecewise ZOH semantics; captures fast continuous decay only unless paired with episodic retrieval |
| `DualPathwayRiccatiObserver` | Fast contractive Riccati state over inter-event gaps paired with content-addressable cosine retrieval over history | Prevents empty-gap erasure of early salient utterances while respecting causal idle masking (`mask_first`) | Episodic weights reflect historical retrieval weighting, not causal counterfactual token edits |
| `estimate_cross_view_lag1_rates` & `solve_axis_trajectory_lp` (`DynamicSetInversion.lean`) | Unbiased cross-view Method-of-Moments separating common vs. private AR(1) retention rates $(r_c, r_{p,1}, r_{p,2})$ feeding HiGHS linear constraints | Eliminates arbitrary private initial offsets across steps whenever $r_c \ne r_{p,i}$; reports exact infeasibility (`None`) on model mismatch | Degenerates toward wide static widths when $r_c = r_{p,1} = r_{p,2}$ (formalized as the equal-rate counterexample) |
| `stack_transition_contrasts` | Multi-window affine contrast stacking of calibrated $A_t - P_{j,t}$ operators across steps and readers | Recovers full-rank initial shared state $c_0$ (`< 1e-12` exactness in well-conditioned systems) even when single transitions are rank-deficient | Assumes calibrated regime transitions and finite observation error bounds |
| Five-Operator Interpersonal Synthesis (`PersonalityDynamics.lean`, `LiquidResearchLoop.lean`) | Decomposes situated behavior into Evocation, Response/Conditioning, Plasticity, Trajectory Viscosity, and Cross-Task Consensus | Formalizes dynamic disattenuation identities, variance-decomposition bounds, and non-equivalence of self-report vs. observer criteria | Behavioral and peer-rated criteria remain conceptually distinct from questionnaire self-reports and clinical diagnosis |

## S3. Computational, dynamic set-inversion, and multi-corpus empirical findings

### S3.1 Continuous-time semigroup invariance and dynamic set inversion

On `N = 108` archived real trajectories from the companion development loop,
standard `ncps LTCCell` discretizations exhibit a 2x time-subdivision error of
`3.8665e-5` (`30.68%` of `wall_vs_equal_median = 1.2601e-4`) and lose memory of
initial events across long gaps (`wall_first_at_end_median = 0.0`). In contrast,
`ConstrainedRiccatiCell` and `DualPathwayRiccatiObserver` achieve exact
subdivision errors of `1.73e-18` to `2.52e-18` (`5.07e-13%` to `7.33e-13%`),
sequential vs. $O(\log T)$ parallel scan discrepancy `1.388e-17`, and nonzero
long-horizon opening-event sensitivity (`wall_first_at_end_median = 2.710e-4`).

On synthetic dynamic set-inversion testbeds, separating common and private
rates (`distinct_rates`) shrinks trajectory polytope widths from `4.949` down to
`0.1070` (`0.0216x`) and lifts branch disjointness from `0/60` to `60/60`. When
all three retention rates coincide (`equal_rates_counterexample`), the same LP
system explodes to width `3.8343` with `0/60` branch disjointness, confirming
the Lean 4 non-identifiability theorem in code. On archived multi-view
trajectory tensors (`actual_slopes/paths.npz`, using cross-view Method of
Moments with whole-trajectory max conformal quantile `0.90`:
`r_c = (0.165, 0.350, -0.031)`, `r_p1 = (0.193, 0.150, 0.104)`,
`r_p2 = (0.021, -0.015, 0.179)`), held branch sets achieve `216/216`
feasibility with width ratios `0.281x`–`0.614x` (`12/216` and `4/216` branch
disjoint on axes 0–1), while held-out PANDORA pairs yield `78..83/96` feasible
and reject `13..18` misspecified paths.

### S3.2 Multi-corpus validation across disjoint public human cohorts

Across four strictly disjoint public human cohorts (analyzed independently
without pooling individuals across datasets), dynamic and interpersonal GRAIN
operators demonstrate replicable behavioral signal where self-reports or naive
static summaries fail, while clarifying scope boundaries:

1. **RealPersonaChat** (`13,583` dialogues; `138` dual-panel speakers / `129`
   balanced speakers with disjoint `117` vs. `116` partner panels; `332,677`
   real timestamps with `dt <= 300s`):
   - **Disjoint-partner conditional validation.** Conditional self-response
     (`v_given_v`) predicts unseen partner peer ratings, surviving 200-draw
     permutation FWER across four dimensions: `informativeness r_cross = 0.3545`
     (`p_FWER = 0.00498`), `familiarity = 0.3338` (`p_FWER = 0.00498`),
     `comprehension = 0.2921` (`p_FWER = 0.0149`), and `interest = 0.2874`
     (`p_FWER = 0.0249`). Simultaneously, `0/252` questionnaire correlations
     pass BH-FDR (`q_BH >= 0.1245`).
   - **Self-report vs. static vs. interpersonal-dynamic head-to-head.** For
     composite peer adjustment (`PeerAdj_Composite`), 28 self-report scales
     (`S_self28`) yield out-of-fold `r_OOF = -0.0216, R² = -0.0719`; static
     behavioral rates (`B_static5`) reach `R² = +0.2082`; the full 22-feature
     interpersonal-dynamic set (`B_full22`) reaches pooled `r_OOF = +0.5869,
     R² = +0.3388` (within-panel dynamic increment `ΔR²(B_full|B_static) = +0.1306`;
     `proactiveness ΔR² = +0.1933`; strict cross-panel generalization
     `r = 0.4031`), beating self-report on `7/7` peer targets (bootstrap 95% CI).
   - **Evocation, plasticity, and cross-script invariance.** Dyadic evocation
     and plasticity exhibit strong split-partner reproducibility (`evoke_p_log_chars
     r_AB = 0.7399`, `evoke_p_q_rate = 0.5643`, `evoke_p_v_rate = 0.2781`;
     `slope_v_to_p_v = 0.2762`, `slope_q_to_p_q = 0.2343`,
     `slope_chars_to_p_chars = 0.1806`, orthogonal to `SMS` and `PeerAdj`) and
     cross-script uptake invariance (`E_katakana <-> E_kanji r = 0.7705`).
   - **Self–Peer discrepancies and nulls.** Raw `V_aff <-> V_cog` correlation
     (`0.2479`) attenuates to `0.0628` under conditional lift, and
     no-re-inking excess lift is null (`r_AB = -0.0016`). Discrepancy scores
     (`Self - Peer`) correlate with lower `v_aff_given_v` (`-0.3665,
     q_BH = 0.00052`), higher `deflect_v_to_e_lift` (`+0.2605, q_BH = 0.020`),
     and higher `evoke_p_q_rate` (`+0.2853, q_BH = 0.0107`); because `-Peer`
     appears in the discrepancy definition, these relations reflect both partner
     judgment and self–peer divergence rather than pure self-deception.
2. **ESConv** (`1,300` supportive dialogues): `Reflection of feelings`
   (`v_prob = 0.7156, e_uptake = 0.0701`) separates from `Restatement or
   Paraphrasing` (`e_uptake = 0.1531`), while `v_prob -> seeker_empathy` is
   weak (`r = 0.0654, p = 0.0266`) and downstream `emotion_relief` is null
   (`p >= 0.309`).
3. **PSE-Database v1.0** (`4,570` writers, `26,389` stories, disjoint `27 vs. 27`
   picture sets, `N = 1,949`, holdout `N = 351`): `story_arc_pow` predicts
   disjoint picture power motive (partial `r = +0.0725, p = 0.0059`; holdout
   `N = 151 r = +0.2831`, beating 32 sentence permutations at `p = 0.0303`),
   `sent_visc_ach` reaches partial `r = +0.0993` (`p = 0.0062`; holdout
   `N = 80 r = +0.1411`), and `3×3` motive-shape alignment (`N = 548`) yields
   `rho = +0.0531` (`p = 0.0303`).
4. **Bar-Anan & Nosek (2014)** (`N = 24,015` multi-task respondents): zero-DoF
   multi-task implicit consensus adds `ΔR² = +0.0442` (`Race -> mrs, N = 331,
   95% CI [+0.0015, +0.0808]`) and `ΔR² = +0.0266` (`Politics -> willVoteDem,
   N = 244, 95% CI [+0.0022, +0.0515]`). Explicit–implicit divergence
   `|z_exp - z_imp|` is domain-specific (`Race <-> Politics r = -0.036,
   p = 0.63`; `Race <-> Self r = -0.081, p = 0.29`), with Self-Esteem double
   dissociation (`IAT+ -> rsnbrg r = 0.1374` vs. `AMP+EPT -> _estmDrct
   r = 0.0814`).

## S4. Retained counterexamples, nulls, and methodological boundaries

1. **Subdivision drift in naive discrete ODEs:** `LTCCell` accumulates `30.68%`
   2x subdivision discrepancy and forgets initial events (`0.0` wall-first end
   sensitivity) whenever irregular delays replace constant step indices.
2. **Equal-rate non-identifiability:** Dynamic LP bounds collapse to wide
   intervals (`3.8343`, `0/60` disjointness) whenever common and private decay
   rates match (`r_c = r_p1 = r_p2`).
3. **Conformal boundary on unseen distributions:** Finite-sample conformal
   trajectory calibration (`0.90`) rejects `13..18/96` held-out PANDORA paths;
   infeasibility must be surfaced as abstention (`None`), never coerced.
4. **Corpus-specific boundaries & honest nulls:** In RealPersonaChat, self-report
   questionnaires yield `0/252` FDR-passing links to dialogue dynamics, and
   no-re-inking excess lift is null (`-0.0016`). In ESConv, strategy process
   separation does not imply downstream emotional relief (`p >= 0.309`).
5. **Non-interchangeability across cohorts:** RealPersonaChat, ESConv,
   PSE-Database, Bar-Anan, and PANDORA represent distinct populations, tasks,
   and outcome channels; results must never be pooled as a single trait model.

## S5. Additional falsifiable propositions (P12–P15)

- **Proposition 12 (Semigroup invariance under irregular sampling).** A valid
  continuous-time language reader must preserve trajectory outputs up to
  floating-point tolerance under gap subdivision and `dt = 0` concatenation.
  Falsified if subdivided runs diverge by more than numerical precision.
- **Proposition 13 (Rate-separation condition for dynamic set inversion).**
  Multi-time linear programming shrinks latent common-state intervals across
  views if and only if common and private transition operators differ on
  observed steps; equal decay rates fail identifiability.
- **Proposition 14 (Interpersonal evocation vs. intrapsychic self-report).**
  In dyadic conversation, evoked partner behavior and contingent dialogue slopes
  explain variance in disjoint-partner peer impressions that self-report scales
  cannot recover (`ΔR² > 0` cross-panel). Falsified if self-report composites
  subsume disjoint-partner dynamic variance.
- **Proposition 15 (Domain-specificity of explicit–implicit divergence and
  sub-domain timing).** Explicit–implicit divergence and implicit consensus
  correlate only weakly across unrelated attitude domains, while episodic
  narrative arc requires ordered sequence rather than unordered sentence bags.

## S6. Relationship to Manuscript V1 and Mechanism Supplement V1

Sections S1–S5 extend the formal measurement chain of Manuscript V1 (Eqs. 6–15)
and the mechanism package of Supplement V1 without altering frozen historical
estimates: (a) continuous-time trajectory transport is now grounded in exact
Möbius semigroups and dual-pathway episodic retrieval (`grain_core.riccati_mobius`),
(b) multi-view partial identification is extended from static ellipses to dynamic
cross-view LP polytopes with machine-checked non-identifiability boundaries, and
(c) dyadic evocation, plasticity, and conditional response are demonstrated across
disjoint human cohorts while strictly honoring self-report null regimes,
cross-panel generalization splits, and the non-clinical scope boundary.
