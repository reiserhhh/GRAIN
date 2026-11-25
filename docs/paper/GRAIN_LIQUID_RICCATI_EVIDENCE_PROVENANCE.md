# GRAIN Liquid & Riccati Supplement (Version 2): Evidence Provenance Map

**Status:** internal reproducibility companion to the trilingual Version 2
supplement (`GRAIN_LIQUID_RICCATI_SUPPLEMENT_EN_V2.md`,
`GRAIN_LIQUID_RICCATI_SUPPLEMENT_ZH_V2.md`,
`GRAIN_LIQUID_RICCATI_SUPPLEMENT_JA_V2.md`).

**Scope boundary:** This map locates every verified software guarantee,
formal proof reference, and empirical public-corpus value cited in Supplement
V2. Formal Lean 4 proofs establish deductive soundness under declared axioms;
archived synthetic and public-corpus summaries establish bounded behavioral
relations within each cohort and never establish a clinical diagnosis or unified
personality scale.

## 1. Standalone GRAIN Implementation & Test Provenance (This Repository)

| Supplement Object | Local Module / Document | Local Verification Test | Scope / Guarantee |
|---|---|---|---|
| Exact Riccati-Möbius ZOH cell & Kogge-Stone $O(\log T)$ parallel scan (`ConstrainedRiccatiCell`) | `suica_core/riccati_mobius.py` | `tests/test_riccati_mobius.py` | Strictly bounded `[-B, B]` contractive flow, `dt = 0` identity preservation, subdivision semigroup consistency (`<= 1e-12`), sequential-vs-parallel agreement (`<= 1e-12`). |
| Dual-pathway continuous decay + episodic cosine retrieval (`DualPathwayRiccatiObserver`) | `suica_core/riccati_mobius.py` | `tests/test_riccati_mobius.py` | Preserves opening-event sensitivity across idle gaps while honoring causal idle masking (`mask_first`). |
| Clipped cross-view moment ratios and dynamic polytope solver (`estimate_cross_view_lag1_rates`, `solve_axis_trajectory_lp`) | `suica_core/riccati_mobius.py` | `tests/test_riccati_mobius.py` | Historical ratios can return fallback zero; feasible bounds are conditional on the calibration. Rate separation alone does not guarantee a decision increment. |
| Multi-window linear transition contrast stacking (`stack_transition_contrasts`) | `suica_core/riccati_mobius.py` | `tests/test_riccati_mobius.py` | Eliminates arbitrary view-specific initial offsets across multiple calibrated transitions to recover full-rank $c_0$. |
| Frozen liquid node readers, event-time checks, task-path endpoints, common-range geometry & dialogue attribution | `suica_core/liquid_instrument.py`, `suica_core/liquid_event_time.py`, `suica_core/liquid_task_paths.py`, `suica_core/liquid_common_range.py`, `suica_core/liquid_dialogue.py` | `tests/test_liquid_instrument.py`, `tests/test_liquid_task_paths.py`, `tests/test_liquid_common_range.py`, `tests/test_liquid_dialogue.py`, `tests/test_liquid_reference_error.py`, `tests/test_liquid_axis_series.py`, `tests/test_liquid_frozen_bank.py`, `tests/test_liquid_inverse.py`, `tests/test_liquid_opportunity.py` | Native deterministic node-map readers and uncertainty/range estimators; `docs/GRAIN_LIQUID_TOOL.md` and `docs/GRAIN_RICCATI_AND_INTERPERSONAL_UPDATE.md` document usage boundaries. |
| Addressed correction response and conditional card | `suica_core/event_response.py` | `tests/test_event_and_view_increment.py` | Illustrative fixture: one supported reply, reference bounds [0.75, 1]. Companion D05 replay: 4 supports and 3 unresolved using existing annotations, not blind detection. |
| Same-history threshold comparison | `suica_core/dual_view_increment.py` | `tests/test_event_and_view_increment.py` | The shipped JSON gives simple bounds [−0.4, 2.0] and joint bounds [0.8, 1.2]; a conditional identification result in a constructed example, not forecast accuracy. |

## 2. Companion Repository Formal Proofs & Public Multi-Corpus Provenance

Numbers summarized in Sections S2–S4 of Supplement V2 originate from the
audited formal Lean 4 suite and deterministic public-corpus scripts executed in
the companion `project-persona` repository (referenced with `project-persona:`
prefix below so repository-local path assertions apply solely to `GRAIN` files).

| Supplement Claim | Value(s) Reported | Companion Source (`project-persona`) | Evidential Boundary |
|---|---|---|---|
| Machine-checked Lean 4 verification | `58` theorems across 7 modules (`RiccatiMobiusFlow.lean` 7, `DynamicSetInversion.lean` 6, `PersonalityDynamics.lean` 7, `LiquidResearchLoop.lean` 14, core 24), `0` `sorryAx` (`propext`, `Classical.choice`, `Quot.sound`) | `project-persona:formal/GrainFormal/`, `project-persona:scripts/verify_grain_lean.sh`, `project-persona:docs/GRAIN_LEAN4_FORMAL_VERIFICATION_20261124.md` | Certifies mathematical deduction from stated axioms only; never replaces out-of-sample human empirical validation. |
| Exact Riccati-Möbius ZOH vs. `ncps LTCCell` on `N = 108` real trajectories | `ncps LTCCell` 2x error `3.8665e-5` (`30.68%` of `1.2601e-4`), `wall_first_at_end_median = 0.0`; Riccati subdivision error `1.73e-18..2.52e-18` (`5.07e-13%..7.33e-13%`), parallel diff `1.388e-17`, dual-pathway `wall_first_at_end_median = 2.710e-4` | `project-persona:reports/grain_theory_leap_20261124/`, `project-persona:docs/GRAIN_THEORY_LEAP_RICCATI_DYNAMIC_DUAL_CLOCK_20261124.md` | Archived trajectory benchmark comparing numerical discretization laws; dual-pathway sensitivity does not imply causal token-intervention effects. |
| Dynamic multi-time set inversion (synthetic + archived real tensors) | `distinct_rates`: `4.949 -> 0.1070` (`0.0216x`), `0/60 -> 60/60`; `equal_rates_counterexample`: `3.8343`, `0/60`; `actual_slopes/paths.npz` (`r_c=(0.165, 0.350, -0.031)`, `r_p1=(0.193, 0.150, 0.104)`, `r_p2=(0.021, -0.015, 0.179)`): `R4_branches` `216/216` feasible (`0.281x..0.614x`, `12/216` & `4/216` disjoint); `pandora_heldout_PA` `78..83/96` feasible (`13..18` rejected) | `project-persona:reports/grain_theory_leap_20261124/`, `project-persona:project_persona/riccati_dynamics/dynamic_set_inversion.py` | Conformal `0.90` calibration on whole trajectories; rejected paths remain reported as `None` (abstention). |
| RealPersonaChat (`13,583` dialogues, `138` dual-panel / `129` balanced `117 vs 116`, `332,677` `dt <= 300s`) | `v_given_v` 200-perm FWER: `informativeness r_cross = 0.3545 (0.00498)`, `familiarity 0.3338 (0.00498)`, `comprehension 0.2921 (0.0149)`, `interest 0.2874 (0.0249)`; questionnaire `0/252` (`q_BH >= 0.1245`); SOKA `PeerAdj_Composite`: `S_self28 r_OOF = -0.0216, R² = -0.0719`, `B_static5 R² = +0.2082`, `B_full22 r_OOF = +0.5869, R² = +0.3388`, `ΔR² = +0.1306`, `proactiveness ΔR² = +0.1933`, cross-panel `r = 0.4031` (`7/7` CI > self); evocation `0.7399/0.5643/0.2781`; plasticity `0.2762/0.2343/0.1806`; `E_katakana <-> E_kanji r = 0.7705`; lift attenuation `0.2479 -> 0.0628`; no-re-inking null `-0.0016`; `Self - Peer`: `-0.3665`, `+0.2605`, `+0.2853` | `project-persona:reports/grain_theory_leap_20261124/`, `project-persona:reports/grain_personality_theory_20261124/`, `project-persona:docs/GRAIN_PERSONALITY_PSYCHOLOGY_THEORY_SYNTHESIS_20261124.md` | The [original questionnaire (§3.5)](https://aclanthology.org/2023.paclic-1.85/) mixes rater, partner, topic and dialogue targets. Strictly disjoint partner panels within RealPersonaChat; pooled `R²`, within-panel `ΔR²`, and strict cross-panel `r` kept distinct; `Self - Peer` reflects right-hand-side `-Peer` variance as well as self-rating. |
| ESConv (`1,300` dialogues) | `Reflection of feelings` (`v_prob = 0.7156, e_uptake = 0.0701`) vs `Restatement or Paraphrasing` (`e_uptake = 0.1531`); `v_prob -> seeker_empathy r = 0.0654 (p = 0.0266)`; `emotion_relief p >= 0.309` | `project-persona:reports/grain_theory_leap_20261124/` | Process markers separate strategies weakly without predicting downstream emotional relief. |
| PSE-Database v1.0 (`4,570` writers, `26,389` stories, `27 vs 27` disjoint pictures) | `story_arc_pow` partial `r = +0.0725 (p = 0.0059)`, holdout `N = 151 r = +0.2831`, beats 32 sentence perms (`p = 0.0303`); `sent_visc_ach` partial `r = +0.0993 (p = 0.0062)`, holdout `N = 80 r = +0.1411`; `3×3` motive-shape (`N = 548`) `rho = +0.0531 (p = 0.0303)` | `project-persona:reports/grain_personality_theory_20261124/` | Non-mergeable picture-story motive cohort; sequence-order permutation control retained alongside partial correlation. |
| Bar-Anan & Nosek (2014) (`N = 24,015`) | 0-DoF multi-task consensus: `Race -> mrs ΔR² = +0.0442 (N = 331, CI [+0.0015, +0.0808])`; `Politics -> willVoteDem ΔR² = +0.0266 (N = 244, CI [+0.0022, +0.0515])`; divergence `Race <-> Politics r = -0.036 (p = 0.63)`, `Race <-> Self r = -0.081 (p = 0.29)`; Self-Esteem double dissociation (`IAT+ -> rsnbrg r = 0.1374` vs `AMP+EPT -> _estmDrct r = 0.0814`) | `project-persona:reports/grain_personality_theory_20261124/` | Structured implicit-measure task cohort; domain-specificity and zero-DoF aggregation do not transfer automatically across domains. |

## 3. Provenance Governance Rules

1. Every claim cited in active documentation must preserve its cohort, design split, and null findings verbatim.
2. Within-panel `ΔR²`, pooled cross-validated `R²`, and strict disjoint cross-panel `r` must never be collapsed into a single summary number.
3. No local host path or restricted dataset reference may appear in public GRAIN artifacts.
