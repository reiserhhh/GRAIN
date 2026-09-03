# GRAIN Construct Registry v1

Created: 2026-07-05 (referee sweep). Operational names ONLY — psychological
names are earned at T4 (lockbox) or human-criterion evidence, not before
(jingle-fallacy rule, R1-2). Every entry lists channel, evidence tier
(see CLAIMS_LEDGER taxonomy), and caveats.

## Tier A — style-register measures (person-stable text behavior)

| ID | Operational name | Channel | Disjoint-occasion r (T3 where marked) | Caveats |
|----|------------------|---------|----------------------------------------|---------|
| first_person_usage_v2 | First-person pronoun rate | C2 style + C3 signature (provisional) + state deviation (best available) | raw retest SB 0.79-0.86 (T2); E6 signature +0.154 (T3-ish, single cohort) | the single strongest measure in the system |
| directive_action_v2 | Directive+second-person blend | C2 style + C3 signature (provisional) | SB ~0.57-0.65 (T2); E6 +0.148 | contains "don't" — see apostrophe note |
| wcl_60 | Apostrophe-omission register (was: contraction-style cluster) | C2 style | r_B 0.649 (T3) | measures typed orthographic habit; ~3.8% of slices carry smart quotes (device confound bounded); normalize U+2019 in v4 scorer |
| wcl_03 | Positive-enthusiasm vocabulary rate | C2 style | r_B 0.579 (T3) | |
| wcl_45 | Epistemic-argument vocabulary rate | C2 style | r_B 0.522 (T3) | shares words with certainty/mentalization lexicons — disjointify in v4 |
| wcl_54 | Analytic-formal function-word rate | C2 style | r_B 0.473 (T3) | |
| wcl_13 | Negation-argument rate | C2 style | r_B 0.454 (T3) | partially apostrophe-splitting fragments (don/doesn) |
| wcl_15 | Casual-interjection rate | C2 style | r_B 0.444 (T3) | |
| wcl_23 | Profanity-intensity rate | C2 style | r_B 0.432 (T3) | |
| wcl_22 | Concrete-narrative function-word rate | C2 style | r_B 0.459 (T3) | interpretation weakest of the register set |

## Tier B — content-interest measures (choice-coupled; report separately)

| ID | Operational name | r_B | Choice coupling |
|----|------------------|-----|-----------------|
| wcl_36 | Family/personal-narrative vocabulary | 0.571 (T3) | joint-axes R^2 low |
| wcl_11 | Technical-practical vocabulary | 0.541 (T3) | joint R^2 ~0.55 (borderline choice duplicate) |
| wcl_25 | Media-narrative vocabulary | 0.516 (T3) | |
| wcl_02 | Sports vocabulary | 0.502 (T3) | joint R^2 0.63 — NEAR CHOICE DUPLICATE; do not report as independent style |
| wcl_07 | Politics/civic vocabulary | 0.495 (T3) | |
| wcl_35 | Media-consumption vocabulary | 0.451 (T3) | |
| wcl_20 | Gaming-mechanics vocabulary | 0.378 (T3) | joint R^2 ~0.56 (borderline) |
| novelty_play_v2 | Novelty/play vocabulary rate | raw SB ~0.65 (T2) | **RELABELED 2026-07-07: VENUE-SIGNATURE (C1-dominant)** — choice-mediated share 0.587 [0.428,0.723], class-disjoint retest 0.033 (c2_purity decomposition). NOT a style trait; fails the v4 flesh-purity gate; model in the choice family. Consistent with lockbox H3 failure + PRED-1 top gain + E9a MTMM fail (round-9 wording) |

## Tier C — state-unstable / retired-as-trait measures

| ID | Operational name | Status |
|----|------------------|--------|
| tension_core_v2 | Negative-affect+uncertainty vocabulary rate (was: Tension-Uncertainty Core) | NOT a trait (P1 SB 0.35; person share 2-6%), NOT an occasion state (<= ~2% at month/quarter); slice-level topical fluctuation. Retained only as an ingredient for future session-scale state designs (OP-16) |
| adversity_recovery_v2 | Strict adversity-recovery pattern score | near-zero everywhere at cap-400; weak positive on deep data (0.129); keep at T1 |

## OP-22 blind-coding outcomes (2026-07-06; 4-family LLM panel + human, pre-committed rules)

Legibility classification added to every candidate (guide rule G10):

| Legibility class | Constructs | Coding evidence |
|---|---|---|
| human+machine legible | wcl_45 epistemic (LLM 0.938/T2 1.00/human 7/8), wcl_23 profanity (human 6/8; T2 0.708 borderline-fail) | full pass (wcl_45); near (wcl_23) |
| content-tier, trivially legible | wcl_02/07/11/20/25/35/36 — all pass (LLM 0.73-0.98, human 5-8/8) | **coding NOT counted as validity** (topic labels are self-identifying); trait evidence remains T3 stability + choice-coupling |
| machine-mostly | wcl_03 positive-enthusiasm, wcl_13 negation, wcl_15 casual, wcl_54 analytic, wcl_22 concrete-narrative | T1 marginal (0.42-0.63), T2 fails (wcl_15 0.25, wcl_03 0.46 — readers' intensity intuition INVERTS vs count-based score); keep as T3-stable measures with reduced label confidence; rename toward operational wording |
| machine-only | **wcl_60 apostrophe-omission: human 0/8**, LLM pooled 0.469 (DeepSeek alone 0.75) | the researcher's own introspection ("読めない") exactly matched the data; validity basis = mechanism description + r_B 0.649, not reader recognition |

Panel quality: pairwise agreement 0.575-0.783 across 4 genuine families
(deepseek/qwen/llama/mistral-base) — real diversity, unlike the retired
same-family v1 panel. Human uncertainty markers: 11% on content items vs 34%
on style items.

## Candidate state dimensions (NOT constructs; E7, T2)

| ID | Working label | Evidence |
|----|---------------|----------|
| state_f3 | Reflective-opinionated mode | above-null monthly persistence +0.100, decaying; cell-state reliability 0.084-0.105 |
| state_f1 | Positivity/growth-evaluation mode | +0.069, decaying |
| state_f5 | Second-person/agency drift mode | +0.074-0.090, non-decaying at month (drift); best cell-state reliability (0.121-0.134) |

Naming rule: "candidate state dimension" until (a) lexicon-disjoint
recomputation (R2-2) and (b) any external state criterion exists.

## Choice axes (C1; held-out confirmation in OP-6a)

Named by content class: gaming, sports, politics/news, life-planning,
tech/beauty, narrative-media (+ 7 minor). choice_ax_11 (MBTI communities) is
LEAKAGE-EXCLUDED from all external validation. Compositional (sum-constrained)
scores — ipsativity note in Rulebook A3/E3; Thurstonian modeling listed as a
design upgrade for the M-choose-K module.

## v4 channel-family classification (2026-07-07; flesh-purity gate,
## results/suica_c2_purity_all19_v1; gate provisional per THEORY v4 sec.3.2)

**ROUND-10 CORRECTION (2026-07-07): the first classification run REFIT the
wcl space on pass-A (same-ID vocabulary Jaccard <= 0.018 vs the registry
constructs) — its wcl labels named DIFFERENT constructs and the table
published here for a few hours was wrong (wcl_35 "F" and wcl_60
"venue-coupled" were swap artifacts, both REVERSED under the correct
transported fit). Standing rule: cluster/embedding-derived constructs are
BOUND TO THEIR FIT; reusing an ID requires transporting the fitted
transform.** The table below is the corrected, transported classification
(auditor-reproduced independently before the fix landed).

Gate: F-family iff class-disjoint retest >= 0.15 AND choice-mediated share
< 0.30. NOTE the gate is CONSERVATIVE: class-disjoint arms are
attenuation-uncorrected (SB-corrected gating = OP-34); directive sits at
class_disj 0.0997, strictly below threshold, with bootstrap flip
probability ~0.49 (CI95 [0.065, 0.137]) — C vs composite is a coin flip
for it.

| Family | Members | Reading |
|---|---|---|
| **F-family (flesh traits, 4)** | first_person (0.268/0.279), **wcl_60 apostrophe-omission (0.365/0.053 — the STRONGEST flesh construct in the inventory)**, wcl_13 negation (0.169/0.014), wcl_23 profanity (0.169/0.172) | channel-pure style traits. Pattern: **flesh lives in FORM** — pronouns, orthographic habit, negation, profanity are mechanical/register habits, exactly the idiolect literature's person-pure markers. wcl_60's machine-only legibility + strongest flesh purity now cohere: the construct humans cannot read is the most person-pure |
| **C-family (venue signatures, 8)** | novelty_play (0.033/0.587), directive_action (0.0997/0.471 — borderline, see note), wcl_35 media-consumption (0.074/0.611), wcl_02 sports (0.041/0.592), wcl_25 media-narrative (0.075/0.538), wcl_54 analytic-formal (0.096/0.329), wcl_22 concrete-narrative (0.100/0.376), wcl_20 gaming-mechanics (0.021/0.595) | stability rides venue choice; predictive but belongs to the CHOICE channel. Pattern: **rind carries CONTENT** — the topic-vocabulary clusters land here |
| **composite (6)** | wcl_03 positive-enthusiasm (0.140/0.508 — the TF best edge is majority choice-mediated), wcl_36 family-narrative (0.159/0.385), wcl_45 epistemic (0.159/0.320), wcl_11 technical (0.102/0.492), wcl_07 politics (0.110/0.336), wcl_15 casual-interjection (0.135/0.333) | report two-channel; wcl_36/wcl_45 sit just above the class-disjoint bar but fail the share arm — nearest F-candidates under OP-34 SB correction |
| **undetermined (1)** | tension_core | shared-matched < 0.10 — consistent with its no-trait verdict |
