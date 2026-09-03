# GRAIN Worked-Example Manual v1 — Building the Instrument on PANDORA

Created: 2026-07-06. Role: paper Section 2 — the existence proof. Section 1
(the method) says what GRAIN is; this manual shows one complete, audited
construction on a real corpus, with every number traceable to a script and a
ledger row. Companion docs: THEORY (rind model), RULEBOOK (fixation rules),
AI_ANALYST_GUIDE (operating standard), CLAIMS_LEDGER (audit trail),
CONSTRUCT_REGISTRY (inventory).

Population scope for every number below: English-writing Reddit users who
publicly disclosed MBTI typology (PANDORA); Essays numbers = psychology
students. Nothing here is claimed beyond these populations (Rulebook D1).

## 1. Materials

| Asset | Content | Role |
|---|---|---|
| PANDORA official dump | 17.6M comments, ~10k users, subreddit+timestamp | raw behavior stream |
| Tier U (dev) | 8,678 MBTI-axis users minus all lockbox users | all discovery/tuning |
| Tier D (dev labels) | Tier-U users' MBTI axis labels | orientation only (never selection) |
| Tier L (historical lockbox) | Big5 1,401 + bridge 375 + Essays confirm-half | opening #1 spent 2026-07-07; overall 2/7 fail, H2/H6 individual pass; one opening remains; Essays confirm-half unopened |
| Essays dev-half | 1,255 students, text + (this manual only) labels | regime tests + dev-tier Big5 orientation |
| Working samples | 5,000-user cap-400 sample; 2,000-user deep (cap 1,200) | standard vs thick-cell designs |

## 2. The construction, step by step (commands are literal)

### S1. Freeze tiers — `build_suica_tiers_v2.py`
Label values never read; membership only. Output: tier CSVs + manifest
(lockbox budget stated). Verified: 0 lockbox users in any dev extraction.

### S2. Slice and score — `prepare_suica_v2_phase2_slices.py`
128-token slices (96 as multiverse check), personality-vocabulary masking,
conditions = subreddits, temporal halves with >= 90-day gap. Scale: 205k
slices / 4,983 users (standard), 2.3M comments (deep).

### S3. Validate the machinery BEFORE real data — `run_suica_synthetic_ground_truth_v2.py`, `run_suica_p0b_thin_cell_regime_v3.py`
P0: planted-truth recovery 0.96-0.98, null FPR 0.06, scorer equivalence
exact. P0-B: multi-start REML certified for thin cells (recovers planted
0.05/0.10; MoM demoted to screening). Anyone can run these with zero data.

### S4. Trait stability, honestly — `run_suica_p1_disjoint_retest_v2.py`
Disjoint-occasion (>= 90-day gap) test-retest, n=1,437: first-person SB 0.73
(0.79 deep), directive 0.53-0.57, others lower; volume curve gives the
tokens-for-precision table (first-person ~0.86 at ~4.5k tokens/half). The
same script run on same-text repartitions would print 0.9+ — the manual's
first lesson: only disjoint-occasion numbers are trait evidence.

### S5. The mechanism test that FAILED — `run_suica_p2_condition_centering_v2.py`, `run_suica_e1_e2_centering_rescue_v2.py`
Condition-mean centering reduces stability in every variant (naive, partial,
FE out-of-sample; pooled delta -0.09). This falsification is part of the
method: conditions carry person signal through self-selection; control rind
by design, not statistics. Choice IS a channel, not a nuisance.

### S6. The choice channel — `run_suica_e3_e4_choice_scale_class_react_v2.py` + `run_suica_op6a_choice_axes_holdout_v3.py`
12 content classes; venue-choice log-ratio axes; held-out confirmation
(cohort-A fit/selection, cohort-B confirm): 5/5 axes confirm, r_B 0.48-0.68,
shrinkage 0.027. Same-user choice-profile AUC 0.84-0.91 (0.839 = class-profile artifact in
e3_e4_results.json; 0.909 = ledger-recorded exploratory computation on
top-300 subreddit vectors, n=2,758, no results artifact — T1). Leakage rule
shown live: the MBTI-community class is excluded from all anchor work.

### S7. Growing the inventory — `run_suica_op5_construct_discovery_v3.py`
Open-vocabulary discovery with A-discover/B-confirm user split: 15/15
candidates confirmed on unseen users (r_B 0.38-0.65), inventory 4 -> 19.
Registry assigns operational names and tiers (style-register vs
content-interest vs state-unstable).

### S8. Blind coding with a real panel — `build_suica_op22_item_bank_v2.py`, `run_suica_op22_llm_coding_v2.py`, `evaluate_suica_op22_coding_v2.py`
240 7-way items + 90 pairs; four model FAMILIES + the researcher (120 items,
blind). 8/15 constructs pass; legibility classification: human+machine
(epistemic argumentation), content-tier (trivially codeable — not counted as
validity), machine-mostly, machine-only (apostrophe-omission: human 0/8 —
machine-legible constructs are real but need mechanism-based validity).

### S9. The interpretation layer — `run_suica_e9_embedding_interpretive_channel_v3.py`, `run_suica_e11v2_adjective_projection.py`
E9: construct definitions translate into embedding directions (11/19 MTMM);
neighbor reading interprets a person via measured profiles of embedding
neighbors (18/19 above random null, deltas +0.30..+0.59). E11 v2 turns
directions into METRIC adjective profiles (bipolar marker axes, permutation
calibration, bootstrap CIs, sense-collision flags):
- trait mode: direction stability SOLVED (median bootstrap rank stability
  0.868) and profiles discriminant (Jaccard 0.080), but under strict
  permutation calibration only 1/19 constructs clears the marker-axis
  99th-percentile bar — v1's ten "pole enrichments" are exposed as
  calibration artifacts. CI-coherent weak alignments remain (novelty ->
  E+/O+ ~ +0.07; tension -> N+; directive/profanity -> A-). RULING: adjective
  profiles ship as DESCRIPTIVE aids with CIs; the licensed construct-trait
  bridge is the dev-anchor label table (section 3), not adjective space.
- state mode (horizontal-cut modes f1/f3/f5, within-person centered month
  embeddings): direction stability 0.90-0.95, discriminant (0.044), no
  marker axis clears the strict permutation bar; CI-coherent weak leans are
  interpretation-consistent (positivity/growth mode f1 -> A+/O+/N-;
  reflective-opinionated f3 -> N+; f5 -> C-/A-). Same ruling as traits:
  descriptive aids with CIs, not licensed bridges.

### S10. Performance orientation WITHOUT opening the lockbox — `run_suica_dev_anchor_performance_v1.py`
MBTI from Tier-D labels (n=3,183); Big5 from Essays dev-half labels
(n=1,255, battery TRANSPORTED from PANDORA without refit); lockbox sealed.

## 3. Analysis performance (T2 development-tier orientation)

| Target | n | Ridge CV r (full battery) | Ridge CV r (style only) | Best single edge | q<.05 edges |
|---|---|---|---|---|---|
| MBTI TF (thinking-feeling) | 3,183 | **0.346** | 0.339 | positive-enthusiasm +0.316 | 22 |
| MBTI JP | 3,183 | 0.182 | 0.172 | casual-interjection -0.108 | 17 |
| MBTI EI | 3,183 | 0.139 | 0.128 | choice-axis-02 -0.086 | 15 |
| MBTI SN | 3,183 | 0.120 | 0.112 | epistemic-argument +0.099 | 5 |
| Big5 Openness (Essays) | 1,255 | — | 0.194 | first-person -0.127 | 7 |
| Big5 Neuroticism (Essays) | 1,255 | — | 0.160 | **first-person +0.122** (matches prereg H2 direction) | 7 |
| Big5 Extraversion (Essays) | 1,255 | — | 0.141 | media-consumption -0.112 | 3 |
| Big5 Conscientiousness (Essays) | 1,255 | — | 0.117 | profanity -0.114 | 4 |
| Big5 Agreeableness (Essays) | 1,255 | — | 0.109 | profanity -0.133 | 3 |

Two context anchors for these numbers:
- TF at 0.346 from ~31 interpretable scores exceeds the project's entire
  earlier exploratory record (best single edge 0.267 from hundreds of cells)
  and dwarfs the retired direct generative-MBTI route (chance).
- The Essays column is a TRANSPORT test: a battery fitted entirely on Reddit
  applied unchanged to student essays yields mean Big5 CV r ~ 0.144 —
  on par with an opaque bge-large transfer baseline (~0.153 in the repo's
  earlier work) while remaining 19 auditable scores.

Reading rules for this table:
1. These are ORIENTATION numbers on development data; confirmatory external
   validity is reserved for the sealed, preregistered lockbox opening.
2. Expectation frame: official PANDORA full-text ML baselines reach mean
   Big5 r ~ 0.25-0.32 with thousands of opaque features; an interpretable
   ~30-score battery is expected below that. The comparison of record is
   information-per-interpretable-score, not leaderboard position.
3. Class imbalance: EI positive rate ~ 0.21, SN ~ 0.89 — report r together
   with base rates.

## 4. Score interpretation guide (for human readers and R-INTERPRETER)

1. **Trait track** (style-base, uncentered): report score + registry
   reliability band + tokens-observed vs tokens-required (volume curve).
   Below the volume floor -> "insufficient text", no interpretation.
2. **Choice track**: 12-axis venue/interest profile (compositional; compare
   within population; ax_11-type leakage exclusions per deployment).
3. **State track**: per-session first-person deviation is the only currently
   measurable state channel, and its per-occasion precision (contiguous SB
   ~ 0.27 at ~1k tokens; audited value from ledger row E5-P-E5c — the
   results file retains the superseded parity-split values) is below the
   0.30 feasibility bar — dashboards must
   show it as tentative. Affect-vocabulary rates are NOT occasion states
   (<= ~2% variance) — do not report "tension" as a mood measure.
4. **Signature track** (provisional): first-person and directive if-then
   patterns across situation classes (stranger-null validated, single
   cohort). Report only with the "provisional" tag.
5. **Adjective profiles** (E11 v2, where licensed): "anxious +2.1 [+0.7,+2.6]"
   style metric descriptors; only constructs passing V2 criteria; flags
   (!lex sense-collision) must be displayed.
6. **Language rules**: ledger-status vocabulary only (T1 exploratory ... T4
   confirmatory); forbidden words without license: validated, diagnostic,
   personality type, clinical significance.
7. **Report template**: see AI_ANALYST_GUIDE section 3.3.

## Appendix A. Scorer v3.1 robustness (OP-18)

Smart apostrophes (U+2019, 3.8% of slices) split tokens in the frozen v3
scorer. Re-running the full P1 battery with pre-tokenization normalization
(scorer v3.1, prep flag `--normalize-apostrophes`) changes no quoted
headline value by more than 0.002 (first-person 0.570 vs 0.569; directive
0.355 vs 0.357; tension 0.213 vs 0.215; largest any-metric shift is the
tension Spearman-Brown value at 0.0022). The v3 numbers therefore stand as reported;
normalization is the recommended ingestion default going forward. Lexicon
disjointification is deliberately deferred to scorer v4 (formula changes
restart the validation chain).

## 5. What this section proves — and what it does not

Proves: a reproducible pipeline exists that takes a raw text-behavior corpus
to (a) a 19-construct provisional inventory with held-out stability, (b) a
held-out choice-axis profile, (c) coding-validated legibility classes, (d) a
metric interpretation layer, (e) orientation-grade anchor relations — with
every step estimator-validated, adversarially audited (7 rounds), and
governed by pre-committed criteria. That is the "本当に作れる" claim, held to
the same falsification standard that killed the method's own first mechanism.

Does not prove: population generality (one platform + students), dialogue
readiness (OP-7), Japanese deployment (OP-8), confirmatory external validity
(sealed lockbox), clinical utility (OP-14). These are the opening moves of
the next study, not gaps hidden in this one.
