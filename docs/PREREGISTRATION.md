# SUICA Lockbox Opening #1 — PREREGISTRATION (SEALED)

> Historical preregistration under the former project name. Its terminology
> and identifiers remain part of the sealed record; the current method is GRAIN.

Status: FINAL as of the v0.1.0 freeze. The lockbox is NOT opened; budget
2/2 openings remain. The initial commit hash of the independent release
repository is the seal for this document.

Disclosure of prior label contact: development-tier orientation (Tier-D MBTI
and Essays dev-half Big5; see WORKED_EXAMPLE_MANUAL section 3) was computed
on 2026-07-06, AFTER hypotheses H1-H8 were drafted (2026-07-05) and WITHOUT
modifying the hypothesis set. The dev-tier observation consistent with H2
(first-person -> Neuroticism +, r=+0.122) is disclosed here, not used to
alter the set. Lockbox users' labels remain untouched.

## What will be frozen at commit time

- Scoring code: `scripts/suica_v2_lib.py`, `scripts/run_suica_e1_e2_centering_rescue_v2.py`
  (twoway_fe_condition_effects), `scripts/run_suica_e3_e4_choice_scale_class_react_v2.py`
  (condition class map building), at the commit hash of the prereg commit.
- Battery (SUICA v3, REVISED after 2026-07-05 audit — FE-base channel removed):
  - Style base (raw, uncentered): `tension_core_v2`, `directive_action_v2`,
    `first_person_usage_v2`, `novelty_play_v2`.
  - Choice axes: log-ratio scores over the 12 Tier-U-frozen content classes
    (class map transported, not refit), plus choice entropy.
  - **Excluded: `choice_ax_11`** (contains MBTI subreddits — leakage) and
    `adversity_recovery_v2` (retest ~0, dead construct).
- Anchor data: PANDORA Big5 prepared (n=1,401) and bridge strict377 MBTI; scored once.

## Directional hypotheses (final set to be fixed at commit; <= 8)

| # | Predictor | Target | Direction | Grounding |
|---|-----------|--------|-----------|-----------|
| H1 | raw_base tension_core | Neuroticism | + | negative-affect language <-> N (Pennebaker; Park et al. 2015) |
| H2 | raw_base first_person_usage | Neuroticism | + | I-usage <-> N/depression (Edwards & Holtzman meta) |
| H3 | raw_base novelty_play | Openness | + | novelty/creativity vocabulary <-> O (open-vocabulary literature) |
| H4 | raw_base directive_action | Agreeableness | - | imperative/directive tone <-> lower A |
| H5 | raw_base directive_action | TF_cont (bridge) | thinking direction | prior exploratory orientation + TA literature |
| H6 | choice axis politics/news | Openness | + | interest breadth/ideas engagement |
| H7 | choice entropy (venue breadth) | Openness | + | interest breadth <-> O |
| H8 | choice axis gaming | Extraversion | - | weakest; may be dropped at commit |

## Analysis (fixed)

- Pearson r per hypothesis, users with >= 4 conditions and >= 12 slices.
- BH-FDR within the hypothesis set only.
- Success rule: >= 50% of the set q < 0.05 AND every significant effect in the
  preregistered direction. Anything else = P5 fail; no re-analysis, no
  additional cells, no second look without spending opening #2.
- Also reported (non-confirmatory): incremental delta R^2 of the battery over
  Empath-194 ridge for Neuroticism and Openness.

## Eligibility and power (added 2026-07-05; metadata-only Tier-L read, logged)

Eligibility (>= 4 subreddits with >= 4 comments AND >= 12 estimated slices,
counted from dump metadata only — no text scored, no labels read):

| Cohort | In dump | Eligible |
|--------|---------|----------|
| Big5 (n=1,401) | 1,401 | **1,108** |
| Bridge strict (n=375) | 375 | **326** |

Analytic power (two-tailed alpha=.05) at eligible n:

| True observable r | Big5 (n=1,108) | Bridge (n=326) |
|---|---|---|
| 0.08 | 0.76 | 0.30 |
| 0.10 | 0.92 | 0.44 |
| 0.15 | 0.999 | 0.78 |
| 0.20 | 1.00 | 0.95 |

Attenuation ceilings (observable r = true validity x sqrt(rel_x x rel_y),
rel_y ~ 0.85 for questionnaire anchors; rel_x = disjoint-occasion SB from the
registry): first_person factor 0.82, novelty 0.74, directive 0.70, choice
axes 0.64-0.76, tension 0.60. Example: a true validity of 0.20 for directive
would appear as observable r ~ 0.14.

Power statement (FINAL): Big5-side hypotheses (H1-H4, H6-H8) are adequately
powered for observable r >= 0.10. H5 (bridge TF_cont) is underpowered below
observable r ~ 0.15 (power 0.44 at 0.10). FINAL RULE: the confirmatory set
is H1-H4 + H6-H8 (7 hypotheses; success = >= 4 of 7 at BH-FDR q < 0.05
within-set, all significant effects in the preregistered direction).
H5 is SECONDARY: reported with CI, direction noted, excluded from the
denominator.

## Expectation management

PANDORA official text baselines: mean Big5 r ~ 0.25-0.32 with full TF-IDF
models. Single-construct directional r in the 0.10-0.25 band is the realistic
success zone for interpretable subscales.

## Preconditions before commit+open

1. E1-E4/P3 audit closed (adversarial pass upheld).
2. User approves spending lockbox opening #1.
3. Prereg committed to git; results reference the commit hash.
