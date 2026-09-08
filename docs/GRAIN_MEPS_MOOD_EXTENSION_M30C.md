# M30C: local adapted mood-score extension

2026-09-09. Newly scoped finite follow-up within the user's all-factor request.
M30's22 relational/session criteria remain frozen. This packet separately adds
the existing processed mood measurements after an item and occasion audit.
No diagnostic use, fitting on participants, raw-survey/Study1 access or API upload.

## Data and target definitions

Only `research2_anal/processed/mood_long.csv` is newly allowlisted. It has46
participants x4 phases x13 items, with no missingness; responses are1-6.
Item1-8 are the locally adapted PANAS negative-affect selection. Item9-13 are
the locally labeled POMS_SHORT negative/dejected-feeling selection. Preserve
these as **adapted item composites**, not canonical PANAS/POMS norms or factors.
Strict complete-item means, no imputation, no reverse items in this selection.

The imported study code defines baseline -> pre (after worry recall) -> mid
(after MEPS) -> post (after free AI chat). Direct-answer evidence is compared
primarily with mid levels; free-chat evidence with post levels. Preceding phase
and after-minus-before associations are secondary, retrospective descriptors.
Future/post-questionnaire outcomes never enter the text or prompt.

## Fixed experiment

3 queries: adapted NA8, adapted dejected5, combined negative13. No outcome-driven
wording; English prompts describe the administered Japanese item content, while
the participant's Japanese text is unchanged. Ask for the participant's own
expressed state, not a fictional scenario character and not a psychiatric diagnosis.
Use the same fixed0-9 native conditional expectation as M30, locally.

Both complete46-author text views are already bounded and fit the observer:
3 queries x46 participants x2 views =276 native calls, no search grid. Collect
and hash scores/P/Q before joining audited mood means. Keep each view separate.
Report the full3-query x12-phase-criterion matrix per view, intended endpoint
correlations, Fisher and participant-bootstrap intervals, and BH families for
endpoint versus descriptive temporal comparisons. No day-to-day or causal
therapy-effect interpretation of these within-session observations.

This packet supplies actual state criteria to the all-factor ledger. A useful
association advances a state-sensitive query function; a weak relation maps
evidence opportunity. It is not a reason to silently turn a state into a trait.
