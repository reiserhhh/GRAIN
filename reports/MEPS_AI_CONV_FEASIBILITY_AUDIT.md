# MEPS + AI Conversation: SUICA Feasibility Audit

## Historical access clarification (2026-09-06)

This historical feasibility report included the export's within-session mood
table. It is not a no-label VM06 result. The importer now defaults to text-only;
reproducing the state-table portion requires the explicit `--include-mood` option
and separate authorization. No state-table rerun was performed in the current
mechanism loop. The newer route does not open adjacent registry/scale tables;
mixed Markdown containers are scanned only to retain direct-answer sections.
See [the VM06 material audit](grain_vector_mechanisms/VM06_MEPS_MATERIAL_PREFLIGHT_REPORT.md).

## Decision

**PILOT_READY_WITH_BOUNDARIES**. This private 46-participant corpus is suitable for a bounded
SUICA protocol pilot: fixed-prompt response comparison, fixed-versus-free
condition contrast, and an exploratory association with measured within-session
mood change. It is **not** evidence for a stable personality trait, a
cross-context response operator, or the V6 two-epoch-by-two-replica dynamic
object.

## Data handling boundary

The importer reads private text only in memory. Its outputs contain aggregate
counts or pseudonymous metadata; no source participant IDs, user text, or AI
text are written by this script. AI turns are never treated as participant
responses. They remain turn/character-count opportunity covariates.

The text exporter is resolved through `decrypted_output/` when the complete
experiment bundle is supplied. Adjacent questionnaire tables are registered as
withheld references only: they are neither loaded into this audit nor used for
SUICA training or external-validity claims.

## Imported text views

| view_type        | condition   |   n_views |   n_participants |   total_user_chars |   median_user_chars |   minimum_user_chars |   maximum_user_chars |   median_user_turns |   median_assistant_turns |
|:-----------------|:------------|----------:|-----------------:|-------------------:|--------------------:|---------------------:|---------------------:|--------------------:|-------------------------:|
| free_ai_chat     | free_chat   |        46 |               46 |              21044 |              458.50 |                  126 |                 1023 |                8.00 |                     9.00 |
| meps_ai_chat     | meps_q1     |        27 |               27 |               7770 |              177.00 |                   53 |                  857 |                5.00 |                     4.00 |
| meps_ai_chat     | meps_q2     |        27 |               27 |               4945 |              169.00 |                   13 |                  852 |                2.00 |                     2.00 |
| meps_ai_chat     | meps_q3     |        26 |               26 |               5104 |              155.00 |                    5 |                  889 |                2.00 |                     2.00 |
| meps_answer      | meps_q1     |        46 |               46 |              11475 |              227.50 |                   66 |                  775 |                1.00 |                     0.00 |
| meps_answer      | meps_q2     |        46 |               46 |              10289 |              206.50 |                   53 |                  477 |                1.00 |                     0.00 |
| meps_answer      | meps_q3     |        46 |               46 |               9977 |              204.00 |                   61 |                  743 |                1.00 |                     0.00 |
| meps_answer_slot | meps_q1     |       138 |               46 |              11383 |               65.00 |                    8 |                  336 |                1.00 |                     0.00 |
| meps_answer_slot | meps_q2     |       138 |               46 |              10197 |               68.00 |                    5 |                  231 |                1.00 |                     0.00 |
| meps_answer_slot | meps_q3     |       138 |               46 |               9885 |               62.00 |                    7 |                  279 |                1.00 |                     0.00 |

## Direct-answer recovery

The Markdown export supplied all three direct MEPS answer blocks. The fixed
Japanese rhetorical scaffold is removed before future scoring and retained only
as a slot indicator (`first`, `then`, `next`). `reported_answer_chars` is an
export-side field that includes scaffold/formatting conventions, so it is an
integrity check rather than a text-score denominator.

|   question_index |   n_expected |   n_recovered |   n_three_slot_format |   median_reported_chars |   median_recovered_chars |   median_reported_minus_recovered |
|-----------------:|-------------:|--------------:|----------------------:|------------------------:|-------------------------:|----------------------------------:|
|             1.00 |        46.00 |         46.00 |                 46.00 |                  244.50 |                   227.50 |                             17.00 |
|             2.00 |        46.00 |         46.00 |                 46.00 |                  223.50 |                   206.50 |                             17.00 |
|             3.00 |        46.00 |         46.00 |                 46.00 |                  221.00 |                   204.00 |                             17.00 |

## Mood-state coverage

| phase    | source_scale   |   n_participants |   n_items |   mean_response |   sd_between_participants |
|:---------|:---------------|-----------------:|----------:|----------------:|--------------------------:|
| baseline | PANAS_NA       |               46 |     8.000 |           2.329 |                     0.952 |
| baseline | POMS_SHORT     |               46 |     5.000 |           2.526 |                     1.011 |
| mid      | PANAS_NA       |               46 |     8.000 |           2.160 |                     0.935 |
| mid      | POMS_SHORT     |               46 |     5.000 |           2.143 |                     0.981 |
| post     | PANAS_NA       |               46 |     8.000 |           2.193 |                     1.082 |
| post     | POMS_SHORT     |               46 |     5.000 |           2.222 |                     1.138 |
| pre      | PANAS_NA       |               46 |     8.000 |           3.057 |                     1.045 |
| pre      | POMS_SHORT     |               46 |     5.000 |           3.278 |                     1.192 |

## What this corpus can test now

1. **Condition-fixed response profile:** model a score as `score[u, prompt] =
   prompt effect + author-within-protocol deviation + residual`. This is a
   conditional text-behavior comparison, not a personality trait score.
2. **Free versus fixed response:** compare the same person's task-answer view
   with their free-chat view while retaining text volume, turn count, and AI
   exposure as opportunity variables.
3. **Exploratory state linkage:** after freezing a Japanese representation and
   score set, relate free-chat descriptors to the pre/mid/post mood trajectory;
   do not train the descriptor on mood labels.

## What must wait for a new design

- Two or more sessions/days per participant;
- at least two independent replicas of each common condition per epoch;
- a frozen Japanese or multilingual embedding/segmentation layer and its
  language-specific reliability audit;
- a preregistered external criterion if any trait or clinical claim is wanted.

## Current protocol facts

- Participants discovered: `46`
- Participants loaded: `46`
- Missing source files: `0`
- The correct analysis unit is **user-authored text in its declared condition**.
  Assistant language is context, never a participant score input.
