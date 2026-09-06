# SUICA V6 MEPS Fixed-Condition Vector Profile v1

## Access-boundary erratum (2026-09-06)

The historical runner discarded `_mood`, but its shared loader still opened and
parsed `mood.csv`. Those values were not supplied to the embedding or correspondence
statistics; "unused as outcomes" must not be read as "never loaded". The loader now
defaults to `include_mood=False`, and this runner specifies that explicitly.
Mixed Markdown exports are still scanned to recover direct answers; non-answer
sections are discarded, not interpreted. Original metrics below are preserved,
not relabeled as a rerun. The new text-only access and speaker-window audit is in
[VM06 material preflight](grain_vector_mechanisms/VM06_MEPS_MATERIAL_PREFLIGHT_REPORT.md).

## Decision boundary

This is a local-only, same-session Japanese **vector correspondence** pilot. It tests whether the frozen multilingual embedding places a participant's answers to two shared MEPS prompts closer than other participants' answers. It does not estimate personality, clinical state, external validity, cross-language numerical equivalence, or a cross-session response operator.

- source text: `decrypted_output/` user-authored direct MEPS answers only
- assistant turns: excluded from embeddings; retained nowhere in this output
- questionnaires/mood: not loaded as labels or outcomes
- raw text, participant IDs, token IDs, and embeddings: not persisted

## Frozen representation

- model: `intfloat/multilingual-e5-large`
- revision: `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`
- input convention: `query: ` for every symmetric text view
- max sequence length: `512`
- device used: `mps`

## Primary fixed-prompt comparisons

| condition_a   | condition_b   |   n_participants |   same_person_auc |   same_person_auc_ci_low |   same_person_auc_ci_high |   linkage_permutation_p | holm_pass   |   length_matched_coverage |   length_matched_same_person_auc |   length_matched_auc_ci_low |   length_matched_auc_ci_high |
|:--------------|:--------------|-----------------:|------------------:|-------------------------:|--------------------------:|------------------------:|:------------|--------------------------:|---------------------------------:|----------------------------:|-----------------------------:|
| meps_q1       | meps_q2       |               46 |            0.546  |                   0.4773 |                    0.6119 |                  0.0008 | True        |                    0.9783 |                           0.5166 |                      0.4532 |                       0.5719 |
| meps_q1       | meps_q3       |               46 |            0.5651 |                   0.4985 |                    0.6343 |                  0.0002 | True        |                    0.9783 |                           0.5316 |                      0.469  |                       0.6019 |
| meps_q2       | meps_q3       |               46 |            0.6033 |                   0.5377 |                    0.668  |                  0.0002 | True        |                    0.9783 |                           0.5559 |                      0.5005 |                       0.6245 |

## Length / truncation audit

| condition   | analysis_role             |   n_texts |   median_tokens_before_truncation |   max_tokens_before_truncation |   n_texts_over_max_seq_length |   max_seq_length |
|:------------|:--------------------------|----------:|----------------------------------:|-------------------------------:|------------------------------:|-----------------:|
| meps_q1     | primary_fixed_prompt      |        46 |                             119   |                            367 |                             0 |              512 |
| meps_q2     | primary_fixed_prompt      |        46 |                             111.5 |                            252 |                             0 |              512 |
| meps_q3     | primary_fixed_prompt      |        46 |                             108   |                            379 |                             0 |              512 |
| free_chat   | descriptive_free_vs_fixed |        46 |                             248   |                            543 |                             2 |              512 |

## Descriptive free-versus-fixed contrast

| condition_a                | condition_b   |   n_participants |   same_person_auc |   same_person_auc_ci_low |   same_person_auc_ci_high |   linkage_permutation_p |
|:---------------------------|:--------------|-----------------:|------------------:|-------------------------:|--------------------------:|------------------------:|
| fixed_prompt_mean_q1_q2_q3 | free_chat     |               46 |            0.5497 |                   0.4729 |                    0.6237 |                  0.0006 |

The free-chat comparison is descriptive only. Its token audit is shown above; any text above the frozen 512-token cap is encoded with the model's standard truncation and cannot affect the primary direct-answer decision.

## Registered interpretation

Primary promotion rule: every one of the three fixed-prompt comparisons must have AUC >= `0.60`, participant-bootstrap lower CI > `0.55`, and Holm-corrected linkage permutation support at familywise alpha `0.05`. Result: `NOT PROMOTED`.

A PASS licenses only the statement that this frozen multilingual vector representation preserves protocol-conditioned same-person correspondence in this Japanese corpus. It does not license language equivalence, a stable trait claim, or generalization outside this single session.

## Artifacts

- `results/meps_fixed_condition_profile_v1/metrics.csv`
- `results/meps_fixed_condition_profile_v1/token_audit.csv`
- `results/meps_fixed_condition_profile_v1/run_manifest.json`
