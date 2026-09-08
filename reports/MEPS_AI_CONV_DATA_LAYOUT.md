# MEPS + AI Conversation: Data Layout and Reference Map

> 2026-09-09 scope addition: the user has now authorized a separate **local
> exploratory external-linkage branch** for Study 2 scored pre/post measures.
> See [the V5 all-factor plan](../docs/GRAIN_ALL_FACTOR_RESEARCH_GOAL_V5.md).
> Only its allowlisted numeric items/scores may enter that branch, with no
> training, private API upload, Study 1 merge or old lockbox opening. The
> reference-only policies below remain the historical VM06/default importer
> contract; they do not retroactively describe the new M27/M30 branch.

## Authoritative route

`decrypted_output/` is the sole SUICA text source for
the fixed-condition pilot. It contains `46` participant export
directories. The importer resolves this route automatically when given the
complete experiment bundle.

## Directory map

| relative_path                 | role                               | policy                                  | present   |
|:------------------------------|:-----------------------------------|:----------------------------------------|:----------|
| decrypted_output              | authoritative_private_text_export  | READ_FOR_FIXED_CONDITION_PILOT          | True      |
| research2_anal/processed      | same_session_deidentified_analysis | REFERENCE_ONLY                          | True      |
| research2_anal/questionnaire  | same_session_questionnaire_import  | REFERENCE_ONLY                          | True      |
| research2_anal/survey_rawdata | raw_questionnaire_export           | EXCLUDE_PII_AND_DO_NOT_READ             | True      |
| research2_anal/coding         | duplicate_free_chat_content_coding | EXCLUDE_DUPLICATE_TEXT_BY_DEFAULT       | True      |
| research1_anal                | different_study_analysis           | NEVER_AUTO_MERGE                        | True      |
| research_docs                 | protocol_and_scale_provenance      | READ_FOR_DOCUMENTATION_ONLY             | True      |
| output_formal_encrypted       | encrypted_archive                  | DO_NOT_USE_WHEN_DECRYPTED_EXPORT_EXISTS | True      |
| keys                          | cryptographic_material             | EXCLUDE_SECRET                          | True      |
| dist                          | experiment_application_bundle      | EXCLUDE_RUNTIME_ARTIFACT                | True      |

## Registered but withheld references

| key                          | relative_path                                          | source_class                    | policy                                  | description                                                                                                | present   | id_column      | reference_unique_codes   | matched_text_export_codes   | text_export_coverage   | note           |
|:-----------------------------|:-------------------------------------------------------|:--------------------------------|:----------------------------------------|:-----------------------------------------------------------------------------------------------------------|:----------|:---------------|:-------------------------|:----------------------------|:-----------------------|:---------------|
| r2_pre_questionnaire_scored  | research2_anal/processed/pre_questionnaire_scored.csv  | same_session_pre_measure        | WITHHELD_NOT_FOR_TRAINING_OR_VALIDATION | Pre-session AI mind-attribution, attachment, trust, and usage measures.                                    | True      | participant_id | 46                       | 46                          | 1.0                    | id_column_only |
| r2_post_questionnaire_scored | research2_anal/processed/post_questionnaire_scored.csv | same_session_post_measure       | WITHHELD_NOT_FOR_TRAINING_OR_VALIDATION | Post-session perceived empathy, session evaluation, mind attribution, attachment, and disclosure measures. | True      | participant_id | 46                       | 46                          | 1.0                    | id_column_only |
| r2_participant_protocol      | research2_anal/processed/participants.csv              | same_session_protocol_metadata  | STRUCTURAL_CONTEXT_ONLY                 | Experiment timing, condition exposure, and AI-use opportunity metadata.                                    | True      | participant_id | 46                       | 46                          | 1.0                    | id_column_only |
| r2_analysis_master           | research2_anal/processed/analysis_master.csv           | derived_joined_analysis         | DO_NOT_LOAD_BY_DEFAULT                  | Convenience join containing many downstream fields; not an authoritative SUICA input.                      | True      | participant_id | 46                       | 46                          | 1.0                    | id_column_only |
| r2_raw_qualtrics             | research2_anal/survey_rawdata                          | raw_questionnaire_export        | EXCLUDE_PII_AND_DO_NOT_READ             | Raw survey exports may contain direct identifiers and must remain outside SUICA.                           | True      |                | <NA>                     | <NA>                        | <NA>                   | not_read       |
| r2_content_coding            | research2_anal/coding                                  | duplicate_human_text_and_codes  | EXCLUDE_DUPLICATE_TEXT_BY_DEFAULT       | Coding files duplicate free-chat content and must not become a second text source.                         | True      |                | <NA>                     | <NA>                        | <NA>                   | not_read       |
| r1_prior_study_panel         | research1_anal/data_cleaned.csv                        | different_study_partial_overlap | NEVER_AUTO_MERGE                        | Larger Study 1 panel; any overlap requires a separately declared longitudinal analysis.                    | True      | anon7          | 177                      | 32                          | 0.6956521739130435     | id_column_only |

## Binding use rules

1. Use only `decrypted_output/` user-authored text for the current SUICA
   fixed-condition verification.
2. Treat Research 2 pre/post questionnaires as registered, linked, **withheld**
   data. They are not training labels and are not external-validity outcomes in
   this phase.
3. Do not read `survey_rawdata/`: raw Qualtrics exports can contain direct
   identifiers. Do not use `coding/` as a duplicate text path.
4. Treat Research 1 as a different study. Its partial overlap cannot enter any
   SUICA analysis without a separately declared longitudinal design and merge
   audit.

This document records data routing only. It makes no psychometric or causal
claim and does not serialize participant IDs, text, or questionnaire values.
