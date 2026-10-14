# C2 dedup sensitivity packet: protocol V1 (frozen before inventory and fits)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_07_DECISION.md` §1.
- **Status.** A post-result sensitivity, not retrospective preregistration.
- **Preservation.** The r1 inputs, code, caches and results (`P05/c2_venue_dependence_r1`) are preserved.
- **New files.** Wrapper `scripts/run_grain_c2_dedup_sensitivity.py`, analysis
  `scripts/analyze_grain_c2_dedup_sensitivity.py`, output `P05/c2_dedup_sensitivity_r1/`, version `c2_dedup_v1`.

## Stage I: equivalence inventory (before any fit)

- **Key.** `(user_id, comment_id)`. Within each duplicate-key group, rows are compared in source-file order.
- **Fields consumed by MCD/C2** (from the original code): user_id, comment_id, body_raw, subreddit, the context
  numerics (score, word_count, token_count, char_count) and the 19 response columns
  (`RESPONSE_COLS` in `run_context_fixed_deviation_benchmark.py`).
- **All other columns are unused provenance:** created_utc, link_id, parent_id, lang, author_flair_text_present and
  sample_score.
- **Classes.** Values are equal if identical, or if both are NaN.
  - **exact:** all 33 columns equal to the group's first row;
  - **consumed-equivalent:** all consumed fields equal, some unused field differs;
  - **conflicting:** any consumed field differs.
- **Report.** Class counts, differing-column names with counts, and sha256 hashes of the differing values. No raw
  text is reported.
- **Stop rule.** If any group is *conflicting*, STOP at this checkpoint before fitting. No version is chosen and no
  author is removed.
- **Dedup rule.** Otherwise, collapse each group of exact or consumed-equivalent rows to its first source-file
  occurrence. Unused-field differences are reported explicitly.
- **Role eligibility, computed separately for np_mcd and mcd_only.**
  - The frames are the original `trait_frame(use_p=True/False)` for every trait.
  - For each condition, report the input rows and authors outside that condition's frame, with the reason
    (for example, no P features).
  - Out-of-frame rows are condition-ineligible; they are not corrupt or missing data. No dummy roles are assigned.
  - The evaluation author sets per condition and trait must equal r1's. The packet aborts otherwise.

## Stage II: A′/B′/C′ fits (only if Stage I passes)

- **Arms.**
  - A′: deduplicated input.
  - B′: the frozen r1 trait-named rule, applied after dedup.
  - C′: the r1 hash-order deletion (seed 20260926, row key `user|comment_id|occ`), with per-author d_u taken from
    B′.
  - B′/C′ retained counts are re-checked per author. Zero-comment authors get the r1 training-reference imputation
    and flags.
- **Scope and fitting.**
  - Five traits, five official folds. np_mcd is primary and mcd_only secondary.
  - N/P matrices, preprocessing, inner-validation grid and fixed clustering seeds (42 + 1000·fold, +31) are all
    unchanged, and every transform is fitted on the training fold only.
  - The Ridge call is identical to r1. The actual solver (`solver_`) is recorded per cell; r1 showed that sparse
    input with an intercept dispatches to `sparse_cg`, so no SAG randomness is claimed.
  - Cache keys hash {version c2_dedup_v1, arm, retained-input hash, train/val/test IDs, settings}.
- **Resources.** Two CPU workers, one BLAS thread each. No GPU, neural calls, downloads, API or remote work.
  - One-hour cap including analysis. On hitting it, checkpoint partial results rather than drop cells.
  - No seed sweep.

## Analysis (declared)

- **Bootstrap.** Same paired 2,000-author bootstrap as r1: seed 20260926, stratified by fold. The universe order is
  identical, so the draws are shared across A/B/C (r1), A′/B′/C′, LRNP and ArchA, and across traits.
- **Reproduction check.** r1's B−C must be reproduced exactly from the r1 cells.
- **Estimands.** Pooled r and fold-average r, reported separately.
- **Contrasts.**
  - A′/B′/C′ versus LRNP;
  - B′−C′ and A′−C′;
  - A′−A;
  - (B′−C′) − (B−C).
- **Interpretation limits.**
  - A′−A is the sensitivity to event weighting plus refitted geometry, not an isolated causal duplicate effect.
  - Missingness (unobserved authors) and multiplicities (duplicates per author, per venue) are reported.
- **Reading.** The packet records whether signs, magnitudes or intervals change the venue-dependence reading.
  - No acceptance rule rests on nonsignificance.
  - No rescue run, no clipping, no re-tuning on test.
