# C2 event-version sensitivities: protocol V1 (frozen before audit and fits)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_08_DECISION.md` Packet B.
- **Files.** Version `c2_event_version_v1`; runner `scripts/run_grain_c2_event_version_sensitivity.py`; analysis
  `scripts/analyze_grain_c2_event_version_sensitivity.py`; output `P05/c2_event_version_sensitivity_r1/`.
- **Status.** A post-result sensitivity: not a replay rescue, not a baseline-tuning campaign.

## Provenance (bounded; producer code only)

- `run_raw_text_signal_cartography.collect_raw_comments` streams the raw PANDORA comments file. The fields it uses
  are author, body, created_utc, subreddit, link_id, parent_id, id, lang, score, word_count and author_flair_text.
- There is **no retrieval, snapshot or edit-time field**. Top-k sampling scores rows by a hash of (author, id), so
  repeated versions of an id tie and are selected together.
- The final sort by (created_utc, comment_id) is stable, so within-key order is **source-file order, not edit
  chronology**.
- Emotion/length features (token_count, char_count and the 19 rates) are computed from each row's own body by
  `run_situation_emotion_dynamics.comment_emotion_features`. `score` and `word_count` come from the source file.
- No authoritative snapshot rule exists. Therefore: one event key can have several observed versions, and the
  authoritative version order is unknown.

## Views (both retained; no third view; no selection by performance)

- **FIRST.** Per (user_id, comment_id), the first source-file occurrence.
- **LAST.** The last occurrence.
- **Both views.** One row per key. Every selected record is kept whole (body and all consumed features), with no
  content normalization and no rewriting. Row order is the file order of the retained rows.
- **Alignment audit (before fits).**
  - For every selected record, recompute `comment_emotion_features(body_raw)` and compare it with the stored
    body-derived consumed features (token_count, char_count, 19 rates); absolute tolerance 1e-9.
  - STOP before fitting if any selected record is inconsistent. Report exact counts.
  - Differences in source-provided `score`/`word_count` between versions are reported as provenance, not as
    inconsistency.
- Reports contain metadata only; no raw text is exported.

## Arms within each view (r1 rules unchanged)

- **Arms.**
  - A′: all retained events.
  - B′: the frozen r1 trait-named-venue rule.
  - C′: the r1 hash-order deletion (seed 20260926; row key `user|comment_id|occ`, where occ is recomputed after
    selection and is 0), matched per author to B′'s deletion counts. B′/C′ retained counts are rechecked.
- **Unchanged from r1.** Eligible authors per condition; official repeat-0 folds; five traits; np_mcd primary and
  mcd_only secondary; N/P matrices; train-fold-only transforms; inner grid; clustering seeds; zero-comment imputation
  with flags and exclusion roles.
- **Scale.** Exactly 150 cells per view (300 total). No seeds or algorithm alternatives.
- **Solver.** The Ridge call is identical to r1, and the actual `solver_` (sparse_cg) is recorded.
- **Cache identity.** sha256 over {version, view, arm, content hash, train/val/test IDs, settings}. The content hash
  is the sha256 of the sorted (row key, sha256 of consumed field values) of the retained rows, so FIRST and LAST
  cannot share cached MCD states.
- **LR-NP reuse.** The archived LR-NP predictions are reused only after verifying the author/fold/label alignment and
  the file hashes used by r1.

## Analysis (declared)

- **Bootstrap.** The r1 2,000-author bootstrap (seed 20260926, stratified by fold, same universe order), so draws are
  shared with r1 A/B/C, ArchA and LRNP.
- **Estimands.** Pooled r and fold-average r, reported separately.
- **Contrasts per view.**
  - A′/B′/C′ − LRNP;
  - B′ − C′ and A′ − C′;
  - A′ − A;
  - (B′ − C′) − (B − C).
- **Contrasts across views.** FIRST − LAST for A′ and for (B′ − C′).
- **Interpretation limits.**
  - A′ − A changes weights, versions and refitted geometry together; it is not an isolated duplicate effect.
  - Missingness and multiplicities are reported.
  - Both views are shown with their sign and interval changes. No favourable view is selected.
- **Resources.** 2 CPU workers, 1 BLAS thread each, no other heavy work at the same time. 90-minute cap including
  analysis; checkpoint partial results, and no rescue rerun.
