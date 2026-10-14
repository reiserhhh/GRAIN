# C2 paired MCD venue-dependence packet: protocol V1 (frozen before any fit)

Date 2026-09-26 JST. Authority: `project persona/reports/claude_research_loop_20260926/SEMINAR_FABLE_06_DECISION.md` item 2.
Runner: `scripts/run_grain_c2_mcd_venue_dependence.py` (new, versioned `c2_v1`). It imports the original
functions read-only from `project persona/scripts/run_pandora_baseline_mcd_integration.py` and
`scripts/run_context_fixed_deviation_benchmark.py`. Their sha256 values are recorded in the manifest. The old
runner, archived outputs and code are never modified. Output goes to
`results/grain_gptpro_foundation_v3/P05/c2_venue_dependence_r1/`.

## Question

How does the historical MCD increment (NP+MCD over official LR-NP) depend on comments in trait-named venues?
This is not a test of whether those venues are invalid. Choice of venue is a legitimate possible signal channel,
and dependence on it is not by itself direct label leakage.

## Input and arms (rules fixed from subreddit metadata only)

The input is `project persona/analysis/output/situation_emotion_dynamics/20260613_041057/comment_emotion_features.csv`
(sha256 868c3e70…). All 83,355 rows are restricted to PANDORA users, exactly as `load_comments(version="paper_visible")`
does. The deletion unit is one input row. The file contains 4,775 exact duplicate rows (user, comment_id), and the
original pipeline kept them, so every arm keeps them unless the arm deletes them.

- Row key: `user_id|comment_id|occ`, where `occ` is the 0-based occurrence index of that (user, comment_id) pair in file order.
- **Arm A.** All rows, which is the original input.
- **Arm B.** Remove every row whose subreddit (lowercased, stripped) is trait-named. A subreddit is trait-named if it either:
  - matches `^[ie][ns][tf][jp]s?$` (the 16 MBTI type names), or
  - contains any of `mbti`, `personality`, `introvert`, `extrovert`, `extravert`, `enneagram`, `socionic`, `typology`, `typeme`.

  Borderline names are declared NOT trait-named, for example `jung` (Jungian psychology in general), `frankocean` and
  `typescript`. The resulting subreddit list is written to `trait_named_venues.txt` and hashed before any fit.
- **Arm C.** Per-author count-matched random deletion. Let d_u be the number of rows arm B removes for author u.
  - Arm C removes the d_u rows of u with the smallest `sha256("20260926|" + row_key)`, drawn from all of u's rows.
  - This is one fixed-seed realization. There is no seed search, and it is not an average over deletions.
  - Retained counts in B and C must be equal for every author. This is checked, and the run aborts if it fails.

## Estimator (original settings; deterministic variant)

- **Splits.** Official Big5 repeat-0 folds. Test fold f, validation fold (f+1) mod 5, all other folds for training.
  The frame and valid-author rules are those of the original `trait_frame`, and the N/P matrices are the original
  released feature matrices in every arm.
- **MCD fit.** `fit_fold_mcd`, unchanged, run inside each train fold. It fits the comment TF-IDF (16,000 max features,
  1–2-grams, min_df 2) and SVD 64, the context metadata space (≤32 dims), the response space, StandardScaler,
  MiniBatchKMeans k=48 and the soft membership at tau scale 1.0 with eps 1e-4. The feature is `mcd_choice_logratio`,
  51 dims. The seed is 42 + 1000·fold, with k-means seed +31 as in the original.
  - MCD fits are label-free.
  - Cache key: sha256 over {version `c2_v1`, arm, retained-input hash (sha256 of the sorted retained row keys),
    train/val/test author lists and all MCD settings}. Cached as npz and json.
- **Authors with no retained rows.** The original `aggregate_mcd_by_user` zero-fills their 51 dims:
  - log-ratio 0 = training-reference imputation;
  - entropy 0 and top-share 0 are zero-fills from the original code, not measured values;
  - log1p(count) = 0.

  This is imputation, not a measured reference-like person. A flag `mcd_unobserved` is written to the artifacts and
  is never used as a feature. Authors are never dropped.
- **Selection and fit.** The original `fit_select_predict` loop:
  - SelectKBest(f_regression) with the original feature-count candidates;
  - Ridge over alpha 2^-10..2^4, fitted with `solver="sag"` and `max_iter=20000` at the sklearn 1.8.0 default tolerance;
  - selection by maximum validation Pearson, with the strict `>` and first-found tie order of the original.
- **Only change.** `random_state` is set explicitly to the paired seed
  `int(sha256("c2_v1|{condition}|{trait}|{fold}|{feat_n}|{alpha}")[:8], 16) % 2147483647`. It is identical across
  arms A, B and C. SAG is not replaced by Cholesky.

  This is a labelled deterministic replication variant, not a bit-identical Windows reproduction.
- **Conditions.** `np_mcd` is primary and `mcd_only` secondary. All five traits, folds 0–4. `n_mcd` is out of scope
  and not claimed replicated.
- **Resources.**
  - Local CPU; the environment sets OMP/OPENBLAS/MKL/VECLIB threads to 1.
  - Alpha fits run through joblib loky with n_jobs=2, so at most 2 worker processes. MCD fits run serially in the main process.

## Stage 1: replay and profiling gate (at most 30 minutes)

Stage 1 runs arm A, condition np_mcd, agreeableness (the original's first trait), fold 0. It is compared with the
archived `results/pandora_baseline_mcd_integration/{predictions,metrics_by_fold,diagnostics}.csv` rows for
(np_mcd, agreeableness, 0). Tolerances are declared here, before the run.

- **Exact gates (hard).**
  - Test author list and order, and true labels.
  - n_train, n_val and n_test; total_features.
  - Diagnostics: train_comment_count, target_comment_count, comment_text_features, context_metadata_feature_count,
    mcd_k_actual, mcd_feature_dim, comment_svd_dims and context_metadata_dims.
- **Numeric gates (hard).**
  - mcd_tau: relative difference ≤ 1e-3.
  - comment_svd_explained_variance: absolute difference ≤ 1e-3.
  - Test r: |Δ| ≤ .02.
  - Pearson correlation between replayed and archived test predictions ≥ .98.
- **Reported, not gated.**
  - Equality of best_alpha and best_feat_n; the best_val_pearson difference.
  - k-means min and median cluster sizes; context explained variance (NaN in the original when unreduced).
  - Maximum absolute prediction difference divided by the archived prediction SD.
  - A diagnostic-only correlation of the replayed test predictions at the archived selected (feat_n, alpha) with the
    archived predictions. It is never used for selection.

If any hard gate fails, the run checkpoints and stops for diagnosis. Tolerances are not loosened afterwards, and the
historical reference is not replaced. Stage 1 also measures the wall time of one MCD fit and one np_mcd cell. It
projects the total, and the scientific settings do not change as a result.

## Stage 2: full paired run (only if Stage 1 passes)

- **Order.** For each fold 0..4, run the P-filtered-author MCD fits for arms A/B/C, then np_mcd for 5 traits × 3 arms.
  After all folds, run mcd_only: its unfiltered-author MCD fits, then 5 traits × 3 arms per fold.
- **Persistence.** Each completed cell (condition, trait, fold, arm) is written to disk immediately.
- **Budget.** Four hours of wall time from the start of Stage 1. A cell is not started if the projected cell time
  exceeds the remaining budget. At the cap the run saves partial evidence with a completion mask and stops cleanly:
  no silent dropping and no automatic retry.

## Outcomes and analysis (declared)

- **Per cell.** Out-of-fold predictions with the `mcd_unobserved` flag, fold/trait metrics, the selected
  hyperparameters, MCD diagnostics, deletion counts per author and per venue, manifests, resources and failures.
- **Bootstrap.** Paired author bootstrap, 2,000 draws, seed 20260926. Authors are resampled with replacement within
  each fold's test set, and the same draw is used for all traits, arms and the archived LR-NP.
- **Two separate estimands:**
  - (i) pooled r per trait (all OOF authors), averaged over traits, which is the archived `paired_bootstrap` estimand;
  - (ii) fold-average r (mean of fold r), averaged over traits.
- **np_mcd contrasts.** A−LRNP, B−LRNP, C−LRNP, B−C (targeted content) and A−C (count removal); A−B is also reported.
  A replay line gives the archived NP+MCD−LRNP under this bootstrap.
- **mcd_only.** r for A, B and C, plus B−C and A−C.
- **Reading.**
  - Effect sizes, signs and 95% percentile intervals are reported.
  - An interval that includes zero is inconclusive: it does not show equivalence or that leakage is absent. No
    survival claim rests on an interval merely containing the old point estimate.
  - B−C also contains retrained cluster geometry and the remaining topic mixture, and N/P stay uncleaned. B−C is
    therefore not a uniquely identified leakage effect.
- **Label access.** Logged in the F00 hygiene log before the first label read. The public PANDORA Big5 labels and
  the official test folds were already reused in 2026-06 and 2026-09. This packet is exploratory historical reuse,
  not a confirmatory test. No model choice uses test outcomes.
