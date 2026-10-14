# R08 exact-cache raw-vector and VM05 GRAIN history comparators: protocol V1 (packet23 B)

Frozen 2026-09-27 JST before any fit or held score.
- **Authority.** `NEXT_PACKET_23_RAW_GRAIN_HISTORY.md`; `SUPERVISOR_CHECK_20260927_0048.md`.
- **Original clause.** P03 p.23 falsifier comparator: "ordinary PCA / embedding factors / GRAIN static components".
- **Code.**
  - Module `grain_research/r08_raw_grain_history.py` (bound).
  - Runner `scripts/run_grain_r08_raw_grain_history.py`, which records its sha at run time.
- **Output root.** `results/grain_gptpro_foundation_v3/F14/raw_grain_history_r1/`, created in x-mode.
- **Budget.** ≤ 10 min CPU, one process, single-thread BLAS. 0 neural calls. No text, label/fold values, MEPS or
  corpus scan. Private per-event files only.

## Question

Does the keyed projection hide predictive information that raw-vector or GRAIN prefix summaries retain? Does order add
information beyond those static summaries?
- **Common target.** The next event's 30 source-standardized keyed values, so every arm has identical events and units.
  Using raw vectors as inputs does not make the target a raw vector.
- **Not claimed.** No Big5/MBTI accuracy, no do(C), and no PSR minimality.

## A. Bridge gates (run first; a failed gate marks its arms NOT EXECUTED, with the reason)

- **A1, raw E5.** Source: `predecessor_integration_v1/material/private_arrays.npz`; only `comments` (624, 36, 1024)
  and `author_ids` are read for the raw arm, and `grain` for A2.
  - Author IDs must equal `conditional_reference_r2/pandora/private_authors.json` in order.
  - The model revision must be e5-large-v2 f169b11e….
  - Units per author must be exactly 0–35.
  - For ALL 22,464 rows (author_index, unit): (v/‖v‖)_float64 · direction must equal the pinned keyed scores. Tolerance
    1e-6, float32-appropriate. Maximum error and mismatch count are reported. There is no nearest-vector reordering.
- **A2, VM05 GRAIN.** Source: `vm05_preflight_r1/grain_frozen_features.npz` `grain_units_raw` (7343, 36, 24), selected
  by unique author ID. Checks:
  - all-36 named mean/std (`raw_author_features` formula, manifest `feature_names`) against the preflight all36 and
    the material 624 × 48 array (1e-12);
  - view blocks 0–11 / 12–23 / 24–35 against the preflight view features;
  - address view_index = unit // 12;
  - the material manifest's recorded common- and parent-manifest hashes against those files;
  - preflight source identity = parent identity.
  - **Order basis.** Hash-chained manifests plus identical views-flatten construction code. It is NOT a per-comment
    content check: the mask ledger holds only cue comments, and the view files contain text and are not loaded.
- **A3, representation audit.** The e1 TF-IDF + SVD24 runtime population (run manifest), and the cohort overlap with
  `old_e1_member` / `old_reference_member` by role (`budget_by_author.csv`). Any cohort overlap stops the GRAIN arm.
- **Name.** The arm is the "VM05 lexical/SVD mean-std GRAIN comparator", not full GRAIN or new psychological factors.
  Every prefix is recomputed from units before t. all36 and landmark features are never used for prediction.

## B. Four standalone affine ridge readouts (fixed; no grid, no blend, no winner selection)

- **Roles.** 403 / 101 / 120, identical to packets 15/16/22. Fitting rows are source-train positions 25–35 (4,433).
- **KEYED-STATIC.** Prefix mean of the 30 keyed coordinates (30 columns).
- **RAW-STATIC.** L2-normalize raw event vectors. Centered PCA30 on source-train event vectors only (all 36 events of the
  403 authors), deterministic SVD, with each component signed so its largest |loading| is positive. Features: prefix
  mean and population SD of the 30 coordinates (60 columns).
- **GRAIN-STATIC.** The 48 named mean/std columns of the 24 VM05 units over the prefix.
- **RAW-ORDERED.** RAW-STATIC plus (last − prefix mean) and (recent-4 mean − prefix mean) in the same PCA coordinates
  (120 columns). One predeclared history extension.
- **Fit, for all four.**
  - Standardize design columns on the fitting rows. A zero-SD column is set to 0 and counted.
  - Objective: (1/n)‖Y − XB − 1bᵀ‖²_F + 1·‖B‖²_F, with an unpenalized intercept. With centered Xc and Yc:
    (XcᵀXc + nI) B = XcᵀYc and b = Ȳ − X̄B. Units are source-standardized keyed coordinates. The loss sums the 30 output
    dimensions and averages rows.
  - Differing input sizes make this a declared representation/capacity contrast.
- **Scoring.** Source-validation is descriptive only. Held authors are scored once.
- **Frozen references.** The packet-22 direct prefix mean and coupled-history losses are recomputed from the saved
  coefficients and must match packet 22 (1e-12). They use a different parameterization, so they are not a pure feature
  contrast.
- **Reported.**
  - Mean and per-coordinate MSE.
  - Paired whole-author deltas (2,000 draws, `default_rng(20260926)`).
  - Known/novel raw-venue strata: counts and contributions.
- **Order control.**
  - Permute only the observed prefix (seed `[22310, role, author, t]`), recompute RAW-ORDERED with the prefix mean/SD
    held at the original values (their invariance is checked), and apply the SAME frozen scaler and readout.
  - Not a p-value.
- **Checks.** Roles disjoint; PCA, scalers and fits source-train only; future targets do not alter earlier features
  (test); static summaries are permutation-invariant; cold bundle replay without fitting; ridge gradients; bootstrap
  grouped by author.

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/r08_raw_grain_history.py` | `206fca5b6ba52953ab93f7fc82b804acaa5babdf771dcf4b9267dbe0a234b435` |
| `grain_research/r08_history_increment.py` | `bd50efb40a66f8275ca7f0ec6bfcb66b3ff76bddd8fc376b37ba1fdaa4210590` |
| `results/grain_gptpro_foundation_v3/F14/history_increment_r1/coefficients.npz` | `a1fa08d28ba4109139cc1ebde7446bf2ecd0d549925da22eeec7dcdbc06b9db7` |
| `results/grain_gptpro_foundation_v3/F14/history_increment_r1/private_held_keyed_event_losses.csv` | `93f7a6ba6018780b43f0bede3d7eb201e4dd7bdcf969ba74265a5834afac0193` |
| `results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/author_roles.json` | `cfda01dcc0653561d19cd2cd57d98ba5fa0de59ed570c5e6700a0587334acb0b` |
| `results/grain_vector_mechanisms/predecessor_integration_v1/material/private_arrays.npz` | `559deb7135dddbd22ccb9d570b0eb46955eef86e2a280b71158fad475e650ec4` |
| `results/grain_vector_mechanisms/predecessor_integration_v1/material/material_manifest.json` | `73c92b282b3bb10991de329d0400e403333fa06c7f567d58230198e74ab59f80` |
| `results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/private_directions.npz` | `71710bb7380f2a2991357f976531e108251d7f33de0664b7b904480bfdc7e8bd` |
| `results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora/private_authors.json` | `8f15e571c3b8edfc970cf6e5d62e516ad730a8ca686cebd0f5f218a62b5a6ec7` |
| `results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora/private_units.npz` | `184887cb4962c8872b670557a988fbaf85e0bef696a97466cd9836f4a5d30a89` |
| `results/grain_vector_mechanisms/vm05_preflight_r1/grain_frozen_features.npz` | `85cf4bf267b07fd887cc81a3c120a633d3d47fc972de513c059d97ac30d752e6` |
| `results/grain_vector_mechanisms/vm05_preflight_r1/manifest.json` | `721fd18d51ed21ebed6ce17ecd5e0579032330211532f0f14d4a7b581784a582` |
| `results/grain_vector_mechanisms/vm05_preflight_r1/common_material_manifest.json` | `9e0043cb4ca029733d15f4d326cf5dd67fcc4f93412ef659a96e6e616ba3c792` |
| `results/grain_vector_mechanisms/vm05_preflight_r1/budget_by_author.csv` | `7800ac389b750d86975b39dd5d9ccaa428b4d245583c0693050c9746398e80a7` |
| `results/grain_vector_mechanisms/vm05_full_source_cohort_r1/manifest.json` | `8d9fe8e6599ddf4ed92f8d2f3eb03763b42ccf8f200898fccf2c1d5a939599e7` |
| `results/v7_multiview_projection/e1_v72_full_20260715/run_manifest.json` | `a97f253ec1c5276d5fe8575c73cda3a7f8439373b06259b82a158af901802993` |
| `results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv` | `61c98ca01fc7acd067ef924611009b2e9ca2cf5d9888a4514a91f6cfb4a10a7e` |
