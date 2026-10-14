# R34 source-half structural reconstruction of T1 / T2 / T3: protocol V1

Frozen 2026-09-26 JST before any computation.
- **Authority.** `NEXT_PACKET_19_RECONSTRUCTION_AND_REACTIVITY.md` Part A; `SUPERVISOR_CHECK_20260926_2042.md`.
- **Inventory.** `F16/target_structure_inventory_r1`.
- **Runner and output.** `scripts/run_grain_r34_source_halves.py`, writing to
  `results/grain_gptpro_foundation_v3/F16/source_half_reconstruction_r1/`.
- **Execution.** CPU ≤ 15 min, one process, single-thread BLAS, 0 neural calls.
- **No new source generation.** Existing source rows only.

## Splits (fixed, stored)

- **Per seed.** For each existing seed (20260915, 20260916, 20260917), 20 author-level 150/150 splits of the sorted
  source author IDs, using `default_rng([19001, seed_index, split])`.
- **Structure.** Halves within a split are disjoint. The 20 splits overlap, so they are **not** 20 independent
  populations.
- **Role check.** Every half ID must be a source author. Validation and application IDs are rejected. All split IDs
  and RNG keys are saved.

## Refits (true disjoint-source point fits; POINT_FIT_PLUGIN)

- **T1 (F04 L source rows).** `f16_t1_source.fit_source(statistics, half)`: μ (4 cells), noise-corrected Σ (4×4) with
  spectral floor 1e-6, and σ².
- **T2 and T3 (F05 core source rows).** `f05_recovery.fit_source_dynamics(half_rows)`: prior μ and Σ over (a, b, η),
  plus log σ = (state innovation, observation). One fit per half is shared by T2 and T3, giving 120 dynamics fits.
- **Full-source reference.** The same builders applied to all 300 source authors, one per seed and target family.
- **Half models are single point fits.** They are labelled POINT_FIT_PLUGIN with ensemble count 1. No 200-component
  bootstrap is fabricated, and source-fit uncertainty is kept separate. The existing 200-component full-source bundles
  (`F16/t1_source_full_r1`, `F16/t2_cold_full_r1`) are used **only** as a descriptive spread reference.

## Scoring panel (fixed; original shared units; no per-half standardization)

- **T1.** The 100 F04 L **validation** authors per seed, all 32 events (t ≤ 31).
  - Cell statistics use the same arithmetic as `sufficient`. A local copy differs only in accepting role =
    validation.
  - The score is `posterior(counts, sums, model)` with the contrast c = (−1, 0, 0, 1) over cells (−1, −1/3, 1/3, 1).
  - Reported: contrast mean and conditional variance.
- **T2 and T3.** The 100 F05 core **validation** authors per seed, prefix t = 0…47.
  - Scored with `score_prefix(prefix, model, future_conditions=(1, −1, 1, −1))`.
  - **T2:** the state at t = 47 (mean, variance). The personal intercept a is reported **separately**, and so is the
    sum a + state.
  - **T3:** the mean of y_48…y_51 under the plan. Its mean is the average of the 4 forecast means. Its predictive
    variance uses the full covariance of the 4 future observations:
    Cov(y_k, y_l) = ρ^{k+l} V_s + q Σ_{j=0}^{k−1} ρ^{l−k+2j} + r·1[k = l] for k ≤ l, and Var(mean) = Σ_{k,l} Cov / 16.
    It is plug-in (personal and source parameter uncertainty are excluded, as in `score_prefix`) and includes future
    noise.

## Reported comparisons

- **Structural distances between halves.**
  - T1: per-cell Δμ; contrast Δ(μ·c); relative Frobenius distance of Σ; σ² ratio; ratio of contrast variances
    cᵀΣc.
  - T2/T3: Δμ over (a, b, η); Δρ at the prior mean, with ρ = .99 tanh(η); relative Frobenius distance of Σ; Δ log σ.
  - Each is also given as a ratio to the 200-component full-source bootstrap SD. This is **descriptive only**: for
    disjoint 150-halves, a difference of about 2× the 300-author SD is the heuristic scale under exchangeability. It is
    not a null test.
- **Target differences (per author, per split).** d = pred_A − pred_B in native units, and z = d /
  √((var_A + var_B)/2). Per split: mean |d|, RMS d, mean |z|, 95th-percentile |z|. Also the half vs full-source
  mean |pred_half − pred_full|.
- **Summaries.** The distribution over the 20 splits within each seed (median, 10th/90th percentiles, min, max), and
  seed-level summaries. No confidence interval treats overlapping splits as independent. Seed-to-seed range is
  reported as between-population variation.
- **Limits.** No universal .80 threshold, and no score correlation alone.

## Text-channel extension (cached reader rows; no model call)

- **Sources.** F08 NLI and F11 GRAIN `reader_scores.csv`: 96 materials, with roles source, validation and application
  (32 each).
- **Calibration role = validation** (32 materials): 8 semantic cores (pole × subject_status), each with 4 context
  variants.
  - **Unit of splitting:** a whole core, all 4 variants kept together.
  - Each (pole, qualification) stratum holds exactly 2 cores, so every admissible half takes one core per stratum.
    The **8 distinct core partitions are enumerated exhaustively**, not sampled.
- **Channel per half.** `f16_t1_text.channel_counts(half_validation_rows, arm)` for each arm (nli_joint,
  grain_joint).
- **Reported.**
  - Channel total variation between the halves, per (pole, qualification) stratum and overall.
  - A held-out check on the 32 fixed **application** materials: the log-probability of each material's observed reader
    category under each half's channel (from normalized counts, with ½ add-one smoothing declared), and the
    difference between halves. Application materials are never calibration items.
- **Author-level text T1/T2/T3 scores.** Computed only if the existing text builders accept a substituted channel
  without refitting. Otherwise that combination is labelled NOT_SUPPORTED_BY_EXISTING_BUILDER, and no numeric score is
  substituted.

## Checksum bindings (verified at run start)

| file | sha256 |
|---|---|
| `results/grain_gptpro_foundation_v3/F04/source_full_r1/L_20260915/source.jsonl` | `bb235d54d43496b0483615191c5c8ece516ff90cb6790abd59cdcae52c7b4896` |
| `results/grain_gptpro_foundation_v3/F04/source_full_r1/L_20260916/source.jsonl` | `102f115751553ab9d8e5420fadefce7cdeafdb23d6c3b300ac65a96c4d891c61` |
| `results/grain_gptpro_foundation_v3/F04/source_full_r1/L_20260917/source.jsonl` | `c58eb17ed3adb91410055557a3b3e95c3a86dd727daf943b1e524f4308ce1c88` |
| `results/grain_gptpro_foundation_v3/F04/source_full_r1/L_20260915/validation.jsonl` | `fe220b83e71d9b4e1568e203f3b24f8504cc19da084b089b51ea09792a777aaa` |
| `results/grain_gptpro_foundation_v3/F04/source_full_r1/L_20260916/validation.jsonl` | `ab132da3c90f5ee16e491e5895c5d01193674fb40aaf776998101129531e56ca` |
| `results/grain_gptpro_foundation_v3/F04/source_full_r1/L_20260917/validation.jsonl` | `3d27d606c7b6f372b90deacda73f71e2e749c5b9e55ebe278d21bf3c4d036c02` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260915/source.jsonl` | `0f5e6cefe4a90bba4268ae0fa5e18b2cd4067c31858aef71a0b1f30362343457` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260916/source.jsonl` | `aa5165f8b6e0ce14ee2adcd334f5fbc7eabf5727f8d59bd826209e6e5a094ca6` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260917/source.jsonl` | `a0c6485c82a4e4b415fec6e87ca69025bdc465ba573a9a1f786ac667a29e7c89` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260915/validation.jsonl` | `13951e8e93d941ba8e47eee6c3da9661173129280a868c8c9bf25c7e0949babe` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260916/validation.jsonl` | `e93fae42e557b80bb406ce0e17143d796fbd18668a7a86949f928659278baefc` |
| `results/grain_gptpro_foundation_v3/F05/source_full_r2/core_20260917/validation.jsonl` | `d7fd532e0133a22ae43497d1dac9c79dbe800ba31ba8c389e34ffd33b2c96c08` |
| `results/grain_gptpro_foundation_v3/F16/t1_source_full_r1/20260915/bundle.json` | `1c403036bc32e654e943fb3e2e6f56eaf8ae1c3e1746d730f1ba3a8951302a58` |
| `results/grain_gptpro_foundation_v3/F16/t1_source_full_r1/20260916/bundle.json` | `d04831ad2f45ea590268f1ac44f57d075525d9f41a0c23e71c7ae090895d69d7` |
| `results/grain_gptpro_foundation_v3/F16/t1_source_full_r1/20260917/bundle.json` | `0b5cc56e183730637955d78ee66c45ce80ed8f5146ad37f952febf83ff1bddac` |
| `results/grain_gptpro_foundation_v3/F16/t2_cold_full_r1/20260915/bundle.json` | `a07066fb1ad42b2084e2cad4517c2d65afc3e0f51726f4001b7c3b90d6f3f2b8` |
| `results/grain_gptpro_foundation_v3/F16/t2_cold_full_r1/20260916/bundle.json` | `606fcb28ea2776ebb164a35e618ba221cedca8eff13a401b0cf8f83324f84a0a` |
| `results/grain_gptpro_foundation_v3/F16/t2_cold_full_r1/20260917/bundle.json` | `ffa4af3a943a14329c16ecff5c5a15be383504c19b71bf3259fe918d6db286c6` |
| `results/grain_gptpro_foundation_v3/F08/nli_reader_r1/reader_scores.csv` | `ec280aa5fcbf384563599e326eabdfa5d514ba6728b789b3a0c00304f7a967bf` |
| `results/grain_gptpro_foundation_v3/F11/grain_comment_reader_r1/reader_scores.csv` | `28280d77653a31c19dd3f823f48b2794d9a94234b3d266f3c125cefdfe4be4ad` |
| `grain_research/gptpro_foundation_v3/f16_t1_source.py` | `d2c5e0da84744f2baf458745810a1d5bf2e60fce34ee7397626e9026434f2cd9` |
| `grain_research/gptpro_foundation_v3/f05_recovery.py` | `0f90ef1e3bc44d7228cdb7ed8c408ce115a4ca556610db3aa594728c530332e4` |
| `grain_research/gptpro_foundation_v3/f16_t1_text.py` | `32cfddf7233b24cc690fe0c5d096e29dcc5d0f9e49f4cf06304ad6ad7b69bd83` |
