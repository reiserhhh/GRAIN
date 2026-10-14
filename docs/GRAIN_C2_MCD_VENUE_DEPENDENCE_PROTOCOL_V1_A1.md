# C2 protocol V1, amendment A1: expansion after a diagnosed replay-gate failure

Date 2026-09-26 JST. Written after Stage 1 and its label-free diagnosis. Written BEFORE any B/C arm fit and
before any other cell. V1 (sha 9a4a4e3a…) is unchanged.

## What happened (not changed or re-labelled)

Stage 1 (A / np_mcd / agreeableness / fold 0) returned **FAIL_STOP_DIAGNOSE**
(`results/.../c2_venue_dependence_r1/replay_check.json`).

- **Exact gates all passed.** Author IDs and order, labels, n_train/val/test, total features (52,680), comment counts
  (49,733/32,598), text features (16,000), context features (4,181), k 48, MCD dimension 51, SVD dimension 64,
  context dimension 32.
- **Passing numeric gates.**
  - Test r .2737 vs archived .2599 (Δ .0138 ≤ .02).
  - Prediction correlation .987 ≥ .98.
  - SVD EV Δ 2.6e-5.
- **Selection.** The selected hyperparameters (alpha .5, k 20,000) are identical.
- **Failed gate.** `mcd_tau` rel diff .0093 > the declared 1e-3. Median cluster size is 569 vs archived 654.5.

The tolerance is not loosened, and the failure is reported as such.

## Diagnosis (`replay_diagnosis.json`; label-free, 6 s, labels_used false)

- **D1.** The local fit is bit-deterministic when rerun.
- **D2.** Text-SVD EV and context EV match the archive, and the context feature counts are equal. The drift
  therefore arises at MiniBatchKMeans.
- **D3.** Twenty alternative k-means seeds on identical inputs give tau 47.72–49.56 (SD .50, relative range .038).
  In 95% of seeds tau differs from the declared-seed tau by more than 1e-3. The archived tau 48.58 lies inside
  this range.

  The same seed gives a different k-means optimum under a different platform or library path (the original ran on
  Windows with an unknown sklearn version; this run uses sklearn 1.8.0 on macOS). The declared 1e-3 tau tolerance
  was stricter than the estimator's own k-means variability. The historical MCD numbers are themselves conditional
  on one k-means realization.

## Amended execution (decision item 2: "reported as a new estimator variant only with the discrepancy explicit")

1. The full stage runs as the labelled variant **`c2_v1-local`**. Arm A is NOT a historical replay. Every A/B/C
   result is reported as this variant, with the stage-1 failure and diagnosis cited.
2. The historical reference is not replaced. For every cell and every estimand, `A − ArchA` (the variant drift) is
   reported next to the contrasts.
3. **Interpretation rule, declared now.**
   - B−C and A−C are within-variant paired contrasts. All arms share code, platform, seeds and the paired SAG seeds.
   - Arms differ in their inputs, so each arm also gets its own k-means solution. The author bootstrap does not
     include k-means-solution variability.
   - A contrast counts as distinguishable from implementation/clustering drift only if both conditions hold:
     - its 95% interval excludes 0;
     - its |point| exceeds |A − ArchA| on the same estimand (pooled and fold-average separately).
   - Otherwise it is reported as not separable from drift, whatever its interval.
4. Everything else in V1 is unchanged: arms, settings, order, budget, bootstrap and estimands. No k-means seed search
   or multi-seed averaging is added; that would be a separate packet.
