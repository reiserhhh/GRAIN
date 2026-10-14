# F13 actual missingness, AR and shifted-population reference

## Replacing a placeholder with a real experiment

The legacy F13-04 runner merely generated four biased normal samples named MCAR,
MAR, MNAR and source shift. It did not mask observations or generate correlated
events. Its PASS remains historical invalid evidence, not this experiment's
acceptance. Old outputs are retained.

The new runner `run_grain_gptpro_f13_missing_reference.py` generates author-level
intercept/slope parameters and 32 ordered observations at four conditions.
Three seeds, 200 application authors, three worlds and four masks produce
28,800 target/arm rows and 144 seed-specific coverage cells.

Masks actually remove observations: MCAR .2; condition-dependent MAR .2 for
negative conditions/.6 for positive; MNAR .2+.4*sigmoid(y). The MNAR function is
an explicit operational choice frozen before execution, not a function specified
in the original prose protocol. Full arrays, masks and probabilities are saved.
AR uses stationary covariance .25*.6^|t-s|. The shifted population uses F12's
mean [1,.2]/variance [1.4,.7] versus old [0,.5]/[1,.25]. Latent and noise ranks
are shared across comparison worlds; no application result chooses parameters.

## Main finding: correlation affects different targets differently

Complete observations, average across seeds:

| AR=.6 arm | Target | Coverage | Width | MSE |
|---|---|---:|---:|---:|
| known current law | level | .946667 | .653333 | .028729 |
| incorrectly IID | level | .695000 | .345137 | .030105 |
| known current law | condition contrast | .945000 | .601694 | .026530 |
| incorrectly IID | condition contrast | .986667 | .904615 | .030383 |

Ignoring correlation understates level uncertainty but overstates alternating
condition-contrast uncertainty. A common interval inflation factor is therefore
not a general correction. Target direction, condition schedule and covariance
must be retained jointly. Averaging these target coverages would hide the error.

Across all masks/worlds, known-current-law Gaussian updates pass 68/72 Wilson
checks; the old-IID-law arm passes 47/72. Failures are retained. In MNAR even
known unconditional parameters do not make the selection-ignoring Gaussian update
an exact posterior: its numerical coverage cannot establish valid MNAR inference.
The shift arm isolates using the wrong population prior, not new pilot learning.

## Evidence and remaining scope

Canonical output: `results/grain_gptpro_foundation_v3/F13/missing_reference_full_r1`;
smoke: `missing_reference_smoke_r1`. Eleven related tests and Ruff passed.
All 28,800 mean/variance rows were independently recomputed with precision-form
Gaussian conditioning; maximum discrepancy was 8.91e-14.
The analytic update uses the corrected centered-support conditioning routine.
The simulation does not yet fit source models or bootstrap source authors; it
establishes the known-parameter reference and misspecification comparisons that
the old runner lacked. Do not call this full F13-04 completion.

Next integrate the existing learned-source/bootstrap Delta layer with these
actual masks and covariance structures. Preserve separate targets, covariance
components, missing-support status and seed-level calibration results. Dynamic
and binary missingness integrations, common record adapter and canonical card
manifests remain open. No human data/API/training/commit/push/lockbox action.
