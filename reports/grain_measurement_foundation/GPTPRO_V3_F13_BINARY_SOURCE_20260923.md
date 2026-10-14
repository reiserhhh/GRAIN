# F13 / F08 source uncertainty: actual bootstrap refits

## Completed

Both frozen F08 worlds (`c2_xor`, `c2_iid`), three original seeds, 300 source
authors per cell and 48 ordered events per author were regenerated from the
original numeric simulator. Each cell has 200 author-bootstrap samples, giving
1,200 actual joint B1/B2/M source refits. Validation/application records are not
used in fitting. The original reader is unchanged and no text is regenerated.

The canonical output is
`results/grain_gptpro_foundation_v3/F13/binary_source_full_r2`.
Every record preserves source author IDs, sampled indices, RNG recipe and fitted
parameters. Protocol/code/input hashes were checked after completion.

Each author has a 4-by-2 ordered transition count table C_u. For bootstrap
multiplicity m_u, the exact pooled counts are C*=sum_u m_u C_u. B1 uses add-one
state probabilities. B2 solves the original unit-ridge normal equations formed
from C*. M recomputes the mean and rank-two SVD of the *resampled individual*
smoothed tables, including repeated authors; it does not reuse an old loading.
Initial pair counts and source mean are also refit.

Every one of the 1,200 fits was checked against separately constructed row-level
source copies: augmented least-squares B2, explicit transition B1, and raw-table
SVD projection for M. Copies receive distinct author keys so resampled authors
are not collapsed into one long history. This is exact sufficient-statistic
refitting, not a surrogate perturbation or a label-derived posterior.

## Diagnostics

Mean bootstrap standard deviation across B1's four transition probabilities:

| World | Seed 20260915 | Seed 20260916 | Seed 20260917 |
|---|---:|---:|---:|
| XOR | .007037 | .007362 | .007327 |
| IID | .007959 | .008357 | .008422 |

These are source-fit fluctuations, not personal-score confidence intervals or
psychological measurement errors. Filtering can change their effect on a final
target, so no target coverage conclusion is drawn yet.

Full r2 completed in 18.34 seconds because the four-state model has exact small
sufficient statistics. Four-fit smoke and five binary/source tests passed;
Ruff passed. r1 is retained: an input validator was strengthened during that
run to reject fractional outcomes before integer conversion. r2 reran the full
calculation under the final hashed code. Valid binary source data are unchanged.

## Remaining work and mathematical boundary

- Replay application histories separately under each refitted source model;
  decompose within-history and between-source next-probability variation.
- Implement the stipulated inner event/parameter simulation and compare with
  exact finite-state quantities where available.
- B2/M are regularized estimators, not Bayesian likelihood models. Do not turn
  fractional reader counts into an invented conjugate Beta posterior. Specify
  and retain the working simulation law separately from empirical reader error.
- Preserve missingness, prior shift, evidence-history and calibration failures
  under the original F13 requirements.

This closes source fitting only. F13 parent cards and the full research goal
remain open. No application scoring, human validity, new training, API calls,
commit, push or lockbox opening occurred in this packet.
