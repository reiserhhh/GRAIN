# F13 dynamic target uncertainty: joint temporal covariance

## Question and completed work

Does temporal covariance matter for four-step average predictions, separately
from the still-open uncertainty in fitted dynamics parameters?

Reused F05 source_full_r2 and application_full_r1 (hash linkage verified): three
worlds (core, near_root, shared_shock), three seeds, 200 application authors in
each world/seed, three prefix budgets (16/32/48). There are 1,800 distinct authors,
not 5,400 independent authors: budgets reuse the same people. No parameters were
refit and no application truth was used in the plug-in scorer. The oracle
comparison explicitly uses simulation parameters, never as a feasible estimator.

Targets are separately named: current latent state, the noisy observation four
steps ahead, and the average of four future noisy observations. These are not
interchangeable with an average latent state or a mean without future innovations.

Completed 2,160,000 conditional future trajectory draws (200 per author/budget/
method), 75,600 prediction rows and 378 metric cells. This is the conditional
event layer, not the full source200 x personal200 hierarchy. Earlier REML work
concerned the F04 linear target and does not close dynamics parameter uncertainty.

## Formula

For s_(T+k)=rho^k s_T + sum_(j=1..k) rho^(k-j) w_(T+j), with filtered terminal
variance P and innovation variance Q, the future observation covariance is

    Cov(y_(T+k), y_(T+l) | prefix, parameters)
      = rho^(k+l) P
        + Q sum_(j=1..min(k,l)) rho^(k+l-2j)
        + R 1[k=l].

Current-state/future covariance is rho^k P. The four-step mean has variance

    P (sum_(k=1..4) rho^k / 4)^2
    + Q/16 sum_(j=1..4) (sum_(l=0..4-j) rho^l)^2 + R/4.

Shared future innovations must be summed before squaring. The diagnostic
diagonal-only calculation deliberately omits cross-time terms; it is retained
as an incorrect comparator, not as an eligible production method.

## Observed results

Equal-weight descriptive coverage across the three seeds and three budgets for
the four-step average (dependence from repeated budgets is not ignored in claims):

| World | Parameter arm | Diagonal-only | Correct joint covariance |
|---|---|---:|---:|
| core | oracle | .887778 | .957778 |
| near_root | oracle | .789444 | .953333 |
| core | plug-in | .809444 | .881667 |
| near_root | plug-in | .700556 | .895556 |
| shared_shock | oracle | .891111 | .961111 |
| shared_shock | plug-in | .830556 | .901667 |

Means and MSE are identical between diagonal-only and analytic-joint arms:
this is an uncertainty correction, not a prediction-accuracy improvement.
Core oracle mean width 1.49879 -> 1.91055; near_root 1.87728 -> 2.98613.

Correct time covariance restores the known-parameter benchmark's aggregate
coverage near nominal, but plug-in intervals remain undercovered. This identifies
a separate remaining task: uncertainty in personal and source dynamics parameters.
It does not demonstrate that a bootstrap will necessarily fix misspecification.

Among the 54 independent-world oracle analytic seed/budget/target cells, 52 pass
the frozen Wilson criterion, two do not. These failures remain. The corresponding
plug-in analytic arm has only 14/54 passing cells. No multiple-cell pass rate is
used as a substitute for the frozen per-cell result.

All shared_shock Wilson limits are explicitly descriptive: common episode shocks
induce dependence between authors. No valid independent-author binomial decision
is claimed there. The oracle conditions on the person's own history only, not on
all other authors; it is a marginal oracle, not a full joint-panel oracle.

## Verification and outputs

Runner: `scripts/run_grain_gptpro_f13_dynamic_targets.py`.
Core: `grain_research/gptpro_foundation_v3/f13_dynamic.py`.
Audit: `scripts/audit_grain_gptpro_f13_dynamic_targets.py`.
Results: `results/grain_gptpro_foundation_v3/F13/dynamic_full_r1` and
`dynamic_audit_r1`; smoke in `dynamic_smoke_r1`.

Every native cached state and single-horizon prediction agrees within 2.23e-15.
Every joint covariance agrees with stationary dense Gaussian conditioning within
3.56e-15. Separate scalar recursive generation replays every draw-derived row,
truth, analytic target and aggregate metric, maximum discrepancy 1.68e-14.
Input hashes/source-manifest linkage, author separation and unique output keys
checked. Six new tests include negative/zero/positive/near-unit rho and irregular
observation times; 15 related tests and Ruff pass. Same-agent alternate algebra,
not independent third-party replication.

## Open work and next action

Source-author bootstrap with actual two-round source re-estimation, personal
parameter uncertainty (not just fixed-parameter state draws), F08 soft-history
target uncertainty, specified missing/correlated/history/shift worlds and the
common card contracts remain. F13 and the full research goal stay open.
No API, LLM training, private text, MEPS fitting, remote run, lockbox or publication.
All jobs in this packet terminated. No prior failed result was overwritten.
