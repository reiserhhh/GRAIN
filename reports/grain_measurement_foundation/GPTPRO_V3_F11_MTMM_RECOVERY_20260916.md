# F11: crossed reader/trait/occasion model recovery

## Result

The old `f11.py` covariance proxy is retained but not used as evidence of MTMM
completion. New `f11_mtmm.py` implements the actual crossed random-effects model,
source-only moment fitting, Gaussian REML and application BLUP.

    y_urkt = lambda_r(t) f_uk + m_ur + c_ut + e_urkt
    Cov(y) = L Sigma_f L' + v_m M M' + v_c C C' + v_e I

Three traits have unit variance and pairwise covariance .3; reader/error variance
.25. Worlds are base (no c), shared occasion (v_c=.25), and known conditional
loading lambda=1+.3(r-1)(2t-1) with shared occasion. The two occasions are distinct
events; multiple readers within an occasion share its c. Conditional loadings are
specified by the simulation, not learned or demonstrated for a natural reader.

Three seeds x three worlds x (300 source +100 validation +200 application) =
5,400 simulated authors. Sources alone estimate parameters; truth is separately
saved and only used in evaluation. REML estimates an unrestricted fixed vector
of 18 cell means and covariance using n-1 residual degrees of freedom. Moment
fit preserves raw estimates and explicitly records PSD/nonnegative repair.
No parameter tuning or method selection uses application scores.

## Application results

Three-seed mean MSE, REML; different panels have different measurement budgets.
These are not equal-cost strategy comparisons.

| World | All readers, two occasions | Three readers, one occasion | One reader, two occasions |
|---|---:|---:|---:|
| Base | .106886 | .140281 | .262956 |
| Shared event | .180908 | .265891 | .295624 |
| Conditional loading + shared event | .136111 | .266706 | .296257 |

Plug-in 95% interval coverage with all observations is .9483/.9539/.9289.
The last is below nominal; source estimation uncertainty is not included yet.
Oracle coverage also fluctuates (.9556/.9550/.9356), so no single cause is assigned
without the planned uncertainty analysis. Moment and REML nearly coincide in the
balanced constant-loading worlds, but differ with conditional loading; this is
not a duplicated-metric placeholder.

Mechanism: adding readers of the same event cannot independently replicate away
that event's common deviation. Replication across events adds different information.
The exact allocation must depend on the target and source-estimated covariance,
not an assumption that reader agreement alone establishes a stable trait.

## Exact nonidentifiability certificate

With loading one, adding author-constant common method variance tau to all 18
cells is observationally identical to replacing Sigma_f by Sigma_f+tau*11'.
At tau=.25 the covariance discrepancy is exactly zero. An independent noisy trait
anchor changes the cross-covariance by .25, and hence could distinguish the models
under its independence assumptions. No such real-world anchor is assumed present.
Occasion-specific common variance, unlike author-constant common variance, is
separable with the present two-occasion design (moment design rank 9).

## Evidence and limits of this packet

- Nine actual REML optimizations converged; maximum reported numerical gradient
  2.28e-4. Optimizer success is not yet an independent optimum certificate.
- Alternate cell-by-cell covariance/inverse audit reconstructed 72,900 predicted
  scalars and posterior variances; largest discrepancy 9.11e-15.
- All REML objectives replayed from residual quadratic forms, discrepancy <=3.42e-12.
- Five focused tests cover covariance, constant-method equivalence, anchor contrast,
  conditional Gaussian scoring and moment design rank. This is same-agent alternate
  arithmetic, not independent-agent reproduction.
- Current audit does not yet replay all generated latent records, aggregate metric
  tables, uncertainty intervals or likelihood optima. These remain explicit work.

Artifacts: `results/grain_gptpro_foundation_v3/F11/mtmm_full_r1` and
`mtmm_audit_r1`. Source/validation/application observations, separate truth, fitted
models, all predicted values and uncertainty matrices are preserved.

Remaining F11: real F08 lexical/NLI/GRAIN reader interface (missing readers remain
missing), target-specific signatures, uncertainty/aggregate audit, constant-method
simulation plus independent-anchor estimation, and full card/DAG contracts.
This packet alone does not close F11 or the global plan. No API, private data,
neural training, MEPS, remote machine, lockbox, commit or push; no live background job.
