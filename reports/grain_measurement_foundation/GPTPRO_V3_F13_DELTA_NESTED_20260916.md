# F13 actual nested computation: linear Delta, source moment arm

## Executed

Reuses F04 L source300/application200 authors for each of three seeds. The first24
application observations update a two-dimensional Gaussian parameter distribution.
The source estimator is the existing noise-corrected moment model, explicitly not
a replacement for the pending REML bootstrap arm. No application truth enters fitting.

For each seed:200 source-author bootstrap resamples, each refitting mean,
coefficient covariance and observation variance;200 joint conditional parameter
draws for each application author per source fit. Total600 source refits and
24,000,000 conditional person draws. Checkpoints saved every10 source resamples;
bootstrap author indices, eigenvalues, floor flags and full fitted parameters saved.
No fabricated RNG-only metrics: each sample updates from the actual saved prefix.

Delta=2b. Exact finite-mixture variance is mean(V_Delta|fit)+Var(mean_Delta|fit),
with ddof0. Monte Carlo first/second moments are stored separately and compared
with that analytic mixture; normal intervals are moment-based, not exact mixture
quantiles or an exact hierarchical Bayesian posterior.

## Results by seed

| Seed suffix | Oracle coverage | Plug-in coverage | Source mixture coverage | Nested MC coverage |
|---|---:|---:|---:|---:|
| 915 | .930 | .925 | .925 | .915 |
| 916 | .950 | .950 | .950 | .950 |
| 917 | .955 | .955 | .955 | .955 |

Source mixture mean variance .070254/.068735/.070923; plug-in
.070170/.068853/.070831. Mixture uncertainty need not exceed the one original
plug-in variance: the within-fit average can change as well as adding a nonnegative
between-fit term. Monte Carlo differences remain reported; they are not hidden.
Coverage calibration with Wilson intervals and full alternate replay are pending.

This complete balanced linear world shows small source uncertainty at n300, not
a claim that source uncertainty is always negligible. F05 state/forecast,F08 next-p,
missingness,correlated events,history updates,shifted priors and source REML are
still executable open tasks. Full F13 is not complete.

## Evidence

Artifacts: `results/grain_gptpro_foundation_v3/F13/delta_full_r1`,
`delta_smoke_r1`. Source identities/order are deterministic sorted author IDs in
existing F04 inputs; new loader rejects duplicate events/role mixing. Joint
posterior and balanced-source moments pass alternate algebra unit tests;7 related
tests pass. Full source-refit/sample replay has not yet run. Checkpoints hold draw
moments and deterministic RNG recipes rather than all24m samples.

## Completed follow-up: nested replay and coverage check

Alternate scalar balanced-OLS source fitting, observation-space Gaussian
conditioning and explicit two-dimensional triangular sampling replay all600
source fits and24million conditional draws. Max source parameter discrepancy
1.78e-15; Gaussian moments1.71e-14; sampled moments9.60e-14. No PSD flooring
occurred in these600 resamples. Aggregate metric recomputation is exact.
Audit: `delta_audit_r1`, script `audit_grain_gptpro_f13_nested.py`.

Per-seed Wilson95% intervals include nominal .95 in11/12 method cells. The
20260915 nested-MC cell has coverage183/200=.915, Wilson[.868104,.946254],
so it does NOT pass the frozen coverage criterion. The corresponding analytic
mixture is185/200=.925, Wilson[.879956,.954025], which contains .95. No samples
were rerun or removed to improve this outcome. Binomial intervals are conditional
on shared source fits; they do not quantify source-population uncertainty.

This completes full source/draw replay, not all F13 scenarios or inference claims.
The empirical bootstrap mixture is a declared approximation, not an exact
posterior. Remaining REML, F05/F08, missingness/correlation/history/shift and
production-contract tasks remain explicitly open.

No API/neural training,private text,MEPS fitting,remote execution,lockbox,commit or
push. Full process ended; no live background job. This is actual200x200 execution
of one named arm, not total-plan or total-F13 acceptance.
