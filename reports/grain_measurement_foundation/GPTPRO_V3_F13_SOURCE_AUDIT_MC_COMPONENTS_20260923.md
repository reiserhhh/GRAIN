# F13 source audit and inner Monte Carlo component contamination

## Source-fit evidence

`dynamic_source_audit_r1` checks 1,589 durable source fits (211 not yet available
at audit time). All recorded bootstrap identities and RNG draws agree. The
two-round prior mean/covariance updates recompute to4.45e-16; vectorized author
likelihoods recompute stored noise objectives to3.64e-12. No source task,
personal optimizer or noise optimizer failures in that snapshot.

This is an alternate arithmetic audit of recorded estimators, not 1,589 fresh
optimizer refits or proof of global optimality. Every source/code hash checked.
Balanced contiguous event times validated, so the vectorized recurrence does not
silently replace an irregular-time process. A dense Gaussian likelihood test
checks the independent vectorized implementation.

## Component correction derivation

The nested estimator uses B source fits and J independent personal parameter
draws per source fit. Let P be the average ddof0 within-source covariance of
conditional target means, and S the ddof0 covariance of their source means.
Conditional on fixed source fits, unbiased Monte Carlo component estimates are

    P_corrected = J/(J-1) P
    S_corrected = S - (1-1/B) P/(J-1).

Their sum increases by only P/[B(J-1)]. Thus finite inner draws inflate the
apparent source component while deflating the personal component; most of these
errors cancel in the total. This is not correction for finite source-population
sampling, model misspecification, or the Laplace approximation itself.

Exact enumeration of all sixteen outcomes in a two-source/two-draw toy verifies
expectations: raw personal/source1.25/1.625 become2.5/1.0, the known targets.
No fitted simulation or approximate significance gate is used for this identity.

## Applied diagnostic, primary results unchanged

Using B=J=200 on the seven previously complete cells (4,200 author-prefixes):

| World/target | Raw source variance | Corrected component | Total variance correction |
|---|---:|---:|---:|
| core/current state | .000375 | .000213 | .000000813 |
| core/future average4 | .000445 | .000154 | .000001462 |
| core/future step4 | .000405 | .000117 | .000001447 |
| near_root/current state | .006036 | .004600 | .000007215 |
| near_root/future average4 | .001442 | .000816 | .000003143 |
| near_root/future step4 | .001969 | .001180 | .000003967 |

About65% of the raw core/average4 source component is inner Monte Carlo
contamination. Its subtraction should not be confused with removing a
psychological noise mechanism. The tiny total correction provides no plausible
explanation for the large near_root current-state undercoverage already found.
Primary frozen interval tables were not changed or tuned.

718 corrected source covariance matrices have a negative eigenvalue. An unbiased
finite-sample component estimate need not be PSD; these values are retained,
not clipped and not used as standalone posterior covariances. They indicate
component estimation noise, not negative physical variance. No claim of an
identifiable positive source component is made for each individual prefix.

## Artifacts and continuation

Scripts: `audit_grain_gptpro_f13_dynamic_sources.py`,
`diagnose_grain_gptpro_f13_mc_components.py`.
Core: `grain_research/gptpro_foundation_v3/f13_monte_carlo.py`.
Results: `F13/dynamic_source_audit_r1`, `F13/dynamic_mc_components_r1` under
`results/grain_gptpro_foundation_v3/`.

Two new tests pass; existing primary results preserved. Producer/consumer were
observed advancing to1,620/1,800, now on the final world/seed. Continue existing
sessions94813/80490; do not restart. Remaining final aggregation, all-personal
replay, F08 uncertainty and specified F13 scenarios remain open. No API, private
text, new human data, neural training, lockbox, commit or push.
