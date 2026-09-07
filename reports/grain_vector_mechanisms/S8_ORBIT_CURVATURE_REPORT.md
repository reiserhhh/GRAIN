# S8: local curvature explains part of the N response, not every query

2026-09-08. COMPLETE. Fixed protocol:
[S8 plan](../../docs/GRAIN_CURVATURE_TEST_S8_PLAN.md). This is an internal
mechanism experiment, not a new personality-prediction benchmark.

## Finding

The mean-null configuration response has measurable local curvature. For the
neuroticism query, coefficients estimated at radius .125 predict part of the
response at held radii .5 and1 without fitting either endpoint. The same formula
does not work well for T/F. Thus query-specific nonlinear response is useful,
but a universal constant-Hessian shortcut is not the current answer.

Eight fixed authors, three queries, two existing comment sets, one predetermined
permutation per set:48 paths.384 new forwards,0 backwards,544.8seconds. All center
scores matched the earlier mean-collapse endpoints within1e-5. No labels read.
All private scores remain in ignored
`results/grain_vector_mechanisms/orbit_curvature_s8_r1/`.

## Held-radius results

Let F(a)=s(m+a*r), m=Pi H. Estimate L and Q with symmetric .125 differences,
and predict F(a)-F(0) by a*L+a^2*Q. The even response is
[F(a)+F(-a)]/2-F(0); it removes odd powers and is predicted by a^2*Q.
MSE/zero compares squared error with predicting zero intervention effect;
below1 is an improvement. A high correlation alone does not ensure correct size.

| Query | Radius | Full-response r [author-bootstrap95% CI] | Full MSE/zero | Even-response r | Even MSE/zero |
|---|---:|---|---:|---:|---:|
| T/F | .5 | .2833 [-.1927,.6614] | 8.051 | .3246 | 13.234 |
| T/F | 1 | -.2056 [-.6852,.4004] | 6.214 | -.1369 | 10.884 |
| Neuroticism | .5 | .8096 [.1030,.9367] | .571 | .8402 | .571 |
| Neuroticism | 1 | .7537 [.0757,.9189] | .718 | .7676 | .866 |
| Confirmation query | .5 | .7338 [.6045,.9290] | 1.861 | .7551 | 2.308 |
| Confirmation query | 1 | .3389 [-.3581,.6894] | 3.134 | .3601 | 6.035 |

N full-response RMS is .00453/.01987 at radii .5/1; prediction RMS is
.00317/.01181. The correlation is useful, with substantial remaining amplitude
error. Confirmation has correlated half-radius changes but overpredicted size.
T/F is not a failure of its previously successful distillation result: it is a
failure of this local quadratic extrapolation on these paths.

## Numerical and shared-query evidence

| Query | Q correlation: .125 vs .0625 | RMS coefficient difference / primary RMS |
|---|---:|---:|
| T/F | .9915 | .3423 |
| Neuroticism | .9944 | .1304 |
| Confirmation query | .9984 | .0770 |

Coefficient ordering is similar, but T/F coefficient amplitudes change noticeably
when the finite-difference step shrinks. Report all three estimators (primary,
smaller-step, Richardson) rather than selecting the best on held endpoints.
Richardson does not rescue T/F; N full-radius r remains .7123 and MSE/zero .7834.

Matched, edit-centered N/confirmation coefficient correlations are:

- Linear response: r=.9503 [.8526,.9879].
- Quadratic response: r=.7117 [.4245,.8717].

These are shared local **model responses**, not a correlation between two human
scales. Their full-radius behavior differs, so shared local sensitivity does not
imply one globally interchangeable trait function.

## Formula revision

For a twice continuously differentiable suffix and a fixed direction r,

\[
s(m+r)-s(m)=\nabla s(m)^T r+
\int_0^1(1-t)r^T\nabla^2s(m+tr)r\,dt.
\]

For a finite uniform permutation group, E_g[r_g]=0, hence exactly

\[
E_g[s(m+r_g)]-s(m)
=\int_0^1(1-t)E_g[r_g^T\nabla^2s(m+t r_g)r_g]\,dt.
\]

The constant-curvature .5*tr(K(m)*Sigma_G) form follows only by replacing the
path-dependent Hessian by K(m). S8 supports a useful approximation for N in this
panel, not its universality. The integral form retains the very nonlinear effects
that a one-point formula drops. S8 itself sampled one orbit direction per
condition, not the full group expectation or a full Hessian matrix.

This is Taylor's integral remainder applied to a finite-group intervention,
not a newly discovered calculus theorem. The feature-interaction perspective is
informed by [Janizek et al., JMLR2021](https://www.jmlr.org/papers/v22/20-1223.html);
our finite-difference estimator is not their Integrated Hessians implementation.

## Mainline decision

Proceed to [S9](../../docs/GRAIN_SOURCE_REMAINDER_S9_PLAN.md), the actual source
teacher-minus-student connection. Do not spend another grid searching a universal
quadratic fit. Preserve S6's accurate integrated response and S7's shared
nonlinear distribution response as established findings.

The projected endpoint retains the query state and all unselected states. Those
states already contain information from earlier layers. Therefore a projected
endpoint is not an independent mean-only text encoder, and its remaining score
must not be called pure aggregate information. S9 reports an explicit intervened
suffix comparison and separates the CUDA/BF16 versus MPS/FP32 observer difference.

Artifacts: `summary.json`, `metrics.csv`, `coefficient_stability.csv`,
`shared_coefficients.csv`, `private_radial_scores.csv`, `private_coefficients.csv`,
`private_radius_predictions.csv`, `manifest.json`, author packets,
and `run_status.json` in the ignored output directory. The frozen script and four
synthetic tests are versioned. No global goal-completion claim follows from S8.

Validation:200 focused mechanism tests passed; four additional integrated-curvature
tests then passed alongside the four prior permutation-covariance tests. Ruff is
clean for the new/modified numerical code. The integrated-curvature helper has
only synthetic validation so far; S8's neural numbers remain finite differences.
