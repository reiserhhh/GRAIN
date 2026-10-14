# F16-R03 T1 target-matched transport and pilot uncertainty

Frozen before this packet's application evaluation, 2026-09-24. Reuse the actual
F12 four transforms and shifted-population data, not its old OLS scores. The
estimator is the frozen F16 T1 four-condition posterior, all200 source fits.
No model selection or parameter sweep. This is the numeric T1 part of R03;
text and dynamic-target transport remain distinct requirements.

## Matched target and data

T1 is m(+1)-m(-1). In F12's linear-response application world its evaluator is
2*b. The four-cell source model can score this response law, but source prior
is deliberately kept old: transport does not silently normalize away the new
population or pretend that an old prior is the new population truth.

Use all three original seeds, source-fitted F16 bundles, F12 pilot100 paired
authors with first60 fitting and last40 validation, and F12 application200.
F12new population mean(1,.2), covariance diag(1.4,.7) and four original channels
stay unchanged. Use32events to match the F16 T1 source/reference design; the old
F12 OLS benchmark used24 and is not a directly comparable estimator score.
Fit/validate data are only paired observations; hidden b is read only after all
calibrators are frozen. Source, pilot and application remain distinct roles.

Arms: uncalibrated, global affine inverse, conditional affine inverse, known
inverse observation mapping. Nonlinear channel adds the prescribed cubic
inverse [1,r',r'^3]; retain derivative<=0 or out-of-range rejection, no rescue
using a different nonlinear fit. Known inverse is an oracle for the reader
mapping, not for the author or noiseless responses. Missing-condition pilot
must report rank deficiency, never silently infer an anchor.

## Uncertainty, not just a refitted point score

Resample the60fit authors with replacement200times, shared bootstrap indices
across arms within seed. Refit calibrators on whole-author copies. No pilot
validation/application rows enter those fits. Keep all rejected draws and their
reasons; do not silently condition uncertainty on surviving calibrators.

For complete balanced condition support, the source model has an exact kernel
`m_s(v)=a_s+w_s'v` on four corrected condition means v and conditional variance
V_s. For independent source bootstrap s and pilot bootstrap j,

`mean = mean_j mean_s m_s(v_j)`;
`variance = mean_s V_s + mean_j Var_s(m_s(v_j)) + Var_j(mean_s m_s(v_j))`.

This is a working cut/bootstrap product, not a universal psychological posterior.
Use every source/pilot combination, not the diagonal pairing s=j. Report plugin
score/variance, mixed score, conditional/source/pilot components, coverage and
width. Exact affine paired observations can give almost zero calibration
variance; do not fabricate independent reader noise to make intervals move.
For a rejected fit or any unsupported bootstrap draw, keep the available plugin
result but mark full-mixture result unsupported with the explicit reason.

## Outputs and checks

Save calibrators, bootstrap indices/statuses, per-author plugin/mixture scores,
three variance components, pilot validation errors and condition residuals,
paired application MSE contrasts, missing-anchor certificates and a provenance
manifest. Reconstruct kernels by direct observation-space Gaussian conditioning;
verify variance components with all-component raw second moments and compare
selected actual public cold scores with corrected observations. No claim of
transport closure from metadata or only a few matching field names.

Matched affine inverses should reproduce original-unit reference scores at1e-10
arithmetic tolerance; scientific coverage keeps the original Wilson95 rule,
not every arm must pass. The prespecified nonlinear inverse is allowed to fail.
No new neural calls, source refitting, private/MEPS text, remote training,
lockbox, commit/push. Original F16 public integration and44+6 remain open until
their own requirements have actual evidence.
