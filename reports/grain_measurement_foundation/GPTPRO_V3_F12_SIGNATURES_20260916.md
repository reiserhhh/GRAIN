# F12 finite functional signatures and coordinate invariance

## Operational definition

The original F12 protocol requests four fixed interventions without specifying
their values. This execution elaboration was written before the new run in
`configs/grain_f12_signature_recovery.json`; it is not retroactive preregistration.
For m(z)=a+bz, interventions are (+.1,0),(-.1,0),(0,+.1),(0,-.1) in (a,b).
Evaluate at the original four conditions. These are synthetic parameter changes,
not natural-language edits, personality manipulation or native attention paths.
No amplitudes or ranks were selected using application outcomes.

## Results

600 existing application authors, four interventions, four conditions, four channel
worlds and their existing calibration methods produced 124,800 signature rows.
The rejected cubic inverse remains rejected (9,600 missing signature rows).
Rotation by pi/5 and scales(2,.5) preserve means and finite actions when both
parameters and readout are transformed: maximum discrepancy2.67e-15.

    theta'=S theta; x'=S^(-T) x; delta'=S delta
    x'^T(theta'+delta')-x'^T theta'=x^T delta

Global-channel signature MSE uncalibrated=.00194444; calibrated approximately0.
Conditional-channel MSE uncalibrated=.00033179; global calibrated=.00036025;
conditional calibrated approximately0. Nonlinear MSE uncalibrated=.01726159;
global=.00263742; conditional=.00306921. Calibration can recover a known affine
response law, but a global correction need not improve conditional signatures.
These errors concern finite response differences, not human trait accuracy.

## Exact counterexample

Define h(z)=product_j(z-z_j), over the four observed conditions, and
m*(z)=a+bz+.2*a*h(z). At every observed condition both baseline functions and all
four intervention signatures coincide exactly. At unobserved z=0, an intercept
increase .1 yields a signature difference .00222222. Thus finite functional
agreement does not establish global function identity. Four signed actions span
rank2, not four independent mechanisms. The numerical size is a chosen witness,
not a psychological threshold; the exact nonidentity is the result.

## Evidence and continuation

Artifacts: `results/grain_gptpro_foundation_v3/F12/signatures_r1`, including the
copied pre-run protocol, all signature rows, metrics, configuration hash and exact
certificate errors. Three new tests plus five transport tests pass; Ruff passes.
The exact algebra/coordinate checks ran, but all124800 signature rows have not yet
received a separate alternate-formula replay. That distinction remains explicit.

### Completed follow-up: full signature replay

`audit_grain_gptpro_f12_signatures.py` uses polynomial expansion of finite
differences rather than subtracting two production function evaluations.
All124800 rows,9600 missing cubic entries and aggregate signature MSE rows replay,
maximum difference8.66e-15. Artifact: `F12/signature_audit_r1`. This closes the
previous numeric signature-audit gap, not semantic or human functional identity.

Next: transport/source uncertainty through F13,
and standard task-card artifacts. Existing multilingual metadata remains a
descriptive interface, not same-human invariance evidence. No human participation,
private text, API, neural fitting, remote execution, commit/push or lockbox changes.
