# F09 composite-null detection calibration

## Executed design

The repaired nine-atom main family crosses M1/M2/M3 with beta=.5/1/1.5. Development uses the four even-parity condition corners, three trials per corner; future uses the disjoint four odd-parity corners, three trials each. Parity refers to coordinates, not alternating record indices.

For each of three seeds and each atom,2,000 independent development/future authors are simulated. Each development sample produces its own exact posterior predictive distribution; the four-cell binomial deviance is then calculated on its future outcomes. Thus the54,000 calibration statistics include both parameter inference uncertainty and future sampling variability, rather than evaluating future residuals against a known generating probability.

For an evaluated statistic T, each null atom supplies p_h=(1+count(T_sim>=T))/2001. The composite p is max_h p_h and rejection is p<.05. This protects against the declared composite null conservatively; it is not a pooled95% quantile or an average of per-author cutoffs.

The900 independent check authors comprise300 validation in-family,450 application in-family and150 application M4. M4 occurs only in application at25%, with the declared sum-of-three-coordinates response law. No threshold is adjusted using validation or application results.

## Result

| Check | n | Rejection rate | Wilson95% interval |
|---|---:|---:|---:|
| Validation, in family |300|.0300|[.0159,.0560]|
| Application, in family |450|.0333|[.0203,.0543]|
| Application, M4 |150|.0267|[.0104,.0666]|

The in-family results are consistent with a conservative nominal5% check. M4 is not effectively distinguished under this short12-development/12-future observation budget. Non-rejection therefore cannot establish that a person's process belongs to this family. A family can be mathematically different yet operationally indistinguishable by the chosen finite test. The M4 rows are retained, not excluded because power is weak.

This finite diagnostic result is distinct from class recovery: the full nine-atom and18-atom/12-equivalence-class posterior classification panels, intermediate-condition process contrast, explicit-belief positive control, and odd-prefix-to-odd-tail prediction target remain to be executed. This subpacket does not substitute for the default300 source cohort of those main experiments; its source resource is the separately required parametric calibration simulation.

## Verification

Seven rival/detection tests pass. All54,000 calibration statistics replay with direct product likelihoods and explicit deviance sums, max difference2.13e-14. All900 evaluated statistics replay, max1.60e-14; all per-atom and composite p-values replay exactly. Recorded statistics are used for exact floating-point equality ties after verifying their alternate-formula values; no threshold is changed. This is same-agent alternate-formula verification.

Artifacts: `results/grain_gptpro_foundation_v3/F09/calibration_r1` and `calibration_audit_r1`. Code: `f09_detection.py`, `run_grain_gptpro_f09_calibration.py`, `audit_grain_gptpro_f09_calibration.py`. Old F09 outputs and incorrect generator remain untouched.

Next: complete the rival/process main panels and observed-equivalence reporting. Do not rescue this detector by tuning thresholds or discarding M4. No API, neural training, private data, MEPS fitting, remote job, lockbox, commit or push. Global goal remains active; no job is running from this packet.
