# F09 rival mechanisms and observable equivalence

## Execution

The full repaired run has3,600 simulated authors: two panels x three seeds x300 source/100 validation/200 application. Each author receives12 even-corner development observations and12 odd-corner future observations. Two distinct targets are scored: even12->odd12, and odd6->odd6 using only the first six odd observations for that second posterior. No future outcome enters the corresponding posterior.

The main panel has nine family-by-slope atoms. The process panel has18 descriptions but only12 distinct observable laws at binary corners. Priors are uniform over classes and split equally within duplicate descriptions, so adding a synonymous mechanism does not double its prior weight. All roles use that declared finite prior; no empirical source prior is learned. M4 replaces25% of application authors only in the main panel and is explicitly excluded from in-family classification accuracy.

## Application results

| Panel/target | Eligible n | Class accuracy | Brier | Probability MSE |
|---|---:|---:|---:|---:|
| Main, even12->odd12 |450|.4022|.20642|.01453|
| Main, odd6->odd6 |450|.2467|.23624|.04369|
| Process, even12->odd12 |600|.3250|.21991|.01800|
| Process, odd6->odd6 |600|.2067|.24321|.04006|

These accuracies score full slope-by-family observable classes, not merely three broad families. The process result does not identify which synonymous latent description generated a response. The two targets differ in observations, conditions and prediction horizons; their difference is not a controlled estimate of the effect of six additional observations.

M4 future Brier is .23236 and probability MSE .02551 for the even->odd target. Reasonable prediction does not prove that the fitted family contains the generating mechanism, especially given the separately demonstrated weak out-of-family test.

## Exact process certificates

At z1=+/-1, the two-stage belief/report marginal equals a negative-slope logistic law exactly, while the audience law equals M2. More samples at only those same corners cannot distinguish the synonymous descriptions.

At z1=+/-1/3 the belief and matched negative-logistic marginals separate, but sometimes very slightly: at beta=.5 the probability difference is only .000290; at beta=1 it is .002199. Mathematical distinguishability alone is not a claim of usable finite-sample classification.

Observing the simulated intermediate B also distinguishes the specified belief/report joint from a comparator with the same B and Y marginals but independent B and Y. Twelve exact joint/marginal certificates are saved. This positive control introduces an explicitly defined extra observable and comparator; it does not imply that an unspecified real-world competitor or unmeasured human belief has become identified. The independent comparator does not match all possible psychological alternatives.

## Verification and remaining closure

All7,200 posterior/prediction targets replay through direct product likelihoods, maximum difference8.88e-16. Observable classes are independently recovered from probability signatures rather than copied from saved class labels. Brier/log-loss values replay as well. This is same-agent alternate-formula verification.

Artifacts: `F09/panels_full_r1` and `F09/panels_audit_r1` under `results/grain_gptpro_foundation_v3`. Old incorrect F09 generator/results remain unchanged. Sampling uncertainty tables, standard common records/card/DAG packaging and source-content evidence crosswalk remain agent work; no parent card is automatically promoted. The extra-B positive control here is an exact joint-law certificate, not a new finite-sample classification panel.

The mechanism lesson is that a named process must be linked to distinguishable observations. Predictive success or class names alone cannot establish internal mechanism identity. The next substantive family is F10's action-weighted vector geometry; retain the F09 nonidentifiability and weak detection as production limits, not problems to hide by retuning.

No API/neural training/private raw data/MEPS/remote/lockbox/commit/push. No background process remains. Global goal is active.
