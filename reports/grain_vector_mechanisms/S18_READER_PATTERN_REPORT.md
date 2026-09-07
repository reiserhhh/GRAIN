# S18: a useful decoder weight is not the whole operative pathway

2026-09-08. COMPLETE:24 authorized source N authors,view0,168full forwards,
0backwards,242.76seconds. No questionnaire access, new reader fit or calibration.

## Main finding

Most of the measured layer17 pairing intervention effect survives inside a
component that the frozen scalar reader annihilates exactly. The covariance
pattern is more responsive than the raw Euclidean decoding direction, but
neither selected component accounts for most of the final decoder response.
This gives a concrete next mechanism to recover, not a reason to discard the
successful source contrast or the conditional-moment results.

| Quantity | Euclidean pattern | Training-covariance pattern |
|---|---:|---:|
| Full pairing-removal effect RMS |.001319375|.001319375|
| Selected-component effect RMS |.000016812|.000268733|
| Reader-null component effect RMS |.001314414|.001244920|
| Selected effect vs full effect r |.387806|.486020|
| Null effect vs full effect r |.999935|.983771|
| Selected MSE/full effect energy |.992910|.887806|
| Null MSE/full effect energy |.000179|.041078|
| Interaction RMS/full effect RMS |.001241|.002893|
| Median selected-vector norm/pairing norm |.006151|.031122|

The covariance selected component has larger effects, but also a larger vector
norm; this comparison does not isolate a norm-matched directional advantage.
Both geometries reproduce the same beta*d scalar by construction. Maximum
absolute reader score in their null components is below9e-18.

Selected/full correlation95%bootstrap intervals are[-.2372,.7248] and
[-.0505,.7662]. Null versus full remains much closer in absolute error, not just
rank. The source reader component beta*d itself correlates.4357 with the full
pairing-removal effect. These are24-author internal intervention quantities,
not source-questionnaire correlations or new human prediction scores.

## Why this is a useful theoretical result

The learned reader is a covector beta. A forward pattern a satisfies beta^T a=1;
d_s=a beta^T d retains its score and d_n=d-d_s is invisible to that one readout.
The actual frozen suffix F has another local covector, grad_h F. Its action on
d_n need not vanish. S18 measures this difference rather than inferring it from
a plotted weight vector. Reader-null does not mean meaningless, nuisance or
personality-free.

The native/self scores and source pairing vectors replay exactly; captured head
reconstruction error is2.41e-6. The finite nonlinear interaction is small here
relative to full effect. This makes direct suffix-gradient recovery a useful
next step: the missing action may be accessible through functional directions
that the single teacher-regression weight did not capture.

S18 removes layer17 KV pairing. S9's teacher contrast instead collapses body
states at block16. These are related but distinct interventions. The S12 reader
was trained to predict the latter; it was never guaranteed to explain every
direction of the former. Keep these targets named explicitly.

## Next action and reproduction

S19 will measure the actual suffix gradient along the full pairing-removal path,
check its integrated response, and test whether a shared gradient learned from
other authors can reconstruct held-out effects. An active-gradient subspace
offers a bounded extension if a single shared covector is insufficient.
No more unbounded approximation-order or generic student grid.

Protocol/code62d4da0. `scripts/run_grain_reader_pattern.py` modes
prepare/run/evaluate. Private output `results/grain_vector_mechanisms/reader_pattern_s18_r1/`
contains frozen fold patterns, provenance,24author packets,private_effects,
metrics,summary andrun_status. Pattern covariance uses the other source folds'
three-view author means; no held-out author's data enters its own pattern fit.
The seven-forward design and all conditions were fixed before collection.
Four synthetic tests verify forward-OLS equivalence, score conservation,
coordinate equivariance, minimum supported Mahalanobis displacement and nonlinear
interaction. Broader focused suite:42passed, two existing SWIG warnings.
No neural collector remains running at this checkpoint.
