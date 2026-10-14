# HM26: Source-Only Hermite Action-Field Compilation

Status: planned; freeze this specification and code before source fitting.

## Scientific Decision

HM25 identifies finite decoder curvature, not just final-score curvature. Can
source directional observations improve a cold scalar-potential approximation
with the same representation/kernel/regularizer as a value-only approximation?
This is a mechanism comparison, not a hyperparameter leaderboard.

## Fixed Design

Reuse all48 HM25 units (16 lexical families x3 queries), all9 fixed nodes and
their exact cached states. Source8 / validation4 / application4 families per
query remain disjoint. These are controlled hidden-state paths, not48 people.
No new neural calls, API, private text or model training.

Two declared representations: the old12 typed moments, and all24 degree0-3
moments over all6 spans. Both append the original block16 question-end state
and13 layout variables (six lengths, six starts, total input length). All6
spans contain <=4 tokens in this bank, so full moments plus lengths reconstruct
their states. Other token/context states are NOT completely represented.
Question-end state is fixed along each patch path, as in the actual intervention.
These inputs still require the lower fixed observer: not raw-text/LLM-free inference.

At node t, x(t)=O H(t) with fixed layout/Q and v=d x/dt=O Delta. Source values
f and source directional observations dF(H(t))[Delta] are the HM25 actual
outputs/derivatives. No validation/application outputs or gradients enter fit.
One source scalar RMS and source median positive squared distance set units
and Gaussian bandwidth; no outcome-dependent selection or grid.

For k(x,y)=exp(-||x-y||^2/(2 l^2)) and observation operators L_i (identity or
directional derivative), K_ij=L_i^x L_j^y k(x,y). The Hermite potential is

    f_hat(x) = m_source + sum_i c_i L_i^y k(x,x_i)
    (K + alpha diag(K)) c = b, alpha=1.

The prior is the source-value mean; derivative prior is0. Kernel-diagonal
normalization defines regularization units, not a calibrated likelihood or
claimed uncertainty. Value-only uses the identical kernel and value rows.
Three queries x2 representations x2 methods =12 bounded CPU fits.

Cold scoring uses f_hat(x1)-f_hat(x0) and the derivative of THAT SAME potential.
Its action telescopes and infinitesimal field is integrable by construction.
This is a known kernel/derivative method, not a new mathematical theorem.

## Outputs / Acceptance

Freeze source models before loading held outputs for metrics. Save all432
values/derivatives per condition, 48 endpoint actions per condition, normalized
Gram matrices/coefficient units, source manifests and exact input hashes.
Report source/validation/application separately: action RMS/RMSE/sign/correlation,
value error and derivative error. A source-only fit that improves held actions
supports compiling response variation; no gain redirects to omitted context or
source support, not a new alpha/bandwidth sweep. Do not claim personality validity.
Retain every sign failure and the old HM23/HM24 failures.

Tests: directional kernel finite differences, mixed-derivative PSD/symmetry,
direct augmented solve, affine unit transport, potential derivative/oddness/
telescoping, source-only dependence and invalid input. Alternate verifier must
recompute Gaussian derivative Gram entries directly, coefficients and every
prediction/action/summary without importing the production kernel.

## Sources

Solak et al., derivative observations in Gaussian-process dynamic-system
identification: https://www.dcs.gla.ac.uk/~rod/publications/SolMurLeiLeiRas03.pdf
de Roos et al. (2021), linear-operator derivative covariance:
https://proceedings.mlr.press/v139/de-roos21a.html
We use directional observations, not their full-gradient structured algorithm.
