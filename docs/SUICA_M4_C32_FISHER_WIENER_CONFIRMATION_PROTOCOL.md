# GRAIN M4-C.3.2 Fisher-Wiener Confirmation Protocol

## Status

This protocol was frozen before opening the independent confirmation endpoint.
The preceding relation-kernel development result is not part of the
confirmation sample.

## Estimand

For each author, retain the discovered chart and the baseline response and
choice edges. Replace only creation:

\[
L_u^{FW}=A_u^{FW}R_u^DD_u^D.
\]

The primary endpoint is the Spearman correlation between author-pair
distances in \(L^{FW}\) and the oracle physical loop. Oracle quantities are
read only after the estimator and gates are frozen.

## Estimator

The maximal nested `gate` hazard family is fixed for every author. Even/odd
calibration and selection occasions produce two feedback-coefficient
estimates. Stable author signal and split noise are estimated from their
cross-covariance and difference covariance. Negative signal eigenvalues are
projected to zero. For each author, the filter and population center exclude
that author:

\[
W_{-u}=S_{-u}
(S_{-u}+N_{-u}+\epsilon I)^{-1}.
\]

The fixed epsilon scale is `1e-6`; no rank, kernel, bandwidth, geometry, or
endpoint selection is permitted. The train-half filter is applied to both
independent path views. Response state is a declared creation-hazard input;
response-head loss is not available to the filter.

## Confirmation population

- repetitions: `8`;
- mechanism authors: `16`;
- events per occasion: `120`;
- calibration/selection/evaluation occasions: `5/2/5`;
- main worlds: matched source partition, creation expansion, source rotation,
  history gate, and selection-creation compensation;
- no-creation controls: linear null, exogenous selection, and equal-marginal
  fast return; and
- refusal controls: condition alias, hidden opportunity source, evaluation
  support shift, author leakage, and response leakage.

## Frozen gates

- oracle creation headroom at least `.05` with repetition-cluster bootstrap
  lower bound above zero;
- Fisher-Wiener loop gain at least `.03` with clustered lower bound above
  zero;
- final loop geometry at least `.70`;
- recovery of at least 50% of oracle creation headroom;
- at least `6/8` repetitions with positive gain and geometry at least `.70`;
- held-out creation-hazard log-loss degradation no greater than 2% relative
  to the legacy selected hazard;
- author-label-permuted split-half gain no greater than `.01`;
- no-creation false success rate no greater than `.05`;
- orthogonal gauge difference no greater than `1e-6`;
- declared alias/support refusal rate at least `.95`;
- legacy selected-hazard physical replay difference no greater than `1e-8`;
  and
- common physical-creation shift geometry error no greater than `1e-10`.

## Stop rule

If oracle headroom remains positive but Fisher-Wiener gain fails, the current
opportunity budget does not identify the headroom through either relation
kernels or cross-author stable-subspace shrinkage. Further kernel/embedding
replacement stops. The next theoretical object must alter repeated
opportunity information or experimental intervention, not the condition
feature map.

## Frozen hashes

```text
e9c0d6304b019362f437d6cedbb06e9fd82ed9b634ffe1ddc4fd11ecaa0abfb4  configs/m4_fisher_wiener_creation_confirmation.json
6667aee92a3ac32ddb3063c61dae2a7de456138f9ab55b7129cd5029b643c946  scripts/run_suica_m4_fisher_wiener_creation_confirmation.py
a7efa3b7aa0f9244b1d3cea3901f677030aef4001bbec2d5676ea20b237e5c0c  suica_core/m4_fisher_wiener_creation.py
7c5fc10b857eb20a7e9c9e3b7fca16b986f3af89f9a3ba32ef7125480f89e5b3  suica_core/m4_creation_intervention.py
ff2b7a48f7948f122e1cb6e10d69b92186de750cd02bd310e17a78a0e3d3acdf  tests/test_m4_fisher_wiener_creation.py
```

## Boundary

A pass can support only response-safe recovery of a stable author derivative
subspace in the registered finite synthetic ecology. It cannot establish
personality, validate natural text, reopen M4-C.2, or authorize M4-D.
