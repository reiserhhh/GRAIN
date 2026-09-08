# Conditional baseline construction V1

2026-09-08, frozen before extracting these representations or fitting this target.
This is a source-exposed discovery panel, not a new preregistered human test.

## Question

Can the measured conditional anchor `b(E)=F(E,Pbar,Qbar)` be read from cached
block16 states using only the next attention layer's known linear projections,
normalization, RoPE and finite weighted moments? Main query N; TF is the distinct
transport-dominated comparison. Do not fit personality labels or change J_ref.

## Frozen protocol

- Exact M11 48 authors, 24/query, three original views each; all cached states.
- Fixed salted SHA256 ordering per query, round-robin four folds, six test authors
  per fold. Average each author's three feature vectors and targets before fitting.
- Four fixed representations: E boundary mean; native P mean plus Q boundary;
  anchored full-head attention mean; anchored mean plus diagonal variance, E
  attention mass and entropy per head. First two are controls, third is primary,
  fourth is a prespecified moment extension. No feature-family selection on Y.
- P and Q use the original two-reference replacement. Do not remove other tokens.
- One fixed linear kernel Ridge: train-only feature z scores, constant dimensions
  removed, K=XX'/p, alpha=1.0, intercept=train b mean. No hyperparameter search.
- Report all families outside author training against train-fold b mean and old
  constant b_ref. Collect no more suffix, prefix, gradient or text-generation calls.
  Only frozen layer17 Q/K/V linear maps, input norm and rotary embeddings execute.
- Freeze predictions before joining already-exposed source outcomes. Report b
  fidelity, full-score reconstruction `b_hat+PQ-b_ref`, and the covariance ledger.
  Preserve actual b, transport c, old PQ and local native results separately.

## Decision

Primary attention moments beating fold-mean MSE supports a constructible b(E).
Compare E mean and native PQ to say whether organized attention adds information
for this target, rather than merely renaming an ordinary mean reader. If richer
moments do not help, retain that result without another rank/search grid. Next
mainline question follows the largest *remaining functional term*, not Y maximization.

## Mathematical basis and limits

`p_t(q)=exp(q'k_t)/sum exp(q'k_i)`, `A(q)=sum p_t v_t` and weighted second moments
are finite, explicit functions. Attention/kernel-regression constructions are
established, e.g. [Goel and Bartlett (2024)](https://proceedings.mlr.press/v242/goel24a.html).
Their Kalman-filter existence result is not a theorem about this Qwen checkpoint
or psychological inference. Our contribution is the frozen conditional-anchor
target and its measured role in the real-author inference decomposition. A
weighted moment is not asserted to be a sufficient statistic without evidence.

Private cache and author-fold manifest stay in ignored
`results/grain_vector_mechanisms/conditional_baseline_construction_v1/`.
No public raw texts, identities, labels or scores; publish aggregate evidence only.
