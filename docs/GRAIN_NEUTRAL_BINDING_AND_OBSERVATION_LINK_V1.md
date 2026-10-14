# Neutral binding, observation links, and qualified author evidence

2026-10-10. HM128 theoretical companion, written during the frozen collector.
It does not change source inputs, intervention rank, layer, budget, or acceptance.
The original18 protocols/100 cards/44 computational questions remain open.

## 1. The mainline decision, not another accuracy target

We want to distinguish two architectures for an author evidence observation:

\[
  F_q(h,u)=D_q(B(h,u))
\]

with a reusable subject/evidence binding operation B, versus a jointly
conditioned interpretation F_q whose attribution action changes with q and
context. This is an operational distinction, not an assertion that the
factorization is mathematically unique. Every arbitrary function can be
trivially factored; reusable, independently identified interventions are what
give this distinction empirical content.

Let pi exchange two people while retaining behavior contents and conditions.
A source-identified activation operation T_pi supports a reusable attribution
channel if the person-indexed readouts approximately satisfy

\[
  F_q(T_\pi h) \simeq \pi F_q(h)
\]

on new materials and targets, beyond a norm-matched unrelated operation, while
an untouched third person remains unchanged. HM128 also compares against the
baseline of actual assignment-swapped text, rather than assuming that its
model readouts are perfectly permutation-equivariant.

A large common score shift is not this property. Neither is a low-rank
activation matrix by itself. The operator must first work on held-out facts;
otherwise failed behavioral routing cannot distinguish the two architectures.

## 2. A probability Hessian includes observation-link curvature

For one factual trial, let m(x,y) be the correct-versus-other candidate logit
margin and p(x,y)=sigmoid(m(x,y)). Denote gradients by g_x and g_y and the
mixed margin Hessian by H_m. The ordinary chain rule gives

\[
  H_p=p(1-p)H_m+p(1-p)(1-2p)g_xg_y^\top.
\]

For the source average, take the average of these trial-specific matrices.
Do not replace trial-specific probabilities by their mean in this equation.

The second term is rank at most one per trial, even when H_m is zero.
Therefore low rank and nonzero H_p do not, alone, identify relational binding.
When every trial is individually balanced (p=0.5), this logistic-link term
vanishes. A mean probability of0.5 does not guarantee that each term vanishes.

An exact counterexample is m=6.6+a^T x+b^T y, with independent x/y contributions
and no mixed margin interaction. H_m=0 but H_p is a nonzero rank-one matrix.
At m=0 the same counterexample has H_p=0. Both statements are checked by
double-precision autograd in
`tests/test_grain_binding_probability_curvature.py`. They are established
calculus, not a new theorem, and do not identify which term dominates HM128.
The frozen run did not retain g_x/g_y separately, so that empirical
decomposition cannot be inferred from its Hessian sketch alone.

## 3. What this changes in interpretation

HM128's source midpoint probabilities must be reported as observed. If they
remain near one, describe the recovered matrix as a local mixed derivative
of factual retrieval, not an established pure binding operator. The factual
interchange and untouched-person tests provide separate causal evidence.
An absent interchange effect at this site is an instrument result, not a
proof against binding in the model or against all reusable attribution.

Do not start an accuracy rescue from this distinction. A new layer/rank sweep
could improve some swaps without answering whether a common attribution
operation exists. A next study would need a different, independently defined
observation or intervention that can distinguish remaining mechanisms.

The GRAIN construction consequence is to keep the following objects separate:

1. The qualified event and person to whom it belongs.
2. Its conditional contextual representation and retained complementary data.
3. The target-conditioned interpretation action.
4. The probability or other numerical reporting link.
5. The author's evidence composition, choice and state over events.

Output-link curvature belongs to item4; it must not silently become a named
personality factor in item3. Existing external-scale associations are not
transferred to a new mixed derivative or to its unnamed coordinates.

## 4. Prior art and the actual candidate contribution

[Feng/Russell/Steinhardt2024](https://arxiv.org/html/2406.19501v1),
sections4.1--4.2, already identify binding using a mixed Hessian and causal
subspace swaps. Their procedure perturbs across all40 layers in a13B model;
their midpoint construction is intended to erase binding. HM128's single
block, small observer and randomized sketch are a restricted adaptation, not
an exact replication or a refutation of their result.

[Halko/Martinsson/Tropp](https://arxiv.org/abs/0909.4061) already provide
randomized matrix range approximation. Differentiation, SVD, projection and
the logistic chain rule are mature mathematics.

[OpenAI's pinned selected-jets manuscript](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex)
motivates complementary differential observations and checking their support.
Its PDE and smoothness assumptions have not been established for GRAIN;
no theorem is transported to human psychological measurement here.

The candidate new application is whether an operation identified from
person-country facts can route new behavioral evidence into multiple
person-specific judgments, and whether it changes GRAIN's construction.
The mathematics alone is not the contribution. Final HM128 numerical results
and limits belong in its result report, not in this a-priori derivation.
