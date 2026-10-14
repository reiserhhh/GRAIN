# Observation Fibers and Context-Dependent Action Fields

HM23-24 theory update, 2026-10-09. Preserve historical scalar/vector results.

## Observation Before Compression

For a fixed query and receiver context, let Fq map intervened block16 states to
the candidate-conditioned score. Let O denote the declared typed moment map.
A local factorization Fq=phi_q o O requires dFq to annihilate ker(O).
For linear O and a connected convex domain, annihilation throughout the domain
makes Fq constant on connected fibers; a zero derivative at one point is not
sufficient. This is a standard differential factorization criterion.

The current property spans have at most4tokens, so moments0-3 recover their
complete temporal states. Actor/neutral spans retain only means. HM24 finds
readout-active directions in those summary fibers with actual small patches.
These interventions are off the natural-text manifold and adaptive to dFq.
They diagnose a representation, not human psychological validity.

## Two Distinct Remainders

Summary omission: a delta in ker(O) changes Fq. Finite nonlinear remainder:
Fq(H+delta)-Fq(H) differs from dFq(H)[delta] even when O retains delta exactly.
The latter cannot be repaired simply by adding more property moments.

HM23 vector-logit prediction under the old source kernel shrinks almost to zero,
not a restoration of joint action. Fit-then-nonlinear-readout and direct scalar
fitting need not commute. The failed model does not reject all vector methods.

## Query Families

At a common addressed body state, gradients define a query-indexed cotangent
field. Gram/cosine relationships describe shared versus distinct sensitivities
in a declared hidden-state metric. They are not psychological factor correlations,
and coordinate/metric transport must be paired. An arbitrary coordinate rotation
cannot turn a lost observation into a recoverable one.

## Finite Action

For a differentiable path gamma from H0 to H1:

    Delta Fq = integral <dFq(gamma(t)), gamma'(t)> dt.

This standard exact-differential identity underlies Integrated Gradients. The
total is endpoint determined; individual coordinate attributions depend on
the path, baseline and decomposition. A scalar score does not have nonzero
holonomy merely because coordinate allocations differ around routes.

The next frozen HM25 experiment tests this identity numerically and separates
decoder variation from known candidate-readout curvature. It does not assert
that evaluating this field is already LLM free. The mainline next challenge is
source-learned, independently evaluable compression preserving these actions,
then author-level measurement under defined evidence and opportunity laws.

## Sources

[Zahm et al.](https://arxiv.org/abs/1801.07922): observation projection and gradient
metrics, without importing Gaussian/Poincare guarantees for text.
[Sundararajan et al.](https://proceedings.mlr.press/v70/sundararajan17a.html): path
integrals and attribution, without claiming a new mathematical theorem.
OpenAI/math metric/unitary-connection manuscript at fixed adc7f124: the discipline
of preserving observations, maps and units together, not its PDE uniqueness result.
