# Query information geometry for the GRAIN vector mechanism

2026-09-08. Mathematical development alongside S12, with finite numerical tests.
The immediate question is which value directions a query can access, not how to
name an attention head as a personality factor.

## 1. What the existing data already give us

S9 links a mean-null text-configuration contrast to source N measurements.
S10 identifies an exact local operation: a contextualized query changes the
weighted readout of paired keys and values. S11 rules out a useful recovery by
two ordinary E5-mean reader classes. S12 measures where the contrast is accessible
in the actual model's source-author representations. These are successive links,
not interchangeable measures of human personality accuracy.

## 2. A statistical geometry beneath a fixed attention head

Freeze the token keys k_i, values v_i, and active causal support. The effective
query q already contains the attention scale. Write

\[
p_q(i)=\exp(q^\top k_i-\psi(q)),\quad
\psi(q)=\log\sum_i \exp(q^\top k_i),\quad A(q)=E_q[V].
\]

Then I(q)=Cov_q(K,K) is the Fisher information of this finite token distribution,
and J(q)=Cov_q(V,K) is the query-to-value Jacobian. For a fixed downstream linear
readout beta, let S_i=beta'v_i and g=Cov_q(K,S)=J(q)'beta. For a query displacement d,

\[
\partial_d\{\beta^\top A(q)\}=g^\top d,\qquad
(g^\top d)^2\leq\operatorname{Var}_q(S)\,d^\top I(q)d.
\]

The inequality follows directly from weighted Cauchy-Schwarz. It states that a
large response requires both varying value evidence and a query change that
actually redistributes attention. Large raw Euclidean query distance alone says
little. This is a local fixed-head result, not an assumption that attention
weights equal psychological importance.

## 3. Solve for accessible value variation

Let I^+ be the Moore-Penrose inverse on the supported key subspace. Because g is
a covariance with those same keys, it belongs to range(I). Thus

\[
\kappa_\beta^2=g^\top I^+g\leq\operatorname{Var}_q(S),\qquad
\max_{d^\top Id\leq\epsilon^2}g^\top d=\epsilon\kappa_\beta.
\]

For kappa>0, one optimal local direction is d*=epsilon I^+g/kappa. Components
in ker(I) do not change token weights locally. If kappa=0, this first-order
direction is zero; finite nonlinear changes can still exist along other queries.

An equivalent derivation is weighted least squares: project centered S onto
centered keys. Explained variance is kappa^2 and remaining variance is
Var(S)-kappa^2. For all value coordinates simultaneously,

\[
H_{\rm accessible}=J I^+ J^\top,\qquad
\Sigma_V-H_{\rm accessible}\succeq0.
\]

This Schur-complement/linear-projection identity supplies a coordinate-free
quantity to study before interpreting text. Under exact nonsingular linear
coordinate changes, the scalar response and Fisher-constrained optimum are
unchanged. Numerical rank thresholds can affect nearly singular estimates.

Crucially, these variances run **over tokens under p_q**, not over authors or
questionnaire scores. An accessible direction is a computational signal route,
not automatically a psychological trait. Different author/query pairs may have
different I and J; shared directions are an empirical next question.

## 4. Relation to query-conditioned representation learning

A deterministic contextual query vector does not create information outside the
input and fixed model. It can convert nonlinear associations into a form a small
reader can recover. For a source contrast C and representation Z, the ideal
squared-error reader is E[C|Z]. Nested information fields have the projection
identity recorded in S12; the body/query/KV summaries are not generally nested.

This gives two distinct notions of access:
- within-model: local redistribution of values through J and I;
- across-authors: out-of-fold recoverability of the measured contrast from Z.

S12 tests the latter. The helpers here quantify the former, enabling a future
query intervention to target a reachable direction rather than an arbitrary
Euclidean perturbation. No new model or free-running query search is launched.

## 5. Primary-source grounding and limits of borrowing

- Wainwright & Jordan (2008), [Graphical Models, Exponential Families, and
  Variational Inference](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf):
  cumulant derivatives and covariance/convexity provide the finite-family starting
  point. Our weighted-regression and constraint solution above are explicit
  applications, not newly discovered general theorems.
- Amari (1998), [Natural Gradient Works Efficiently in Learning](https://doi.org/10.1162/089976698300017746):
  motivates respecting the Fisher metric rather than Euclidean parameter distance.
  This is a local query intervention calculation, not a natural-gradient training
  implementation or a borrowed convergence guarantee.
- Wibisono & Wang (2025), [Exponential Family Attention](https://arxiv.org/abs/2501.16790):
  attention-based context-dependent latent factors offer a close statistical
  precedent. Their identifiability is up to linear transformations under stated
  assumptions, not a proof that our frozen heads identify unique traits.
- Calvello et al. (2025), [Continuum Attention for Neural Operators](https://www.jmlr.org/papers/v26/24-0879.html):
  motivates treating attention as a nonlocal operator. Their function-space
  approximation theorem is not automatically applicable to human text distributions
  or evidence that GRAIN can replace all pretrained semantic computation.

Implementation: grain_research/vector_mechanisms/query_geometry.py.
Tests cover finite-difference slope, weighted-regression equivalence, Fisher-norm
constraint optimum, nonsingular coordinate changes, and degenerate keys.
No source labels are needed by these mathematical tests.
