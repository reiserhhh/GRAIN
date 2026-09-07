# GRAIN: aggregate information and conditional token configuration

2026-09-08. Mathematical continuation of
[functional theory V2](GRAIN_FUNCTIONAL_MECHANISM_THEORY_V2.md).
This supplements the established GRAIN measurement framework; it does not
replace it with a new psychological scale or rename its factors.

## 1. Why successful distillation and lost association can coexist

Let S be a frozen model's trait-query score, Z a fixed lightweight representation,
Y a psychological observation, and f(Z) the student. Exact covariance accounting:

\[
\operatorname{Cov}(Y,S)=\operatorname{Cov}(Y,f(Z))+
\operatorname{Cov}(Y,S-f(Z)).
\]

Improving E[(S-f(Z))^2] need not increase Cov(Y,f(Z)). The direction of the
remaining error in relation to Y matters, not just its magnitude. With square-
integrable variables, m(Z)=E[S|Z] is the orthogonal projection in L2 onto
Z-measurable functions, giving

\[
E[(S-f)^2]=E[(S-m)^2]+E[(m-f)^2],
\]
\[
\operatorname{Cov}(Y,S-f)=E[\operatorname{Cov}(Y,S\mid Z)]
+\operatorname{Cov}(E[Y\mid Z],m-f).
\]

Thus a finite student's remainder combines conditional information inaccessible
to that Z and estimation/approximation error within Z. One unsuccessful estimator
does not identify the first term. S5 tested additional linear and distributional
estimators; all retain the T/F connection but leave much of the N connection in
the remainder. Their measured result is stronger than the original single-PCA
observation, while remaining scoped to these finite classes and source authors.

Counterexample explaining the decision: take independent centered U,V, teacher
S=U+V, representation Z=U, and outcome Y=V. Exact teacher projection m(Z)=U can
capture most teacher variance if Var(U) is large, yet its Y covariance is zero.
Replacing Y by U reverses the conclusion without changing any observed (S,Z)
training data. Teacher fidelity alone therefore cannot universally identify
psychologically useful teacher components. In GRAIN the trait query and
evidence-response experiments supply additional structure; outcome connections
remain separately reported rather than silently optimized.

## 2. Distribution is not the same object as mean position

For user u's comment vectors h_ut, a distribution representation is

\[
\widehat\mu_u=\frac{1}{T_u}\sum_t\phi(h_{ut}).
\]

Nonlinear phi generally gives a different object from phi(mean_t h_ut).
A Gaussian kernel mean can distinguish distributions with the same ordinary
mean; finite random Fourier features approximate that construction. S5B tested
this distinction at the comment level. It did not produce the missing N
association, so the next target is conditional token configuration inside the
decoder, not an unbounded kernel search.

This use follows [Szabo et al., JMLR2016](https://jmlr.org/papers/v17/14-510.html)
and [Rahimi & Recht, NeurIPS2007](https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html).
We borrow the operators and finite approximation, not an automatic consistency
guarantee for dependent Reddit comments.

## 3. Mean-preserving configurations: an exact diagnostic

Let H be the entire block16 token-state matrix for one fixed prompt and q the
trait query. For selected, disjoint comment token groups, Pi replaces states
within each group by their own mean and leaves all other rows unchanged. With
the Frobenius inner product, Pi is an orthogonal projection:

\[
\Pi^2=\Pi,\quad \Pi^*=\Pi,\quad H=\Pi H+R,\quad \Pi R=0.
\]

The operation preserves each comment mean, the unselected states, token count,
position indices, and the block16 query state. It removes within-comment state
variation, not a predeclared psychological noise component. The intervention
is explicitly internal; it need not correspond to a natural sentence.

For a differentiable fixed decoder suffix s_q, the line integral is exact:

\[
s_q(H)=s_q(\Pi H)+
\int_0^1\langle\nabla s_q(\Pi H+tR),R\rangle_F\,dt.
\]

S6 computes the reverse path H+aD with D=-R, so its measured effect is
s_q(Pi H)-s_q(H). Its first-order approximation is <grad s_q(H),D>.
Numerical integration error is compared against the actual endpoint difference;
the fundamental theorem is exact but a three- or five-node rule need not be.

### What a nonzero effect establishes

If a fixed readout were exclusively a function of Pi H, it would be constant
along these mean-null directions. A measured nonzero endpoint difference is
a direct counterexample to that sufficiency claim for this readout and input.
Conversely, a continuously differentiable function whose directional derivative
vanishes along every mean-null direction at every point of a connected affine
fiber is constant on that fiber, by integrating its zero derivative.

This is standard projection/differential reasoning applied to a new measurement
question, not a newly invented theorem. It tests sufficiency of this particular
averaging operation in this decoder; it does not test every embedding model or
demonstrate a causal personality trait in a human.

## 4. What becomes measurable

For each author, query and selected comment set, retain:

- Aggregate-preserving endpoint effect: the actual change of s_q.
- Local functional sensitivity: the derivative in the declared direction D.
- Integrated contribution: the directional gradient accumulated along the path.
- Nonlinear remainder: endpoint minus the local derivative.
- Matched-query shared response: the correlation of the same intervention's
  effects for different trait queries, preserving author pairing.

These give query-conditioned operators on configurations rather than just
distances between pooled author vectors. Shared effects can then guide compact
multi-query models; query-specific effects remain private branches. No scalar
correlation here is automatically a psychological measurement coefficient.

The path calculation follows
[Sundararajan, Taly & Yan, ICML2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
Here the baseline is the declared projection Pi H in an internal suffix, not
an allegedly neutral text or a semantic explanation of individual words.
Pairwise interaction methods such as
[Janizek et al., JMLR2021](https://www.jmlr.org/papers/v22/20-1223.html)
motivate a possible subsequent decomposition; S6 itself is not Integrated Hessians.

## 5. Evidence and continuation

[S5 results](../reports/grain_vector_mechanisms/S5_STUDENT_REMAINDER_REPORT.md)
are complete. S6 follows the bounded protocol in
[the active goal](GRAIN_ACTIVE_RESEARCH_GOAL_V3.md). Its result report must state
actual mean-preservation error, endpoint effects, quadrature gaps and computation
cost before choosing a further experiment. Do not resume the old observer queues
or interpret student-grid exhaustion as proof that the mainline is impossible.
