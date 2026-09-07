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

## 6. Group averaging separates two mechanisms inside that configuration

The mean-null effect combines a change of dispersion with a change of arrangement.
Take G to be the product of permutation groups acting within the selected comment
rows. Every gH preserves the complete unordered multiset of states per comment.
There are two distinct averages:

\[
\Pi H=|G|^{-1}\sum_{g\in G}gH,
\qquad (\mathcal R s_q)(H)=|G|^{-1}\sum_{g\in G}s_q(gH).
\]

The first averages representations; the second averages judgments. They commute
for an affine readout, but not for a general nonlinear readout. The identity

\[
s_q(H)=s_q(\Pi H)
+\underbrace{[s_q(H)-(\mathcal R s_q)(H)]}_{\text{arrangement-sensitive response}}
+\underbrace{[(\mathcal R s_q)(H)-s_q(\Pi H)]}_{\text{nonlinear distribution response}}
\]

keeps both mechanisms without declaring either to be noise. The last term is
a signed Jensen-type gap; a neural readout is not assumed convex. The middle
term describes this state's position on its permutation orbit, not an author's
population norm or a psychological score in its own right.

The exact group-average function is invariant because multiplication by a group
element only reorders the sum. Under an appropriate group-invariant measure,
averaging is also an orthogonal projection on invariant functions. The natural
author/text distribution is not assumed to have that symmetry, so the three
components above are not reported as independent psychological variances.

S7 approximates the judgment average with three fixed uniform permutations per
comment set. The empirical accounting remains algebraically exact; approximation
to the full group average has Monte Carlo error, which is recorded separately.
An effect under permutation directly shows sensitivity beyond the unordered bag
of these hidden states. It can include the decoder's positional machinery; it
does not mean that shuffling real words preserves their meaning.

This draws on finite-group representation theory (Etingof et al.,
[Sections4.1 and4.6](https://math.mit.edu/~etingof/reprbook.pdf)) and
[Deep Sets](https://papers.nips.cc/paper/6931-deep-sets.pdf). The latter motivates
the distinction between invariant functions and order-sensitive functions; its
sum-representation results have domain/size assumptions and do not make our
finite256-feature bag approximation universally sufficient.

## 7. S7 points to a curvature-weighted covariance operator

[S6](../reports/grain_vector_mechanisms/S6_MEAN_NULL_PATH_REPORT.md) and
[S7](../reports/grain_vector_mechanisms/S7_ORBIT_RESPONSE_REPORT.md) are complete.
The matched N/confirmation association lies mainly in the nonlinear distribution
component (r=.9730), not the estimated arrangement component (r=-.0776).

Write r_g=vec(gH-Pi H), K_q=Hessian(s_q)(Pi H). Under a twice differentiable
local expansion with a controlled remainder,

\[
(\mathcal R s_q)(H)-s_q(\Pi H)
=\tfrac12\operatorname{tr}(K_q\Sigma_G)+E_g[\mathrm{Rem}_q(r_g)],
\qquad \Sigma_G=E_g[r_g r_g^T].
\]

The linear term cancels because the orbit displacement has zero group mean.
This expression is a proposed approximation when the remainder is omitted,
not a claim that S7 already estimated a Hessian. Functional sensitivity selects
which variance matters, reconnecting S0's function-preserving reduction to the
token-configuration mechanism.

There is useful exact structure in Sigma_G. For a selected comment of n>1
centered state rows R_i with C=(1/n)sum_i R_i R_i^T, uniform permutation gives

\[
\operatorname{Cov}(R_{g(i)},R_{g(j)})=
\begin{cases}C&i=j,\\-C/(n-1)&i\ne j.\end{cases}
\]

Different independently permuted comments have zero cross-covariance. For n=1
the residual is zero. This permits an analytical covariance or matrix-vector
product without materializing a gigantic token-by-hidden covariance matrix.
The corresponding Hessian contraction combines diagonal token sensitivity and
cross-token interactions with known coefficients. This supplies the local
approximation tested in S8, not an established lightweight personality predictor.

Implementation seed: `grain_research/vector_mechanisms/orbit_geometry.py`
applies Sigma_G without constructing the full covariance. Four synthetic tests
enumerate complete small permutation groups and verify both the covariance
action and the quadratic Hessian contraction. The neural S8 curvature experiment
is now complete; see the refinement below.

## 8. S8 refines local curvature into a path-dependent operator

[S8](../reports/grain_vector_mechanisms/S8_ORBIT_CURVATURE_REPORT.md) estimates
one fixed directional contraction per condition, not a full Hessian trace.
Small-radius coefficients predict N's held full-radius response with r=.7537
and MSE/zero=.7177. T/F gives r=-.2056 and MSE/zero=6.2142. Confirmation shows
half-radius ordering (.7338) but miscalibrated amplitude (MSE/zero=1.8611).
Shared N/confirmation local quadratic response is r=.7117 [.4245,.8717].

The correct global object, for a C2 suffix, is the integral remainder:

\[
s_q(m+r)-s_q(m)=\nabla s_q(m)^T r+
\int_0^1(1-t)r^T K_q(m+tr)r\,dt.
\]

Uniform averaging over the finite group eliminates only the initial linear term:

\[
(\mathcal R s_q)(H)-s_q(m)
=\int_0^1(1-t)E_g[r_g^T K_q(m+t r_g)r_g]\,dt.
\]

The Hessian can depend on the direction and the position along its path. Pulling
it outside that expectation and integral is an approximation, not an identity.
Thus S8 narrows the constant-curvature proposal without discarding the established
configuration effect. It directs development toward query-conditioned integrated
operators rather than assuming one quadratic law for all traits.

`integrated_hessian_contraction` in `orbit_geometry.py` now implements bounded
Gauss-Legendre integration of an externally supplied Hessian-vector product.
Four synthetic tests cover exact quadratic contraction, a quartic mean-null
effect with zero local curvature, full finite-group nonlinear averaging, and
invalid outputs. The quartic witness shows why a zero Hessian at the center
does not imply no nonlinear configuration signal. This is verified algebra and
reusable numerical code, not an additional neural collection or psychometric result.

Important representational boundary: Pi H retains the query and unselected token
states, which already contain earlier-layer contextual information. The mean-null
experiment tests a suffix's use of selected token configurations conditional on
that retained information. It does not turn the whole decoder into a mean-only
encoder and does not exhaust all routes that carried configuration upstream.

## 9. Connect the operator to the observed source remainder

[S9](GRAIN_SOURCE_REMAINDER_S9_PLAN.md) keeps the frozen S4 student f, the exact
old three-view texts, and all213 N source authors, with64 hash-selected T/F controls.
Let S_old be the original CUDA/BF16 score, S_new the paired MPS/FP32 origin,
and M the all-comment projected suffix score. Then

\[
S_{old}-f=(S_{old}-S_{new})+(S_{new}-M)+(M-f).
\]

Linearity gives the same equality after Cov(Y,.) is applied. The terms measure
observer-condition difference, token-configuration intervention response, and
the remaining projected-suffix/student difference. The last term can include
retained query-state information, not just comment means. None is automatically
an independent fraction of human personality variance. Scores are frozen before
the already authorized source outcomes are joined; no estimator is selected on Y.

## 10. A concrete connection: within-comment key/value coupling

S10's [source-grounded plan](GRAIN_QUERY_COVARIANCE_S10_PLAN.md) now supplies
an elementary link beneath the suffix derivatives. For fixed, position-resolved
keys/values and effective scaled query q, A(q)=sum_i p_i(q)v_i,
p_i(q)=exp(q'k_i)/sum_j exp(q'k_j). The finite exponential-family derivative is
J(q)=Cov_p(V,K); integrating J along a query segment recovers its readout change.
These are actual vector operations, not a generated verbal explanation.

For a comment group G, let P_G=sum_{i in G}p_i(q), and let A_G(q) be the
conditional weighted value within that group. Hold **native group mass** fixed
while considering its within-group response. The exact coupling is

\[
C_G(q)=P_G[A_G(q)-\bar v_G]
=P_G\int_0^1\operatorname{Cov}_{p_G(tq)}(V,K)q\,dt
=\sum_{i\in G}(p_i-P_G/|G|)v_i.
\]

The entire head vector therefore separates non-comment output, group-level
mean values weighted by P_G, and sum_G C_G. This preserves the choice/coverage
mass of every comment while measuring within-comment relational contribution.
It is a controlled computational comparison, not a proposal to erase the
remaining text or a claim that P_G is a psychological importance score.

The same raw keys and values with different pairing can change A(q), despite
identical marginal distributions. A joint KV object differs from two independent
bags. Contextualized embeddings could already contain such relations, so this
counterexample does not establish that a particular encoder has lost them.
Likewise, the q above is a head's contextualized query, not literally a single
trait word and not a human trait coordinate.

Implementation and eight algebra tests are in `query_tilt.py` and its tests;
two tiny random Qwen CPU tests verify native reconstruction, copying and hook
cleanup. The planned real-model patch subtracts sum_G C_G at layer17's final
query head output and observes downstream trait-query scores. This connects
covariance geometry to an intervention without assuming that attention weight
alone is an explanation. No S10 real-model result has been collected yet.
