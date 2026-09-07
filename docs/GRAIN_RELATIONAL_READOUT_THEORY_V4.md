# GRAIN V4: a relational readout of contextual evidence

2026-09-08. Continuation of the aggregate/configuration theory V3, not a new
release of the GRAIN measurement system. S9-S13 provide the empirical starting
point; S14 tests the shared-query decomposition below.

## 1. The object is a joint relation, not just a bag of vectors

For one captured attention head, keep the finite pairs (k_i,v_i), their positions,
and disjoint comment-body groups G fixed. Let

\[
\widetilde v_i=v_i-|G|^{-1}\sum_{j\in G}v_j\quad(i\in G),
\qquad \widetilde v_i=0\quad(i\notin\cup G).
\]

All tokens still participate in attention normalization. For the effective,
already-scaled and rotated query q, define

\[
p_q(i)={e^{q^Tk_i}\over\sum_j e^{q^Tk_j}},\qquad
A(q)=\sum_i p_q(i)\widetilde v_i.
\]

Expanding the group centering gives exactly the S10/S12 vector
sum_G sum_(i in G) (p_i-P_G/|G|) v_i. This algebra isolates within-comment
pairing; it does not delete comments or claim their mean is psychological noise.
The frozen source reader is

\[
\widehat C_u=b_f+{1\over3}\sum_{v=1}^3\sum_h\beta_{f,h}^TA_{u,v,h}(q_{u,v,h}),
\]

where f is the held-out author fold. The beta coefficients were fitted only to
the teacher configuration contrast C, not to the source questionnaire. This is
an exact representation of the implemented affine reader, an approximation to C,
and not an exact formula for human personality.

### A concrete non-sufficiency witness

Take a single four-token group, k=(-2,-1,1,2). Author A has v=k; author B has
v=-k. Both have the same key multiset, value multiset, and all marginal moments.
Their separate empirical distributions, and any functions of those separate
distributions, are identical. For q>0 the paired readouts have opposite signs.
What differs is the coupling of K and V. Separate universal kernel summaries of
each marginal would still miss this distinction; a joint summary can retain it.

This exact finite example explains what a relational representation can add.
It does not prove that E5 lacks the relationship: E5's coordinates may already
encode it nonlinearly, and S5/S11 tested only specified finite reader families.
S12's observed comparison nevertheless favors a direct pairing representation
for recovering the measured C response on the current source authors.

## 2. A common query can still reveal author differences

For each held-out fold, S14 averages native pre-RoPE queries over training
authors only, averaging views equally within author. Rotate this reference into
each held-out prompt's endpoint position. Set q_u=q_0+delta_u, with K,V fixed.
For each head write R_i=beta^T vtilde_i and U_i=delta_u^T k_i. Then

\[
E_{q_0+\alpha\delta_u}[R]
={E_{q_0}[R e^{\alpha U}]\over E_{q_0}[e^{\alpha U}]}.
\]

This is an exact finite exponential tilt, not a regression assumption. At the
author-score level it yields

\[
\widehat C_u(q_u)=\widehat C_u(q_0)+\Delta_u^{\rm query\mid KV}.
\]

The first term is not an author-independent constant: each author supplies
different K,V and therefore a different attention distribution even at q_0.
The second term isolates query adaptation given those K,V, not all identity.
Both terms may contain useful author information; no orthogonality is assumed.

The general mechanism is shared statistical operation plus different input
relations and possibly different queries. It is not necessary for each author
to have a unique mathematical law or for all authors to share a scalar axis.

## 3. Local derivatives make the query adjustment measurable

Differentiating the finite normalized expectation along alpha gives, at zero,

\[
L=\operatorname{Cov}_{q_0}(R,U),\qquad
Q=E_{q_0}[(R-E R)(U-E U)^2].
\]

Thus Delta(alpha)=alpha L+alpha^2 Q/2+remainder. At any point t the first
derivative is Cov_(q0+t delta)(R,U), and integrating it gives the exact finite
response. A bad second-order approximation means curvature varies along the
path; it does not contradict this integral identity. S14 records both fixed
doses .5 and1 and both orders without fitting an order or dose to human scores.

These operations follow the finite exponential-family derivative calculus in
[Wainwright & Jordan2008](https://www.cs.princeton.edu/courses/archive/fall11/cos597C/reading/WainwrightJordan2008.pdf).
The query Fisher form Var_q(U) connects them to the accessibility calculation in
`GRAIN_QUERY_INFORMATION_GEOMETRY_V1.md`. That variance is over weighted tokens,
not a fraction of human personality variance.

## 4. Uniform attention exposes the first joint moment

Since each group's centered values sum to zero, A(0)=0 exactly, including any
outside tokens. Let n be the full attention-support size. At zero query,

\[
D_q A(0)={1\over n}\sum_G\sum_{i\in G}
(v_i-\bar v_G)(k_i-\bar k_G)^T.
\]

Proof: use Cov_uniform(Vtilde,K), the global mean of Vtilde is zero; subtracting
each group's key mean leaves the sum unchanged. Hence near zero, the first
nonzero readout is a query-contracted within-comment key/value cross-moment.
This gives a precise covariance alternative to ordinary mean pooling. A useful
large native query may still need higher joint moments or the exact softmax.

This calculation links the earlier group-projection diagnostic to actual
associative reading. [Ramsauer et al., ICLR2021](https://arxiv.org/abs/2008.02217)
provides the attention/associative-memory precedent; we do not inherit a global
Hopfield energy or convergence guarantee for the whole Qwen decoder.

## 5. How much of the language model is still necessary?

S13 verifies this actual path:
text -> blocks0-16 -> layer17 QKV projections -> pairing -> frozen affine reader.
No upper decoder suffix or generated explanation is used. Existing pretrained
semantic computation is still used by the prefix. The reduction is therefore
operationally real but not a claim that all semantic computation has vanished.

The next compression question is whether an author-out reference query and a
small set of joint moments preserve the useful response. S14 answers the first
part. Only its result should determine a further moment approximation experiment.
Do not substitute an unbounded search for this clear sequence.

[Le et al.2026, CCQ](https://arxiv.org/abs/2606.01294) suggests a related read-side
curvature intervention for linear attention. GRAIN here uses exact normalized
softmax on a frozen Qwen prefix; it does not apply CCQ's high-variance contraction
or assume crowded directions are irrelevant. Its relevance is the separation
of stored relations from how a query reads them.

## 6. Shared across authors is not yet shared across traits

S14 changes the internal query while retaining one N prompt's entire KV support.
A different natural-language trait prompt also changes its own token keys and
the normalization support. Body token states precede the question in this
causal template, but that alone does not make the full attention denominator
query-independent. A future cross-trait reuse experiment must declare whether
it keeps full support or uses body-only conditional attention; those are
different operators. A reused common N query does not by itself prove a common
TF/N/attachment query or a universal psychological readout.

## 7. Evidence ledger

- S9: source N configuration contrast r=.2116; native versus projected delta
  r=.0773, paired95%CI[.0236,.1330]. A useful empirical target exists.
- S10: within-comment pairing intervention changes final scores; fixed-head
  query derivative is a measured KV-covariance operation.
- S11: E5 mean readers do not recover C in the declared finite families.
- S12: KV/PCA teacher-C r=.3203, MSE/null=.9063, better target recovery than body
  mean. Source N r=.1429, not a demonstrated external superiority over all controls.
- S13: exact vector/reader replay with prefix-only execution; paired time ratio
  .6121. No additional prediction-accuracy claim from the replay itself.
- S14 complete: common/native query target-C r=.315627/.320283, source N
  r=.133706/.142869; paired differences span zero. Native queries are already
  close to the reference (median cosine.999191). Second-order covariance/cumulant
  formula reconstructs the small personalized query increment at r=.999416,
  MSE/zero=.001185. This is internal response reconstruction, not personality r.
  Details: `S14_SHARED_QUERY_REPORT.md`. The next test uses the common query.

The contribution is the experimentally connected chain from mean-null response
to source outcome, joint KV access, and a cheaper mathematical readout. The
general covariance, Taylor, projection and exponential-family identities have
prior theory; the theory does not need invented mathematical names to be useful.

Finite tests: `tests/test_grain_query_decomposition.py` (derivatives and multihead
implementation), `tests/test_grain_relational_readout.py` (joint-only witness,
uniform-query annihilation and cross-moment derivative).

## 8. A precise next compression question, not a wider hyperparameter grid

Inside comment G, use its uniform empirical distribution and define
L_i=q^T k_i, Z_G=sum_(i in G) exp(L_i), P_G=Z_G/sum_all exp(L_i).
The exact readout can also be written

\[
A(q)=\sum_G P_G(q)
\left\{ {E_G[V e^L]\over E_G[e^L]}-E_G[V]\right\}.
\]

This separates **which comment receives mass** from **how the comment's paired
evidence is read**. It is conditional expectation algebra, not an assumption
that comments are independent observations or that tokens are Gaussian.

The conditional cumulant expansion begins with

\[
E_G[V e^L]/E_G[e^L]-E_G[V]
\approx \operatorname{Cov}_G(V,L)
=\Sigma_{VK,G}q,
\]
\[
\log Z_G\approx\log|G|+E_G[L]+\tfrac12\operatorname{Var}_G(L).
\]

For a truly joint-Gaussian (V,K) population these moment expressions are exact
for conditional tilted means and log normalizers. Finite token clouds are not
assumed joint Gaussian. The useful experiment compares: exact pairing; uniform
comment masses plus covariance; exact comment masses plus covariance; and
Gaussian-approximated masses plus covariance. Outside-comment normalization
terms remain explicit in every approximation, rather than being dropped.

If exact masses rescue covariance but Gaussian masses fail, normalizer/tail
structure is the next target. If covariance fails despite exact masses, retain
within-comment higher-order relationships. If the approximation succeeds, it
offers a concrete sufficient-statistic candidate for this readout and data,
not universal sufficiency for all psychology. S14 decides whether the frozen
common query or native query should be the fixed operating point for that test.
