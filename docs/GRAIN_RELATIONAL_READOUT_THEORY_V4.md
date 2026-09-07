# GRAIN V4: a relational readout of contextual evidence

2026-09-08. Continuation of the aggregate/configuration theory V3, not a new
release of the GRAIN measurement system. S9-S20 connect the empirical source
response, shared-query decomposition, conditional curvature and functional reader.

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

## 9. Empirical closure of the finite moment question: S15-S17

S15 found that preserving exact comment mass is critical: uniform covariance
has MSE/exact-author-variance1034.28; exact-mass covariance.7011. Gaussian mass
approximation gives3.7979. S16 independently decomposed A=B sum_G r_G D_G,
where B=sum_G P_G and r_G=P_G/B. Total body gain repairs most scale distortion,
but conditional comment allocation and response curvature still contribute.

The working finite approximation is now

\[
\widehat C^{(2)}_u=b_f+\frac13\sum_{v,h,G}\beta_{f,h}^T P_{u,v,h,G}(q_0)
\left[\operatorname{Cov}_G(V,L)
+\frac12 E_G[(V-E_GV)(L-E_GL)^2]\right],\quad L=q_0^TK.
\]

S17 reuses24 pilot authors and measures189 additional source authors. Internal
exact-reader correlation is.9207 on new189 and.9250 on all213; corresponding
normalized errors.3188/.3018. Source N r=.1254 versus exact.1337; paired delta
-.0083[-.0621,.0461]. This supports a finite mathematical approximation to the
relational readout, not a human prediction advantage or a universal truncation.
It still uses contextualized prefix K,V and exact masses; it is not yet a
standalone raw-text formula or a complete replacement for pretrained semantics.

The next mathematical distinction is between a decoding covector beta and an
intervention vector. A training-covariance forward pattern
a=Sigma beta/(beta^T Sigma beta) defines d_s=a beta^T d and d_n=d-d_s;
the reader sees d_s and annihilates d_n. The nonlinear suffix need not do so.
S18 tests that difference directly, retaining selected/null interaction instead
of forcing linear additivity. See `GRAIN_READER_PATTERN_S18_PLAN.md` for the
source literature, metric assumptions, finite tests and fixed intervention.

## 10. A readout-null vector can remain functionally active

S18 has now performed that intervention on24 source authors/view0. Native and
self patches replay exactly. Full pairing-removal effect RMS=.001319; Euclidean
and covariance-selected components give.0000168/.0002687. Their reader-null
components retain.001314/.001245 and correlate.99994/.98377 with full effects.
The interaction RMS is only.00124/.00289 of full effect RMS. Pattern norms differ,
so their effect-size contrast is not a norm-matched geometry test.

The useful distinction is between beta, learned to decode the body-collapse
contrast, and the actual suffix covector grad_h F for a pairing intervention.
For d_s=a beta^T d and d_n=d-d_s, beta^T d_n=0 does not imply grad_h F^T d_n=0.
S18 makes the latter response empirically visible. It does not negate S9's
source association, S12's learned target recovery, or S17's approximation.

This separates three mathematical levels: the joint KV evidence, a statistical
reader of that evidence, and the decoder's functional response operator. They
can overlap without being identical. The next exact expression is
F_u(h)-F_u(h-d)=<integral_0^1 grad F_u(h-t d)dt,d>.
S19 tests how much of that covector can be shared across authors, using a
training-only mean or active-gradient subspace rather than another ordinary
regression-weight interpretation. See `GRAIN_FUNCTIONAL_GRADIENT_S19_PLAN.md`.

## 11. The functional covector can be estimated and transported

S19 recovers the actual integrated suffix gradient on24source authors. The
path integral reconstructs their pairing effects at r=.99999977, maximum
completeness gap2.32e-6, and native gradients already give.99954. The first
uncentered gradient second-moment direction contains95.117% of energy; median
between-author cosine=.9593. This is response geometry, not human trait variance.

Using other23 authors' mean gradient gives held-author effect r=.8865,
bootstrap95%CI[.7149,.9675], error/effect-energy=.2147. Let g_ref be the mean
over a fixed reference panel. The exact finite error equation is

E_u-g_ref^T d_u=(g_bar_u-g_ref)^T d_u,

up to numerical path integration error. A shared direction is useful without
assuming that all authors have identical gradients or identical evidence.
Author-specific displacement and gradient variation remain distinct terms.

S20 freezes the24-author reference and computes z_u=mean_v(g_ref^T d_uv) from
all213 cached prefix vectors. On24 different authors, actual effect transport
has r=.9431[.9009,.9809] and error/effect-energy=.1653. On the189 authors outside
the reference panel, z has source N r=.2000,p=.0058,FisherCI[.0589,.3332].
All213 r=.1749; discovery24 r=-.0079. These exposed source results are reported
without human-label fitting, calibration or outcome-chosen sign.

The formula is therefore connected to both actual vector intervention and a
real-author source outcome. It no longer relies only on a fitted statistical
reader beta. Prediction still uses a pretrained prefix for contextual K,V;
the frozen suffix direction replaces the upper decoder at readout time.

E and C remain different interventions. z-C correlation is.2212 on primary189;
this route need not recover all of C to carry useful source association.
The primary difference versus S12's statistical-reader source r is+.0425 with
CI[-.1521,.2377], so do not turn the association into an established superiority.

The next question is query specificity: E_uq=g_bar_uq^T d_uq, compared with
crossed readers g_ref,r^T d_uq. Numeric-answer controls distinguish a shared
suffix scoring direction from trait-specific evidence retrieval or a distinct
trait covector. See `GRAIN_CROSS_QUERY_FUNCTIONAL_S21_PLAN.md`.

Sources: the internal-path completeness adaptation of
[Sundararajan et al.2017](https://proceedings.mlr.press/v70/sundararajan17a.html)
and finite sensitivity-spectrum algebra from
[Constantine & Gleich2015](https://arxiv.org/abs/1408.0545).
Reports: `S19_FUNCTIONAL_GRADIENT_REPORT.md`, `S20_FUNCTIONAL_TRANSFER_REPORT.md`.

## 12. Shared evidence does not imply the same functional reader

S21 holds numeric answers, native template and author text fixed across N,
confirmation-seeking, numeric Feeling and quantitative-content questions.
It measures12 reference and12 different transfer authors. Transfer pairing
vectors have same-author cosine .891-.983 across questions, yet own-query
functional effect reconstruction r=.931/.949/.801/.637. For N, borrowing
confirmation gives .827, Feeling -.256 and content -.072. The response reading
direction is therefore part of the mechanism, not only the retrieved evidence.
Question length is not equalized; these are query-condition comparisons.

An ordinary cosine between covectors depends on the chosen coordinates.
For the actual reference displacement distribution, define the empirical
response-function Gram matrix

\[
K_{rs}=\frac{1}{nQ}\sum_{u,q}
 \big[g_r^T(d_{uq}-\bar d_q)\big]
 \big[g_s^T(d_{uq}-\bar d_q)\big]
 =g_r^T\Sigma_D g_s.
\]

This is positive semidefinite and invariant under d'=Ad, g'=A^{-T}g for any
invertible A. It is a reference-specific L2 geometry, not a coordinate-free
claim about every text distribution. Authors, not their repeated questions,
are the independent units. Singular support is allowed; no covariance inverse
is needed. The question-conditional centering controls query mean differences
without removing within-query author variation.

S22 finds functional N/confirmation correlation .715, N/Feeling -.578 and
N/content -.379, versus native covector cosines .850/.316/.371. Thus a positive
coordinate angle need not mean positive response covariation on the observed
evidence. The first two of four function-space eigenvalues contain94.76% of
Gram trace; these are combinations of query functions, not personality factors.

Fix the content anchor c without using human outcomes. The retained decomposition

\[
\lambda_r=K_{rc}/K_{cc},\quad g_r^{\parallel}=\lambda_r g_c,
\quad g_r^{\perp}=g_r-g_r^{\parallel},\quad
z_{ur}=z_{ur}^{\parallel}+z_{ur}^{\perp}
\]

has zero pooled reference covariance between anchor and perpendicular function.
It does not guarantee orthogonality on other populations or within each query.
No channel is discarded. Own-query transfer response r for perpendicular parts
remains .925/.924/.784 (N/confirmation/Feeling), versus .072/.475/.207 for
aligned parts. This reveals distinct functional content without requiring an
arbitrary Euclidean vector-space split.

S23 uses S20's original24-author g_N, preserving its213 scores exactly, and
the S21 content anchor/reference displacement distribution. Lambda=-.4810823.
On189 nonreference authors, source N r=.2000 total, -.0344 aligned, .1946
perpendicular (p=.00730, bootstrapCI [.0529,.3298]). The covariance identity is
.0000610554=-.0000032820+.0000643374. Both signed parts remain in the formula.
The source association resides primarily in the function distinct from this
content anchor, not merely in a common numeric-answer direction. It is a source
mechanism connection, not a new prediction gain or a pure psychological axis.

The metric choice is informed by the distinction between measurement and
intervention representations in
[Park, Choe & Veitch2024](https://arxiv.org/html/2311.03658v2).
Their causal inner-product identification requires extra conceptual and
unembedding assumptions; this empirical displacement metric does not inherit
that causal interpretation. The earlier readout/pattern distinction also
follows [Haufe et al.2014](https://pubmed.ncbi.nlm.nih.gov/24239590/).
Reports S21-S23 contain full matrices, uncertainty, retained components and
source provenance. The next mainline target is the actual larger S9 C pathway,
not further polishing of the already useful small pairing response E.

## 13. The actual configuration contrast reaches the measured functional head

S24 directly measures the useful S9 C, not just the pairing effect E. On24
authors/view0, B denotes native/projected block16 body states, and H donates
their corresponding layer17 final-query head. The four final scores yield

\[
C_u=F_{11}-F_{00}=\phi_{H,u}+\phi_{B,u},\qquad
\phi_{H,u}=\tfrac12[(F_{11}-F_{10})+(F_{01}-F_{00})].
\]

The other allocation is the analogous average for B. Keep the interaction
I=F11-F10-F01+F00 explicitly. This is the exact two-player order average of the
declared computational game, as in
[Lundberg & Lee2017, Eq4](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf).
It is not a unique natural mediation estimand for every possible intervention.

Writing delta_h=h1-h0 and h(t)=h0+t*delta_h, the head allocation itself is

\[
\phi_{H,u}=\left\langle
\frac12\int_0^1[\nabla F_{1,u}(h(t))+\nabla F_{0,u}(h(t))]dt,
\Delta h_u\right\rangle.
\]

This exact differential expression averages two body backgrounds. Substituting
the frozen S20 g_N gives a concrete error term: the covector difference dotted
with delta_h. S24 finds that this approximation predicts the actual allocation
at r=.95433[.88975,.97925], MSE/allocation-energy=.022989 without further fitting.
Thus the shared functional direction transfers to a larger, source-relevant
configuration displacement, not only the small pairing direction it was built on.

Head/C r=.70647; head and other RMS=.00659190/.01165312 against C=.01350153.
Their means .00577776 and-.00313095 partly cancel. RMS ratios are not variance
fractions, and other/C r=.97978 includes an algebraic part-whole relation.
Interaction RMS=.00113006. All source endpoint and allocation replays are0.
Full runtime96forwards/139.51seconds; no new human outcomes are used.

The next practical source score is mean_v g_N^T delta_h_uv, extracted with
native/projected prefixes and no upper decoder. It must be compared with the
existing pairing score rather than blindly added to it, because both read the
same head space. S25 tests its source N connection; no new gradient grid is
justified by S24's already small head approximation error. Other-path structure
remains a separate useful mechanism, not a discarded nuisance term.
