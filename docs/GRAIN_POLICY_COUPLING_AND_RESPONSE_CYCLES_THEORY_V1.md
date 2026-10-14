# Qualified evidence, policy coupling and author response coordinates

2026-10-10. HM118 actual finite construction and result-after cycle companion.
Original 18 protocols /100 cards /44 computational requirements remain open.
This is a construction of a measurement interface, not new personality factors.

## 1. What this adds to the existing theory

HM91 already retains relation pairing, HM102 separates author and partition
mass, HM109 includes response observations and HM112 compiles a joint reader.
HM117 distinguishes stable native person contrasts from a focus-dependent
physical intervention. Those results are not newly claimed here.

HM118 connects that distinction to author scoring: identical own-actor evidence
and identical focus counts within every condition still yield different readings
when different evidence is assigned to the same observation focus. Consequently
neither an event histogram nor a focus histogram defines the scored object alone.

The computational object is a qualified joint ledger plus a response kernel:

\[
S_{u,q}^{\nu}=\sum_{e,r}J_u(e,r)K_q(e,r),\qquad
J_u(e,r)=P_u(e)\nu(r\mid e).
\]

The law is defined only for the registered finite support. Kernels for unobserved
expressions have not been inferred by this experiment. Native readings and each
actual intervention-response profile are different objects and are kept separate.

## 2. Pairing coordinates, without fitting an axis

Let g index six behavior families times two conditions. Within g, b denotes
low/high behavior polarity and r denotes focus B/A. For its count table C_g,
row margins n_g, column margins m_g and group total N_g, define

\[
I_g=\frac{n_gm_g^\top}{N_g},\qquad
\lambda_g=C_g(0,0)-I_g(0,0),\qquad
D=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
\]

Then C_g=I_g+lambda_g D. Proof: C_g-I_g has zero row and column sums, so once
its (0,0) entry is fixed the other three entries must be its signed copies.
For N_g=0, the group is explicitly unobserved. It contributes zero mass, not a
fabricated conditional score. There is one coordinate per complete 2x2 support,
not one psychologically discovered factor per group. Twelve groups were imposed
by this design, not selected by factor analysis or predictive performance.

Define each target's rectangular response from actual registered readings:

\[
\Gamma_{g,q}=K_q(g,0,0)-K_q(g,0,1)
             -K_q(g,1,0)+K_q(g,1,1).
\]

Writing N=sum_g N_g gives a usable contraction

\[
S_{u,q}=S_{u,q}^{\mathrm{independent}}
        +\frac1N\sum_g\lambda_{u,g}\Gamma_{g,q}.
\]

The coordinates depend on the observed pairing, not on a personality query.
Gamma depends on the query and observation kernel. Thus a single qualified
coordinate log can be used with all registered targets without relabeling authors
or learning a new scalar predictor. This does not claim a new unseen-text encoder.

The independent table retains both marginal laws. It is NOT the balanced common
reference: if the focus proportions differ from 1/2, their contribution remains.
The exact account is balanced reference + focus-margin contribution + pairing.
No context, state, topic selection or author level is automatically centered out.

## 3. Identification rather than fidelity

Nonnegativity gives the exact feasible interval

\[
\ell_g=\max(0,n_{g,0}+m_{g,0}-N_g)-I_g(0,0),\qquad
h_g=\min(n_{g,0},m_{g,0})-I_g(0,0).
\]

For a nondegenerate fixed-margin fiber, scalar readings are determined by the
margins alone exactly when Gamma=0. In that case the kernel is additive in b/r.
When Gamma is nonzero, two feasible coordinates have different readings, even
if every original endpoint was read without error. This is an information
requirement, not a demand for an ever more accurate classifier.

For a zero row or column, the interval may collapse even when Gamma is nonzero;
the nondegeneracy condition matters. Across independent groups, sharp scalar
bounds are

\[
S_q^{\min}=S_q^{\mathrm{independent}}+
 \frac1N\sum_g\min(\ell_g\Gamma_{g,q},h_g\Gamma_{g,q}),
\quad
S_q^{\max}=S_q^{\mathrm{independent}}+
 \frac1N\sum_g\max(\ell_g\Gamma_{g,q},h_g\Gamma_{g,q}).
\]

Both endpoints have explicit feasible count-table witnesses. They are exact
finite identification bounds, not bootstrap/Fisher intervals or psychological
calibration. For integer margins their scalar extrema are also attained by
integer count tables. Each target may require a different witness.

## 4. Joint target geometry

For one object and one output unit, all six targets jointly form

\[
\mathcal Z_u=S_u^{\mathrm{independent}}+
 \sum_g[\ell_g,h_g]\,\Gamma_g/N.
\]

Under continuous mass this is a zonotope: a Minkowski sum of line segments.
For integer count tables it is the convex hull of the discrete attainable set.
The scalar intervals above are projections of this joint set, not evidence
that all six upper endpoints occur for the same policy. The unit test with two
opposite kernels demonstrates the distinction: both marginal maxima are +1,
but their common-policy scores always sum to zero.

This geometry offers a joint response object without inventing one universal
personality direction. Its generators are observed instrument responses, not
six named latent human traits. No optimal transport distance is introduced.

## 5. Actual finite results

HM118 uses 18 distinct simulated evidence ledgers repeated across two instrument
aliases, yielding 36 scoring instances of 512 events. They are not independent
human subjects. All five policies, five objects, six targets and four units are
reported. Author ownership and queried author agree in every admitted kernel.

Positive/negative coupled policies have identical event margins and identical
focus margins in each of 12 groups. Their natural score contrast has RMS
.004592 for alias A and .002640 for alias B. Fixed-margin identification widths
have RMS .012344 / .015601, with maxima .027140 / .033366. The log-odds widths
have RMS .067119 / .090220. These are model reading units, not personality r.

The comparison table keeps three designed contrasts. Under independently
unknown couplings but known balanced margins, the direction of the natural
score contrast is identified in 18/18 A and 15/18 B level-contrast cells; only
1/18 A and 0/18 B slope-vs-flat cells; 7/18 A and 8/18 B choice-composition cells.
Each count is three seeds times six registered targets, not independent people.
This shows why marginal logs can obscure condition-response comparisons more
than a large level contrast on these materials. It is not a universal ranking
of psychological measurement sensitivities.

Complete joint logs remove this particular ambiguity. Changing the provenance
flag from instrument to author leaves numbers unchanged but changes their
interpretation. The author-origin copy is only an interface test: actual focus
in HM117 was supplied by the experimenter. Real author choices must not be
deleted as instrument error merely because they occupy the same algebraic term.

## 6. Verification and implementation

Core: grain_research/construction/math_policy_coupling.py and math_policy_cycles.py.
Runner: scripts/readback_grain_policy_cycles_hm118.py run / verify.
Canonical outputs: results/grain_math_response_v1/policy_coupling_r1 and its
cycles_readback subdirectory. All original HM118/HM117 seals remain intact.

The companion reconstructs 21600 complete readings and verifies 43200 per-output
endpoint witnesses by independently constructing their four count cells. Maximum
account error is 1.63e-15; count/margin/endpoint-feasibility errors are zero.
47 focused tests pass (13 new companion tests), Ruff passes. No new neural call,
model fit or private reading occurred. The upstream kernels still come from
Qwen observations; this is not full removal of LLM dependence. The verifier is
the same agent's alternate computation, not independent third-party replication.

## 7. Mathematical ancestry and next scientific decision

Fixed-margin coupling polytopes and their cycle perturbations are classical;
see Peyre/Cuturi, Computational Optimal Transport, equation (2.10), section
3.4.1 / Proposition 3.4 and section 3.5.2:
https://arxiv.org/html/1803.00567v4 . This work uses the 2x2 specialization,
not an optimal transport solver or a new theorem.

OpenAI/math 03b-isotypic at adc7f1241b42e322a6451854ab7e4b4c146bf78a supplied
component-retention inspiration only. Signed-density, compact-group and angle
theorems have not been transferred to this neural observer.

The addition is an executable author-evidence/observer-policy interface and its
finite consequences. The next missing link is constructing conditional Gamma
responses for new qualified expressions from source-only vector operations,
then testing whether they preserve author comparisons and multi-target effects.
Inventory HM112's compiler and HM115's changed-expression data first. Do not
repeat fixed-direction failure, tune strengths/head/rank, or call lower numeric
reconstruction error a new discovery. This question is registered, not executed
in the present closeout; the full original goal remains active.
