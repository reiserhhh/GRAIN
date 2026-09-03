# GRAIN V8 Conditional Response Geometry

Status: `TECHNICAL_MEASUREMENT_CORE__PSYCHOLOGICAL_INTERPRETATION_CLOSED`

## 1. Measurement object

GRAIN does not define a person as a single text vector or a short list of
named factors. It estimates how an author selects expression opportunities
and how the author's observable response changes under shared conditions.

Let \(u\) index authors, \(o\) available opportunities, \(c\) conditions,
\(t\) occasions, and \(r\) observers:

\[
c_{ui}\sim\pi_u(c\mid o_{ui}),
\]

\[
y_{uictr}
=
\mu_{c,o}+a_u+
\left[B_0+F(\eta_u)+\varepsilon D_u\right]z_c
+Hs_{ut}+\delta_r+e_{uictr}.
\]

The terms are:

- \(\pi_u\): C1 choice structure over topic, situation, and interaction
  opportunities;
- \(a_u\): static author position after condition and opportunity control;
- \(B_u=B_0+F(\eta_u)+\varepsilon D_u\): C2 response operator under shared
  conditions;
- \(\eta_u\): position in the population geometry of response operators;
- \(D_u\): individual response deviation not represented by the population
  geometry;
- \(s_{ut}\): occasion-specific state;
- \(\delta_r\): observer effect;
- \(e_{uictr}\): remaining measurement error.

Under the 2026-07-30 foundation ontology, the anonymous L4 technical author
object is

\[
\mathfrak A_u
=
\left(
a_u,\pi_u,[B_u],R_u^{\mathrm{stable}},
U_u,E_u;\mathcal P_{\mathrm{ref}}
\right),
\]

where \(U_u\) records uncertainty and refusal, \(E_u\) binds the estimate to
auditable text evidence, and \(\mathcal P_{\mathrm{ref}}\) identifies the
reference population used by the technical procedure. Without a licensed E45
reference-scoring edge, \(\mathfrak A_u\) is an author conditional-response
structure, not an L5 person score or personality profile.

## 2. Distinct estimands

### C1: selected exposure

C1 measures which opportunities an author enters, avoids, revisits, or
creates. It is psychologically interesting but cannot substitute for
response under a common condition.

### C2: shared-condition response

C2 measures the change in observable expression when multiple authors face
identified shared conditions with adequate overlap and repetition. Private
condition coordinates, insufficient rank, or inadequate repeated support
must refuse output.

### Stable position and response

Static position \(a_u\) and conditional response \(B_u\) are not
interchangeable. An author can have a stable average expression but a weak
response operator, or a similar average expression but a distinctive
condition sensitivity.

### State and measurement error

Occasion state, observer fingerprint, and finite-text dispersion are not
individual traits merely because they produce large residuals. Stability must
be demonstrated across independent halves and observers.

## 3. Population hierarchy

The response operator is decomposed as

\[
B_u=B_0+G_{k(u)}+\varepsilon D_u.
\]

Ordinary author matching mixes within-group and between-group negatives.
Therefore every individual-structure analysis must report:

\[
A_{\mathrm{author-all}},\qquad
A_{\mathrm{author-within-group}},\qquad
A_{\mathrm{group}}.
\]

The vanishing-individuality battery showed why this distinction is necessary:
a group-only world produced author-all AUC 0.864 while within-group AUC was
0.500. High ordinary author AUC alone is not evidence of individual
structure.

After discovery-only group centering, the oracle stable residual obeys

\[
R_\varepsilon
=
\mathbb E
\left\|
B_u-\mathbb E(B_u\mid\eta_u)
\right\|_F^2-\tau^2
=
\varepsilon^2\mathbb E\|D_u\|_F^2.
\]

The planted oracle exponent was 1.999; the finite estimator exponent was
1.687. The current registered design could not promote individuality below
\(\varepsilon=0.50\). Below-threshold structure is reported as undetectable,
not absent.

## 4. Factors, modes, and topology

A traditional factor is one possible coordinate system for stable
cross-author covariance. It is not the default ontology.

A dynamic mode is a response-operator object. In a linear approximation,

\[
B_u=U_u\Lambda_uV_u^\mathsf T,
\]

where \(V_u\) gives condition directions, \(U_u\) gives response directions,
and \(\Lambda_u\) gives response strengths. If directions rotate across
samples, GRAIN reports stable subspaces and principal angles rather than
forcing a common named axis.

Population topology is

\[
\mathcal T
=
\operatorname{Topology}
\left(
\operatorname{supp}P([B_u])
\right).
\]

The topology-margin planted battery identified separated mixtures,
continuous rings, and continuous open curves at its registered low-noise
margin, while refusing ambiguous worlds. This demonstrates technical
identifiability in simulation only. It does not imply that real authors form
personality types, rings, or curves.

## 5. Dual estimation

Stable scoring and unbiased operator inference are separate tracks:

\[
S_u=\phi(\widehat B_u^{\,\lambda})
\]

uses a calibration-selected regularized estimate for repeatable scoring,
whereas

\[
\widetilde B_u
=
\operatorname{Firth/debiased\ refit}(Y,Z)
\]

is used for operator direction, effect size, intervals, and refusal.
Regularization may improve ranking while biasing inferential coefficients;
one estimator cannot silently serve both purposes.

## 6. Evidence hierarchy

GRAIN claims advance in the following order:

1. the technical object is identifiable under a declared design;
2. the object is detectable above a calibrated noise floor;
3. the object is stable across independent views and observers;
4. its population geometry is identifiable or explicitly refused;
5. a deterministic score can be produced with uncertainty;
6. external evidence supports a psychological or behavioral interpretation;
7. independent replication supports transport beyond the original reference.

Passing an earlier layer does not promote a later layer.

## 7. Role of the LLM

Deterministic GRAIN code estimates \(\mathfrak A_u\). A frozen LLM may observe
explicit behavior and render evidence-bound explanations, but it may not
invent response operators, scores, construct names, diagnoses, or validity.

The later construct mapping

\[
\Psi:\mathfrak M_u\longrightarrow
\{\text{trait, state, emotion, behavior, or clinical construct}\}
\]

starts only after an E45-licensed L5 measurement \(\mathfrak M_u\) exists and
remains closed until supported by separately designed external evidence.

## 8. Statements retired by V8

- High author-all AUC proves individual personality differences.
- Residual magnitude is noise and should be removed.
- A stable cluster direction is automatically a personality factor.
- An unnamed dynamic axis is automatically an unknown personality dimension.
- A discrete group, ring, or curve is a psychological type.
- Cross-half or cross-corpus stability alone establishes trait status.
- One regularized estimator can provide both stable scores and unbiased
  operator inference.
- Unsupervised text geometry is sufficient for psychological validity.

## 9. Current boundary

GRAIN is a method for estimating author choice structure, conditional response
operators, population geometry, and stable individual deviation from text.
It supplies a technical foundation from which future natural-language
assessment instruments may be built. It is not itself a finished personality
scale, diagnostic system, or clinically validated assessment.

## 10. Response-set incidence extension

An author response operator also defines an image and a lifted graph:

\[
S_u=f_u(\mathcal Z),\qquad
\Gamma_u=\{(z,f_u(z)):z\in\mathcal Z\}.
\]

This creates three noninterchangeable relation levels:

\[
d_{uv}^{=}
=
\inf_z\|f_u(z)-f_v(z)\|,
\]

\[
d_{uv}^{\mathrm{free}}
=
\inf_{z,z'}\|f_u(z)-f_v(z')\|,
\]

and distribution-weighted whole-map discrepancy:

\[
d_{\mathrm{map}}^2(u,v)
=
\mathbb E_{z\sim P_Z}
\left[
\Delta_{uv}(z)^\mathsf T
\Sigma_\Delta(z)^{-1}
\Delta_{uv}(z)
\right].
\]

The first asks whether two authors reach the same response under the same
condition. The second permits different conditions and therefore cannot by
itself define a shared conditional response. The third asks whether their
response maps are broadly similar.

The affine V1 planted battery technically distinguished the first two at its
registered low-noise margin. It also demonstrated that a common condition
anchor can create a complete raw incidence graph without pair-specific map
equivalence. This supports condition-indexed relation graphs while rejecting
the inference that geometric intersection alone means psychological affinity.

For future nonlinear work, exact edges are replaced by persistent uncertainty
tubes:

\[
R_{uv}(z,\epsilon)
=
\mathbf 1
\left[
\|\Delta_{uv}(z)\|_{\Sigma_\Delta(z)^{-1}}\le\epsilon
\right].
\]

A candidate pair-specific relation must persist across radii and independent
views and exceed a condition-matched random-pair baseline.

The subsequent V2.3 planted battery verified the finite-dimensional
persistence and multiplicity object at its registered low-noise margin. A
population relation is represented as a hypergraph, not as a binary
intersection graph:

\[
\mathcal H_{\epsilon,z}
=
\left\{
H:
\bigcap_{u\in H}T_u(z,\epsilon)\ne\varnothing
\right\}.
\]

Raw multiplicity is not monotone evidence of author similarity. A
population-wide common anchor had median raw multiplicity 24 and zero group
claims, while planted response groups had recovered multiplicity 6 and
median F1/ARI 1.0. All registered null worlds had 0/160 claims separately,
and isolated transverse contacts also had 0/160 group claims. The technical
object is therefore persistent hyperedge identity plus scale, view,
specificity, and population-coverage structure. V2.3 records tangent
compatibility as a diagnostic but does not use it as a group-decision gate;
its cross-view gate is population-level pair-score Spearman rather than
per-hyperedge identity replication.

Decision: `V8_INCIDENCE_MULTIPLICITY_PLANTED_PASS`. This establishes
recoverability of anonymous planted response mechanisms only. Real-text
hyperedge stability, unsupervised threshold selection, arbitrary-manifold
coverage, and psychological interpretation remain open. See
`reports/V8_INCIDENCE_MULTIPLICITY_REPORT.md`. Independent methodology
decision is `PARTIAL` because whole-map distance also separates every clear
world perfectly and complete hyperedge/chaining controls remain open; see
`reports/V8_INCIDENCE_MULTIPLICITY_INDEPENDENT_AUDIT.md`.

V3 neutralized that competing whole-map signal by constructing positive and
counterfactual flattened paths with the same centered author Gram matrix:

\[
HF^+F^{+\mathsf T}H
=
HF^-F^{-\mathsf T}H
=
\gamma H.
\]

Integrated whole-map Euclidean distance was consequently non-discriminative,
while same-condition local recurrence recovered the planted groups. The
registered contrasts separately rejected asynchronous visits to the same
anchor, transitive chaining of overlapping relations, and rotating
high-multiplicity membership.

V3.1 corrected the persistent-core estimand. For author set \(A\), support is
now aligned within the same condition-radius cell across every independent
view:

\[
P_\cap(A)
=
\frac1{|Z||E|}
\sum_{z,\epsilon}
\min_v
\max_{\substack{G\in\mathcal H^{\max}_{v,z,\epsilon}\\A\subseteq G}}
\frac{n-|G|}{n-2}.
\]

This support is anti-monotone under inclusion. A threshold-complete Apriori
search from all triples can therefore recover a stable subset hidden inside
different maximal hyperedges without materializing the full intersection
lattice.

All three positive worlds had incidence AUC/F1/ARI 1.0 over 200 confirmation
populations each, while each registered counterfactual and null had 0/200
group claims. Independent decision:
`PASS_TECHNICAL_PLANTED_ONLY`.

This does not make an intersection a personality signal. Minimum
same-condition distance and a soft local kernel also solved the strongly
separated positive worlds. The current information-bearing mathematical
candidate is the identity and recurrence of a simultaneous conditional
response set, not its raw multiplicity and not an exact coordinate. Its
possible psychological interpretation is “shared response mechanism under
comparable conditions,” which remains unvalidated in human text. See
`reports/V8_INCIDENCE_INCREMENTAL_V31_REPORT.md`.

### 10.1 Condition index versus aggregate pair structure

V3.2 planted a balanced strong chain in which every author pair recurred as
often as in a true simultaneous six-author core, but only overlapping
triplets ever co-occurred. Aggregating over conditions,

\[
W_{ij}
=
\frac1{|Z||E|}
\sum_{z,\epsilon} A_{ijz\epsilon},
\]

made the two worlds indistinguishable. V3.3 then gave the comparators the
unaggregated tensor \(A_{ijz\epsilon}\). Persistent maximal cliques,
condition-aware complete link, and an aggregate-component concurrency gate
all recovered the core and rejected the chain in 500/500 paired
populations.

The accepted result is an aggregation theorem:

\[
A_{ijz\epsilon}\longmapsto W_{ij}
\]

is many-to-one and can erase simultaneous-set structure. It is not evidence
that a hypergraph is always superior to a condition-aware pairwise method.

### 10.2 Pairwise intersection versus a common intersection

For one condition and view, assign each author response point
\(x_{u,z,v}\) a closed uncertainty ball

\[
\mathcal B_{u,z,\epsilon,v}
=
\left\{q:
\lVert q-x_{u,z,v}\rVert_2
\le \epsilon R
\right\}.
\]

Pairwise adjacency records

\[
A_{ijz\epsilon v}
=
\mathbf 1
\left[
\mathcal B_{i,z,\epsilon,v}
\cap
\mathcal B_{j,z,\epsilon,v}
\ne\varnothing
\right]
=
\mathbf 1
\left[
\lVert x_i-x_j\rVert_2\le2\epsilon R
\right].
\]

A \(k\)-author common intersection is the stronger statement

\[
C_{G,z,\epsilon,v}
=
\mathbf 1
\left[
\bigcap_{u\in G}
\mathcal B_{u,z,\epsilon,v}
\ne\varnothing
\right]
=
\mathbf 1
\left[
\min_q\max_{u\in G}
\lVert x_u-q\rVert_2
\le\epsilon R
\right].
\]

Thus a pairwise clique is a Vietoris-Rips simplex, while a verified common
ball is a Cech simplex. In two dimensions, pairwise intersection alone is
insufficient; Helly's theorem shows why triple-wise common intersection is
the decisive local order for convex balls.

V3.4 held the complete per-view tensor \(A_{ijz\epsilon v}\) identical while
changing \(C_G\). The positive world used six points with
minimum-enclosing-ball radius \(0.90R\). The negative world duplicated the
vertices of an equilateral triangle with circumradius \(1.10R\); every pair
remained closer than \(2R\), but the six-author common intersection was
empty through the largest registered radius \(1.06R\).

Across 500 paired confirmations:

- per-view and view-intersected adjacency mismatches were zero;
- all three pairwise methods returned identical outputs between worlds;
- feasible six-author cores were recovered 500/500;
- infeasible six-author claims were 0/500;
- the infeasible geometry instead exposed overlapping four-author
  substructures and was ambiguity-refused 500/500.

The result identifies a mathematical object beyond finite binary pair
adjacency:

\[
\text{persistent same-condition common-ball membership}.
\]

It does not identify a psychological construct. A full continuous Euclidean
distance matrix determines minimum-enclosing-ball feasibility, so the result
does not exceed complete distance geometry. In real text, a recurring common
ball can still arise from prompt design, topic, platform, language,
demographics, opportunity structure, or measurement artifacts. Only
condition-matched replication plus external psychological or behavioral
evidence can promote it to a personality-similarity claim.

### 10.3 Continuous scale, matched null, and local intersection type

V3.5 replaced finite radius-hit counting with the exact cross-view birth
scale

\[
b^\cap_{H,z}
=
\max_v
\frac{\rho_{\mathrm{MEB}}(X_{H,z,v})}{r_{z,v}}.
\]

Its survival over the registered interval is integrated continuously and
compared with a condition/opportunity-restricted maximum-statistic
permutation null. This separates three questions that earlier versions had
mixed:

1. at what uncertainty radius does a common response set become feasible;
2. whether its repeated membership exceeds condition-matched chance;
3. what local relation the underlying paths or sheets have at contact.

For two same-condition response maps \(f_u,f_v\), local transversality is
identified from

\[
Dg_{uv}(z)=J_u(z)-J_v(z),
\qquad
\dim I_{\mathrm{same}}
=
\dim\mathcal Z-\operatorname{rank}Dg_{uv}(z).
\]

When first-order rank is insufficient, the normal Hessian difference
distinguishes the registered tangent template from the coincident template;
a grey rank interval is refused. V3.5 recovered all registered low-noise
curve and nonlinear-sheet templates and passed its matched-null controls.
Independent audit limits this to local jet-template classification. The
nonlinear AUC reuses the transversality rank score, and the reported
intersection dimension is assigned from the classified template rather than
independently recovered from an unknown manifold. It does not imply that
real-text trajectories are smooth manifolds or that their intersections carry
psychological meaning.

### 10.4 Spacetime junctions and routing kernels

The number of authors sharing a junction and the number of available
directions at that junction are different quantities:

- author multiplicity \(m(J)\) counts trajectories participating in a local
  common-response set;
- branch valence \(K(J)\) counts distinguishable incoming/outgoing tangent
  classes;
- the transition kernel \(\Pi_J\) describes which outgoing class is selected.

A compact junction signature is therefore

\[
\mathcal J
=
\left(
m,\ b^\cap,\ \dim I,\ \Theta,\ K,\ \Pi
\right),
\]

where \(\Theta\) records transverse/tangent geometry. No one component alone
defines similarity, personality, or cognition.

V3.6 lifts repeated spatial nodes into

\[
d_\lambda((x,t),(y,s))^2
=
d_X(x,y)^2+\lambda^2(t-s)^2
\]

and preserves causal time order. At each junction it estimates

\[
G_C
=
\frac{I(B^+;C\mid B^-,G,J)}{\log K},
\qquad
G_T
=
\frac{I(B^+;B^-\mid C,G,J)}{\log K},
\]

\[
H_{\mathrm{res}}
=
\frac{H(B^+\mid B^-,C,G,J)}{\log K}.
\]

The three registered worlds share static geometry and branch marginals:
pass-through has high \(G_T\), random branching has high
\(H_{\mathrm{res}}\), and cue-guided routing has high \(G_C\). Static
marginals were non-discriminative, while event-level correspondence
separated the worlds. Explicit time distinguished deliberately repeated
stages that spatial coordinates alone collapsed; this is a construction
sanity check, not latent-time discovery.

The result narrows the possible interpretation of an intersection. Geometry
defines available continuation, the transition kernel defines selection,
and an external reward defines effectiveness. Random and pass-through worlds
show that the registered common ball is not sufficient for cue-sensitive
routing; a cue-guided near miss shows that this radius-specific common-ball
criterion is not necessary. It does not exclude every mathematical form of
intersection. The frozen tree-capacity diagnostic also assumes a pooled
stage-independent channel, so it is not a general tree-search capacity. The
supported object is a **spacetime routing junction in the registered DGP**.
"Thought node" requires real-text recurrence and interpretable cues;
"intelligence" additionally requires goal success, cost, and novel-cue
generalization; "personality" requires author-specific kernels stable across
contexts and linked to independent evidence.

The V3.6.1 post-seal scope correction adds a statistical warning. Complete
three-stage paths have 27 categories; direct plug-in conditional mutual
information is strongly upward biased at the registered sample size. In the
random world, raw conditional path MI was 0.411 and its matched permutation
null was also 0.411, leaving adjusted MI -0.0001. Cue-guided routing retained
adjusted MI 0.451 and composed over held-out cue-sequence combinations;
pass-through retained perfect path prediction from incoming direction with
zero cue MI. Path entropy therefore measures route diversity, not control.
Bias-corrected conditional information and held-out local-operator
composition are required before any routing interpretation.

### 10.5 Author-specific stochastic routing operators

V3.7A moved from pooled routing worlds to an author-relative probability
operator. For a frozen reference opportunity design \(Q\), the synthetic
estimand is

\[
D_u
=
\operatorname{ilr}(\Pi_u^Q)
-
\operatorname{ilr}(\Pi_{\mathrm{reference}}^Q).
\]

This is not a named factor. It is a multivariate deviation in the simplex
geometry of conditional route selection. The registered primary world
planted \(D_u\) in a fresh hidden rank-6 Haar subspace for every population.
The estimator received neither that basis nor author links during fitting.

Across 200 sealed confirmation populations, the recovered operator had truth
correlation 0.862, split-session reliability 0.650, unseen-context
reliability 0.824, and within-group author AUC 0.963. A group-only world
raised ordinary author AUC to 0.848 while its within-group AUC remained 0.500.
Thus population grouping can imitate identity unless scoring conditions on
group structure.

The result supports a **compressible stochastic response operator** under the
registered synthetic design. It does not establish that the latent rank is
exactly six: ranks 6 and 8 were nearly tied in discovery likelihood. The
rank-12 stress failure marks a capacity boundary for the estimator and event
budget, not absence of an author-specific operator.

The independent audit also retired three stronger readings:

- direct noise-energy subtraction produced negative or above-one variance
  shares, so source-specific variance attribution is not closed;
- the reported coverage belongs to an empirical projection-residual band,
  not a derived single-author confidence interval;
- the estimator used a known four-component mixture count and the scorer used
  true groups for centering, so group-free recovery is not established.

The 48D contrast should likewise be called a registered budget/reliability
threshold. At low budget, reliability failed even though truth correlation
and author matching remained nontrivial; this is not an information-theoretic
impossibility theorem.

The current technical object is therefore

\[
\mathfrak R_u^Q
=
\left(
\widehat D_u,\widehat{\mathcal S},
\operatorname{Rel}_{\mathrm{split}},
\operatorname{Rel}_{\mathrm{unseen}},
U_u
\right),
\]

where \(\widehat{\mathcal S}\) is a learned compressible subspace and \(U_u\)
must remain empirical until a valid hierarchical uncertainty model is
provided. Personality, cognition, intelligence, and clinical meaning remain
closed.

### 10.6 Blind event localization

V3.7B removed the oracle transition window. A geometry-and-order-only locator
searched for the registered conjunction of a local speed dip and a cusp
component perpendicular to the incoming tangent:

\[
L_t
=
\sqrt{
\left[
1-\frac{v_t}{\operatorname{med}(v_{\mathrm{flank}})}
\right]_+
\cdot
\left\|
(I-\widehat d_t\widehat d_t^\mathsf T)
\Delta^2 x_t
\right\|/
\operatorname{med}(v_{\mathrm{flank}})
}.
\]

The threshold was selected on discovery paths and frozen before canonical
confirmation. In the registered synthetic family, localization F1 was
0.9996, P95 location error was one sample, and false detections were 0.0025
per 1000 negatives. The resulting blind operator correlated 0.959 with the
oracle-window operator and retained essentially all registered predictive
gain.

This is a family-specific identification result. The generator explicitly
plants the speed-dip/perpendicular-cusp conjunction that the locator tests.
Smooth, bend-only, pause-only, cue-bend, near-crossing, and speed-wave
controls show that neither named component alone was sufficient in this
battery, but they do not constitute an open-world set of alternative
mechanisms.

V3.7B also introduced equal-context, mask-aware aggregation. Missing local
cells are omitted and complete packet-cell non-overlap is refused rather than
filled. Canonical missingness was nearly zero, so robustness to
missing-not-at-random localization remains untested.

The supported progression is:

\[
\text{registered trajectory signature}
\rightarrow
\text{blind localized event}
\rightarrow
\text{conditional route count}
\rightarrow
\widehat D_u.
\]

No reverse implication is licensed. A recovered \(\widehat D_u\) does not
prove that the localized event is a thought node, and a localized event does
not prove personality, intelligence, planning, or clinical relevance.

The independent audit adds four mathematical restrictions. First,
blind-versus-oracle correlation on relocalized copies of the same events is a
retention measure with shared sampling error, not an independent replication.
Second, available-case aggregation estimates a common \(Q\)-profile only
under ignorable missingness:

\[
M_{ucik}\perp Y_{ucik}
\mid
(u,c,i,k,\text{registered difficulty}).
\]

When localization failure depends on outcome or author, authors are averaged
over different effective \(Q\) designs. Common-support or
inverse-probability weighting is then required. Third, localization error
must be scored over every true event, with misses represented explicitly;
conditioning P95 on error already being at most the gate is circular.
Fourth, a matched generator-detector pair establishes model-family
identifiability only. General localization requires held-out mechanism
families and joint hard negatives.

### 10.7 Group-free recovery and the identity-crowding boundary

V3.7C replaced true-group centering with a group-free cross-session estimator.
For split-session matrices \(X\) and \(Y\), it estimates

\[
\widehat C
=
\frac{
(X-\bar X)^\mathsf T(Y-\bar Y)
+
(Y-\bar Y)^\mathsf T(X-\bar X)
}{2(n-1)},
\]

truncates negative eigenvalues, and applies a spectral-Wiener weight in the
selected stable subspace. Candidate rank and author shrinkage are chosen on
discovery-context log loss; true groups open only in a post-primary
sensitivity score.

The sealed canonical experiment passed 20/20 gates. Core truth correlation
was 0.963, independent-oracle correlation 0.935, split-session reliability
0.740, unseen-context reliability 0.936, and hard-neighbor author AUC 0.932.
The 32-event capacity arm fell to truth correlation 0.526 and cross-context
reliability 0.363. The registered out-of-family and high-noise mechanisms
retained operator truth correlation near 0.96 while locator F1 declined to
0.984 and 0.962.

The central theoretical correction is that structural recovery and author
identification have different margins. Define

\[
\epsilon_u=\|\widehat D_u-D_u\|,
\qquad
m_u=\min_{v\ne u}\|D_u-D_v\|.
\]

Accurate operator recovery requires small \(\epsilon_u\). Stable identity
resolution additionally requires separation, approximately

\[
2\epsilon_u<m_u.
\]

Across planted ranks 2/4/6/8, truth correlation remained
0.967/0.958/0.965/0.960, but hard-neighbor AUC was
0.777/0.961/0.992/0.998. Rank-2 split reliability was 0.797, so the weaker
identity result is not explained by unstable measurement. It is consistent
with geometric crowding: many authors can occupy a reliably measured
low-dimensional region while having insufficient nearest-neighbor margin.

V8 therefore distinguishes three empirical properties:

\[
\text{operator recovery}
\ne
\text{repeated-measurement reliability}
\ne
\text{individual identifiability}.
\]

Under the registered MAR mechanism, common-reference AIPW reduced excess
NRMSE from 0.103 for available cases to 0.021 and improved all 80 paired
populations. MNAR remained unidentified: sensitivity coverage 0.919 required
a wide mean envelope of 0.941.

The next mathematical object is a pair of finite-sample critical surfaces,

\[
N_{\rm recover}^{*}(K,\mathrm{SNR})
\quad\text{and}\quad
N_{\rm identify}^{*}(K,\rho_{\rm author},\mathrm{SNR}),
\]

where \(K\) is operator dimension and \(\rho_{\rm author}\) is local author
density. V3.7C licenses only this synthetic finite-sample distinction. It
does not identify personality, intelligence, thought nodes, clinical states,
or real-text junctions.

### 10.8 Crossing recovery and identity phase surfaces

V3.7D tested the distinction prospectively on entirely new authors. Each
fresh latent world used 96 authors to fit the reference router and stable
basis, then evaluated nested sets of 32, 96, or 192 disjoint authors. The
registered joint states were

\[
R=\mathbf 1(r_{\rm truth}\ge .80,\ R_{\rm split}\ge .50),
\qquad
I=\mathbf 1(\mathrm{AUC}_{\rm hard}\ge .80,\ \mathrm{Top1}\ge .50).
\]

All 13 sealed gates passed. At 128 events and rank 2, recovery without
identity occurred in every 96- and 192-author population. Truth correlation
was approximately .98, while hard-neighbor AUC fell from .892 at 32 authors
to .605 at 192 authors. The paired loss was

\[
\Delta_{\rho} \mathrm{AUC}_{K=2}
=
.288\ [.281,.294].
\]

The corresponding losses were .104 at rank 4 and .022 at rank 8, producing
a registered dimension-by-density interaction of .266 [.259, .273].
Random-neighbor AUC changed by only .006. Thus the density effect concerns
the local hard-negative margin rather than generic same-author
discrimination.

For a regular local \(K\)-dimensional author distribution, nearest-neighbor
distance contracts approximately as

\[
m_u(U)=O\!\left(U^{-1/K}\right).
\]

Increasing \(U\) can therefore destroy identity resolution without changing
the recovered population operator surface.

The reverse separation also occurred. At true rank 12, a frozen rank-8
estimator retained hard-neighbor AUC from .935 to .986 but had truth
correlation near .77. A true-rank sensitivity restored mean truth
correlation to .940. If \(P_r\) is the fitted rank-\(r\) projection, then

\[
\|D_u-\widehat D_u\|^2
=
\|(I-P_r)D_u\|^2
+
\|P_rD_u-\widehat D_u\|^2,
\]

when \(\widehat D_u\in\operatorname{range}(P_r)\). The retained coordinates
can preserve enough author separation for matching even while omitted
coordinates prevent complete operator recovery.

The resulting finite-sample measurement state is not a ladder:

\[
(R,I)\in\{(0,0),(1,0),(0,1),(1,1)\}.
\]

This is a stronger correction than the one-way V3.7C crowding observation.
Recovery, reliability, and identification are distinct estimands with
crossing critical surfaces. The confirmation used fixed total signal,
synthetic categorical routing, a fixed rank-8 primary estimator, and 40
worlds per cell. It does not establish a universal density law, an adaptive
unknown-rank solution, a real-text dynamic object, or psychological meaning.

### 10.9 Adaptive capacity and fixed-reference contrasts

V3.7E replaced the fixed-rank assumption with an author-disjoint
cross-session selector. For candidate projection \(W_r\), calibration risk
was

\[
L_r
=
\frac{
\|Y-c-W_r(X-c)\|_F^2
+
\|X-c-W_r(Y-c)\|_F^2
}{
\|Y-c\|_F^2+\|X-c\|_F^2
},
\]

and the smallest rank within one standard error of minimum risk was selected.
The true rank, evaluation authors, truth profile, and identity outcomes were
unavailable to this rule.

All 16 sealed gates passed. Under a flat spectrum, ranks 4/8/12/16 were
selected exactly in all 60 worlds at both event budgets. At true rank 12,
fixed rank8 produced truth correlation .783 and identity without recovery in
.933 of worlds. Adaptive rank produced truth correlation .952, matched the
oracle, and changed every world to recovery plus identity.

The V3.7D error decomposition can therefore be sharpened:

\[
\|D_u-\widehat D_u\|^2
=
\underbrace{\|(I-P_{\hat r})D_u\|^2}_{\text{capacity bias}}
+
\underbrace{\|P_{\hat r}D_u-\widehat D_u\|^2}_{\text{sampling error}}.
\]

V3.7D's rank-12 reverse state was generated by the first term. The low-rank
crowding result is different: it is governed by the local author margin and
is not repaired by selecting a larger rank. Recovery and identification
remain logically non-equivalent, but the registered reverse example is now
classified as estimator misspecification rather than a persistent population
phase.

The fixed-reference arm defines

\[
D_u^{(0)}
=
\operatorname{ilr}(\Pi_u^{Q_0})
-
\operatorname{ilr}(\Pi_{P_0}^{Q_0}).
\]

With a planted \(B_u z_c\) interaction, equal-\(Q_0\) standardization reduced
eight-panel opportunity drift by .686 [.669, .703]. A planted population
movement was recovered with direction cosine .998 and projected amplitude
1.006. These are contrast results. The reference router is external, but the
stable projection center is estimated on calibration authors, so the absolute
score origin is not yet a fully external norm. Single-author uncertainty and
minimum detectable change also remain open.

### 10.10 External origin, unresolved channels, and error limits

V3.7F separated four author-disjoint roles: router reference, zero/norm
reference, calibration, and evaluation. The absolute origin is now

\[
\hat z_0
=
\frac{1}{2|R_Z|}
\sum_{i\in R_Z}(x_{i1}^{Q_0}+x_{i2}^{Q_0}),
\]

and calibration may estimate a stable operator around \(\hat z_0\) but may
not redefine it. Evaluation-cohort composition changed a fixed score by
exactly zero in the sealed implementation.

The hard unresolved channel is

\[
R_{us}=(I-\hat P_{\hat r})(x_{us}-\hat z_0).
\]

It is not named noise before cross-session and permutation tests. In the
exact-rank scorer control, residual AUC was .498 [.486, .511]. In a
power-law world with nonzero stable energy in all 48 directions, finite-budget
residual AUC was .863 [.844, .882]. Thus a low-energy distributed signal can
survive outside a selected stable subspace.

The nested score is \(T_u(R,K,E)\), where \(R\) resamples external reference
authors, \(K\) resamples calibration pairs and reselects rank/subspace, and
\(E\) resamples target events. Event-only and full held-out synthetic Monte
Carlo coverage were .944 and .952. These are oracle-DGP calibration results,
not observable-only deployment intervals.

Across event budgets,

\[
\operatorname{MSE}(m)=F_\infty+A m^{-\alpha}
\]

gave an exact-rank event-floor estimate .0035 [.0022, .0049], practically
near zero under the registered threshold. A correlated state-alias world
retained an infinite-event floor .243 [.239, .248]. Therefore the supported
statement is:

\[
\text{event uncertainty}\to 0
\quad\not\Rightarrow\quad
\text{total uncertainty}\to 0.
\]

More generally, \(I(X_j;A_u\mid C)>0\) for every synthetic direction does
not imply
\(\operatorname{Var}(\Theta_u\mid X,C)=0\). V8 consequently replaces
deletion-oriented denoising with information-preserving reliability
weighting. The unresolved channel remains available for future estimators
and cannot be interpreted as personality without external evidence.

### 10.11 Reliability spectrum and resolution filtration

V3.7A-F jointly suggest that a fixed `factor + residual` representation is
too coarse. The next candidate object is a family of externally normed
measurements,

\[
\mathcal M_u(m)
=
\left(
\widehat G_u^{(m)},
\widehat{\mathcal U}_u^{(m)};
P_0,Q_0
\right),
\]

indexed by event budget \(m\). Stable cross-session covariance and event
variation define a generalized reliability spectrum. Directions receive
continuous evidence-dependent weights rather than permanent signal/noise
labels.

Under the ideal conditional-expectation definition

\[
M_u(m)=\mathbb E[G_u\mid\mathcal F_{u,m},P_0,Q_0],
\]

the score process is coherent across nested event budgets and mean-square
error cannot increase with additional evidence. These are theorem-level
properties of the ideal object, not established properties of the current
plug-in estimator.

This upgrade also separates mathematical object types. Static position is a
vector, dynamic response is an operator or transition kernel, hybrid
structure is a coupling object, and the residual is an unresolved channel.
They should not be forced into one interchangeable factor catalog.

The complete candidate theory and its V3.7G falsification protocol are in
`docs/V8_RELIABILITY_SPECTRUM_RESOLUTION_FILTRATION_V37G.md`.

V3.7G uncertainty is represented by a model-assisted 95% content / 95%
confidence measurement-error tolerance ball, not by an axis-wise confidence
ellipsoid. At the registered \(p=48,n_f=192\) resolution, axis-specific
deconvolution was empirically unstable; only the mapped radial error
distribution supported a defensible tolerance statement. This uncertainty
object remains technical author-response geometry and carries no personality
interpretation.
