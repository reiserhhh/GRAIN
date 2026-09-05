# GRAIN Development and Theory Route Index

Opened: 2026-07-30; historical route through 2026-08-22; next-program links added 2026-09-05
Status: active development map; not a release seal or a scientific promotion

Current vector research program (in progress): the
[Project Persona-to-GRAIN retrospective](GRAIN_PROJECT_RETROSPECTIVE_20260905.md)
reconciles historical results and the scope of the completed cycle. The
[LLM vector inference mechanism plan](GRAIN_VECTOR_INFERENCE_MECHANISM_PLAN_V1.md)
extends the observation layer with task-conditioned hidden states, personal
context contrasts, and activation interventions. The historical objects below
remain the foundation; an event-process generator is not itself evidence of an
LLM's internal computation. The [execution log](GRAIN_VECTOR_RESEARCH_LOG.md)
records the completed instrument and first natural-context cycles, including
the shared-input confound and independent-history correction. This does not
mean all VM02-VM07 requirements or external personality links are complete.
The 2026-09-06 [relational-binding report](../reports/grain_vector_mechanisms/VM03_RELATIONAL_BINDING_REPORT.md)
adds same-word relational controls, exact token-span interventions, and a
standard product-kernel comparator. Readability, causal use, and personality
validity remain separate claims; the frozen GRAIN scorer is unchanged.

The [multi-comment follow-up](../reports/grain_vector_mechanisms/VM02_MULTICOMMENT_DISCOVERY_REPORT.md)
adds an independent author cohort, complete-evidence encoder/whitebox comparisons,
and a finite-population calculation of sampling-induced reference-distance
inflation. It does not identify that fluctuation with psychological noise or
delete it from scores. Natural evidence composition and external links remain open.

## 1. Purpose

This file is the single navigation entry for how GRAIN was developed, what
mathematical object each stage added, how those objects now connect, and which
mechanisms remain to be invented.

GRAIN has three different maps. They must not be confused:

1. the **development map** records how the method acquired new objects;
2. the **mechanism map** records how text events may generate author and
   population structure; and
3. the **evidence constitution** states what may be claimed about an object.

The first two maps drive current research. The third is a supporting
constraint, not the development roadmap.

Recommended reading order:

| Priority | Question | Primary documents |
| --- | --- | --- |
| 0 | What is the complete current system and its claim boundary? | [Unified Theory System V8](SUICA_UNIFIED_THEORY_SYSTEM_V8.md) |
| 0.5 | How do condition, expression, identity, and time form one person-interpreter architecture? | [4W Person Interpreter Theory](SUICA_4W_PERSON_INTERPRETER_THEORY.md) |
| 1 | How did the method evolve? | This index, [Rind Theory v3](THEORY.md), [Rind Theory v4](THEORY_V4.md), [Theory V6](THEORY_V6.md) |
| 2 | What is the current mathematical object? | [V8 Conditional Response Geometry](V8_CONDITIONAL_RESPONSE_GEOMETRY_THEORY.md), [V8 Mathematical Route](V8_MATHEMATICAL_RESEARCH_ROUTE.md) |
| 3 | What micro mechanisms can generate it? | [M3 Foundation](SUICA_M3_MICRO_MESO_MACRO_FOUNDATION.md), [M3 Mechanism Atlas](SUICA_M3_MECHANISM_ATLAS_THEORY.md) |
| 4 | Which objects survived each experiment? | [Claims Ledger](CLAIMS_LEDGER.md), [V7 Discovery Ledger](V7_DISCOVERY_LEDGER.md), named reports |
| 5 | What are the inference boundaries? | [Foundational Closure Standard](SUICA_FOUNDATIONAL_CLOSURE_STANDARD.md), [Cross-Level Inference Contract](SUICA_CROSS_LEVEL_INFERENCE_CONTRACT.md), [Foundation Gap Ledger](SUICA_FOUNDATION_GAP_LEDGER.md) |

## 2. Development lineage

The versions are not a succession of failed systems. Each stage discovered a
larger object and retained the useful part of the earlier one.

| Stage | New object or mechanism | Problem it solved | What remains in the current system | Source status and links |
| --- | --- | --- | --- | --- |
| MCD origin | Condition-relative deviation \(x_{u,c}-E[X\mid c]\) | How to compare authors after locating topic/situation regions | Common-condition comparison and population-relative position | Historical source incomplete; [MCD-style node experiment](../scripts/run_suica_p2_condition_centering_v2.py), [later worked example](WORKED_EXAMPLE_MANUAL.md), and [claims reconstruction](CLAIMS_LEDGER.md) are secondary sources |
| H-MCD / V-HMCD bridge | Token slices, recursive clusters, local blocks, author aggregation | How to turn long text into many item-like observations and return local structure to an author profile | Slicing, recursive local structure, and author position in a reference population | Historical theory source incomplete; [surviving recursive implementation](../suica_core/suica.py), [slice preparation](../scripts/prepare_suica_v2_phase2_slices.py), and [later token-slice compatibility audit](V6_LINEAGE_TOKEN_SLICE_COMPATIBILITY_PROTOCOL.md) are secondary reconstruction sources |
| GRAIN V3/V4 rind theory | Choice \(\pi_u\), expression base \(f_u\), condition reaction \(\gamma_{u,c}\), distinctive residual | Why subtracting topic/situation also removes stable author information | Choice/base/react channels; free and fixed conditions expose different information | [Theory v3](THEORY.md), [Theory v4](THEORY_V4.md), [Lineage synthesis](../reports/V6_LINEAGE_COMPATIBILITY_SYNTHESIS.md) |
| V6 path theory | Path \(w_k=p+k s+g_k\), response operator \(B_u\), nonlinear conditional law \(\Pi_u(dy\mid h)\) | How to represent movement, memory, switching, and position information lost by a mean | Path is a first-class object; dynamics are not merely static residuals | [Theory V6](THEORY_V6.md), [Measurement Objects](V6_MEASUREMENT_OBJECTS_FORMAL.md), [Nonlinear Path Objects](V6_NONLINEAR_PATH_OBJECTS.md) |
| V7 operator geometry | Parallel observation operators \(\Phi_j\), shared/private subspaces, relative geometry bundle | Why no single slice or factor chart remains optimal across views | Multiple slices are parallel measurement planes; stable author information may be high-dimensional and axis-free | [Operator-indexed Multiview](V7_OPERATOR_INDEXED_MULTIVIEW_V71.md), [Geometry Object](V7_GEOMETRY_OBJECT.md), [Psychometric Projection](V7_PSYCHOMETRIC_PROJECTION.md) |
| V8 conditional geometry | Response field \(G_u(z)\), stable position \(a_u\), response operator \([B_u]\), response sets, incidence, junctions, routing | How an author changes with condition rather than occupying one fixed coordinate | Author structure is a conditional field, path operator, and relation geometry | [Conditional Response Geometry](V8_CONDITIONAL_RESPONSE_GEOMETRY_THEORY.md), [Mathematical Route](V8_MATHEMATICAL_RESEARCH_ROUTE.md) |
| V8/M3 mechanism atlas | Event kernel and mechanism tuple \((\mu_u,\mathcal R_u,\mathcal K_u,\mathcal C_u,\mathcal P_u)\) | What microscopic processes can create the mesoscopic author geometry | Distribution, response, temporal, interaction, and higher-order path mechanisms | [M3 Foundation](SUICA_M3_MICRO_MESO_MACRO_FOUNDATION.md), [Mechanism Atlas](SUICA_M3_MECHANISM_ATLAS_THEORY.md), [Seam Closure](V8_MICRO_MESO_MACRO_SEAM_CLOSURE.md) |
| Foundation constitution | Typed path from process to technical object, measurement, construct, and use | How to prevent one object from silently being renamed as another | A supporting type system and claim boundary for every new mechanism | [Closure Standard](SUICA_FOUNDATIONAL_CLOSURE_STANDARD.md), [Gap Ledger](SUICA_FOUNDATION_GAP_LEDGER.md) |

The MCD through H-MCD/V-HMCD phases predate the independent repository and
have no single canonical theory file here. The linked implementations and
later audits are secondary reconstruction sources, not substitutes for the
missing historical theory records. A future archival task should consolidate
those phases without rewriting their historical outcomes.

The cumulative development is:

```text
condition-relative deviation
  -> recursive token-slice observations
  -> choice/base/react channels
  -> static, dynamic, hybrid, and residual path objects
  -> parallel observation operators and relative geometry
  -> condition-indexed author response fields
  -> micro mechanism atlas
  -> author-to-relation and population geometry
```

## 3. What changed at each conceptual transition

| Earlier formulation | Discovery that forced the transition | Current formulation |
| --- | --- | --- |
| Context is noise to remove | Condition centering removed stable author information because people select their contexts | Context is opportunity, selection, and response support |
| One best factorization or slice | Factor axes and slicing advantages changed across samples/operators while author relations remained | Every slice/representation is an indexed observation operator |
| Author is a vector of mean scores | Means discarded order, transition, switching, and condition response | Author is a field/operator plus static position and unresolved structure |
| Dynamics are linear `flow/gust` factors | Shared dynamic axes were unstable and nonlinear paths can agree on low-order moments | Dynamics are conditional laws, generators, spectra, currents, and path functionals |
| Residual is removable error | Residual retained distributed author structure and changed across resolution/operator | Residual is a mixture and a source of candidate mechanisms |
| Population factors define the person | Stable relative geometry survived without stable global axes | Person structure may be a high-dimensional configuration or equivalence class |
| Individual objects and group relations share one score | V8 relation experiments separated author vectors, dyadic relations, and population fields | `V`, `R`, and `P` are different output types with different mathematics |
| V8 and M3 are sequential products | V8 describes mesoscopic/macro geometry while M3 explains candidate micro generators | M3 is the microscopic engine inside the V8 architecture |

These are transformations, not deletions. The original question often
survives after its first estimator is replaced.

## 4. Current object-generation path

Let \(\mathcal N_u^\ast\) denote the unobserved human event process and
\(\mathcal D_u\) the captured record. Slice rule \(\lambda\) and
representation \(v\) produce an observed event measure, not the latent
process:

\[
\mathcal N_u^\ast
\xrightarrow{\mathsf P_{\mathcal D}}
\mathcal D_u
\xrightarrow{S_\lambda,R_v}
\widehat{\mathcal N}_{u,\mathrm{obs}}^{(v,\lambda)}
\xrightarrow{\mathcal C_m}
\widehat{\mathcal A}_u^{(m)}
\xrightarrow{\mathscr L_s}
\mathcal J_s
\xrightarrow{\Gamma}
(\mathcal J_W,\mathcal J_B,\mathcal J_T,H,\kappa)
\longrightarrow
\mathcal T_{\mathrm{population}}.
\]

Here \(\mathcal C_m\) is a mechanism-specific coarse-graining,
\(\mathscr L_s\) a context-indexed relational lift, and \(\Gamma\) the
context-to-macro aggregation operator.

The latent process may be written conceptually as:

\[
\mathcal N_u^\ast
=
\sum_{o,t}
\delta_{(o,t,O^\ast_{uot},C^\ast_{uot},S_{uo},\Xi^\ast_{uot})}.
\]

The observable measure contains only captured counterparts:

\[
\widehat{\mathcal N}_{u,\mathrm{obs}}^{(v,\lambda)}
=
\sum_j
\delta_{(\widetilde t_j,\widetilde O_j,\widetilde C_j,
\widetilde\Xi_j^{(v,\lambda)},\mathrm{provenance}_j)}.
\]

Latent state \(S\), uncaptured opportunities, intentions, and the complete
human process are not produced by tokenization or embedding.

The current mechanism atlas is:

\[
\mathcal A_u
=
\left(
\pi_u,\,
\mu_u,\,
\mathcal R_u,\,
\mathcal K_u,\,
\mathcal C_u,\,
\mathcal P_u
\right),
\]

where:

- \(\pi_u\) is opportunity/condition choice;
- \(\mu_u\) is the distribution of expressed states;
- \(\mathcal R_u\) is conditional response;
- \(\mathcal K_u\) is temporal evolution;
- \(\mathcal C_u\) is interactional coupling; and
- \(\mathcal P_u\) is higher-order path information.

Within a declared condition measure \(Q_0\), conditional response can be
projected as:

\[
G_u(z)=a_u+B_u\varphi(z)+h_u(z).
\]

Here \(a_u\) is stable position, \(B_u\) a response operator, and \(h_u\)
unresolved nonlinear response outside the registered basis. None is
automatically noise.

Across authors and contexts, the system may then form relation objects:

\[
\{\mathcal A_u\}_{u=1}^{n}
\longrightarrow
\mathcal J_s
\longrightarrow
(\mathcal J_W,\mathcal J_B,\mathcal J_T,H,\kappa)
\longrightarrow
\mathcal T_{\mathrm{population}}.
\]

This is the present micro-to-meso-to-macro skeleton. It is not yet a complete
generative theory because the coupling laws between its components remain
underspecified.

## 5. Current mechanism inventory

| Mechanism family | Existing object | What it can express | Present development gap |
| --- | --- | --- | --- |
| Opportunity and selection | \(O_t,\pi_u,K_u^C\) | entered, avoided, revisited, or selected contexts | Opportunity is still mostly treated as an available menu rather than something the author helps create |
| Distribution | \(\mu_u,D_u^\Omega\) | modality, tails, dispersion, occupancy beyond a mean | Need richer mixed-family and nonstationary distribution dynamics |
| Conditional response | \(G_u,\mathcal R_u,B_u,h_u\) | how expression changes under shared conditions | Condition coordinates are usually externally supplied rather than discovered |
| Temporal evolution | \(\mathcal K_u\), routing kernel, spectral/path objects | persistence, switching, recovery, current, path dependence | No general author generator has replaced the historical linear motion axes |
| Interaction | \(\mathcal C_u\) | response to another actor or input | Current form is mostly one-way and does not separate actor, partner, dyad, and group effects |
| Higher-order path | \(\mathcal P_u\), path signatures and within-author motifs | loops, returns, ordered motifs, and path information beyond first-order transitions | Need a common algebra connecting path functionals to response and temporal mechanisms |
| Observation multiverse | \(\{S_\lambda,R_v,\Phi_j\}\) | different slices and representations as parallel views | Need a theory of how mechanism information flows across scales/operators |
| Author geometry | \(a_u,[B_u],\widehat{\mathcal G}_u\) | high-dimensional relative author position/configuration | Need to derive geometry from interacting mechanisms rather than fit it as a terminal object |
| Relation field | \(\mathcal J_s,\mathcal J_W,\mathcal J_B,\mathcal J_T,H,\kappa\), incidence and hypergraph objects | local, between-context, total, heterogeneous, ecological, dyadic, and joint-feasibility relations | The map from individual mechanisms to relation topology is incomplete |
| Resolution and residual | \(\mathcal F_m,R_u^{(m)}\) | finite-information limits and unresolved structure | Need a mechanism-discovery rule that separates alias, finite resolution, omitted mechanism, and private stable structure |

## 6. Mechanisms still to be developed

The main theoretical gaps are conversion mechanisms between existing
objects, not missing low-dimensional factors.

### M4-A. Mechanism composition algebra

Development status (2026-07-31):
`FINITE_STATIC_AND_DYNAMIC_SYNTHETIC_DISCOVERY_SUPPORTED`. The implemented
V1 composition
grammar separates structural interaction, dependence distortion, directional
gating, projection non-commutation, and alias refusal. See
[M4-A Mechanism Composition Grammar](SUICA_M4_MECHANISM_COMPOSITION_THEORY.md)
and
[the discovery report](../reports/SUICA_M4_MECHANISM_COMPOSITION_DISCOVERY.md).

M3 currently supplies coarse-grained outputs but not a common grammar for the
joint event generator. Begin with an abstract composition operator:

\[
K_u^{\mathrm{joint}}
=
\mathfrak C\left(
K_u^O,K_u^S,K_u^C,K_u^X,K_u^I,K_u^H
\right),
\]

where the kernels govern opportunity, state, condition choice, expression,
interaction, and higher-order history. Suppose a common path space and
dominating reference measure \(K_0\) are declared. For every mechanism subset
\(A\), let \(\mathcal F_A\) be its generated sigma-algebra and define the
conditional projection

\[
\mathsf P_A\ell
=
E_{K_0}[\ell\mid\mathcal F_A].
\]

A Möbius recursion can then define candidate interaction components:

\[
\ell_A
=
\mathsf P_A\ell
-
\sum_{B\subsetneq A}\ell_B,
\qquad
\ell=\log\frac{dK_u^{\mathrm{joint}}}{dK_0}.
\]

Under a product reference with commuting projections this gives the familiar
unique orthogonal functional-ANOVA decomposition. With dependent mechanisms,
components may be nonorthogonal and projection-order dependent; an ordered
or Shapley-symmetrized decomposition must be declared instead. Finding which
decomposition preserves the intended mechanism meaning is itself part of
M4-A, not a settled theorem. V1 resolves the finite multilinear case with two
declared games: an observational held-out game and a product-reference game.
It does not yet implement a general dependent-input RKHS-HOFD or a true
noncommuting transition-kernel world.

The atlas objects \((\pi,\mu,\mathcal R,\mathcal K,\mathcal C,\mathcal P)\)
are projections of this joint law, not terms that may be added directly. The
new object is a decomposition of how opportunity, choice, response, state,
interaction, and path mechanisms alter one another.

The resulting author composition signature is

\[
\Lambda_u
=
\left(
\{d_u^\otimes(A)\},
\{d_u^{\mathrm{obs}}(A)-d_u^\otimes(A)\},
\{G_{g\rightarrow i}\},
\{\Omega_{ij}\}
\right).
\]

Across the finite discovery worlds, active hyperedge F1 and sign recovery
were 1.00, mean planted-parameter geometry was 0.736, alias refusal was 1.00,
and the null false-positive rate was zero. A subsequent designed
transition-kernel world recovered noncommuting condition/history order
(accuracy 0.828), commutator geometry (0.940), temporal gate strength
(Spearman 1.00), and a stable dynamic composition signature (AUC 0.856).
This supplies the first conversion law between the M3 mechanism list and a
joint author mechanism object in a finite synthetic space. The path-score
gain is a joint kernel/order/gate gain rather than a pure order effect, and
the next stage must separate calibration, mechanism selection, and final
evaluation panels.

### M4-B. Endogenous opportunity ecology

Development status (2026-07-31):
`FINITE_SYNTHETIC_DISCOVERY_PASS_WITH_SCOPE_CORRECTION`. See
[M4-B Endogenous Opportunity Ecology](SUICA_M4_ENDOGENOUS_OPPORTUNITY_ECOLOGY_THEORY.md)
and
[the discovery report](../reports/SUICA_M4_OPPORTUNITY_ECOLOGY_DISCOVERY.md).

The opportunity set must become dynamic:

\[
O_{t+1}
\sim
K_u^O(\cdot\mid H_t,C_t,\Xi_t,\text{others}).
\]

This adds environment creation, avoidance, persistence, and return to the
current choice profile. It connects the original rind insight to a genuine
process model. Across eight repetitions, mechanism macro F1 was 0.991,
feedback-loop geometry was 0.920, matched selection-versus-creation
discrimination was 0.998 while raw-menu half-\(L_1\) was 0.006 and choice TV
was 0.013, return/recovery geometry exceeded 0.94, hidden-source aliases
refused at 1.00, and the null false-positive rate was zero.

The scalar spectral radius of the recovered loop did not preserve planted
author rank (Spearman 0.473 versus the 0.80 target). The validated object is
therefore the full ecology loop and its return/gate operators, not an author
criticality score. This scope correction does not block M4-C: the next
development problem is to discover the condition coordinates instead of
planting them.

### M4-C. Condition-manifold discovery

Development status (2026-07-31):
`FINITE_RESPONSE_SAFE_SYNTHETIC_DISCOVERY_PASS`. See
[M4-C Response-Safe Condition Manifold](SUICA_M4_CONDITION_MANIFOLD_THEORY.md)
and
[the discovery report](../reports/SUICA_M4_CONDITION_MANIFOLD_DISCOVERY.md).

Current \(z_c\) is usually supplied by topic, prompt, platform, or metadata.
GRAIN needs an author-independent and response-safe procedure for discovering
which vector directions function as response conditions:

\[
z_j
=
\zeta_{-u}\left(
\widetilde O_j,\widetilde C_j,X_j^{\mathrm{pre}}
\right),
\qquad
G_u:z\mapsto\mathcal H.
\]

The chart is fitted on an independent reference panel or by author-level
cross-fitting and may use only variables observed before the response being
explained; \(X_j^{\mathrm{pre}}\) denotes pre-response context/provenance, not
the response \(\widetilde\Xi_j\). A joint condition/response latent model is a
separate, explicitly identified alternative. The goal is not to name topics,
but to recover local condition charts and the transition maps that join them
without circularly defining a condition from its own outcome.

The V1 implementation registers robust linear PCA, global Isomap, and a
landmark geodesic atlas. Across eight repetitions of eight finite worlds,
identifiable charts resolved in 22/24 cells, held-out geodesic geometry was
0.913, local-neighbour Jaccard was 0.774, and two-source relation geometry was
0.963. After chart freezing, conditional-response retention was 0.925 relative
to the synthetic oracle and author-response geometry was 0.960. Author
fingerprint leakage, declared response leakage, no-manifold conditions, and
noninjective condition aliases were refused; response permutation left every
chart output invariant.

This result does not establish general topology recovery. The registered
graph/MST topology signature matched 0.833 overall among identifiable worlds,
including only 4/8 circle labels under source rotation. The branched mismatch
was detected after global-chart selection and refused; it was not automatically
routed to the correct atlas. Reference calibration and selection panels use
independent observations from the same reference authors, so strict
author-fold cross-fitting remains unimplemented.

#### M4-C.2. Chart-covariant ecology transport

The discovered chart was inserted into the complete M4-B ecology rather than
only a conditional-response head. For a menu measure

\[
\nu_{u,t}
=
\sum_{k:M_{u,t,k}=1}\delta_{\widehat z_{u,t,k}},
\]

freeze a reference basis \(\phi\) and form

\[
o_{u,t}=\int\phi(z)\,d\nu_{u,t}(z),
\qquad
c_{u,t}=\phi(\widehat z_{Q_t}).
\]

V2 used fresh seeds after V1 exposed a mismatched-menu DGP and an
estimand-misaligned alias rule. Exact matched paths, truth-open alias
information loss, source/null controls, and atomic action transport all passed.
The composite loop did not: geometry was `.652 < .70`, and only `2/8`
repetitions passed every gate. The frozen status is
`M4_C2_NO_GO_CHART_TRANSPORT`.

#### M4-C.3. Physical-edge composition audit

Write the physical loop as

\[
L_u=A_uR_uD_u,
\]

where \(D\) is the menu-to-choice-measure edge, \(R\) the
condition-to-response edge, and \(A\) the response-to-creation edge. M4-C.3
constructs all eight oracle/discovered products, uses Shapley contrasts and
path-conditioned error budgets to localize loss, and compares the Jacobian
product with the full finite intervention.

This diagnostic can explain why M4-C.2 failed but cannot convert it into a
pass. M4-D remains blocked until a later independently frozen repair closes
the composite action gate.

#### M4-C.3.4. Creation residual attribution

M4-C.3.4 held the C3.3 gate-zero creation estimand fixed and crossed
condition chart (`C`), history/source observation likelihood (`S`), and
cross-author pooling (`P`). The full cell recovered only `.3405` of oracle
headroom. `S` was negative in both target worlds across all eight repetitions
(effect `-.0844`, LCB `-.1581`), and `P` was small. The truth-open `C` factor
alone recovered `.7854` of headroom and was the separated Shapley leader
(margin LCB `.0396`), but it is nondeployable and its alias audit passed only
`7/8`. The frozen status is
`M4_C34_NO_GO_THREE_FACTOR_INCOMPLETE`.

The result rejects omitted history/source likelihood as the dominant
residual mechanism and routes one fresh chart-identification experiment. It
does not reopen M4-C.2 or M4-D.

#### M4-C.3.5. Response-safe cross-view quotient chart

R1 replaced chart-family smoothing with an observable replicated-CCA
quotient. Independent author blocks and numeric sources estimate a common
pre-response condition subspace:

\[
\widehat S_1^{-1/2}
\widehat C_{12}
\widehat S_2^{-1/2}
=
U_r\Lambda_rV_r^\top,
\qquad
\widehat\phi_{\mathrm{RS}}
\in
[U_r,V_r]\pmod{O(r)}.
\]

Rank was selected by response-free permutation and replicate stability.
Response, choice, author identity, synthetic truth, and relation endpoints
remained forbidden. R1 confirmed the chart itself across eight repetitions.

R2B then froze all charts in a pre-response Stage A and opened response/oracle
truth only in Stage B. The three accessible worlds showed eligible gain
`.1227` (LCB `.0934`), headroom recovery `.8146` (LCB `.6798`), and accessible
efficiency `.9874` (LCB `.9801`). Equal-rank variance and repeatability
controls, CKA permutation, history, compensation, null, alias, source, gauge,
and hazard gates passed. Null false successes were `0/120`.

The registered zero-native-refusal gate failed: two of 160 cells triggered
`SUPPORT_SHIFT`, one in an eligible main world. The frozen status is therefore
`M4_C35_R2B_PARTIAL_CONDITIONAL_EFFECT`; it does not license unconditional
chart replacement. R2C implemented the abstention-aware policy

\[
\Pi_i =
\begin{cases}
R_i,&A_i=1,\\
B0_i,&A_i=0,
\end{cases}
\qquad
\Delta_\Pi=E[A_i(\Gamma_R-\Gamma_{B0})],
\]

where acceptance \(A_i\) is frozen from pre-response applicability evidence.
Accepted-stratum performance is secondary; the primary estimand is full
planned-population policy value.

Fresh R2C development contained no refusals (`16/16` accepted), so it proved
exact policy replay but not routing utility. Boundary-enriched development
then held response truth fixed and moved evaluation conditions across the
`.80` support boundary. The resulting scalar policy lost `.0528` relative to
always using `R`; refused forced-`R` gain was `.0880`, versus `.0301` among
accepted cells. Coverage predicted only `4.6%` of oracle-error variance and
harm AUC was `.554`. Thus

\[
C=T(q)
\]

is a many-to-one audit summary, not an effect-ordering statistic. The
operational threshold rule is retired; coverage is retained as one component
of a response-safe vector applicability signature

\[
q=\Phi_{\mathrm{pre}}(X)
=
(q^{sup},q^{tail},q^{src},q^{panel},q^{RB},q^{id}).
\]

The first 24-component post-seal pilot did not resolve this signature:
leave-one-world-out harm AUC was `.588`, OOF \(R^2=-.691\), and routing value
remained `-.0616`. A fresh experiment must vary departure direction at fixed
coverage before any vector gate can be evaluated. M4-D remains blocked. See
`SUICA_M4_C35_R1_RESPONSE_SAFE_RCCA_CHART_PROTOCOL.md` and
`SUICA_M4_C35_R2B_CONDITIONAL_CHART_PROTOCOL.md`,
`SUICA_M4_C35_R2C_ABSTENTION_POLICY_DEVELOPMENT_PROTOCOL.md`, and
`SUICA_M4_C35_R2C_BOUNDARY_ECOLOGY_DEVELOPMENT_PROTOCOL.md`.

### M4-D. Dynamic generator discovery

Because text paths are discrete and often irregularly sampled, define an
augmented predictive state \(Y_t=(X_t,H_t)\). The primary object is a
lag-indexed transition operator on that state:

\[
P_{u,\Delta}f(y)
=
E[f(Y_{t+\Delta})\mid Y_t=y,u].
\]

If a continuous-time embedding exists and
\(\{P_{u,\Delta}\}_{\Delta\ge0}\) forms a sufficiently regular Markov
semigroup on the augmented state, a generator may then be defined:

\[
\mathcal L_u
=
\lim_{\Delta\downarrow0}
\frac{P_{u,\Delta}-I}{\Delta}.
\]

Its spectrum, eigenfunctions, probability current, non-normal modes,
change-points, and recovery rates are candidate dynamic properties.
Historical `flow/gust` factors become finite projections of the discrete
operator family, not assumed continuous-time coordinates. If no finite
history makes \(Y_t\) Markov, the development target must remain a
history-dependent kernel, predictive-state representation, or memory
operator rather than a semigroup generator.

### M4-E. Multiscale coarse-graining and renormalization

GRAIN needs an explicit within-author coarse-graining chain:

\[
\text{token}
\rightarrow\text{slice}
\rightarrow\text{utterance}
\rightarrow\text{occasion}
\rightarrow\text{author}.
\]

For coarse-graining maps \(\mathcal C_{s\to s'}\), the research object is the
flow of mechanism parameters across scale. Stable universality classes,
fixed points, and scale-specific mechanisms are more plausible targets than
one universal factor chart. Author-to-dyad/group movement is not another
coarse-graining step; it is a type-changing relational lift:

\[
\{\mathcal A_u\}_{u=1}^{n}
\xrightarrow{\mathscr L}
\mathcal J
\longrightarrow
\mathcal T_{\mathrm{population}},
\]

and belongs to M4-G.

### M4-F. Residual mechanism discovery

The unresolved component has a non-unique candidate mixture:

\[
R
=
R_{\mathrm{resolution}}
+
R_{\mathrm{alias}}
+
R_{\mathrm{omitted}}
+
R_{\mathrm{private}}
+
R_{\mathrm{technical}},
\qquad
\operatorname{Cov}(R_i,R_j)\ \text{unrestricted}.
\]

The terms are not assumed orthogonal, independent, or uniquely identified.
An orthogonal decomposition is allowed only after a reference inner product
and fitted projection operators are declared.
Cross-resolution, cross-operator, and cross-condition trajectories can be
used to infer which class a residual component approaches. The target is to
discover omitted mechanisms, not to force residual energy to zero.

### M4-G. Individual mechanism to relation field

The missing social-scale bridge is:

\[
\{K_u^{\mathrm{joint}}\}_{u=1}^{n}
\longrightarrow
\{G_u\}_{u=1}^{n}
\longrightarrow
J_{uv}(z)
\longrightarrow
\mathbb H_z^{\mathrm{rel}}
\longrightarrow
\mathcal T_{\mathrm{population}}.
\]

It should explain when similar response mechanisms generate proximity,
temporary coalitions, stable relations, or population topology.

### M4-H. Multi-actor interaction dynamics

Interaction should separate:

\[
\Xi_{u\leftarrow v,t}
=
\alpha_u^{\mathrm{actor}}
+\beta_v^{\mathrm{partner}}
+\delta_{uv}^{\mathrm{dyad}}
+\gamma_g^{\mathrm{group}}
+\epsilon_t,
\]

with author, partner, dyad, and group effects. Dynamic interaction is then a
history-dependent transition kernel:

\[
K_{uv}^{\mathrm{int}}
\left(
d\Xi_{t+1}\mid
H_t,\alpha_u^{\mathrm{actor}},\beta_v^{\mathrm{partner}},
\delta_{uv}^{\mathrm{dyad}},\gamma_g^{\mathrm{group}}
\right).
\]

This kernel must represent accommodation, resistance, repair, entrainment,
and turn transition. It is required before dialogue can be reduced to an
individual author profile.

## 7. Proposed development tree

The next mechanism-development program is:

```text
M4-A mechanism composition algebra
  |
  +-- M4-B endogenous opportunity ecology
  |     `-- M4-C condition-manifold discovery
  |           `-- M4-C.2 chart-covariant ecology transport
  |                 `-- M4-C.3 relation-space composition budget
  |                       `-- M4-C.3.2 creation-estimator intervention
  |                             `-- M4-C.3.3 opportunity-budget frontier
  |                                   `-- M4-C.3.4 creation residual attribution
  |                                         `-- M4-C.3.5 response-safe cross-view EIV chart
  |
  +-- M4-D dynamic generator discovery
  |     `-- M4-E multiscale renormalization
  |
  +-- M4-F residual mechanism discovery
  |
  `-- M4-H multi-actor dynamics

{M4-A, M4-C, M4-E, M4-F, M4-H}
  `-- M4-G individual-to-relation field
```

Primary sequence:

1. define the composition algebra and synthetic coupled-mechanism worlds;
2. recover opportunity creation and condition coordinates without using
   outcome labels, transport the complete ecology through that chart, and
   separate chart error, information limitation, latent history, and
   author-specific edge heterogeneity;
3. replace finite motion axes with generator/operator objects;
4. trace those objects across token-to-author scales and keep the later
   relation lift separate;
5. use residual trajectories to propose missing mechanism families; and
6. combine individual, interaction, condition, scale, and residual mechanisms
   to derive relation fields rather than fitting population geometry
   independently.

Parallel exploratory lines:

- state-to-trait as a spectrum of time horizons rather than a binary label;
- cross-language/culture mechanism isomorphism and culture-specific
  deformation;
- LLM translation of frozen anonymous mechanisms into behaviorally grounded
  human explanations; and
- archival reconstruction of the MCD/H-MCD/V-HMCD mathematical lineage.

## 8. Development criteria

A new mechanism is useful when it adds at least one of the following:

1. a new identifiable object not reducible to the existing atlas;
2. a conversion law between two existing levels;
3. an explanation for a previously stable residual or relation pattern;
4. an observable distinction between worlds that existing summaries treat as
   equivalent; or
5. a reusable coarse-graining rule across operators or scales.

Prediction improvement is informative but not sufficient by itself. The
development output should be a mathematical object, its estimator, its
equivalence class, and the transformations under which it persists.

## 9. Supporting evidence and governance map

The evidence constitution remains necessary, but it does not choose the next
mechanism. Its typed chain is:

\[
(\mathcal H,F)
\to B
\to X
\to Z
\to\mathfrak A^{(\tau)}
\to\mathfrak M^{(\tau)}
\to\Theta^{(\tau)}
\to D^{(\tau)},
\qquad \tau\in\{V,R,P\}.
\]

Use the following only after a mechanism question is selected:

- [Foundational Closure Standard](SUICA_FOUNDATIONAL_CLOSURE_STANDARD.md) for
  the layer and output type;
- [Foundation Gap Ledger](SUICA_FOUNDATION_GAP_LEDGER.md) for unresolved
  evidence;
- [Claims Ledger](CLAIMS_LEDGER.md) for historical outcomes;
- [V7 Discovery Ledger](V7_DISCOVERY_LEDGER.md) for frozen operator results;
  and
- the named protocol/report for the local experiment.

The current research decision is therefore:

> GRAIN now has a typed object ontology, a finite mechanism grammar, an
> event-to-relation seam, and explicit reference/applicability rules. The
> fixed-coverage directional experiment closed the current chart-routing
> branch negatively: neither scalar coverage nor the present 24-component
> pre-response signature identifies cell-level `R/B0` relative utility across
> worlds or repetitions. Applicability must be fixed at study-design level,
> with dual-path sensitivity or refusal outside licensed designs. The next
> scientific work should target genuinely new observable mechanisms and human
> repeated-design edges, not threshold tuning on the opened M4 panel.

## M4-D arc and post-arc queue (2026-08-01..02, appended; open-exploration phase)

Thirteen registered legs interrogated the M4-C loop-transport wall and the reserved
R->V bridge. Full record: docs/SUICA_M4_D_CURVATURE_AND_RELATION_BRIDGE_PLAN.md
(registrations + outcomes), docs/SUICA_M4_D_LOOP_WALL_SYNTHESIS.md (the wall in four
parts, final form), ledger rows M4-D.1..M4-D.16. Headlines: curvature hypothesis
dead; route flips repaired by construction (two-stage, flips 196 -> 73, pooled loop
geometry .6519 -> .7605); Leg 4b's "budget-invariant structural floor" retroactively
corrected (V2 penalty self-infliction; de-biased budget slope -.521 = textbook);
paired truth-referenced diagnostics shown to reward common-mode shrinkage and to be
SUBSIDIZABLE by contamination (the C.3 retrofit LOWERED the attribution Spearman
.7264 -> .6677 while improving every loop statistic); the one genuine object-level
residual is a smooth quadratic-basin direction-content deficit of discovery
(gap ~ theta^1.8); the typed R->V bridge exists with a perfect designed-null refusal
record (200/200 everywhere incl. all heteroscedastic arms) and a documented tau
operating frontier (baseline at tau .275 dominates; variants buy specific repairs:
V1 un-caps C2 60/60, V2 repairs deep-noise ordering .944 -> .995; the C2
exact-EDM/PSD floor collapse is identified, debiased cross-half floor queued as
candidate). Registered-vs-outcome across the arc: 5 pivots fired exactly as
pre-committed, 2 planner interpretations killed by their own registered tests, 1
append-only retroactive correction, 1 outcome outside both registered branches
(recorded as such). Next registered thread: discovery-objective displacement
reduction (Leg 14) — the quadratic basin predicts gap ~ alpha^2 under displacement
fraction alpha, making the residual quantitatively attackable.

## M4-D/E/F line completion (2026-08-02..03, appended)

The arc above closed with four further registered experiments; full records in
docs/SUICA_M4_D_CURVATURE_AND_RELATION_BRIDGE_PLAN.md and per-leg reports.

- **Leg 14 (displacement reduction)** — PIVOT FIRED: the discovery residual is a
  systematic objective bias, not consensus-reducible noise (consensus worsens all
  three worlds, gap .215 -> .680); the quadratic-basin alpha^2 prediction held in
  form but the confound-disclosed consensus criterion moved only 4.6% split-half.
  Objective redesign recorded as the arc's closing open problem.
- **M4-E1 (convention gap, real text, label-free)** — PIVOT FIRED ~4,900x: the
  V2-penalty self-infliction lesson is synthetic-scoped for a principled reason —
  on real text the two conventions are near-perfectly projectively equivalent
  (Frobenius ratio tracks lambda_i/lambda_ii to 0.03-0.08%). The deeper finding is
  the SATURATED YARDSTICK: real-text internal split-half agreement is nil at every
  budget under every convention, so the binding constraint is panel signal, not
  penalty choice — reconfirming REALTEXT_SOFT_SUPPORT_ONLY_RELATION_UNRESOLVED.
- **M4-E2 (offset anatomy)** — PIVOT FIRED: the common offset SPREADS (max single
  share .38, S2 near-zero, directions world-specific at the permutation null;
  removing the dominant term DOUBLES the gap). No single objective term and no
  shared direction can absorb it. The M4-D/E line closed with the redesign problem
  properly characterized (freeze-whitening 1/sqrt(eig) scale family recorded as
  lead suspect at .39-.48 mass).
- **M4-F1 (D3 panel sizing law)** — pivot letter did NOT fire (agreement rises off
  zero on both axes) but the measured law kills size-only rescue: events exponent
  gamma=.153 (the only axis that fits; authors degenerate, companion .404), the
  .5-agreement budget extrapolates to 10^14x events (CI [10^2.2, 10^95.9];
  bootstrap lower edge ~155x never enters the registered [4,50] feasibility band);
  a held-out 16x cell validates the law within factor 2. Corollary hand-off: D3
  must change COMPOSITION (shared-context events, within-author condition
  pairing), not scale.

Honesty record across the full 16-experiment M4-D/E/F line: registered pivots
fired exactly as pre-committed in Legs 3, 6, 9, 11, 13, 14, E1 and E2; two planner
interpretations were killed by their own registered tests (per-realization
variance; all-or-nothing floor); one append-only retroactive correction (Leg 4b);
two outcomes landed outside every registered branch and were recorded as such;
F1's pivot letter missed while its measured law still decided the question. The
instruments decided, not the leans.

Governance: the 2026-08-01 V8-drop review and its push-gate remediation are
recorded in docs/V8_PUSH_REMEDIATION_20260803.md (PANDORA bridge headline demoted
to POST_HOC_OPERATOR_SELECTED_EXPLORATORY; Essays confirm-half text consumption;
F16 reconciliation; V37F corrections; process rules).

## M4-F panel design line CLOSED (2026-08-03, appended)

Eight registered legs (M4-F1..M4-F9, including the voided F8) closed the panel
question opened by M4-E1's saturated yardstick. Full synthesis:
docs/SUICA_M4_F_PANEL_DESIGN_SYNTHESIS.md; registrations and outcomes in
docs/SUICA_M4_D_CURVATURE_AND_RELATION_BRIDGE_PLAN.md; ledger rows
M4-F1..M4-F9.

Result in one line: **a finite panel can be made self-consistent, and cannot
be made trait-certifying.** Scale alone is infeasible (events exponent .153,
10^14x budget); composition works at fixed budget (shared occasions lift
agreement from nil to +.0234, paired +.0262 [.0195,.0328]); the author axis is
a clean validated power law (gamma 1.096, x32 holdout predicted .4012 /
observed .3861) — but what it certifies is an occasion-bound, state-inclusive
object: across authors x1->x32 the gauge rises 63x while long-window
(trait-like) recovery plateaus near .15, and spreading observation across
decorrelated occasions does not close that gap on a test that is both causally
live and adequately powered. The gauge is provably blind to the occasion axis.
This is foundation gap F10 (state/trait depends on the occasion universe and
horizon) appearing as a measurement; V8's typing is hardened accordingly — the
panel-certified relation field is a P-type object indexed by its occasion
universe and is never promoted to a trait claim.

Consequences on the books: D3 is permanently restricted to occasion-bound
objects; the .5 agreement target, the 46.1x budget and the ~26,052-author
recommendation produced by M4-F4 were WITHDRAWN by the M4-F5 audit registered
to kill them; the M4-E2 objective redesign is the only remaining route to a
trait-level object and is the line's next registration.

Four planner registration defects of one family were caught and recorded
(unstated aggregation rule; free base scale with no power requirement; an
economized-away causal channel; a nil-significance gate that fails at scale),
producing four standing rules, the last of which subsumes them: every gate
bounds MATERIALITY via an equivalence form, never nil significance on a
known-nonzero quantity.

## M4-K — the identity line (IDT), 2026-08-09

Opened on the program owner's conjecture (identity = the structured
residual; the certificate metaphor; the similarity/direction question) and
closed its registered empirical program the same day: eleven executed legs
(K1..K3), theory document `SUICA_IDENTITY_THEORY_V1.md` (T1–T9, appendices
A–M), synthesis `SUICA_M4_K_IDENTITY_LINE_SYNTHESIS.md`, plan/ledger
`SUICA_M4_K_IDENTITY_LINE_PLAN.md`. Headlines: the deployed gauge AMPLIFIES
common-frame content (T3(f) dead; forged identities pass plain
reproducibility — T6″ frame refreshment is the licensed discriminator); the
F2/F4/F5 re-typings (frame-owned composition; replicate axis; mixture
truths) via dated retrospective notes; T4 CLOSED in composite reader-borne
form (field ≈ λ·r^q − κ·V_person − ε; q = 1.83, κ ≈ 0.72 confirmed
out-of-sample) — the F5 plateau re-attributed to the reader; the similarity
geometry measured (anti-direction bound exact and binding; caricature/
rotation cos-law; the licensed three-coordinate estimator; crowding
falsified in sparse galleries, readability is the driver). Standing rules
9–17 and conventions added, each paid for by a recorded planner defect
(#9–#26). Repairs certified and UNADOPTED per F16: estimated de-framing
(94.4%/73.5%, regime-dependent) beside `colstd_alpha_0.10`; K-R1 (in
flight) tests whether de-framing improves trait reading. EXPLORATORY,
synthetic, instrument-world laws only.

**Dated addendum (2026-08-09, arc complete):** K-R1 resolved AGAINST the
repair intuition — de-framing zeroes trait reading (the scaffold corollary,
IDT appendix N: the trait has no frame-free expression in this family;
counter-operations are diagnostics, never preprocessing); instrument-role
typing follows (card readers for traits, the gauge for occasion-bound
objects). C-NFI is now THEOREM T10 (anchor impossibility, global form —
IDT appendix O). Twelve legs, arc closed; the defense phase charter opens
(`SUICA_DEFENSE_PHASE_CHARTER.md`) per the owner's standing phase
instruction.

**Dated addendum (2026-08-10, M4-L closed):** the reverse reading (IDT-R,
owner's directive) ran L1–L3 and closed: the η-floor law measured three
times (projection-invariant; tax ratio m/k_τ); the floor-invariance and
localization-window lemmas; the taxometer η̂ certified ±0.125; the
completeness meter certified at its measured ~0.005 noise floor with the
partition-propagation correction; "error = 0" converted into a calibrated
audit (IDT appendices P–S; SUICA_M4_L_TYPOLOGY_LINE_SYNTHESIS.md). Rules
18–21 added (defects #27–#38). The pre-authorized defense phase resumes
at D1 (prospective seal #2).

**Dated addendum (2026-08-10, V8 × IDT integration):** the identity era is
folded into the V8 stack of record — integration map
`SUICA_V8_IDT_INTEGRATION.md`; paper-form Part II supplement in
`V8_THEORY_PAPER.md`; the identity-era instruments and pitfalls as
`V8_MANUAL.md` §11; foundation-gap annotations (F10 mechanism, F08
reinforcement, PROPOSED F12/F13) in `SUICA_FOUNDATION_GAP_LEDGER.md`;
State-of-main updates in all three READMEs. Law grades of record:
PREDICTIVE (η-floor, taxometer, tax-ratio algebra), RESPONSE (variance
tax, quadratic link band), LEVEL-interpolative (λ′ − κ′·V_person), DEAD
((d/n)^¼ window conjecture, T4 level composite v1, T8(e) crowding).

## Dated addendum (2026-08-11) — the M4-M line

The level-law completion line: plan
`docs/SUICA_M4_M_LEVEL_LAW_LINE_PLAN.md` (7 registrations, outcomes,
adjudications); synthesis `docs/SUICA_M4_M_LEVEL_LAW_LINE_SYNTHESIS.md`;
theory appendices W–CC in `docs/SUICA_IDENTITY_THEORY_V1.md`; graded
laws updated by the dated addendum in `docs/SUICA_V8_IDT_INTEGRATION.md`.
Headlines: the scoped level law is PREDICTIVE-SCOPED (M2, 3/3 sealed);
the tax is a declining quadratic curve, closure-validated 5/6 (M3);
the response-band exponent is re-attributed (V-shadow +2.245); rules
25–29; defects #43–#49. Successor: the M4-N mechanism line
(`docs/SUICA_M4_N_TAX_MECHANISM_LINE_PLAN.md`).

## Dated addendum (2026-08-14) — the N/P/Q/R lines

Four lines closed since the M-line: **N** (tax mechanism —
transport seal 3/3, closure clean, synthesis
`SUICA_M4_N_TAX_MECHANISM_LINE_SYNTHESIS.md`), **P** (penalty
mechanism — the gauge is a frame-agreement instrument; split-seed
instrument delivered; synthesis
`SUICA_M4_P_PENALTY_MECHANISM_LINE_SYNTHESIS.md`), **Q** (transport
— rule 33; card pure trait; tax frame-borne; synthesis
`SUICA_M4_Q_TRANSPORT_LINE_SYNTHESIS.md`), **R** (identity channel
— certified; the founding answer; synthesis
`SUICA_M4_R_IDENTITY_CHANNEL_LINE_SYNTHESIS.md`). Theory appendices
W–MM in `SUICA_IDENTITY_THEORY_V1.md`; graded laws and corrections
in the four dated addenda of `SUICA_V8_IDT_INTEGRATION.md`; rules
1–33 and defects #1–#64 in `SUICA_DEFECT_REGISTRY.md` dated notes.

## Dated addendum (2026-08-17) — the M4-T line and the 4W era

Codex's T1 (`d9da17d`) and the owner's 4W synthesis (`24d4a80`)
opened the era; five legs under the loop discipline closed its
measurement phase (synthesis `SUICA_M4_T_LINE_SYNTHESIS.md`;
adjudications in `SUICA_M4_T_HIERARCHICAL_SELECTION_IDENTITY_PLAN.md`).
Headlines: selection identity = nested innovations + a
generalizing taste tail; the identity budget (taste 82%); the
dissociation (identity's carrier largely trait-silent); the
carrier METRIC-BORNE — identity lives in the distinctive,
personality in the common. Defects #68–#75; suite 1021.

## Dated addendum (2026-08-18) — the M4-U When line

The 4W theory's unposed quadrant measured on both time scales
(plan + all adjudications `SUICA_M4_U_WHEN_ORDER_PLAN.md`; synthesis
`SUICA_M4_U_LINE_SYNTHESIS.md`). Headlines: the order channel exists
and is inertia (dwell), reproducible but redundant vs the bag; the
person is a moving distribution with a THREE-YEAR core (53%) — and
discrimination scores over-predict geometric persistence (AUC 0.9837
coexists with the 53% floor because the cross floor is 0.05); the
low-rank taste direction is the slowest object measured (λ 0.17/y);
whether common outlasts distinctive is OPEN (7/7 point estimates say
yes, no interval closes; second-corpus charter owns it). Defects
#76–#80; suite 1148; U3 (stage-E trait join) registered-next.

## Dated addendum (2026-08-19) — the M4-W transport line and the 4W empirical state

W1/W2 closed the label-free transport phase on the disjoint cohort
(8,895 authors; plan + adjudications
`SUICA_M4_W_DISJOINT_TRANSPORT_PLAN.md`): the U-line's open middle CLOSED
(COMMON_STANDING, Λ=+0.0911 CI>0 — the When clause lands in two-regime
form), the order channel and T1's branch code transported in cell
(T1's last open item closed; the tail STRENGTHENS on fresh authors), and
the phase's law: structure and ratio laws transport, levels belong to the
cohort. The whole T→U→W arc is consolidated against the owner's theory
in `SUICA_4W_EMPIRICAL_STATE.md` (the §11 feedback: priorities 1–4
answered; priority 5, external connection, remains and is owner-gated).
Defects #76–#83; suite 1279; W3 (MBTI-axis labels) gated on governance
review.

## Dated addendum (2026-08-22) — X-line inference repair and X5-T

The cross-level audit separated common units from measurement invariance and
renamed the historical `Delta_erg` object canonically as the
projection-specific transport gap `Delta_T`. X5-T then reproduced all ten
committed K=2 relation/cohort slopes to machine precision and audited nested
K=2/4/8 event-order partitions on fixed K=8 pools. At K=8, the disjoint cohort
had four of five equivalent within-slope paths but only R5 retained the same
transport family in all segments; the Big5-name-list cohort had one equivalent,
two detected heterogeneous, and two unresolved paths. The result closes the
old compression: temporal persistence of an average within-person response and
temporal persistence of cross-level transport are distinct gates. Their exact
path relation is now represented by
`||b-w||^2 = ||b||^2 + ||w||^2 - 2<b,w>`. Evidence:
[contract](SUICA_CROSS_LEVEL_INFERENCE_CONTRACT.md) and
[report](../reports/SUICA_M4_X5_TEMPORAL_TRANSPORT_AUDIT.md).
