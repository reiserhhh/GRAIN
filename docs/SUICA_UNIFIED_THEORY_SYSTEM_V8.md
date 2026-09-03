# GRAIN Unified Theory System V8

Status: **canonical theory integration, 2026-08-01**

This document is the shortest authoritative path through the current GRAIN
theory. Detailed protocols and reports remain the evidence record; when an old
headline conflicts with this document, the claims ledger, or the
[Cross-Level Inference Contract](SUICA_CROSS_LEVEL_INFERENCE_CONTRACT.md), the
narrower audited claim controls.

## 1. What GRAIN is

GRAIN (Group-Referenced Analysis of Individual Natural-Language Expression) is
a foundational analysis method for developing future natural-language
measurement systems. It specifies how text may be converted into reproducible,
reference-relative, condition-aware technical objects and how those objects
may later be connected to psychological constructs.

Its central technical object is **individual language texture**: the
multiscale organization of a person's local language events relative to a
declared population, opportunity distribution, observation operator, and
calibration version. A sentence or embedding is not itself a texture. Texture
appears in the reproducible arrangement of distributions, orientations,
condition choices, conditional responses, interactions, and paths across
multiple observations. The binding terminology and compatibility boundary are
defined in [GRAIN_NAMING_MIGRATION.md](GRAIN_NAMING_MIGRATION.md).

GRAIN is not:

- a finished scale or personality test;
- a universal set of named factors;
- a language model that directly diagnoses a person;
- a claim that author identification equals personality measurement;
- a claim that finite-synthetic mechanism recovery equals human validity.

The research endpoint is a **personality and state interpreter**, not a generic
text analyzer. Reaching that endpoint requires a psychometric spine:
cross-person scoreable structure, an explicit reference population,
reliability, construct validity, and use-specific validation. Context,
opportunity, path, and interaction layers support that spine; they do not
replace it.

## 2. Type system

GRAIN forbids silent conversion between the following types:

\[
(\mathcal H,F)
\rightarrow B
\rightarrow X
\rightarrow Z
\rightarrow \mathfrak A^{(\tau)}
\rightarrow \mathfrak M^{(\tau)}
\rightarrow \Theta^{(\tau)}
\rightarrow D^{(\tau)},
\qquad \tau\in\{V,R,P\}.
\]

| Symbol | Type |
|---|---|
| \(\mathcal H,F\) | unobserved human history and current psychological process |
| \(B\) | behavior that became expressible under the available opportunity |
| \(X\) | recorded text/event stream |
| \(Z\) | embedding and other technical representation |
| \(\mathfrak A\) | author-level technical object |
| \(\mathfrak M\) | measurement candidate in an explicit reference frame |
| \(\Theta\) | psychologically interpreted construct |
| \(D\) | use-specific decision or assessment output |

The superscripts distinguish three noninterchangeable outputs:

- `V`: an individual vector or score candidate;
- `R`: a relation between people, conditions, or constructs;
- `P`: a population-level distribution or relation field.

Evidence for `R` does not automatically license `V`, and evidence for `P`
does not automatically identify an individual.

## 3. Observation operators and slicing

The latent language process is observed through a corpus and a chosen
operator:

\[
\mathcal N_u^\ast
\xrightarrow{\mathsf P_{\mathcal D}}
\mathcal D_u
\xrightarrow{S_\lambda,R_v}
\widehat{\mathcal N}_{u,\mathrm{obs}}^{(v,\lambda)}.
\]

Here \(S_\lambda\) is a slicing/granularity operator and \(R_v\) is a
representation operator. A slice is therefore one observation plane, not a
piece of psychological truth. No theory requires one globally optimal cut.
Uncut, token-window, sentence, utterance, occasion, and multiscale views may
each preserve different lawful structure.

This resolves the apparent conflict between older slice-centered work and V7:
slicing remains a core experimental operation, but its superiority is an
empirical, construct-local question rather than an axiom.

## 4. Condition, opportunity, and the author object

Topic, situation, format, prompt, interlocutor, and expressive opportunity are
not merely noise. They are antecedent coordinates that determine what can be
expressed. Removing them may also remove author-specific selection and
response.

The minimal author object is

\[
\mathcal A_u=
(\pi_u,\mu_u,\mathcal R_u,\mathcal K_u,\mathcal C_u,\mathcal P_u),
\]

where:

- \(\pi_u\): selection over available conditions or opportunities;
- \(\mu_u\): distribution of expressed positions under those conditions;
- \(\mathcal R_u\): response to a controlled condition change;
- \(\mathcal K_u\): temporal/path dynamics;
- \(\mathcal C_u\): interaction and coupling with another actor;
- \(\mathcal P_u\): higher-order path information not preserved by marginals.

A local response representation is

\[
G_u(z)=a_u+B_u\varphi(z)+h_u(z),
\]

or, at event resolution,

\[
w_{u,r,t}
=\mu_{r,t}^{\mathrm{condition/opportunity}}
+a_u+B_u z_{r,t}+g_{u,r,t}.
\]

`a_u` is a reference-condition position, `B_u` is an author response operator,
and `h_u`/`g_u` is structure not represented by the current basis. None is
automatically personality, state, or noise.

## 5. The reference-relative measurement spine

An author score has no standalone meaning. It is a position relative to a
declared reference population \(P_0\), opportunity distribution \(Q_0\),
observation operator, and calibration version:

\[
\mathfrak M_u
=\Psi\!\left(
\widehat{\mathcal A}_u;
P_0,Q_0,S_\lambda,R_v,\mathcal C
\right).
\]

Consequences:

1. Batch-standardized exploratory scores are not reusable individual scores.
2. Reference composition, opportunity support, and calibration provenance must
   travel with every result.
3. Coordinate axes may rotate while relational geometry remains stable.
4. A future study-specific instrument needs its own G-study/D-study,
   uncertainty, minimum detectable difference, and external validation.

The current L4.5 work supports a finite-synthetic vector score candidate only;
it does not yet license a reusable human psychological score.

## 6. Micro, meso, and macro layers

### 6.1 Micro: event mechanisms

M3 organizes observable path information into five nonexhaustive families:

\[
\mathcal A_u^{\mathrm{M3}}
=\big(
\mathsf D_u,
\mathsf R_u,
\mathsf T_u,
\mathsf I_u,
\mathsf H_u
\big),
\]

representing distribution/selection, conditional response, temporal dynamics,
interaction, and higher-order path structure. These are mechanism classes, not
five personality factors. Synthetic recovery shows that they can generate
distinct author geometries; it does not prove that they span human psychology.

The individual path kernel can be written

\[
K_u(ds',dc,d\xi\mid H_t,s,o),
\]

which separates next-state change, condition transition, and expression event
given history, current state, and opportunity. Lower-order flow and gust are
projections of this kernel, not complete dynamics.

### 6.2 Meso: replicated relation geometry

Replicated author objects are lifted into context-local relation objects:

\[
\{\widehat{\mathcal A}_u\}_{u=1}^n
\xrightarrow{\mathscr L_s}
\mathcal J_s.
\]

Within-, between-, and transported relation components must remain separate:

\[
\mathcal J_s
=\mathcal J_{W,s}+\mathcal J_{B,s}+\mathcal J_{T,s}
+H_s+\kappa_s.
\]

The event-to-relation seam is supported only when raw replicated supports are
observably comparable. Support drift, aliasing, underresolution, and ecological
only structure require explicit refusal rather than forced transport.

Their temporal paths must also remain separate. For centered segment paths
\(b_k=\beta_{B,k}-\bar\beta_B\) and
\(w_k=\beta_{W,k}-\bar\beta_W\), cross-level transport has the exact geometry

\[
\lVert b-w\rVert^2
=\lVert b\rVert^2+\lVert w\rVert^2
-2\lVert b\rVert\lVert w\rVert\cos\phi.
\]

The X5-T audit shows why this distinction is operationally necessary: a
within-person path may be temporally stable while the transport family changes
because the between-person geometry moves. The three coordinates are therefore
within-response energy, between-geometry energy, and their coupling angle;
none is automatically a personality factor. See the
[Cross-Level Inference Contract](SUICA_CROSS_LEVEL_INFERENCE_CONTRACT.md) and
[X5-T report](../reports/SUICA_M4_X5_TEMPORAL_TRANSPORT_AUDIT.md).

### 6.3 Macro: population relation field

Population structure is a field over reproducible contexts, not one universal
correlation matrix:

\[
\mathcal T_{\mathrm{population}}
=\Gamma\big(\{\mathcal J_s,\omega_s\}_{s\in\mathcal S}\big).
\]

Contextual moderation, cancellation, composition reweighting, and ecological
relations can all produce different macro summaries. A macro relation can be
real even when individual prediction is weak, but it remains a relation-level
claim and cannot be back-projected into an individual diagnosis without a
separate bridge.

## 7. Coarse graining and information loss

The sequence

\[
\text{token}\to\text{slice}\to\text{utterance}
\to\text{occasion}\to\text{author}
\]

is a family of coarse-graining maps. Averaging can preserve position while
destroying order, transitions, hysteresis, and coupling. Conversely, very fine
resolution may be dominated by technical variation and insufficient
opportunity. GRAIN therefore treats resolution as a filtration and asks which
properties persist across scales.

This is different from the relation lift in Section 6: coarse graining changes
resolution of one object; relation lifting changes object type.

## 8. Residual theory

Residual is not synonymous with garbage:

\[
R=R_{\mathrm{resolution}}+R_{\mathrm{alias}}
+R_{\mathrm{omitted}}+R_{\mathrm{private}}
+R_{\mathrm{technical}}.
\]

The terms need not be orthogonal or uniquely identifiable. They denote:

- unresolved structure below the current resolution;
- multiple mechanisms with observationally equivalent projections;
- missing condition/opportunity variables;
- lawful individual structure not shared by the fitted group basis;
- scorer, embedding, sampling, and numerical variation.

The development goal is not to force total residual to zero. It is to state
which residual components become negligible under which limit and which remain
scientifically meaningful.

## 9. Chart geometry and applicability

The response-safe RCCA chart is an observable quotient coordinate system:

\[
\widehat\phi_{\mathrm{RS}}
\in[U_r,V_r]\pmod{O(r)}.
\]

Its coordinates are gauge-dependent; distances, subspaces, and relation
geometry are the more defensible objects. R2B showed that the chart can improve
registered downstream geometry in some finite-synthetic worlds, but also
showed that unconditional replacement of `B0` is not licensed.

Applicability must therefore be represented as a response-safe pre-outcome
signature:

\[
q=\Phi_{\mathrm{pre}}(X),
\qquad
\mathcal D_R=
\{q:E[d\mid q]\ge-\delta,
P(d<-\delta\mid q)\le\alpha\},
\]

with routed operator

\[
\Pi(q)=A(q)R+[1-A(q)]B0.
\]

Scalar coverage \(C=T(q)\) is a lossy summary. Boundary Ecology falsified its
use as an automatic `.80` route: similar support counts contained beneficial
source-partition departures and harmful source-rotation departures. Coverage
is retained as support metadata, not a utility gate.

The first 24-component `q` pilot improved harm ranking directionally but failed
gain prediction and routing utility because support amount and departure
direction were confounded. A fresh experiment therefore held exact support at
`12/16` or `13/16` while crossing radial-dispersed, radial-concentrated,
tangential, and source-asymmetric departures.

The audited result is `M4_C35_PRE_RESPONSE_DOMAIN_NOT_IDENTIFIED`. Across 360
main and 72 null cells, all basis/signature/response replay errors and null
gains were exactly zero. Nevertheless, LOWO `Q` had harm AUC `.533` (LCB
`.464`), gain `R2=-1.616`, and routing value over always-`R` `-.0243` (LCB
`-.0417`). LORO had AUC `.512` (LCB `.477`), `R2=-.195`, and routing value
`-.0134` (LCB `-.0314`). `Q` did not outperform the scalar learner; the latter
routed every cell to `R`. The historical threshold also lost `.0503` relative
to always using `R`.

Thus the present pre-response observables do not identify cell-level relative
utility, even after direction is manipulated explicitly. Applicability is now
a **design-level license** unless a future independent observable mechanism is
found: use a chart only inside prospectively declared study classes with
separate evidence; otherwise expose `B0/R` sensitivity or refuse. No further
threshold tuning is licensed on these outcomes.

## 10. LLM role

Stable scoring must remain a deterministic, versioned GRAIN path. An LLM may:

- assist construct discovery and blind coding;
- translate frozen technical evidence into human-readable explanations;
- propose hypotheses and counterexamples;
- render an assessment narrative constrained by supplied scores, uncertainty,
  provenance, and refusal state.

An LLM must not silently replace the scoring path, invent a construct label,
override a refusal, or turn group-level structure into an individual claim.

## 11. Development lineage

| Stage | Retained contribution | Refuted or downgraded claim |
|---|---|---|
| MCD | compare authors conditional on shared antecedents | condition centering as universal denoising |
| H-MCD/V-HMCD | recursive local structure and author aggregation | one unique best cut or few universal axes |
| V3/V4 | choice/base/react separation; fixed and free designs differ | tension as stable trait; emotion-word rate as personality scale |
| V6 | path is first-class and means lose motion information | shared dynamic axes already equal personality factors |
| V7 | multiple observation planes, reference-relative geometry | slice always beats uncut; author AUC equals personality validity |
| V8 | typed micro-to-meso-to-macro hierarchy and relation fields | one universal score can replace level-specific objects |
| M3 | five mechanism families can generate distinct geometries | those families are a complete basis of human psychology |
| M4-A/B | mechanism composition, opportunity, feedback, return | one feedback scalar is a stable author essence |
| M4-C/C.2 | pre-response chart and recoverable atomic edges | atomic recovery guarantees composite-loop recovery |
| R2B | conditional downstream chart benefit | unconditional RCCA replacement |
| R2C | exact executable routing and replay | routing utility identified without abstention events |
| Boundary Ecology | support amount relates weakly to error | `.80` support threshold predicts relative utility |
| Vector pilot | direction-sensitive applicability is plausible | current `q` already defines a general domain |

Earlier versions are not discarded. Their surviving mechanisms become typed
components of the current system; their failed universal claims become refusal
rules or design constraints.

## 12. Evidence status at integration

### Supported within named domains

- Human corpora contain reproducible choice structure, some author-relative
  geometry, and limited exploratory external relationships under explicit
  protocols (the PANDORA cross-scale bridge headline is additionally post-hoc
  operator-selected and sign-unstable across the two operators run that day;
  see `docs/V8_PUSH_REMEDIATION_20260803.md`, item 1).
- Synthetic work identifies distinct selection, response, temporal,
  interaction, and path mechanisms and their failure cases.
- A support-comparability gate can close the synthetic event-to-relation seam.
- R2B establishes a sealed finite-synthetic conditional RCCA effect.
- The `.80` scalar utility gate has been falsified rather than rescued by
  threshold tuning.

### Development evidence only

- The L4.5 vector reference frame is an opened synthetic candidate awaiting
  independent confirmation and human G-study/D-study.
- M4 mechanism composition and opportunity ecology explain why apparently
  similar observed author geometry may arise through different generators.
- Vector applicability signatures may describe departure direction better than
  scalar coverage, but the fixed-coverage experiment failed to identify a
  general or repeated-world route.

### Unidentified

- Which technical structures are personality, state, learned habit, social
  identity, or platform adaptation.
- A complete basis of natural-language psychological mechanisms.
- Same-person cross-context psychological invariance in appropriately designed
  human data.
- A reusable natural-text applicability domain for any chart or construct.
- A response-safe cell-level observable that predicts `R/B0` relative utility.

### Explicitly not claimed

- GRAIN is a clinical instrument, diagnostic system, or finished scale.
- Big Five or MBTI correlations validate the full GRAIN ontology.
- Same-author retrieval is personality validity.
- Synthetic identifiability establishes human prevalence.
- Cross-language embedding compatibility removes cultural bias.

## 13. Building a future study-specific measurement system

GRAIN supplies a development method, not a ready item bank. A future study must:

1. Define the intended construct and decision use before collecting outcomes.
2. Declare population, language/culture, setting, interlocutor, prompt,
   opportunity set, occasion structure, and prohibited leakage.
3. Use a two-phase design where appropriate:
   - free-expression phase for \(\pi_u\) and natural opportunity selection;
   - repeated fixed-condition phase for \(B_u\) and condition-specific style.
4. Distribute fixed conditions across multiple occasions; one document is not
   a retest design.
5. Freeze representation and observation operators before confirmatory scoring.
6. Discover cross-person structure in development data, then estimate reference
   scores and uncertainty without using the final confirmation sample.
7. Run reliability, invariance, opportunity, scorer, resolution, and negative
   controls.
8. Link only frozen technical candidates to external psychological or
   behavioral criteria.
9. Confirm in fresh people and, when transport is claimed, fresh settings.
10. Publish the refusal domain and calibration provenance with every score.

This procedure can later produce many topic- or setting-specific natural
language measures. Those instruments are applications built under GRAIN, not
GRAIN itself.

## 14. Evidence governance

Every claim must point to:

- a prospective protocol or an explicit exploratory label;
- immutable config and source provenance;
- participant/source-disjoint information order where applicable;
- negative and alias controls;
- exact output type (`V`, `R`, or `P`);
- uncertainty and refusal behavior;
- the narrowest licensed interpretation.

Opened development findings cannot be converted into confirmation by changing
thresholds. A new claim requires fresh randomness or fresh data and a new seal.
Generated results remain reproducible local artifacts; theory, code, configs,
tests, and concise reports belong in version control.

## 15. Current theory boundary

GRAIN V8 now has a coherent typed chain:

\[
\mathcal N_u^\ast
\to\widehat{\mathcal N}_{u,\mathrm{obs}}^{(v,\lambda)}
\to\widehat{\mathcal A}_u^{(m)}
\to\mathfrak M_u(P_0,Q_0)
\to\mathcal J_s
\to\mathcal T_{\mathrm{population}}
\to\Theta^{(\tau)}
\to D^{(\tau)}.
\]

The mathematical and procedural links are defined. What remains is not another
unbounded factor search, but evidence for selected edges: human repeated-design
identification, external construct meaning, natural-text study-level
applicability, and use-specific validation. The M4 route has additionally
closed one negative edge: the current scalar and 24-dimensional pre-response
summaries cannot support a general cell-level chart router. Until new edges are
tested, the theory must stop at the highest licensed technical object rather
than fill the gap with an LLM or a familiar personality label.

## Canonical evidence links

- `docs/SUICA_THEORY_ROUTE_INDEX.md`
- `docs/SUICA_M3_MECHANISM_ATLAS_THEORY.md`
- `docs/V8_MATHEMATICAL_RESEARCH_ROUTE.md`
- `docs/V8_MICRO_MESO_MACRO_SEAM_CLOSURE.md`
- `docs/SUICA_L45_REFERENCE_FRAME_PROTOCOL.md`
- `docs/SUICA_M4_C35_R2B_CONDITIONAL_CHART_PROTOCOL.md`
- `docs/SUICA_M4_C35_R2C_BOUNDARY_ECOLOGY_DEVELOPMENT_PROTOCOL.md`
- `docs/SUICA_M4_C35_R2C_DIRECTIONAL_APPLICABILITY_DEVELOPMENT_PROTOCOL.md`
- `docs/CLAIMS_LEDGER.md`
- `docs/RULEBOOK.md`
