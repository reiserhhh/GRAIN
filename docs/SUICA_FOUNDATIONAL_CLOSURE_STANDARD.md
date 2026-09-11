# GRAIN Foundational Closure Standard

Date: 2026-07-30  
Status: binding V8 pre-experiment methodology constitution

2026-09-12 scope clarification: this document supplies inference and use rules,
not proof that a constructive measurement theory is complete. The
[foundational program V2](GRAIN_FOUNDATIONAL_THEORY_PROGRAM_V2.md) now sets the
research target: explanatory mechanisms, discriminating predictions and derived
measurement-design rules. A typed opening or refusal does not solve its mechanism.
Historical statuses below remain unchanged; discovery follows the current
discovery-first policy and execution is currently paused for conceptual review.

## 1. Identity and purpose

GRAIN is **not a scale** and is not a pipeline whose purpose is to produce one
scale. It is foundational methodology for making natural-language observation
mathematically explicit enough that a future researcher may construct,
falsify, compare, or refuse a text-based measurement.

Questionnaire development, projective methods, behavioral observation,
Generalizability Theory, construct validation, and high-stakes assessment are
used here as adversarial sources of requirements. They do not define GRAIN's
identity. A future scale, state monitor, clinical aid, or other instrument is
an application built on GRAIN and must bring its own construct, population,
administration, validity, and use evidence.

GRAIN's foundational question is:

> Given language produced by people under partially observed conditions, what
> mathematical object is identified, under which operator and reference
> universe, with which uncertainty, and how far may it be interpreted?

## 2. What “closure” means

Foundational closure is **inferential closure**, not omniscience.

A framework is closed when:

1. every object has an explicit ontological layer;
2. every transition has assumptions, evidence, and a refusal rule;
3. observationally equivalent explanations are not falsely distinguished;
4. unresolved quantities are registered as bounded openings rather than
   silently reused as premises;
5. technical existence, measurement, psychological interpretation, and use
   remain separate claims; and
6. a failed gate stops promotion but preserves the lower-layer result.

An empirical unknown is not automatically a theoretical hole. It becomes a
hole only when the framework cannot say where it belongs, what would identify
it, or what must be refused while it remains open.

## 3. The eight-layer ontology

The full epistemic chain is:

\[
\tau\in\{V,R,P\},\qquad
\iota_V=u,\quad \iota_R=(u,v),\quad \iota_P=\nu,
\]

\[
(\mathcal H_{ut},F_{ut})
\xrightarrow{\mathsf P_{\mathcal D}} B_{ut}
\xrightarrow{\Gamma_{\mathcal D}} X_{ut}
\xrightarrow{R_v} Z_{ut}
\xrightarrow{\mathcal C_{\lambda,\tau}}
\mathfrak A_{\iota_\tau}^{(\tau)}
\xrightarrow{\mathcal E_{\nu,\mathcal U,\omega,\tau}}
\mathfrak M_{\iota_\tau}^{(\tau)}
\xrightarrow{\Psi_{\kappa,\tau}}\Theta^{(\tau)}
\xrightarrow{\Omega_{a,\tau}}D^{(\tau)} .
\]

| Layer | Object | Licensed statement |
| --- | --- | --- |
| L0 | \(\mathcal H_{ut},F_{ut}\): human process and available facets | A hypothesized human process exists; it is never directly observed from text alone. |
| L1 | \(B_{ut}\): expression behavior under a design | A person selected and produced behavior within a menu of opportunities. |
| L2 | \(X_{ut}\): stored text record | These bytes, turns, timestamps, sources, and provenance were captured. |
| L3 | \(Z_{ut}=R_v(X_{ut})\): representation | A versioned tokenizer/encoder/slice operator produced this representation. |
| L4 | \(\mathfrak A_{\iota_\tau}^{(\tau)}=\mathcal C_{\lambda,\tau}(Z)\): typed anonymous technical object | An author vector/operator (`V`), dyadic relation/distance (`R`), or population relation field (`P`) was estimated. |
| L5 | \(\mathfrak M_{\iota_\tau}^{(\tau)}\): typed reference-relative measurement | The same output type is comparable under a frozen operator, reference population, and generalization universe; this is not necessarily a person score. |
| L6 | \(\Theta^{(\tau)}\): construct interpretation | External evidence licenses a named psychological or behavioral interpretation for the declared output type. |
| L7 | \(D^{(\tau)}\): use or decision | Consequence, fairness, calibration, and authority evidence licenses a type-compatible action. |

The arrows are partial maps. For edge \(E_j\),

\[
E_j(x)=
\begin{cases}
f_j(x), & A_j\land S_j\land V_j=1,\\
\bot_j, & \text{otherwise},
\end{cases}
\]

where \(A_j\) is the registered assumption set, \(S_j\) is support, \(V_j\)
is the validation gate, and \(\bot_j\) is a typed refusal. No later layer may
inherit support merely because an earlier layer passed.

## 4. Foundational axioms

### A1. No direct-access axiom

Text is evidence produced by a human process, not transparent access to that
process. Intent, personality, emotion, effort, disclosure, and cognition
cannot be read directly from an embedding.

### A2. Design-relative observation axiom

Observed expression depends on what could be expressed, what was requested,
what was selected, what was recorded, and what was omitted:

\[
B_{ut}=g(\mathcal H_{ut},F_{ut},O_{ut},C_{ut},I_{ut},A_{ut}).
\]

Here \(O\) is opportunity, \(C\) context, \(I\) interaction, and \(A\)
administration. A study must fix, randomize, observe/model, or explicitly
exclude each relevant facet.

### A3. Choice-response separation axiom

Context is not globally “signal” or “noise.” It can create two different
channels:

\[
\Pr(C=c\mid O,u) \quad\text{and}\quad
\Pr(B\mid C=c,O,u).
\]

The first is selection/choice; the second is conditional response. Removing
context may erase choice signal, while ignoring it may confound response.
They must be estimated separately whenever both are claimed.

### A4. Representation-relativity axiom

Tokenization, embedding, slicing, windowing, and coarse-graining define an
operator-indexed view:

\[
Z^{(v,\lambda)}=R_v\!\left(S_\lambda(X)\right).
\]

There is no presumed uniquely true slice or coordinate chart. A robust claim
is an invariant or an explicitly bounded family across admissible operators,
not the output of one convenient operator.

### A5. Identification-before-naming axiom

If two latent worlds induce the same observable distribution,

\[
P_{\theta_1}(X)=P_{\theta_2}(X),
\]

GRAIN identifies the equivalence class
\([\theta]=\{\theta':P_{\theta'}(X)=P_\theta(X)\}\), not a unique essence.
Naming one member requires additional intervention, anchor, or external
evidence.

### A6. Reference-relativity axiom

A technical object becomes a comparable measurement only relative to a
declared reference and universe while preserving its output type:

\[
\mathfrak M_{\iota_\tau}^{(\tau)}
=\mathcal E_{\nu,\mathcal U,\omega,\tau}
\left(\mathfrak A_{\iota_\tau}^{(\tau)}\right),
\qquad \tau\in\{V,R,P\},
\]

where \(\nu\) is the frozen reference population, \(\mathcal U\) the
generalization universe, and \(\omega\) the operator version. Batch-relative
distance is not an independently reusable measurement, and a relation or
population field is not an author score.

For an L5 candidate, the minimal versioned reference is

\[
R=(\nu,\lambda,\omega,\chi,\mathcal U),
\]

where \(\lambda\) is the target facet/opportunity measure and \(\chi\) the
common chart. The output must include the reference-relative point or
equivalence class, conditional uncertainty, support mass, effective
information, detectable-difference rule, and full provenance. Reference,
fit/calibration, and scored panels must be disjoint or cross-fitted.
Uncertainty must propagate reference, event, occasion, and operator channels.
If any requirement is absent, E45 remains open even when the technical point
estimate correlates strongly with synthetic truth. The executable
specification is `reference_measurement` in
`schemas/suica_foundation_study.schema.json`.

E45 is indexed by the type of its L4 input. At minimum, distinguish:

\[
\mathfrak A_u^{(V)}
\quad\text{(author-indexed vector or operator)},\qquad
\mathfrak A_{uv}^{(R)}
\quad\text{(pairwise relation or distance)},\qquad
\mathfrak A_{\nu}^{(P)}
\quad\text{(population relation field)}.
\]

Their corresponding L5 objects are
\(\mathfrak M_u^{(V)}\), \(\mathfrak M_{uv}^{(R)}\), and
\(\mathfrak M_\nu^{(P)}\). Only the first is an author-indexed measurement.

Their L5 interfaces are not interchangeable:

| Interface | Permitted L5 object | Required uncertainty | Prohibited shortcut |
| --- | --- | --- | --- |
| `E45-V` | reference-relative author score or equivalence class | person, facet, reference, event, and operator uncertainty | no score without an independently frozen reference |
| `E45-R` | reference-relative dyadic relation or distance | node/dyad dependence, relation-valued confidence region, reference and operator uncertainty | no unregistered scalarization or attribution of a pair relation to one person |
| `E45-P` | population- or group-indexed relation summary | sampling, composition, reference, and transport uncertainty | no ecological-to-individual inference and no person ranking |

An L5 mechanism validated for one interface does not license another. In
particular, a population relation field cannot become an individual score by
factorization, averaging, or an LLM narrative without a separately identified
map and validation design.

The current executable `reference_measurement` contract is interpreted as an
`E45-V` contract for the existing L45 V3 study. Before any `E45-R` or `E45-P`
experiment, the machine schema must add an explicit output type and the
type-specific reference, dependence, uncertainty, detectable-difference, and
refusal fields. A generic contract name does not transfer the `E45-V` license.

### A7. Residual-agnosticism axiom

The unexplained remainder is not “noise” by default:

\[
r=
r_{\mathrm{technical}}
+r_{\mathrm{unobserved\ facet}}
+r_{\mathrm{model}}
+r_{\mathrm{innovation}}
+r_{\mathrm{stable\ individual}} .
\]

These components need not be orthogonal or separately identifiable. Only a
component isolated by a declared counterfactual, replicate, randomization, or
validated error model may be called removable technical noise.

### A8. Time-horizon axiom

State and trait are not labels attached to a feature family. They are claims
about generalization over occasions:

\[
\Theta_{\iota_\tau}^{(\tau,h)}
=
\mathbb E_{t\sim\mathcal T_h}
\left[\Theta_{\iota_\tau,t}^{(\tau)}\right],
\qquad \tau\in\{V,R,P\},
\]

where \(h\) indexes the time horizon and occasion universe, while \(\tau\)
continues to index output type. Changing \(h\) or the event opportunity
changes the estimand. Same-session stability cannot establish cross-occasion
trait stability; an author-level trait claim additionally requires
\(\tau=V\).

### A9. Invariance-is-earned axiom

Embedding alignment, similar distributions, or shared labels do not establish
measurement invariance. Language, culture, platform, population, interaction
partner, and operator transport each require a registered test and a refusal
region.

### A10. Evidence-noninheritance axiom

\[
E0\nRightarrow E1\nRightarrow E2\nRightarrow E3\nRightarrow E4\nRightarrow E5.
\]

Engineering correctness does not prove existence; synthetic recovery does
not prove a real-text object; reliability does not prove psychological
validity; validity does not prove beneficial use.

### A11. Observer-renderer axiom

An LLM may propose hypotheses, code bounded evidence, or render a registered
evidence packet. It may not silently alter the measurement object or act as
the final deterministic score:

\[
\text{score}=\mathcal E(\text{registered data}),\qquad
\text{explanation}=L(\text{score,evidence,counterevidence,boundary}).
\]

Explanation stability and fidelity are independent validation objects.

### A12. Refusal-preserves-knowledge axiom

Failure at layer \(L_j\to L_{j+1}\) invalidates promotion, not the supported
lower-layer result. “Anonymous technical structure exists” remains useful
when “this structure is personality” is refused.

## 5. Edge licenses

| Edge | Required assumptions | Minimum evidence | Mandatory refusal |
| --- | --- | --- | --- |
| E01 human process -> expression | response opportunity, audience, assistance, disclosure and selection process declared | response-process study, intervention, or deliberately bounded production model | refuse mental-process or intention inference |
| E12 expression -> record | capture, provenance, attribution, quotation/AI/editing and missingness rules | record audit and chain of custody | refuse authorship or complete-behavior claims |
| E23 record -> representation | frozen encoder/tokenizer/slice/operator and fit/apply separation | perturbation, gauge, slicing and version sensitivity | refuse coordinate or transport claims after operator drift |
| E34 representation -> technical object | estimand, dependence, support, null/alias worlds and uncertainty declared | source-disjoint recovery, negative controls, counterworlds | refuse object existence or unique mechanism |
| E45 typed technical object -> typed reference-relative measurement | output type, frozen reference population, universe, operator, type-compatible uncertainty and detectable-difference rule | type-specific G-study/D-study or dependence analysis, reference transport, information-budget analysis | refuse cross-type conversion, unsupported person score, relation scalarization, ecological-to-individual inference, change, or batch comparison |
| E56 typed measurement -> construct | construct theory, rival explanations, external anchors and response process | convergent, discriminant, behavioral and invariance evidence | refuse psychological naming |
| E67 construct -> use | intended use, horizon, base rates, costs, fairness, human authority and contestability | prospective field validation and consequences analysis | refuse diagnosis, intervention, ranking, or forensic decision |

## 6. Generalization universe and error object

Every study must declare:

\[
\mathcal U=
\mathcal P_{\mathrm{target}}
\times\mathcal P_{\mathrm{reference}}
\times\mathcal C
\times\mathcal O
\times\mathcal T
\times\mathcal M
\times\mathcal L
\times\mathcal I
\times\mathcal A
\times\mathcal R ,
\]

covering target/reference populations, context, opportunity, occasion/time,
operator, language/culture, interaction, administration, and response
process.

Each facet is one of:

- `declared`: sampled and bounded;
- `bounded_open`: the present study directly targets the missing license; or
- `not_applicable`: excluded by the claim's source and target layers.

The uncertainty object must separate at least event count from event
composition. When applicable it also separates person, context, opportunity,
occasion, method/operator, language/culture, interaction partner, response
process, reference population, event sampling, and technical replication.
Independence may not be assumed merely to make the decomposition additive.

## 7. Pre-experiment gates

No new GRAIN study may access expensive data or compute until its foundation
contract passes these gates:

| Gate | Question |
| --- | --- |
| G0 Claim | What exact functional is being estimated, and what statements are forbidden? |
| G1 Ontology | At which source and target layers does the claim begin and end? |
| G2 Alias | Which rival worlds generate the same or similar observables? |
| G3 Support | What population, contexts, opportunities, occasions, languages, interactions, and operators are represented? |
| G4 Operator | Which transformations are fitted, where are they fitted, and what remains invariant across admissible alternatives? |
| G5 Reference | Is any score reference-relative, independently normed, and accompanied by reference uncertainty? |
| G6 Error | Which remainder is identified technical error, and which remains an unresolved mixture? |
| G7 Falsification | Which nulls, knockouts, counterworlds, and negative controls can stop the claim? |
| G8 Transition | Does every crossed edge have exactly one license and refusal rule? |
| G9 Interpretation | Does wording stop at the maximum licensed layer? |

The executable version is:

```bash
python scripts/validate_suica_foundation_study.py \
  --contract configs/suica_foundation_study.template.json
```

Possible outcomes:

- `FOUNDATION_CONTRACT_READY`: all declared transitions are licensed;
- `READY_FOR_BOUNDED_FOUNDATION_TEST`: an opening is localized and the
  proposed experiment directly targets it, but promotion remains blocked;
- `REFUSE_EXPERIMENT_START`: the design contains a hidden jump, unresolved
  declaration, or incompatible use.

## 8. Current V8 placement

Current V8 evidence is strongest on L2 -> L3 -> L4:

- versioned text-to-representation operators exist;
- synthetic and real-text analyses support selected anonymous technical
  relation objects within registered designs;
- support, alias, resolution, and refusal machinery exists.

The weakest transitions are:

- L0 -> L1: human production and response-process evidence;
- L4 -> L5: frozen external reference, type-compatible uncertainty, and
  transportable typed-measurement construction;
- L5 -> L6: external psychological interpretation;
- L6 -> L7: field use, consequences, fairness, and authority.

This does not make V8 a failed scale. It correctly locates GRAIN as a
foundation whose technical core is more mature than its downstream
measurement and interpretation interfaces.

## 9. Binding doctrine for future work

1. Begin from a question and layer path, never from an attractive statistic.
2. Run one contract per estimand; do not bundle technical existence,
   personality naming, and use into one success criterion.
3. Preserve raw records and full transformation provenance.
4. Treat slice/operator alternatives as admissible views until invariance or
   a design-specific choice is justified.
5. Preserve context choice and conditional response as separate channels.
6. Never call an unidentified remainder noise.
7. Never call an L4 object a factor, trait, state, emotion, intelligence, or
   risk without an E56 license.
8. Never emit an individual score without a frozen reference and error
   interval.
9. Never allow an LLM explanation to exceed the deterministic evidence
   packet.
10. A bounded opening may motivate an experiment, but it may not support the
    conclusion the experiment is intended to test.

This standard is the GRAIN foundation constitution. Local protocols may add
stricter rules, but they may not weaken these boundaries.
