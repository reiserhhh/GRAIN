# GRAIN and Scale-Development Methodology: Alignment Audit

Date: 2026-07-30  
Status: external-methodology stress test, not a scale-development mandate or
scale-validation claim.

## Executive ruling

GRAIN is **not a scale and is not defined by scale construction**. It is a
foundational methodology for making text-based observation, estimation,
comparison, interpretation, and refusal explicit. Scale-development
literature is used here as an adversarial source of requirements: if a future
instrument is built on GRAIN, it should not repeat known psychometric
mistakes. This audit therefore constrains downstream designs without changing
GRAIN's identity or requiring the current corpora to produce a scale.

Classic scale-development requirements divide into three classes:

1. requirements already applicable to GRAIN's technical object and
   measurement interface;
2. requirements that become mandatory only when a named psychological
   construct and intended use are declared; and
3. questionnaire-specific procedures, such as item-response modeling, that
   are not yet applicable because natural text events are not fixed items.

GRAIN did not invent sampling, reliability, validity, or person-by-situation
reasoning. Its potentially original contribution is their methodological
combination for **natural-language behavioral observations**, with typed
inference layers, condition/opportunity provenance, axis-free relation
geometry, explicit support refusal, and a strict separation between
deterministic scoring and LLM explanation.

The binding foundation standard is
`docs/SUICA_FOUNDATIONAL_CLOSURE_STANDARD.md`. The present document tests that
standard against one mature family of downstream requirements; it does not
replace it.

The closest historical lineage is:

- behavioral/content-domain sampling and representative design;
- Generalizability Theory (persons crossed with situations, occasions, and
  methods);
- construct validity and nomological networks;
- multitrait-multimethod validation;
- latent state-trait theory;
- measurement invariance and differential functioning; and
- modern argument-based validation of score interpretations and uses.

## 1. Translation dictionary

| Conventional assessment term | GRAIN analogue | Important non-equivalence |
| --- | --- | --- |
| Item | text event, slice, prompt opportunity, or interaction turn | A text event is not a locally independent item with fixed difficulty. |
| Item/content universe | admissible topic/situation/opportunity universe | Reddit comments are a convenience sample, not a designed domain sample. |
| Test form | observation operator plus context/prompt schedule | Different slicing or collection regimes may define different estimands. |
| Response | produced text and its path through representation space | Production also depends on audience, platform, task, language, and AI partner. |
| Scale score | frozen reference-relative geometry functional | Current relation matrices are batch-relative technical objects, not independent norm scores. |
| Parallel forms | source-disjoint event sets under matched opportunity designs | Equal event count alone does not make forms parallel. |
| Norm group | frozen external reference population | Current PANDORA panels are local development cohorts, not population norms. |
| Factor | a replicated low-dimensional common direction, if identified | V8 presently resolves relation geometry without a stable global factor chart. |
| Measurement error | source, occasion, method, reference, and operator variability | Unexplained residual is not automatically noise; it may include stable author structure. |

## 2. What the historical literature requires

The modern validity position is that **interpretations and uses of scores are
validated, not an algorithm in the abstract**. More ambitious interpretations
need more evidence, and validating a score meaning does not automatically
validate its clinical or decision use
([Kane, 2013](https://doi.org/10.1111/jedm.12000);
[Standards for Educational and Psychological Testing](https://www.testingstandards.net/)).

Classic scale construction emphasizes a clear target construct, an initially
broad content pool, comparison with close rival constructs, thoughtful choice
of validation samples, and unidimensionality rather than maximizing internal
consistency alone
([Clark & Watson, 2019](https://doi.org/10.1037/pas0000626)).
The common three-phase summary is item/content development, scale
construction, and evaluation of dimensionality, reliability, and validity
([Boateng et al., 2018](https://doi.org/10.3389/fpubh.2018.00149);
[DeVellis & Thorpe, 2021](https://uk.sagepub.com/en-gb/eur/scale-development/book269114)).

For GRAIN, the decisive extension is that **situations and text opportunities
must be sampled as seriously as people**. Representative-design theory permits
generalization only to the ecology that was sampled
([Brunswik, 1955](https://doi.org/10.1037/h0047470)). Generalizability Theory
then asks how observed scores depend on persons, items/tasks, occasions,
raters, and settings rather than treating all inconsistency as one error term
([Brennan, 1992](https://doi.org/10.1111/j.1745-3992.1992.tb00260.x)).

Content evidence is also population- and use-specific. COSMIN operationalizes
it as relevance, comprehensiveness, and comprehensibility for the target
construct, population, and context of use
([Terwee et al., 2018](https://doi.org/10.1007/s11136-018-1829-0)).
This requirement does not name GRAIN's anonymous geometry; it becomes
mandatory when a future instrument claims to assess a psychological
construct.

## 3. Requirement-by-requirement audit

Status vocabulary:

- `COVERED`: an explicit current mechanism exists.
- `PARTIAL`: relevant machinery exists, but the psychometric requirement is
  not closed.
- `MISSING`: no adequate current closure.
- `APPLICATION_STAGE`: required only after a named construct/use is declared.
- `NOT_YET_APPLICABLE`: requires fixed items or a different response model.

| Requirement | Status | Existing GRAIN correspondence | Remaining gap |
| --- | --- | --- | --- |
| Intended construct and score use | `PARTIAL` | V8 has typed technical estimands, evidence levels, and claim boundaries. | It has no named psychological construct or intended clinical decision rule. This is correct for a method foundation, but no personality validity claim can precede that declaration. |
| Target population | `PARTIAL` | Corpus/support domains and refusal language are explicit. | No target-population sampling frame, inclusion probabilities, range-restriction analysis, or population coverage weights. PANDORA cannot stand for humanity. |
| Reference population and norming | `PARTIAL` | V7 specifies a reference-relative geometry bundle; the L45 V3 opened synthetic candidate now implements an independently frozen reference, composition correction, support checks, and reference uncertainty for an E45-V object. | This is not a sealed confirmation or a real human G/D study. Current V8 relation geometry remains cohort-relative, and E45-R/E45-P interfaces still need type-specific reference and transport rules. |
| Behavioral/content-domain definition | `PARTIAL` | Context, opportunity, event budget, choice, and observation operator are first-class objects. | No table of specifications defines which human situations, topics, registers, and response opportunities the future instrument intends to cover. |
| Topic/situation sampling | `PARTIAL` | Context is preserved rather than blindly residualized; fixed/free designs are distinguished. | Current contexts are available Reddit venues, not a representative sample of a declared ecological domain. The 15-schedule budget audit also shows that which observations enter a form can change the result. |
| Content validity | `APPLICATION_STAGE` | Blind coding and construct registries provide development diagnostics. | For a named application, experts and the target population must assess relevance, comprehensiveness, and comprehensibility; statistical clustering cannot substitute for this. |
| Response-process evidence | `MISSING` | AI echo, prompt versioning, and opportunity variables are recognized. | No cognitive interview/think-aloud evidence shows how people interpret prompts, choose disclosures, react to audience or AI wording, and produce the scored text. |
| Administration standardization | `PARTIAL` | Scorer, prompt, operator, and provenance hashes can be frozen. | Device, input modality, interaction partner/model, language, time pressure, assistance, quotations, and editing opportunities are not yet a complete administration manifest. |
| Observation/sample size | `PARTIAL` | Resolution frontiers and the 4/6/8/10/12 information-budget experiment directly estimate local support. | A budget is not a universal sample-size rule. Required authors and events must be derived for each estimand and use through power simulation and a D-study. |
| Internal structure/dimensionality | `COVERED` for current technical claim | GRAIN refuses a factor chart when stable axes are not identified and retains axis-free relation geometry. | Any future scalar subscale still needs independent-sample dimensionality testing. Factor-analysis sample size depends on communalities and factor overdetermination, not one universal respondent:item ratio ([MacCallum et al., 1999](https://doi.org/10.1037/1082-989X.4.1.84)). |
| Local dependence/autocorrelation | `PARTIAL` | Source-disjoint replicates, path operators, and block-aware diagnostics exist. | Texts from one thread/session are dependent. Classical alpha or ordinary item-factor assumptions are invalid without an explicit hierarchical/local-dependence model. |
| Reliability and Generalizability Theory | `PARTIAL` | Split replicates, event-budget curves, uncertainty components, and future G-study schemas exist. | No fitted `person × context × occasion × method × interaction-partner` G-study and no D-study for a declared administration design. |
| Standard error / minimum detectable difference | `PARTIAL` | V7 separates source, model, and reference uncertainty conceptually; L45 V3 propagates event, occasion, reference, and operator uncertainty and passes opened synthetic coverage/MDD gates for E45-V. | No sealed confirmation, real person-score SEM, relation-valued confidence region, smallest detectable human change, or decision consistency. |
| State, trait, and responsiveness | `PARTIAL` | GRAIN distinguishes level, path, state, choice, and condition response; affective tension was not mislabeled as trait. | Same-person repeated occasions are still insufficient. Latent state-trait theory requires separating consistency, occasion specificity, and measurement error probabilistically ([Steyer et al., 2015](https://doi.org/10.1146/annurev-clinpsy-032813-153719)). |
| Measurement invariance / DIF | `PARTIAL` | Support drift, opportunity sensitivity, and refusal are modeled. | Equal support is not equal measurement. Group, language, culture, platform, device, operator, and interaction-partner invariance/DIF remain open ([Meredith, 1993](https://doi.org/10.1007/BF02294825)). |
| Convergent/discriminant evidence | `APPLICATION_STAGE` | Big Five/MBTI links are kept external and claim-bounded. | A frozen future construct needs a nomological network and rival-construct tests, not one correlation. MTMM requires multiple traits and methods ([Campbell & Fiske, 1959](https://doi.org/10.1037/h0046016); [Cronbach & Meehl, 1955](https://doi.org/10.1037/h0040957)). |
| Criterion/predictive/incremental evidence | `APPLICATION_STAGE` | Behavioral and external-anchor paths are architecturally possible. | No current criterion licenses personality, diagnosis, treatment choice, or change monitoring. Predicting an existing questionnaire is not the only nor necessarily the strongest criterion. |
| Missingness and self-selection | `PARTIAL`, conceptually strong | GRAIN treats choice of topic/situation as an estimand rather than automatically deleting it as nuisance. | Eligibility and nonresponse are still MNAR risks. Selection models, inclusion funnels, and refusal coverage must accompany any population claim. |
| Cross-validation/replication | `PARTIAL`, technically strong | Synthetic falsification, D0/D1/D2 separation, source-disjoint tests, and builder/auditor separation exist. | The current RGC/transport/budget sequence repeatedly inspected the same panels. Fresh D3 or a new corpus is required for confirmation. |
| Fairness and consequences | `MISSING` | Current reports prohibit clinical use. | No subgroup fairness, accessibility, adverse-impact, feedback-harm, or high-stakes decision analysis. Fairness is a primary testing requirement ([NCME Fair Testing Code](https://ncme.org/document/code-of-fair-testing-practices-in-education/)). |
| Interpretability and LLM use | `PARTIAL` | Deterministic scoring is separated from LLM rendering; evidence and counterevidence can be attached. | Explanation fidelity, rater/model drift, calibrated uncertainty language, and human comprehension require direct validation. |
| IRT / item difficulty-discrimination | `NOT_YET_APPLICABLE` | No fixed item bank is assumed. | Many-facet or multidimensional IRT becomes relevant only if a future standardized prompt/task bank yields repeatable, locally modeled responses. It should not be retrofitted to arbitrary comments. |

## 4. Formal upgrade suggested by the audit

### 4.1 Declare the universe of generalization

GRAIN should define every future measurement design against:

\[
\mathcal U =
\mathcal P_{\mathrm{target}}
\times \mathcal C_{\mathrm{context}}
\times \mathcal O_{\mathrm{opportunity}}
\times \mathcal T_{\mathrm{occasion}}
\times \mathcal M_{\mathrm{operator}}
\times \mathcal L_{\mathrm{language/culture}}
\times \mathcal I_{\mathrm{interaction}}.
\]

The observed sample licenses inference only over the facets and ranges it
actually sampled. A fixed event count does not repair missing context,
language, occasion, or interaction support.

### 4.2 Add a relation-valued G-study

The current relation object should not be forced into a scalar additive ANOVA.
For a declared scalar functional \(\psi\) of an author relation matrix, fit:

\[
\psi(K^{obs}) =
\mu + P + C + T + M + L + I
+ P\!:\!C + P\!:\!T + P\!:\!M + \varepsilon.
\]

For the full positive-semidefinite relation object, use a registered
Hilbert-space or Fréchet variance decomposition only after proving that the
chosen metric and common reference operator make the objects comparable. A
matrix decomposition without a common score space is not a G-study.

The corresponding D-study asks how error changes when the design uses
\(n_C,n_T,n_M,n_I\) observations. This is the correct place to derive minimum
text and situation budgets; it replaces universal rules such as “twelve
comments are enough.”

### 4.3 Freeze an external reference panel

A future individual score requires an independent reference panel \(R\):

\[
s_u=\Phi\!\left(\{k(u,r):r\in R\};\,\mathcal U,\nu\right),
\]

with population definition, sampling weights, version \(\nu\), conditional
uncertainty, support checks, and drift rules. Without \(R\), GRAIN can validly
describe geometry inside one cohort but cannot claim a stable normed personal
score.

### 4.4 Separate four evidence objects

1. `TECHNICAL_GEOMETRY`: reproducible mathematical structure.
2. `MEASUREMENT_SCORE`: frozen reference-relative score with error theory.
3. `PSYCHOLOGICAL_CONSTRUCT`: score with content, response-process, internal,
   convergent/discriminant, and criterion evidence.
4. `USE_DECISION`: a declared application with fairness, consequences, and
   utility evidence.

No level inherits the next level automatically.

## 5. What GRAIN already contributes

1. It makes context, opportunity, and observation design explicit instead of
   silently treating naturally available text as interchangeable items.
2. It preserves self-selection as a possible psychological channel rather
   than automatically residualizing every group deviation.
3. It can retain reproducible relation geometry when no stable low-dimensional
   factor chart exists, while refusing to name that geometry prematurely.
4. It exposes information-budget and support frontiers rather than reporting
   a score for every input.
5. It separates deterministic measurement from LLM explanation and keeps an
   auditable claims ledger.

These are meaningful methodological differences. They do not exempt GRAIN
from population sampling, response-process evidence, error theory,
invariance, external validation, or fairness.

## 6. Priority closure sequence

### Before human data collection

1. Publish a `GeneralizationUniverseManifest` schema covering target
   population, context/topic/situation universe, opportunity, occasion,
   operator, language/culture, interaction partner, and intended score use.
2. Build a synthetic relation-valued G-study/D-study harness with planted
   person, context, occasion, method, language, and interaction effects.
3. Define the fixed-reference scoring functional and show batch invariance,
   reference-resampling uncertainty, drift detection, and `SCORE_REFUSE`.
4. Treat the nested-budget result as content-sampling evidence: estimate both
   event-count error and event-composition error rather than choosing one
   count from a single ladder.
5. Specify conditional SEM and minimum detectable difference for every future
   score functional.

### When one applied instrument is designed

1. declare construct, target population, context of use, and forbidden uses;
2. create a representative behavior/situation sampling frame;
3. obtain target-population and expert content/response-process evidence;
4. collect independent development and confirmation samples with repeated
   occasions and alternate forms;
5. test reliability, G-theory, dimensionality where applicable, invariance,
   MTMM/nomological relations, behavioral criteria, and incremental value;
6. set norms or cut scores only with separate justification, uncertainty,
   fairness, and consequence analyses.

## Final judgment

GRAIN's current formulas correspond strongly to the **sampling and
generalizability problem** that traditional scale development often hides
inside fixed items. They correspond partially to reliability, internal
structure, and validation governance. They do not yet cover the full
psychometric chain needed for person-level interpretation: a representative
target/reference population, response-process evidence, complete
person-by-facet error theory, invariant scoring, norm uncertainty, and
consequences of use.

That is not evidence that GRAIN is methodologically wrong. It locates its
current maturity precisely: a technically developed measurement foundation
whose next theoretical advance should be a **generalization-universe and
reference-score theory**, not another unconstrained factor search.

## Core references

- AERA, APA, & NCME. (2014). [*Standards for Educational and Psychological Testing*](https://www.testingstandards.net/).
- Boateng, G. O., et al. (2018). [Best practices for developing and validating scales](https://doi.org/10.3389/fpubh.2018.00149).
- Brennan, R. L. (1992). [Generalizability Theory](https://doi.org/10.1111/j.1745-3992.1992.tb00260.x).
- Brunswik, E. (1955). [Representative design and probabilistic theory](https://doi.org/10.1037/h0047470).
- Campbell, D. T., & Fiske, D. W. (1959). [Convergent and discriminant validation by the MTMM matrix](https://doi.org/10.1037/h0046016).
- Clark, L. A., & Watson, D. (2019). [Constructing validity](https://doi.org/10.1037/pas0000626).
- Cronbach, L. J., & Meehl, P. E. (1955). [Construct validity in psychological tests](https://doi.org/10.1037/h0040957).
- DeVellis, R. F., & Thorpe, C. T. (2021). [*Scale Development: Theory and Applications*, 5th ed.](https://uk.sagepub.com/en-gb/eur/scale-development/book269114).
- Kane, M. T. (2013). [Validating the interpretations and uses of test scores](https://doi.org/10.1111/jedm.12000).
- MacCallum, R. C., et al. (1999). [Sample size in factor analysis](https://doi.org/10.1037/1082-989X.4.1.84).
- Meredith, W. (1993). [Measurement invariance, factor analysis and factorial invariance](https://doi.org/10.1007/BF02294825).
- Steyer, R., et al. (2015). [A theory of states and traits—revised](https://doi.org/10.1146/annurev-clinpsy-032813-153719).
- Terwee, C. B., et al. (2018). [COSMIN methodology for content validity](https://doi.org/10.1007/s11136-018-1829-0).
