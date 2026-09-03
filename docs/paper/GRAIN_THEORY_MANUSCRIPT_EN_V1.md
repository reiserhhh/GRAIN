# Natural-Language Psychological Measurement as Context-Conditional Multilevel Inference: The GRAIN Method

**Article type:** Theoretical and methodological article  
**Manuscript status:** English master draft, version 1  
**Running head:** CONTEXT-CONDITIONAL LANGUAGE MEASUREMENT  
**Authors and affiliations:** [To be supplied before submission]  
**Corresponding author:** [To be supplied before submission]

> Editorial note. This is a journal-neutral APA-style manuscript. It presents
> Group-Referenced Analysis of Individual Natural-Language Expression (GRAIN) as an analysis method for psychological measurement,
> not as a finished test, diagnostic instrument, or validated taxonomy of
> personality. Journal-specific length, anonymization, and disclosure rules
> must be applied after a target journal is selected.

## Abstract

Natural-language data provide dense records of behavior, but prediction from
text is not equivalent to psychological measurement. Texts are produced under
particular opportunities, prompts, audiences, platforms, and histories; people
also select some of these conditions. Consequently, removing contextual
variation can delete person-relevant behavior, whereas ignoring context can
confound persons with their environments. We present Group-Referenced Analysis
of Individual Natural-Language Expression (GRAIN), an integrative method for
developing natural-language measurement systems under this dual constraint.
We call the reproducible multiscale organization of local language events an
individual's **language texture**; a single sentence or embedding is not a
texture. Texture comprises distributions, orientations, condition choices,
conditional responses, interactions, and paths relative to an explicit
reference frame.
The method treats tokenization, segmentation, and representation as explicit
observation operators; separates condition selection from conditional
response; distinguishes stable person differences, time-varying states,
person-by-condition response functions, and path dependence; and requires
separate inferences for individual scores, relations, and population
structure. Scores are defined relative to a declared reference population,
opportunity distribution, observation design, and operator version. Technical
structure may be estimated before it is psychologically named, but construct
interpretation requires convergent, discriminant, behavioral, longitudinal,
and invariance evidence. An audited research program illustrates the need for
these restrictions: strong author-reproducible structure coexisted with weak
links to Big Five labels, and between-person and within-person relations could
differ in magnitude or direction. We specify a development workflow,
falsifiable propositions, abstention rules, and a constrained role for large
language models. GRAIN's contribution is an integrated measurement logic for
turning situated language behavior into testable psychological candidates
without equating predictive accuracy, author identification, or semantic
plausibility with validity.

**Keywords:** natural-language assessment; psychological measurement;
multilevel modeling; person-situation interaction; construct validity;
computational psychometrics

## 1. Introduction

Language is an unusually rich record of human behavior. It can be collected
across long periods, generated without a questionnaire, and analyzed at scales
that are inaccessible to manual coding. Research has linked language use to
personality, affect, demographic characteristics, and behavior using
closed-vocabulary dictionaries, open-vocabulary models, supervised machine
learning, and, more recently, large language models (Kern et al., 2021; Park et
al., 2015; Pennebaker & King, 1999; Schwartz et al., 2013; Yarkoni, 2010).
These developments make natural language attractive as a possible basis for
psychological assessment.

The same developments create a measurement problem. A model can predict a
questionnaire score without measuring the intended construct. It may exploit
topic, platform, demographic composition, test-report leakage, or other
correlates that are useful in one sample but change elsewhere. Conversely,
procedures designed to remove topic or situation can erase stable selection
patterns and conditional responses that are themselves part of behavior.
High prediction accuracy, temporal stability, and convincing model-generated
explanations answer different questions. None alone establishes that a score
has the same psychological meaning across people, occasions, contexts, or
uses (American Educational Research Association et al., 2014; Cronbach &
Meehl, 1955; Jacobs & Wallach, 2021).

This problem is not unique to language. Personality theory has long treated
behavior as jointly determined by persons and situations. Cognitive-affective
system theory describes stable individual differences in conditional
``if-then'' behavioral patterns (Mischel & Shoda, 1995; Shoda et al., 1994),
and density-distribution approaches represent traits through distributions of
states rather than a single behavior (Fleeson, 2001). Latent state-trait theory
formalizes the fact that observations are fallible, situation-dependent,
method-specific, and historically situated (Steyer et al., 2015).
Generalizability theory similarly requires researchers to specify the universe
of observations over which a score is intended to generalize (Brennan, 2001).
Natural-language assessment needs to incorporate these principles at the level
of data representation and computation.

We present Group-Referenced Analysis of Individual Natural-Language Expression (GRAIN) as an analysis method for
that task. GRAIN is not a personality scale and does not prescribe a universal
set of language factors. It is a method for developing study-specific
measurement systems from language. It asks four linked questions: under what
conditions was expression possible, what behavior was recorded, which
cross-person differences are reproducible, and over what temporal horizon do
they persist? These questions correspond to familiar concerns in
person-situation theory, psychometrics, multilevel modeling, longitudinal
analysis, and computational measurement. The contribution lies in integrating
them into one inferential chain with explicit stopping rules.

The paper has five aims. First, we distinguish prediction, technical
measurement, psychological interpretation, and use. Second, we formalize text
as condition-dependent behavior observed through fallible computational
operators. Third, we distinguish individual, relational, and population
claims, including between-person and within-person relations. Fourth, we
provide a reproducible development and validation workflow. Fifth, we define
the appropriate role of large language models as constrained coders and
explanation generators rather than unversioned scoring authorities.

## 2. Conceptual Foundations

### 2.1 Prediction is not measurement

Prediction evaluates whether information in an input improves an estimate of
an outcome under a specified sampling distribution. Psychological measurement
additionally requires an account of the attribute, the response process, the
observation method, the reference frame, error sources, and the intended
interpretation. A predictive association can contribute evidence to a validity
argument, but it does not define the construct (Borsboom et al., 2004;
Cronbach & Meehl, 1955; Messick, 1995). This distinction is especially important when flexible models
can exploit many unknown proxies.

GRAIN therefore separates four questions:

1. **Technical existence:** Is a reproducible structure present in the
   recorded language under a frozen computational procedure?
2. **Measurement comparability:** Can that structure be compared under a
   declared reference population and generalization universe, with quantified
   uncertainty?
3. **Construct interpretation:** Does independent evidence support a named
   psychological or behavioral interpretation?
4. **Use validity:** Does the resulting interpretation support a particular
   decision without unacceptable harm?

Evidence does not automatically transfer from one question to the next. An
author-identification result, for example, demonstrates persistent
individual-specific information but does not establish personality validity.
Likewise, a population-level correlation does not license classification of an
individual. The Standards for Educational and Psychological Testing place
validity on interpretations for intended uses rather than on a test in the
abstract (American Educational Research Association et al., 2014). GRAIN
applies that principle to computational language measurements.

### 2.2 Text is situated and selected behavior

Text is never produced in a situational vacuum. A speaker or writer responds
to available opportunities, prompts, audiences, platform conventions,
interlocutors, and prior events. Some conditions are assigned, whereas others
are selected. Representative-design arguments emphasize that psychological
inference depends on sampling the environments to which it is intended to
generalize (Dhami et al., 2004). In observational language corpora, selection
into contexts is also behavior.

This creates a dual role for context. Within a condition, context is a basis
for fair comparison: two people cannot be assumed to have had the same
opportunity to express a feature when one was prompted to narrate a loss and
the other to solve a technical problem. Across conditions, however, repeated
selection, avoidance, return, and combination can carry person-relevant
information. Globally regressing out context confounds these roles. Globally
ignoring it does the same.

Accordingly, GRAIN separates:

\[
C_{ut}\sim \pi_u(c\mid O_{ut},H_{ut}),
\tag{1}
\]

from

\[
P(X_{ut}\mid C_{ut}=c,O_{ut},H_{ut},u),
\tag{2}
\]

where \(u\) indexes a person, \(t\) an event or occasion, \(O_{ut}\) the
available opportunity set, \(H_{ut}\) the observed history, \(C_{ut}\) the
selected or assigned condition, and \(X_{ut}\) the recorded expression. The
first distribution represents condition selection; the second represents
conditional response. Their psychological meanings must be validated
separately.

### 2.3 Stable differences and variability are complementary

A person's average behavior and their pattern of variation can both be
informative. The person-situation literature shows that stable conditional
patterns can coexist with substantial within-person variability (Fleeson,
2001; Mischel & Shoda, 1995). GRAIN therefore does not define personality as
the context-free remainder, nor state as whatever is left after averaging.
Trait and state are claims about generalization over specified occasion and
time universes.

At minimum, a language-based trait claim requires evidence that a person-level
quantity persists across appropriately sampled occasions and relevant
contexts. A state claim requires sensitivity to within-person change at the
declared temporal resolution and discrimination from technical and
opportunity-driven variation. A quantity may remain unnamed while these
properties are studied. Psychological labels should be attached only after
independent construct evidence is available.

### 2.4 Computational representations are measurement methods

Tokenization, segmentation, embedding, feature construction, pooling, and
normalization are not neutral preprocessing. They determine which distinctions
are observable. Let \(S_\lambda\) be a segmentation or granularity operator
and \(R_v\) a versioned representation operator. Then

\[
Z_{ut}^{(v,\lambda)}=R_v\{S_\lambda(X_{ut})\}.
\tag{3}
\]

An unsegmented document, a fixed token window, an utterance, and an occasion
are different observation methods. None is presumed universally optimal.
Fine segmentation may preserve transition information but increase sampling
error; coarse aggregation may stabilize means while destroying order,
recovery, hysteresis, and coupling. Similarity across admissible operators is
evidence of robustness. Disagreement is information about method dependence,
not a reason to select the most favorable representation after observing an
outcome.

## 3. A Context-Conditional Multilevel Model

### 3.1 The inferential chain

GRAIN distinguishes the human process from each observable and inferential
layer:

\[
(\mathcal H,F)
\rightarrow B
\rightarrow X
\rightarrow Z
\rightarrow A
\rightarrow M
\rightarrow \Theta
\rightarrow D.
\tag{4}
\]

Here \((\mathcal H,F)\) denotes unobserved history and current psychological
processes; \(B\) is behavior made expressible by the design; \(X\) is the
recorded text and metadata; \(Z\) is a computational representation; \(A\) is
an uninterpreted technical variable; \(M\) is a reference-relative measurement
candidate; \(\Theta\) is a psychologically interpreted construct; and \(D\)
is a use-specific output. Every arrow is conditional on assumptions and
evidence. When an arrow is unsupported, inference stops at the preceding
layer.

This ordering prevents two common shortcuts. First, semantic plausibility does
not turn \(Z\) or \(A\) directly into a psychological construct. Second,
reliability at \(M\) does not establish that \(\Theta\) is the intended
construct. A stable technical coordinate can represent a topic preference,
community membership, demographic proxy, learned writing habit, or platform
adaptation.

### 3.2 A minimal event model

A useful working model for represented expression is

\[
Z_{ut}
=\mu(C_{ut},O_{ut},t)
+a_u
+B_u\phi(C_{ut},O_{ut})
+s_{ut}
+h_u(H_{ut},C_{ut})
+\varepsilon_{ut}.
\tag{5}
\]

The terms have deliberately limited interpretations:

- \(\mu\) is the shared structure associated with condition, opportunity,
  and time in the reference population;
- \(a_u\) is a person's position under the reference conditions;
- \(B_u\) is a person-specific response function for observed changes in
  conditions or opportunities;
- \(s_{ut}\) is time-varying structure at the declared horizon;
- \(h_u\) contains history-dependent or nonlinear structure not represented by
  the current basis; and
- \(\varepsilon_{ut}\) is remaining variation under the current model.

Equation 5 is a decomposition of estimands, not a claim that every component
is identifiable from one observational corpus. Assigned conditions, repeated
occasions, cross-fitting, negative controls, or external anchors are needed to
separate particular terms. The model's function is to make confounding and
nonidentification explicit.

### 3.3 Person-specific dynamics

Sequential language can contain information absent from marginal averages.
Let \(K_u\) denote a person-indexed transition kernel:

\[
K_u(ds',dc,d\xi\mid H_t,s_t,c_t,o_t),
\tag{6}
\]

which describes a next state, condition transition, and expression event
given history, current state, condition, and opportunity. Autocorrelation,
change points, recurrence, recovery time, spectral power, and cross-feature
coupling are projections of this object. They should not be treated as
complete dynamics or named as traits solely because they are reproducible.

This distinction also clarifies the status of movement-only structure. A
dynamic pattern can be a real, author-reproducible technical object even when
it does not form a stable low-dimensional factor or admit an immediate verbal
label. Its psychological interpretation remains open until it is linked to
behavior, interventions, or established measures under an appropriate design.

### 3.4 Individual, relational, and population outputs

Three output types must be kept separate:

1. **Individual:** a person-indexed vector or score candidate, \(M_u^{(V)}\);
2. **Relational:** a dyadic distance, similarity, or cross-construct relation,
   \(M_{uv}^{(R)}\); and
3. **Population:** a distribution or context-indexed relation structure,
   \(M_\nu^{(P)}\).

A population structure can be recovered even when individual prediction is
weak. This is mathematically possible because covariance patterns aggregate
small, noisy signals over many observations. It remains a population-level
finding, and group-to-individual generalization requires separate evidence
(Fisher et al., 2018; Molenaar, 2004). Factorizing a relation matrix, averaging group estimates, or asking
an LLM to narrate the result does not convert it into a calibrated person
score.

### 3.5 Reference-relative measurement

An author score has no universal meaning outside its reference frame. A
measurement candidate is therefore defined as

\[
M_u
=\Psi\left(\widehat{\mathcal A}_u;
P_0,Q_0,S_\lambda,R_v,\mathcal U,\mathcal C\right),
\tag{7}
\]

where \(P_0\) is the reference population, \(Q_0\) the opportunity
distribution, \(S_\lambda\) and \(R_v\) the observation operators,
\(\mathcal U\) the generalization universe, and \(\mathcal C\) the calibration
version. Every reported result should carry these elements, together with
support, uncertainty, and conditions under which no score should be returned.

Batch standardization can support exploration but does not create a reusable
individual metric. A study intended to produce individual scores requires a
frozen external reference sample or cross-fitted reference estimates,
person-level uncertainty, a minimum detectable difference, and tests of
reference transport. Measurement invariance is not established by common
embedding dimensions, similar score distributions, or multilingual model
coverage (Meredith, 1993).

### 3.6 Between-person, within-person, and person-specific relations

For two observed channels \(x\) and \(y\), a between-person regression
coefficient can be written

\[
\beta_B=
\frac{\operatorname{Cov}_u(\bar x_u,\bar y_u)}
     {\operatorname{Var}_u(\bar x_u)},
\tag{8}
\]

whereas the pooled within-person coefficient is

\[
\beta_W=
\frac{\sum_{u,t}(x_{ut}-\bar x_u)(y_{ut}-\bar y_u)}
     {\sum_{u,t}(x_{ut}-\bar x_u)^2}.
\tag{9}
\]

This is the fixed-effects within-person slope. It is weighted by each author's
within-person variation in \(x\), rather than being an unweighted mean of the
person-specific coefficients in Equation 10.

A person-specific coefficient is

\[
\beta_u=
\frac{\sum_t(x_{ut}-\bar x_u)(y_{ut}-\bar y_u)}
     {\sum_t(x_{ut}-\bar x_u)^2}.
\tag{10}
\]

The difference

\[
\Delta_T=\beta_B-\beta_W
\tag{11}
\]

is a projection-specific cross-level transport diagnostic. It is not a test of
ergodicity for the full process. Common physical units make the coefficients
comparable, but do not establish equality of meanings, response functions, or
temporal laws. A confidence interval containing zero indicates an unresolved
difference, not equivalence; equivalence requires a prespecified margin.

Person-specific estimates also require reproducibility analysis. If

\[
\widehat\theta_{u,h}=\theta_u+d_{u,h}+e_{u,h},
\tag{12}
\]

then covariance across two partitions includes stable person variation,
shared drift, correlated error, and cross-terms. Split-sample correlation is
therefore evidence of reproducibility under that partition rule and horizon,
not proof of an error-free or timeless trait.

### 3.7 Residuals are unresolved structure, not disposable noise

In high-dimensional language data, the residual can include several sources:

\[
r=r_{\mathrm{technical}}+r_{\mathrm{opportunity}}
+r_{\mathrm{model}}+r_{\mathrm{innovation}}
+r_{\mathrm{stable\ individual}}.
\tag{13}
\]

These components need not be orthogonal or separately identified. A component
should be called removable noise only when a replicate, randomization,
counterfactual, or validated error model supports that interpretation.
Otherwise, residualization may remove psychologically relevant selection or
response structure. The methodological objective is not to drive total
residual variance to zero. It is to determine which variation is reproducible,
which is conditional on opportunity, which changes with the observation
operator, and which remains unidentified.

## 4. Development and Validation Procedure

### 4.1 Define the intended inference before modeling

A study should begin with a construct theory, target population, intended use,
and explicit output type. When no construct is yet justified, the study may
instead declare uninterpreted technical variables and a discovery objective. In
that case, candidate codes should remain semantically neutral until external
evidence is obtained. Exploratory naming based on attractive text examples or
LLM summaries is not construct validation.

The generalization universe should specify at least population, language and
culture, platform or setting, condition, opportunity set, occasion and time
horizon, interlocutor, administration, and computational operator. Each facet
must be fixed, sampled, modeled, or explicitly excluded from the claim.

### 4.2 Use complementary collection designs

Naturalistic and controlled text serve different purposes. A free-expression
phase samples naturally selected contexts and supports analysis of selection
distributions and ecological language behavior. A repeated fixed-condition
phase supports comparison of conditional responses because the expressive
opportunity is held more nearly constant. A conversational phase can identify
interaction, adaptation, repair, and response to an interlocutor.

No phase is universally superior. A fixed prompt can improve internal
comparability while reducing ecological coverage. Naturalistic writing can
sample a person's verbal ecology while confounding response with opportunity
and platform history. The design should therefore match the intended
inference, and multiple occasions are required when temporal generalization is
claimed. One long document is not a test-retest design.

### 4.3 Freeze observation operators before confirmation

Segmentation rules, tokenization, embedding models, feature transformations,
pooling, dimensionality reduction, and model selection must be fitted only in
development data or within resampling folds. The confirmatory sample must not
determine the operator. Because no single segmentation is assumed optimal,
the development stage may compare a registered family of views: unsegmented,
token-window, sentence, utterance, occasion, and multiscale representations.
The confirmatory claim should concern either one frozen view or a prespecified
invariant across views.

### 4.4 Separate discovery, calibration, and confirmation

An adequate workflow uses disjoint information stages:

1. **Discovery:** identify reproducible technical candidates and rival
   explanations without final outcome testing.
2. **Calibration:** fix reference populations, score transformations,
   uncertainty estimators, support thresholds, and abstention rules.
3. **Confirmation:** test frozen candidates and directional hypotheses in new
   participants, events, or a sealed holdout.
4. **Transport:** evaluate the same estimand under new populations, settings,
   languages, or operators without assuming invariance.

Cross-validation protects predictive evaluation only when all preprocessing
and selection occur within folds. It does not replace independent construct
validation or prospective use validation.

### 4.5 Evaluate distinct measurement properties

At minimum, a natural-language measurement study should report:

- event and person sample sizes, missingness, opportunity coverage, and text
  volume;
- score distributions and effective information;
- stability across resampled text, occasions, and admissible observation
  operators;
- variance components for persons, conditions, occasions, and interactions;
- sensitivity to text length, formatting, quotation, platform, assistance,
  and leakage;
- negative controls, including label permutation and identity leakage;
- convergent and discriminant evidence using independent methods;
- behaviorally meaningful criteria and longitudinal predictions;
- measurement invariance or explicit noninvariance across intended groups;
  and
- uncertainty, minimum detectable differences, and abstention frequency.

Generalizability studies can estimate how persons, conditions, occasions, and
operators contribute to score variance, and decision studies can estimate the
number and composition of observations needed for a particular use (Brennan,
2001). Multitrait-multimethod designs are especially relevant because language
features, questionnaire reports, behavioral observations, and informant
ratings have different method variance (Campbell & Fiske, 1959).

### 4.6 Interpret anonymous candidates only after external linkage

A technical coordinate can be useful before it is named. Researchers can
estimate its distribution, reproducibility, conditional response, temporal
dynamics, relations to other coordinates, and sensitivity to operators. A
psychological name should be considered only after a nomological network has
been tested. This should include multiple forms of evidence rather than one
correlation with a familiar scale.

For example, a candidate state coordinate may be linked prospectively to a
subsequent behavior:

\[
Y_{u,t+\delta}
=\alpha+\beta M_{ut}+\gamma^\top C_{ut}+b_u+\epsilon_{ut},
\tag{14}
\]

where \(M_{ut}\) is frozen before the outcome is observed and \(C_{ut}\)
contains registered condition and opportunity controls. Even a replicated
association licenses only the interpretation supported by the broader
validity evidence. It does not justify a clinical or personality label by
itself.

### 4.7 Separate deterministic scoring from LLM-supported interpretation

Large language models can assist codebook development, blind qualitative
coding, hypothesis generation, and human-readable explanation. They can also
introduce prompt sensitivity, stochasticity, model-version drift, semantic
overreach, and hidden training-data dependence (Demszky et al., 2023). Direct
numeric prompting can further produce arbitrary response bunching, and recent
work suggests that alternative elicitation and aggregation strategies may be
needed even for scalar text constructs (Licht et al., 2025).

GRAIN therefore requires a separation of functions:

\[
\text{score}=f_{\mathrm{frozen}}(X),
\qquad
\text{explanation}=L(M,U,E^+,E^-,B),
\tag{15}
\]

where \(f_{\mathrm{frozen}}\) is a deterministic, versioned scoring procedure;
\(M\) is the resulting measurement candidate; \(U\) is uncertainty; \(E^+\)
and \(E^-\) are supporting and countervailing evidence; and \(B\) is the
interpretive boundary. The LLM may render this packet but may not change the
score, invent a construct, suppress uncertainty, override abstention, or turn
a group-level result into an individual diagnosis. Explanation fidelity and
stability require their own validation.

## 5. Empirical Motivation and Current Evidence Boundary

GRAIN developed through repeated attempts to improve personality inference
from naturalistic text. The program began with the intuition that topic and
situation were nuisance variance. Empirical tests showed that unconditional
centering could reduce reproducibility because it removed person-specific
selection and response. Later work showed that author-reproducible information
could remain high-dimensional, operator-dependent, and weakly connected to
Big Five scores. These findings motivated the present separation of technical
structure, measurement, and psychological interpretation.

The current evidence comes primarily from PANDORA, a large Reddit corpus with
partial personality labels (Gjurković et al., 2021), together with controlled
synthetic studies and smaller external corpora. The completed audit covers
10,296 PANDORA authors and 17,640,062 parseable comment rows. These are the
repository's analysis-census counts under its frozen parser and eligibility
rules, not replacements for the corpus article's published summary counts.
The Big Five
subset contains 1,401 eligible authors under the project's prepared protocol;
8,895 other authors form a source-disjoint cohort for label-free structural
checks. The principal findings relevant to the theory are summarized in Table
1. They are methodological evidence within named corpora, not validation of a
clinical or reusable personality instrument.

**Table 1**  
*Selected evidence motivating the framework and its limits*

| Question | Audited result | Licensed interpretation |
|---|---:|---|
| Does condition removal necessarily improve measurement? | Across the initial registered style measures, unconditional condition centering reduced disjoint-occasion stability (mean change approximately -.09). | Context cannot be treated globally as removable nuisance; selection and conditional response require separate models. |
| Can a preregistered first instrument claim broad personality validity? | The sealed first test confirmed 2 of 7 directional hypotheses; its omnibus success criterion failed. | Narrow associations may exist, but the original broad claim was not supported. |
| Is author-reproducible structure equivalent to Big Five structure? | Split-sample reproducibility was high for several author-level selection and timing coordinates, whereas registered multivariate Big Five links were mostly not detected. | Stable author information is not sufficient evidence of personality validity. |
| Do between-person and within-person relations coincide? | For one registered relation, \(\beta_B=.069\), \(\beta_W=-.007\), and \(\Delta_T=.076\), 95% CI [.056, .097]; both slope intervals excluded zero and the point estimates had opposite signs. | The relation differed across levels in this analysis; population and within-person interpretations must be separated. |
| Can persistent person-specific response be estimated? | The registered person-by-context variance-component estimate was .019, 95% CI [.017, .019], and its split-sample reproducibility was \(r=.279\), 95% CI [.260, .293]. | A small positive component was estimated with modest split-sample reproducibility at the tested horizon; it is not yet a named psychological trait. |
| Does current language geometry connect strongly to Big Five geometry? | A registered selection-distance relation was small (Mantel \(r\approx .049\)); several expression and dynamic coordinates were within realized null bands. | External links are weak and coordinate-specific; the framework is not a Big Five replacement. |

The analysis provenance for every numerical statement in Table 1 is recorded
in the companion reproducibility map
`GRAIN_THEORY_EVIDENCE_PROVENANCE.md`.

The program also demonstrated an important positive methodological result: one
dense event stream can support separate between-person, pooled within-person,
and person-specific analyses in common physical units. However, common units
did not imply measurement invariance, and repeated analyses of the same corpus
were treated as analytical recurrence rather than independent replication.
This distinction was added after audit and is retained as a permanent boundary.

Synthetic experiments were used to test estimator recovery, aliasing, and
failure cases before contact with outcomes. They support the mathematical
possibility of separating selection, response, temporal, interaction, and
higher-order path mechanisms under specified generating conditions. They do
not establish that these mechanisms are prevalent in humans or correspond to
particular psychological constructs.

## 6. Falsifiable Propositions

The framework is intended to constrain research rather than merely redescribe
any result. The following propositions can fail.

### Proposition 1: Context has separable selection and response roles

When conditions are partly selected, the person's selection distribution and
conditional response are distinct estimands unless their empirical redundancy
is demonstrated. For a prespecified target domain, the need for separate
estimation is rejected if models that retain versus collapse the two channels
yield equivalent out-of-sample quantities, within a declared tolerance, under
designs that observe or experimentally control opportunity sets. Such a result
is domain-specific; it does not establish universal redundancy.

### Proposition 2: Measurement conclusions are operator-indexed

For a prespecified target and operator family, measurement conclusions are
operator-dependent when the relevant estimates vary beyond a declared
tolerance. A claim of operator robustness instead requires equivalent
relations or bounded sensitivity across that family. Either conclusion is
target-specific: identifying one operator-invariant object does not imply that
other structures are invariant.

### Proposition 3: Reference populations are part of the score definition

Changing the reference population or opportunity distribution can change a
person's relative score even when their recorded text is fixed. A reusable
absolute score requires an explicit invariance argument, not batch
standardization.

### Proposition 4: Stable and dynamic information are nonredundant in some domains

For at least some targets, sequence-sensitive representations will contain
reproducible information not recoverable from marginal means. Conversely, a
dynamic statistic that adds no reproducible or outcome-relevant information
should be discarded for that target.

### Proposition 5: Cross-level transport is relation-specific

Equality of a between-person and within-person relation cannot be assumed from
common variables or units. It must be tested for each registered relation and
time horizon. Full process claims require stronger conditions than agreement
of one slope.

### Proposition 6: Residual variance can contain lawful person structure

Residualization against a group basis can leave reproducible individual
structure. Such structure remains technically meaningful but psychologically
unnamed until competing explanations are tested.

### Proposition 7: Construct validity cannot be inherited from an embedding or LLM

An embedding's broad semantic competence or an LLM's plausible explanation
does not establish the construct validity of a score. If external validity,
invariance, and response-process evidence fail, the psychological
interpretation must be withdrawn while the technical result may remain.

### Proposition 8: Abstention is an informative outcome

When support, invariance, uncertainty, or interpretive evidence is inadequate,
not returning a score preserves the boundary of current knowledge. A system
that is forced to score every text cannot distinguish measurement from
extrapolation.

## 7. Implications for Psychological Science

### 7.1 Natural language can support measurement development without becoming a questionnaire

The framework does not require that language-derived dimensions reproduce the
Big Five, MBTI, or any existing inventory. Existing measures can provide
external evidence, but they are neither exhaustive descriptions of persons nor
automatic ground truth for every language-derived object. Uninterpreted technical
variables can be studied first and later linked to behavior and established
constructs. This allows discovery while retaining the requirements of
psychological measurement.

At the same time, freedom from existing scales does not license unconstrained
interpretation. A new name should summarize a validated nomological network,
not an LLM's reading of representative texts. The distinction between
discovery and naming is central to avoiding reification of computational
artifacts.

### 7.2 State and trait measurement should share a framework but not an estimand

The same event stream may support relatively stable person differences and
time-varying state candidates. Their estimation differs. Trait-oriented
inference emphasizes generalization across occasions and conditions; state
inference emphasizes within-person sensitivity at a declared temporal scale.
Neither is a residual category. A future clinical application might combine
them, but each output would require separate reliability, validity, alerting,
and consequence studies.

### 7.3 Multilingual use requires empirical invariance, not model confidence

Multilingual embeddings make common computation possible, but they do not
remove cultural differences in disclosure, genre, pragmatics, platform use, or
opportunity. International comparison is therefore a high-value use case and a
high-risk inference. The same operator can be applied across languages only as
an initial technical bridge. Measurement invariance, differential functioning,
and reference-population effects must be evaluated empirically.

### 7.4 The framework supports, but does not yet authorize, clinical assessment

Natural language could eventually reduce burden by supplementing interviews,
repeated questionnaires, or behavioral observation. Clinical use raises a
higher evidential threshold: prospective sampling, clearly defined outcomes,
within-person calibration, uncertainty, fairness, human review, contestability,
and analysis of consequences. Current GRAIN evidence does not support
diagnosis, risk classification, treatment selection, or automated judgments
about an individual.

### 7.5 LLMs are useful at the interpretive boundary

LLMs can translate high-dimensional evidence into language and compare a
frozen evidence packet with an explicit coding rubric. This is valuable where
technical objects are difficult to communicate. The model's output should be
treated as a fallible, versioned interpretation method. Human-readable
explanation is not a substitute for score provenance, and fluent language is
not evidence of causal understanding.

## 8. Limitations and Research Priorities

First, most human evidence in the completed development program comes from
self-selected Reddit users. Platform norms, community structure, demographic
selection, and self-report practices limit generalization. The source-disjoint
cohort is not an external population replication.

Second, external construct evidence is limited. Small relations with Big Five
scores and isolated exploratory associations cannot validate the framework's
full ontology. Same-person data collected across free-expression, repeated
fixed-condition, and interaction designs are needed to separate selection,
conditional response, state, and stable person components.

Third, computational operators remain consequential. Embedding models,
tokenization, segmentation, and dimensionality reduction can change geometry.
The framework specifies how to register and compare these methods but does not
identify a universal representation.

Fourth, high-dimensional person-specific dynamics are difficult to estimate.
Substantial text volume may be required, and apparent reproducibility can be
created by shared context, temporal clustering, or correlated scorer error.
Future work should use multiple occasions, negative controls, operator
perturbations, and designs that vary opportunities prospectively.

Fifth, the formal model is intentionally noncommittal about causal direction
where observational data cannot identify it. Selection can reflect
personality, demographics, social position, learned preference, or access.
Conditional response can reflect stable dispositions, current state, audience
design, or assistance. Interventions or stronger quasi-experimental designs
are required for causal interpretation.

Sixth, the completed research program established methodological closure, not
psychological completeness. It provides explicit object types, transition
assumptions, evidence gates, and abstention conditions. It does not establish
a complete basis of human psychological mechanisms, a universal personality
taxonomy, or a ready-to-use individual assessment.

The most informative next study is therefore not another unrestricted factor
search. It is a prospectively designed human study that collects, from the
same participants, (a) naturalistic language across sampled contexts, (b)
responses to repeated fixed conditions, (c) conversational interactions, (d)
independent behavioral and psychological criteria, and (e) repeated occasions.
Such a design would permit a generalizability study, a multitrait-multimethod
analysis, and preregistered tests of which uninterpreted technical variables earn
construct interpretations.

## 9. Conclusion

Natural-language prediction becomes psychological measurement only when the
path from behavior to representation, comparison, interpretation, and use is
made explicit. Context cannot be classified globally as signal or noise: it is
both an expressive condition and, when selected, a behavioral channel.
Segmentation and embedding are observation methods rather than transparent
views of a person. Between-person, within-person, individual, relational, and
population claims are not interchangeable. Stable technical structure can be
scientifically real while remaining psychologically unnamed.

GRAIN organizes these constraints into a context-conditional, multilevel
development framework. Its central practical rule is conservative but
productive: estimate and test what the data identify; preserve anonymous
structure when interpretation is not yet justified; and stop inference rather
than fill an evidential gap with a familiar label or fluent model narrative.
This provides a foundation on which future natural-language measures of
personality, state, behavior, or interaction can be developed and evaluated
without confusing computational success with psychological validity.

## Open Science and Declarations

### Data and materials availability

Theory documents, analysis code, registered protocols, tests, and aggregate
reports are maintained in the GRAIN repository. Raw PANDORA data and other
third-party corpora are not redistributed and remain subject to their original
licenses and access conditions. A public repository URL and archival DOI will
be inserted after the release is prepared for double-anonymous review.

### Preregistration

The development program contains sealed and commit-timestamped protocols for
individual empirical legs. This theoretical synthesis was written after those
results and should not be described as preregistered. The manuscript
distinguishes prospective tests from exploratory or post hoc synthesis.

### Ethics

The present article reports a methodological synthesis and aggregate results
from existing corpora. Any new study involving human participants, clinical
interpretation, or identifiable language will require prospective ethics
review, consent and privacy procedures appropriate to the setting, and a
separate risk assessment.

### Generative AI assistance

Generative AI tools assisted with drafting, code execution, consistency
checking, and multilingual editorial comparison. The authors are responsible
for the theoretical claims, source verification, interpretation, and final
text. No AI-generated explanation is reported as empirical psychological
evidence. This statement should be adapted to the policy of the target journal.

### Funding, conflicts of interest, and author contributions

[To be completed by the authors before submission.]

## References

American Educational Research Association, American Psychological
Association, & National Council on Measurement in Education. (2014).
*Standards for educational and psychological testing*. American Educational
Research Association.

Borsboom, D., Mellenbergh, G. J., & van Heerden, J. (2004). The concept of
validity. *Psychological Review, 111*(4), 1061-1071.
https://doi.org/10.1037/0033-295X.111.4.1061

Brennan, R. L. (2001). *Generalizability theory*. Springer.
https://doi.org/10.1007/978-1-4757-3456-0

Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation
by the multitrait-multimethod matrix. *Psychological Bulletin, 56*(2), 81-105.
https://doi.org/10.1037/h0046016

Cronbach, L. J., & Meehl, P. E. (1955). Construct validity in psychological
tests. *Psychological Bulletin, 52*(4), 281-302.
https://doi.org/10.1037/h0040957

Demszky, D., Yang, D., Yeager, D. S., Bryan, C. J., Clapper, M., Chandhok, S.,
Eichstaedt, J. C., Hecht, C., Jamieson, J., Johnson, M., Jones, M.,
Krettek-Cobb, D., Lai, L., JonesMitchell, N., Ong, D. C., Dweck, C. S., Gross,
J. J., & Pennebaker, J. W. (2023). Using large language models in psychology.
*Nature Reviews Psychology, 2*, 688-701.
https://doi.org/10.1038/s44159-023-00241-5

Dhami, M. K., Hertwig, R., & Hoffrage, U. (2004). The role of representative
design in an ecological approach to cognition. *Psychological Bulletin,
130*(6), 959-988. https://doi.org/10.1037/0033-2909.130.6.959

Fisher, A. J., Medaglia, J. D., & Jeronimus, B. F. (2018). Lack of
group-to-individual generalizability is a threat to human subjects research.
*Proceedings of the National Academy of Sciences, 115*(27), E6106-E6115.
https://doi.org/10.1073/pnas.1711978115

Fleeson, W. (2001). Toward a structure- and process-integrated view of
personality: Traits as density distributions of states. *Journal of Personality
and Social Psychology, 80*(6), 1011-1027.
https://doi.org/10.1037/0022-3514.80.6.1011

Gjurković, M., Karan, M., Vukojević, I., Bošnjak, M., & Šnajder, J. (2021).
PANDORA talks: Personality and demographics on Reddit. In *Proceedings of the
Ninth International Workshop on Natural Language Processing for Social Media*
(pp. 138-152). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2021.socialnlp-1.12

Jacobs, A. Z., & Wallach, H. (2021). Measurement and fairness. In *Proceedings
of the 2021 ACM Conference on Fairness, Accountability, and Transparency*
(pp. 375-385). Association for Computing Machinery.
https://doi.org/10.1145/3442188.3445901

Kern, M. L., Yaden, D. B., Schwartz, H. A., Giorgi, S., Park, G., Hagan, C. A.,
Tobolsky, V. A., Smith, L. K., Buffone, A., Iwry, J., Seligman, M. E. P., &
Ungar, L. H. (2021). Closed- and open-vocabulary approaches to text analysis:
A review, quantitative comparison, and recommendations. *Psychological
Methods, 26*(4), 398-427. https://doi.org/10.1037/met0000349

Licht, H., Sarkar, R., Wu, P. Y., Goel, P., Stoehr, N., Ash, E., & Hoyle, A. M.
(2025). Measuring scalar constructs in social science with LLMs. In
*Proceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing* (pp. 32144-32171). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2025.emnlp-main.1635

Meredith, W. (1993). Measurement invariance, factor analysis and factorial
invariance. *Psychometrika, 58*(4), 525-543.
https://doi.org/10.1007/BF02294825

Messick, S. (1995). Validity of psychological assessment: Validation of
inferences from persons' responses and performances as scientific inquiry into
score meaning. *American Psychologist, 50*(9), 741-749.
https://doi.org/10.1037/0003-066X.50.9.741

Mischel, W., & Shoda, Y. (1995). A cognitive-affective system theory of
personality: Reconceptualizing situations, dispositions, dynamics, and
invariance in personality structure. *Psychological Review, 102*(2), 246-268.
https://doi.org/10.1037/0033-295X.102.2.246

Molenaar, P. C. M. (2004). A manifesto on psychology as idiographic science:
Bringing the person back into scientific psychology, this time forever.
*Measurement: Interdisciplinary Research and Perspectives, 2*(4), 201-218.
https://doi.org/10.1207/S15366359MEA0204_1

Park, G., Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Kosinski, M.,
Stillwell, D. J., Ungar, L. H., & Seligman, M. E. P. (2015). Automatic
personality assessment through social media language. *Journal of Personality
and Social Psychology, 108*(6), 934-952.
https://doi.org/10.1037/pspp0000020

Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an
individual difference. *Journal of Personality and Social Psychology, 77*(6),
1296-1312. https://doi.org/10.1037/0022-3514.77.6.1296

Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Dziurzynski, L., Ramones,
S. M., Agrawal, M., Shah, A., Kosinski, M., Stillwell, D., Seligman, M. E. P.,
& Ungar, L. H. (2013). Personality, gender, and age in the language of social
media: The open-vocabulary approach. *PLoS ONE, 8*(9), e73791.
https://doi.org/10.1371/journal.pone.0073791

Shoda, Y., Mischel, W., & Wright, J. C. (1994). Intraindividual stability in
the organization and patterning of behavior: Incorporating psychological
situations into the idiographic analysis of personality. *Journal of
Personality and Social Psychology, 67*(4), 674-687.
https://doi.org/10.1037/0022-3514.67.4.674

Steyer, R., Mayer, A., Geiser, C., & Cole, D. A. (2015). A theory of states and
traits-revised. *Annual Review of Clinical Psychology, 11*, 71-98.
https://doi.org/10.1146/annurev-clinpsy-032813-153719

Yarkoni, T. (2010). Personality in 100,000 words: A large-scale analysis of
personality and word use among bloggers. *Journal of Research in Personality,
44*(3), 363-373. https://doi.org/10.1016/j.jrp.2010.04.001

Yarkoni, T., & Westfall, J. (2017). Choosing prediction over explanation in
psychology: Lessons from machine learning. *Perspectives on Psychological
Science, 12*(6), 1100-1122. https://doi.org/10.1177/1745691617693393
