# Target Evidence, Ownership and Query Response Geometry

2026-10-08. HM-10B2 is complete at its finite model-observation scope. The
parent research goal and original 18/100/44 requirements remain open.

## 1. What Was Found

The fixed local reader uses more than the presence of affect/trust words.
Changing who holds the same expression changes its response. Moreover,
self-attributed negative affect changes the AI-trust readout even when the
AI-trust statement is unchanged. This supplies a concrete cross-target
mechanism to localize, rather than another score-precision search.

| Query | Changed expression | Self high-minus-low | Other high-minus-low | Self-minus-other binding |
| --- | --- | ---: | ---: | ---: |
| NA subset | Negative affect A | .104150 | .046864 | .057286 |
| Dejection | Negative affect A | .131253 | .049556 | .081697 |
| AI trust | Negative affect A | -.161778 | .046113 | -.207891 |
| AI trust | General AI view T | .332785 | .180042 | .152743 |

NA binding is positive in 15/16 material families; dejection and direct trust
binding in 16/16. The cross-affect trust binding is negative in 16/16. These
are responses of Qwen2.5-1.5B to controlled text and fixed queries, not human
personality effect sizes or questionnaire validity coefficients.

## 2. Five Independent Design Coordinates

The address is g=(A,T,U,V,S) in G=(Z/2Z)^5. A is negative-affect polarity;
T is general AI-trust polarity; U and V separately assign the two statements
to self (0) or colleague (1); S reverses the two sentences. Some A items,
such as ashamed and irritable, are not literal anxiety statements. The older
code label `anxiety` names the address, not a newly validated anxiety scale.

Each ownership intervention swaps a target predicate with a neutral predicate.
Its word multiset is unchanged. For fixed A,T, the U,V,S subgroup therefore
contains eight same-word variants. Independent U and V avoid an alias caused
by swapping both statements globally. All 32 full cells are distinct.

Sixteen lexical/context families were fixed as source/validation/application
8/4/4. There are 800 records but 656 unique raw texts; single-statement order
duplicates are exact raw nulls, not additional observations. Three unchanged
queries required 1,968 actual local forwards in 574.10 seconds. No target
function, prompt, digit alphabet or neural parameter was fitted anew.

## 3. Response Lift, Character Decomposition and Lost Binding

For each fixed family/query, the actual observer is lifted to f_q:G->V.
This is a function on a designed group, not assumed LLM equivariance or a
group symmetry of human personalities. With chi_J(g)=(-1)^popcount(g&J),

\[
c_{q,J}=2^{-5}\sum_{g\in G}\chi_J(g)f_q(g),\qquad
f_q(g)=\sum_J c_{q,J}\chi_J(g).
\]

The decomposition applies separately to centered digit logits, native scores
and the existing affine program. Parseval closes the centered variation exactly.
The matched binding terms are A:U (mask 5) and T:V (mask 10):

\[
B_{q,A:U}=[f_q(A=1,U=0)-f_q(A=0,U=0)]
 -[f_q(A=1,U=1)-f_q(A=0,U=1)]=-4c_{q,5},
\]

where T,V,S are averaged uniformly over the designed cells. T:V is analogous.
Averaging over U removes every U-odd character, including A:U. Thus making
ownership invisible as supposed nuisance can remove the very relation needed
for author-specific measurement. It is not legitimate denoising by default.

These are standard finite character identities. [Cohen and Welling (2016),
Section 4.4](https://proceedings.mlr.press/v48/cohenc16.pdf) gives the
function-on-group perspective; [Finzi et al. (2021), Sections 3-4.2](https://proceedings.mlr.press/v139/finzi21a/finzi21a.pdf)
clarifies representation actions. No group-equivariant network is trained here.
The pinned OpenAI/math hypergraph Fourier proof inspired keeping and accounting
for interaction components; its Boolean influence and hypercontractive bounds
are not transferred to natural language.

## 4. Natural Coordinates and the Exact Signed Link Split

The actual map remains text/query -> normalized hidden state -> digit logits
-> conditional softmax probabilities -> fixed expected digit value. Define
theta=logits-mean(logits), S(theta)=a^T softmax(theta), a=(0,1/9,...,1).
Removing this constant gauge is not topic or population centering.

For every polarity/ownership four-corner table, let
theta_*=theta_10+theta_01-theta_00 and d=theta_11-theta_*.
The scalar interaction separates exactly into link curvature and the action
of the actual representation interaction, as in the categorical geometry doc.
After converting to the self-minus-other sign convention, the mean terms are:

| Query / binding | Total | Link | Representation action |
| --- | ---: | ---: | ---: |
| NA / A:U | .057286 | .013012 | .044274 |
| Dejection / A:U | .081697 | .010432 | .071265 |
| AI trust / A:U | -.207891 | -.011865 | -.196026 |
| AI trust / T:V | .152743 | .027158 | .125585 |

The cross-target trust effect is therefore not merely a nonlinear scalar-link
artifact. This split depends on the declared three-corner endpoint. Signed
components are not psychological information percentages or unique causal shares.

## 5. Cross-Target Action Depends on the Surrounding Statement and Order

Hold the trust statement's owner at self and vary only affect/its owner.
The result-after diagnostic preserves every family and both sentence orders:

| Trust statement | Order | Self affect effect on trust | Ownership contrast |
| --- | --- | ---: | ---: |
| Negative | Affect first | -.031022 | -.065631 |
| Negative | Trust first | -.119491 | -.125724 |
| Positive | Affect first | -.167332 | -.171703 |
| Positive | Trust first | -.269309 | -.249670 |

Every row has negative mean binding. The same nominal evidence therefore has
an order/context-conditioned action. The large marginal effect cannot be
interpreted as a context-free human trait relation.

In the source trust-query scalar spectrum, A:U and T:V have coherence ratios
.953989 and .954932 across eight families, defined as
(mean coefficient)^2 / mean(coefficient^2). A:T is also present; higher/order
interactions remain. Coherence is a finite-family alignment statistic, not
test-retest reliability or a psychological population parameter.

## 6. Source-Supported Cold Completion

Only source-family coefficients are averaged. A cold family supplies its
actual cell-zero natural-coordinate anchor:

\[
\widehat\theta_q(g)=\theta_q^{cold}(0)+
\sum_{J\in\mathcal J}\overline c_{q,J}^{source}
[\chi_J(g)-\chi_J(0)].
\]

All six declared arms are retained. Application RMSEs are:

| Query | Constant | Main | Main + A:U/T:V | Full law | Reversed binding | Scalar binding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NA | .051074 | .033691 | .033300 | .031952 | .042074 | .040284 |
| Dejection | .086664 | .059434 | .056645 | .054329 | .072290 | .060758 |
| AI trust | .297966 | .211151 | .201277 | .127958 | .233090 | .194489 |

The prespecified binding-versus-main source/application family intervals all
include zero. The point effects and their existence are reported separately
from this uncertain completion gain. Full-law trust RMSE is 39.4% lower than
main. A secondary, result-after 2,000-draw family bootstrap gives full-vs-main
MSE gain [.014518,.040426] and full-vs-binding [.008851,.039766]. NA/dejection
secondary intervals include zero. These additional contrasts are explicitly
post-result diagnostics, not new blind confirmations.

The completion uses a real cold anchor and a finite address law. It is not an
arbitrary-natural-text predictor, a personal response operator B_u, or removal
of transformer dependence. The experiment does not identify which additional
term individually causes the full-law gain; all interactions are retained.

## 7. Evidence Qualification Is a Separate Map

Declared NA self-expression and general AI-view evidence each cover only a
subset of their query. Dejection is adjacent inferred content, without direct
dejection statements. Off-target outputs are preserved, not called direct
self evidence. Missing evidence is null, not a personal zero.

This is why a mathematical response map alone is insufficient for measurement:
query relevance, event/actor binding, direct evidence and model extrapolation
are different operations. Their distinction can guide GRAIN construction
without assuming every off-target response is either noise or psychological truth.

## 8. Next Mechanism, Not Another Accuracy Grid

Binding-ID work uses activation swaps to separate entity and attribute roles
([Feng and Steinhardt, 2024, Sections 3-4](https://proceedings.iclr.cc/paper_files/paper/2024/file/9d1b7fc578c0d2d6431fc26d736ecaf3-Paper-Conference.pdf)).
The CBR preprint probes entity/relation indices at attribute activations and
tests subspace interventions ([Dai et al., 2026, Sections 3.1-3.3](https://arxiv.org/html/2604.19052v1)).
These motivate our next test; HM10B2 did not collect attribute-token states
or identify a CBR subspace. A readout contrast is not an already localized path.

HM10B3 should capture explicit actor/property spans at predeclared layers,
swap same-property donor activations between owners, and compare role/content/
order controls on all three unchanged queries. Source-only binding directions
must be distinguished from whole-span copying. A learned nonorthogonal loading
matrix is not inverted by its transpose: use an orthonormal basis/projector
or a checked pseudoinverse. No claim of a bug in the cited code is made; that
repository has not been inspected here.

## 9. Provenance and Reproduction

Primary frozen results: `results/grain_math_response_v1/target_orbits_r2`.
`target_orbits_r1` was zero-forward preparation before a loop-variable binding
fix. Preserve it; do not run the frozen plan's default r1 command blindly.
Result-after companion: `target_orbits_diagnostic_r1`; alternate companion
verification: `target_orbits_diagnostic_verification_r2` (r1 also retained).

Primary alternate arithmetic checked 3,966 artifact hashes, 17 source hashes,
all 1,968 scores, 48 complete tables, 96 bindings, 768 gradient integrals,
2,304 same-word edges, 432 exact raw null query pairs and all cold predictions.
Maximum algebra error 3.55e-15; float32 digit-head replay 3.70e-6 is recorded
separately. Companion alternative grouping/count bootstraps agree to 1.78e-15.
This is same-agent numerical verification, not an independent human audit.

205 focused tests passed, including 8 new orbit tests and 3 new companion
tests. No private text, API, new neural training, Claude, commit/push/tag,
lockbox or historical queue was used. Original source/author uncertainty and
the full research scope remain separate from this finite completion.
