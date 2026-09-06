# GRAIN vector mechanism atlas

2026-09-07, round33. **WORKING SYNTHESIS, NOT PROGRAM COMPLETION.**
The45 primary,45 fixed-layer and35 raw same-ready source-head comparisons are complete; other same-ready,
task/history/direct comparisons, strong controls/nulls, bridge evaluation and
final reusable-observer acceptance remain open. The bounded domain-function packet is complete. This atlas records
what the experiments distinguish; it does not turn every positive statistic
into a new personality construct. See the [remaining requirements](../../docs/GRAIN_VECTOR_REMAINING_REQUIREMENTS.md).

## The current explanatory chain

`author-produced events -> permitted evidence selection -> speaker/condition
assignment -> contextual representation -> task-dependent access -> output
readout -> serialization/calibration -> external validation`.

These links have different experimental objects. A model can encode a source
distinction without using it for the requested task; it can use it while decoding
the requested answer code incorrectly; it can retrieve conversation content
accurately without inferring a stable human trait. This separation is the main
theoretical refinement so far, not a claim to a newly discovered universal law.

| Link | Current evidence | Limit and consequence |
|---|---|---|
| Input/instrument identity | Fixed token/body/position checks, self-replacement/restoration, code/material hashes; several numerical defects repaired before promotion | Batch/precision/execution shape are part of the observation operator, not incidental implementation choices |
| Evidence selection | Natural memory summaries could contain unsupported propositions despite valid quotations; corrections and question wording did not provide uniform gains | Memory fidelity must be checked at proposition/attribution level, not citation-presence alone |
| Binding/combination | Controlled entity/relation conditions distinguish additive summaries from joint encoding; simple product kernels can also recover the relation | Controlled non-additivity is not proof of uniquely LLM reasoning or personality synergy |
| Contextual access | Small frozen history-state interventions transfer to a second current text on independent authors | The answers fail code-reversal controls and non-target effects exist; causal model access is not semantic or psychological specificity |
| Mean geometry vs output | RC5 mean-state cosine .9918 coexists with +.4197 tail logp; final readout admits exact target/partition and KL identities | Euclidean closeness does not imply functional equivalence. The identities are standard softmax mathematics, not new personality formulas |
| Continued access | Cutting later prefix attention can harm more than never reading the prefix | Intermediate states depend on remaining computation; cut damage cannot be interpreted as an additive information percentage |
| Author vs conversation | Complete46-person MEPS fixed/free user correspondence .5908; same-chat user/AI correspondence .9942 | Conversation content can dominate pooled vectors. Role-separated source evidence is required; neither value is personality accuracy |
| External-label readout | All45 primary neural heads/225 fold replays and15 three-observer paired comparisons complete; raw Big5 BGE/E5/Qwen10 r=.177/.202/.150 | No primary Qwen-minus-BGE positive pointwise interval. These are trained source-head readouts, not the LLM's native decisions; task/history/direct controls remain open. Strong sparse controls are a separate head family |
| Ranking versus probability | All36 primary MBTI heads improve log loss and Brier over fold-trained prevalence priors, including near-majority EI/SN classifiers | Weak probability differentiation can exist without crossing the0.5 decision threshold. Proper-score gain is not mutual information, calibrated personality probability or native LLM reasoning |
| Same-ready incremental evidence | Identical144 Big5 /1,539 MBTI authors,35 heads/175 saved folds:19/20 GRAIN-increment intervals include0; Big5 g+BGE pooled+.008295 has a barely positive lower endpoint, mean-fold interval includes0 | g is16 sorted landmark distances, not the full GRAIN framework. Joint-PCA/readout and support-selected population constrain interpretation. Strong combinations minus g do not establish unique g contribution; see [same-ready report](VM05_RAW_READY_COMPARISON_REPORT.md) |
| Additive head capacity | All60 saved hybrid heads have zero mixed block differences on the identity/logit scale within4.44e-16; coefficient pullback replayed | An affine PCA/head cannot test arbitrary cross-block interactions. Nonlinear complementarity remains possible, not demonstrated. Curvature of the logistic probability link is not a log-odds interaction |
| Projection/readout restriction | All9 Big5 original-dimension versus PCA pairs complete; all18 pooled/mean-fold intervals cross0 | No demonstrated simple compression rescue, not equivalence or proof that low-variance directions contain no signal. Removing PCA also changes the regularization problem |
| Fixed-layer accessibility | All45 layer heads/225 saved-fold replays accepted across three arms: EI/SN highest Qwen point estimates at04, TF/Big5 at16, JP at final norm | Nonmonotone, outcome-specific readability is not trait localization or causal layer use. No added Qwen layer's pooled interval has a positive lower bound versus E5. Pointwise comparisons; no outcome-selected layer/blend |
| Cross-domain scope | Frozen167-case Essays/X context gains;469 case-observer geometry comparisons; lexical/cell follow-up independently replayed | Source similarity and model function correlate, partly through repeated content. Japanese adjusted effect remains underresolved; no language/cultural measurement invariance |

## Where the mathematics helps

Let `C=R(H)` be an explicitly recorded retrieval/aggregation of prior evidence,
`h_l=F_l(x,C,q)` the fixed-model state, and `s=g(h)` a declared scalar readout.
The experimental objects are `(x,C,q,F,g)`, not an unqualified vector called
"personality". Matched-context differences estimate the response of this
specified operator to evidence substitution. They do not identify the true
human cause of a language difference.

At the final linear head, `z=Wh+b` and `A(z)=logsumexp(z)` give exactly:

`Delta log p(y) = delta z_y - Delta A = G_m - KL(p_m||p_a)
                = G_a + KL(p_a||p_m)`.

The Jacobian/Fisher pullback explains which state displacement is consequential
for that output distribution. A positive finite change may have a negative
endpoint directional derivative, so one local gradient is not a global summary.
This motivates function-specific rather than distance-only observer variables.
It does not license interpreting KL as psychological uncertainty or a person's
mental distance. Evidence: [RC5](VM06_RESPONSE_READOUT_REPORT.md).

Proper invertible coordinate transport preserves a linear head's predictions;
refitting an isotropically regularized head after whitening is a different
statistical problem. Likewise, high-dimensional residual directions are not
identified by a small collection of landmark distances. Evidence:
[coordinate/readout audit](VM05_READOUT_AND_RANDOM_SUBSPACE_REPORT.md).

On the current domain material, the difference in standalone context similarity
tracks the contextual model's likelihood benefit. English X E5 r=.8987 falls to
.4902 [.1279,.7431] after descriptive lexical/length/cell conditioning; Essays
effects are smaller and less reduced. Excluding repeated four-token spans leaves
positive context gains for Essays/English X, but Japanese X is underresolved.
This joins geometry to a measured function while showing why the join is not
itself personality-specific. See [domain report](VM06_DOMAIN_CONTEXT_GEOMETRY_REPORT.md).

The source tests now distinguish two further objects:

```
R(l,t) = score(y_t, fitted_source_head(l,t)(pool(F_l(x))))
D(t)   = direct_candidate_readout(F_last(x, q_t)).
```

R depends on source labels, pooling, compression and fitted regularization.
D depends on the task prompt, native model computation and output coding.
Neither a high R nor a small R identifies D. The complete fixed-layer profile varies
by outcome, without a uniform depth advantage; removing PCA did not establish
a general rescue. These findings make the already planned task/history/direct
contrasts more informative than adding an outcome-selected layer search. See
[primary comparison](VM05_PRIMARY_NEURAL_SOURCE_REPORT.md),
[projection sensitivity](VM05_PROJECTION_SENSITIVITY_REPORT.md) and
[fixed layers](VM05_FIXED_LAYER_REPORT.md).
The [proper-score diagnostic](VM05_PRIMARY_PROPER_SCORE_REPORT.md) rules out
the stronger claim that all near-majority source classifiers merely repeat
the training prior. The [theory note](../../docs/GRAIN_VECTOR_READABILITY_THEORY_NOTE_V1.md)
connects these distinctions to information-theoretic probing and states the
full-state versus pooled-state data-processing assumptions explicitly.

Observation boundary: the current direct-score collector saves candidate
logits and the vocabulary normalizer, not hidden-state arrays for each of the
nine trait prompts. The secondary state cache uses two social-initiation
questions, not nine trait-specific prompts. Those artifacts can compare output
behavior and the declared task-state channels; they cannot by themselves
localize an internal Big5/MBTI computation or justify subtracting coordinates
from different prompt/pooling protocols. No active collector/schema is changed
here, and no new state experiment is silently counted as completed.

## What may enter a future GRAIN observer

Round34's [native semantic panel](VM05_DIRECT_SEMANTICS_REPORT.md) separates
behavioral ordering from output-code invariance:17/18 primary pairs, but0/10
correctly recoded Big5 inversion pairs, despite high candidate mass and90/90
first-token compliance. MBTI ordering mostly survives listing reversal while
absolute scores shift strongly. These are invented capability/encoding controls,
not natural personality accuracy or circuit identification. A native score must
retain its question/encoding identity instead of being treated as author-only.

- Separate records of user evidence, interaction context and evidence-selection
  transformations, rather than treating all conversation tokens as one author.
- Anonymous function-specific contextual response features with model, task,
  role, layer and reference identity. Do not prematurely name personality axes.
- Access/usage and fidelity controls attached to each interpretation. A trace
  diagram or attention heatmap alone is insufficient.
- Source-only statistical heads for external associations, kept separate from
  the released deterministic GRAIN scorer and from causal model-intervention claims.

## Open synthesis work

The atlas is not yet the final conclusion. Full common-material observers and
the45 primary source heads, no-PCA and fixed-layer comparisons are complete. Finish
declared task-conditioned source sensitivities,
same-ready and task/history/direct comparisons, stronger controls/nulls, then
freeze source choices before the untouched bridge outcome evaluation. Carry
forward the completed domain packet's role/genre limitations;
link every accepted observer function to an audited artifact and explicit
failure/refusal state. The [draft observer specification](../../docs/GRAIN_VECTOR_OBSERVER_SPEC.md)
is a contract for that integration, not a release-ready personality instrument.

Reports supplying the other links: [memory](VM03_NATURAL_MEMORY_REPORT.md),
[binding](VM03_RELATIONAL_BINDING_REPORT.md), [independent access](VM04_INDEPENDENT_ACCESS_REPORT.md),
[answer-code audit](VM03_SECOND_OBSERVER_REPORT.md), [continued access](VM06_RESPONSE_ACCESS_REPORT.md),
[fixed/free](VM06_FIXED_FREE_REPORT.md), [source controls](VM05_SOURCE_CONTROLS_AND_NULL_REPORT.md),
[domain inventory](VM06_DOMAIN_INVENTORY_REPORT.md).
