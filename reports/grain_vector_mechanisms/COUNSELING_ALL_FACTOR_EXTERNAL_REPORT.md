# All-factor counseling external connection: M30/M31

2026-09-09. Completed local exploratory screen, not a fitted questionnaire
predictor. The same46 Study2 participants supply two separate views: user-only
free AI conversation and their three final MEPS scenario answers (AI assistance
allowed).22 phase/referent-specific queries are compared with22 audited scores.

## Findings that change the next construction

1. Narrower targets are readable at a useful exploratory level. Free-chat
   AI trust r=.324, AI-directed attachment anxiety r=.308, and overall session
   evaluation r=.319 extend the inquiry beyond Big Five and MBTI. These are
   correlations with participants' questionnaires, not native/function fidelity.
2. The evidence opportunity matters. For overall session evaluation, free-chat
   r=.319 versus MEPS-answer r=-.023; paired author-bootstrap delta=.342
   [.016,.699]. Session depth delta=.319 [.003,.634]. These are unadjusted
   exploratory intervals across22 comparisons, not a selected confirmatory test.
   General AI trust is positive in both views (.324/.265). More text alone is
   not the explanation: direct answers have more median tokens (577 vs458).
3. Many queries share a broad author-score component. Free-chat standardized
   PC1 explains56.8%, median off-diagonal query r=.583; direct-answer equivalents
   are34.0% and.241. This component may combine evidence, engagement and observer
   appraisal; it is not named a new psychological factor.
4. Shared and specific associations coexist. AI trust's total r=.324 partitions
   into .257 shared and.067 specific covariance contributions; attachment
   anxiety .308=.225+.083. Overall evaluation .319=.141+.179 and disclosure
   level2 .284=.096+.188 retain a larger query-specific contribution. Keep both
   components rather than deleting common variation as noise.

All associations below are unselected, n=46. Free-chat phase-family BH q values
are all>.05 (minimum.149); direct-answer minimum.309. This is a broad discovery
map for choosing mathematical objects, not22 independently validated traits.
AI-avoidance, distrust and privacy concern remain informative negative/weak rows.

## Complete intended-target table

| Questionnaire occasion / target | Free chat r | Direct answers r |
|---|---:|---:|
| Pre AI agency attribution, exploratory split | .225 | .039 |
| Pre AI experience attribution, exploratory split | .203 | .235 |
| Pre AI mind attribution total | .237 | .241 |
| Pre AI attachment anxiety | .308 | .066 |
| Pre AI attachment avoidance | -.124 | .194 |
| Pre positive AI trust | .324 | .265 |
| Pre AI distrust | .031 | -.008 |
| Pre AI privacy concern, one item | .148 | -.034 |
| Post this-AI agency, exploratory split | .260 | .051 |
| Post this-AI experience, exploratory split | .184 | -.016 |
| Post this-AI mind attribution total | .234 | .103 |
| Post this-AI attachment anxiety | .213 | .054 |
| Post this-AI attachment avoidance | .055 | .172 |
| Post perceived AI empathy | .265 | .050 |
| Post session depth | .231 | -.088 |
| Post session smoothness | .223 | .151 |
| Post overall session evaluation, one item | .319 | -.023 |
| Post disclosure level1, one indicator | .258 | -.068 |
| Post disclosure level2, one indicator | .284 | .103 |
| Post disclosure level3, one indicator | .186 | -.096 |
| Post disclosure level4, one indicator | .187 | .041 |
| Post disclosure mean, exploratory composite | .162 | -.211 |

The same query's scores generally do not preserve author order across these
two very different views (cross-view r=-.202 to.300). This favors a conditional
evidence-use object, not one context-free score copied between tasks. Session
ratings target the AI encounter, so MEPS answers are a contrasting opportunity,
not an equally direct source of session-specific evidence.

## Exact formulas and interpretation

Each query is fixed before inference. For frozen observer logits l and answer
digits d=0,...,9, the native score is

```
p_q(d|x,c) = exp(l_d) / sum_{j=0}^9 exp(l_j)
s_q(x,c) = sum_{d=0}^9 (d/9) p_q(d|x,c).
```

This is a candidate-conditioned model judgment, not a calibrated questionnaire
score. No human outcomes fit or rewrite the model, question, or answer alphabet.
P/Q states were cached, but the22 new suffix-free affine functions have NOT yet
been constructed. Do not substitute the nine Big5/MBTI matrices for them.

For descriptive sharing, standardize each query within this observed sample.
For query q, let c_-q be the average of other standardized queries, excluding
all pre/post instances of q's semantic family. Fit only the unlabeled projection:

```
lambda_q = Cov(z_q, c_-q) / Var(c_-q)
z_q = lambda_q c_-q + r_q
Corr(z_q,Y_q) = Cov(lambda_q c_-q,Y_q)/(sd(z_q)sd(Y_q))
              + Cov(r_q,Y_q)/(sd(z_q)sd(Y_q)).
```

The accounting residual is below1e-16. Terms are signed covariance contributions,
not two separate correlations to add, and not causal mechanisms. The projection
describes all46 observations; it is not a held-out regression improvement.
No common component is removed from the original scores or external table.

Fisher95% intervals and1000 participant-bootstrap intervals accompany all
intended links;2000 paired bootstraps compare views. For example trust's Fisher
interval is [.037,.561], bootstrap [.095,.548]; attachment anxiety's Fisher
interval [.020,.549] and bootstrap [-.005,.578] differ near zero. Preserve both.

## Data, literature and scope

Only user turns are scored. The AI replies are excluded; scenario events are
explicitly not assigned to the participant. All46 authors' complete selected
views fit the frozen context, with no clipping. Collection:2024 forwards total,
587.71seconds free chat plus782.99seconds direct answers. Local model only;
no API upload, raw-survey read, Study1 merge, old-lockbox opening, or training.
Prediction bytes were frozen before joining the locally audited criterion cache.
Actual22 scores and reverse keys were recomputed exactly; no missingness.
Agency/experience splits and disclosure means are marked exploratory. Adapted
mood trajectories await their item/time audit and are not silently called PANAS.

[Funder's realistic accuracy model](https://pubmed.ncbi.nlm.nih.gov/7480467/)
separates relevant cue availability, detection and utilization. It motivates
our separation of task opportunity from query readout, not an equivalence theorem.
[Peters, Cerf and Matz](https://arxiv.org/abs/2405.13052) find different personality
inference performance across dialogue conditions. Our view contrast examines a
related condition-dependence question, but is not their randomized intervention.
[Campbell and Fiske](https://pubmed.ncbi.nlm.nih.gov/13634291/) motivate the complete
trait-by-criterion matrix; here no latent MTMM model is fitted to a small sample.

## Reproduction and next decision

- Collector: `scripts/run_grain_counseling_external.py` (`prepare/collect/analyze`).
- Comparison: `scripts/analyze_grain_counseling_views.py`.
- Frozen query wording: `grain_research/vector_mechanisms/counseling_queries.py`.
- Local outputs: `results/grain_vector_mechanisms/counseling_external_m30_{free,direct}_v2/`.
- Paired outputs: `results/grain_vector_mechanisms/counseling_paired_context_m30_v1/`.
- Each view contains484 complete query/criterion associations,22 intended links,
  prediction/provenance hashes and private P/Q packets. The paired folder adds22
  view contrasts, two query matrices and44 common/specific accounting rows.

Next is **function construction and opportunity-sensitive evidence use**, not
choosing a new prompt by these questionnaire r values. Test externally fixed
reference functions for relational/session queries, then ask whether shared
versus specific vector contributions match the observed score-level accounting.
Retain all22 targets; do not build only the three favorable rows.
