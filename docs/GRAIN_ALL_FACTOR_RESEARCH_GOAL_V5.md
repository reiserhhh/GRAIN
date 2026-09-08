# GRAIN all-factor research goal V5

2026-09-09. Active user-authorized research loop; supersedes the completed V4
queue, not its findings. Objective: construct a common, executable mathematical
procedure from author text to trait-conditioned vectors, with separate functions
for each measurable target. This is methodology research, not questionnaire
replacement or a contest to maximize PANDORA accuracy.

## Starting point and missing pieces

The D1 source discovery already contains all five Big Five domains and four MBTI
axes. Its raw native mean Big5 r is .1993; native T/F AUC is .6600. These are
external associations. M25/M26 instead constructed three query functions (T/F,
neuroticism, confirmation seeking), demonstrated responses to evidence changes,
and packaged a suffix-free readout that still requires the frozen Qwen prefix.
Expanding the *mechanism* beyond these three functions is the missing step.

| Packet | Question and executable delivery | Decision enabled |
|---|---|---|
| M27 | All-factor catalog; recomputed MEPS scales; D1 all-query/all-criterion matrices and cue ablations | Which targets have readable information, query-specific use, or unused encoder information? |
| M28 | All 9 core query functions on the same existing source-author material; mean/configuration/final-token terms and evidence responses | Does one construction work across targets, and which terms need query-specific treatment? |
| M29 | Source-only external connection of the frozen functions; human-label correlations separate from native-score fidelity and intervention agreement | Which identified vector functions carry externally relevant author differences? |
| M30 | Fixed MEPS queries and user-only texts; pre/post/session-specific external association with all eligible linked scores | Which narrower targets are observable in fixed tasks versus free counseling? |
| M31 | Factor-by-dataset equation/evidence atlas and signed cross-query covariance ledger | Which mechanisms recur, which are specific, and what next idea changes the construction? |

No requirement that every factor is significant. All computable comparisons,
missing targets and negative responses remain in the atlas. Big Five has five
available domain scores, not invented facets. MBTI has four oriented binary
criteria (E/N/F/J positive), not sixteen independent traits or eight functions.

## Common mathematical program

For author u, material/view c, and query q, the frozen prefix gives ordered
question-token states P and final-token state Q. Write P = 1 m' + C, with token
mean m and centered configuration C. Given an independent reference R:

```
f_q(P,Q) = b_q + <G_q, P-P_R>_F + g_q' (Q-Q_R)
         = b_q + M_q + C_q + Q_q
response_q(edit) = f_q(edited material) - f_q(full material)
position_uq = (f_q(u)-mean_R(f_q)) / sd_R(f_q)
external_link[q,k|corpus,view,phase] = Corr(f_q(text_u), scale_uk)
```

G/g are functional derivatives/integrated local derivatives of the frozen
observer, not human-label-trained coefficients. Teacher-trained Ridge is a
separate baseline. Formula fidelity, response fidelity, author geometry and
human-scale association are four different evidence columns, never one metric.
No clipping is applied to affine function values. Native digit/axis readouts
remain candidate-conditioned expectations/probabilities, not questionnaire norms.

With U_q = native_q - f_q and V_q=(M_q,C_q,Q_q,U_q):

```
Cov(native_q,native_r | c,R) = sum_{a,b} Cov(V_qa,V_rb | c,R)
```

Signed terms may overlap or cancel. Within-query dominance need not explain
between-query sharing. Compare hidden-space operators or matched author effects,
not token-index axes across questions of unequal lengths. Population estimates
become more precise with data; neither individual error nor systematic mismatch
is assumed to converge to zero automatically.

## Data use and phase-aware target catalog

- PANDORA: reuse the frozen D1 source discovery; no old bridge/lockbox opening.
  Source Big5-only and MBTI-only remain separate. Within-block association
  matrices use exact same-author joins; unmatched groups never create a bridge.
- MEPS Study 2: the latest user request explicitly adds local exploratory
  questionnaire linkage. This is a *new branch-specific authorization*, not a
  retroactive change to text-only VM06 or sealed analyses. Read only selected
  numeric columns in processed pre/post scored files, the newly item-audited
  `processed/mood_long.csv` under M30C, and the authoritative
  decrypted user-text exports. No training, API upload, raw survey export,
  identity reconstruction, Study 1 merge, or public participant-level output.
- The local study explanation describes correlations/regressions of user
  characteristics, AI use, empathy, disclosure and mood change. This supports
  the planned local exploratory analysis; it is not a claim that every future
  GRAIN application or publication has independent ethics clearance.
- Recompute scores from administered item lists and reverse keys. Actual scoring
  configuration takes precedence over earlier draft revision memos. Preserve
  missingness; no item imputation or outcome-driven prompt rewriting.
- AI-directed attachment/mind attribution/trust are relational targets, not
  generic attachment diagnoses. Post empathy/depth/smoothness concern this
  session; pre/post differences are session-linked changes, not stable-trait
  change by definition. Disclosure levels are ordered single indicators.
- Agency/experience splits and disclosure-level means are explicitly exploratory.
  Mood trajectories enter only after checking adapted items and time alignment.
  This audit is now complete for13 adapted items at baseline/pre/mid/post;
  [M30C](GRAIN_MEPS_MOOD_EXTENSION_M30C.md) adds3 state queries and12 occasion
  composites in a separate finite packet. They are not additional trait scales.
  Essays/X without matching labels can test geometry/transport, not missing
  anthropomorphism or attachment criteria.

## Literature translated into design choices

1. [PANDORA, Gjurkovic et al., 2021](https://aclanthology.org/2021.socialnlp-1.12/)
   motivates separate partially labeled sources and later cross-model analysis,
   rather than fitting both heads on a tiny bridge.
2. [Schwartz et al., 2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0073791)
   and [Park et al., 2015](https://pubmed.ncbi.nlm.nih.gov/25365036/) motivate a
   broad language/trait atlas and independent external associations, not assuming
   every trait has equally observable language markers.
3. [Campbell and Fiske, 1959](https://pubmed.ncbi.nlm.nih.gov/13634291/) motivates
   retaining the full query-by-criterion matrix: an apparent trait association
   may be shared with other traits. Here this is descriptive multi-trait evidence,
   not a fitted latent MTMM model on 46 participants.
4. [Peters, Cerf and Matz, 2024](https://arxiv.org/abs/2405.13052) compares
   elicitation with naturalistic/helpful dialogue. It makes evidence opportunity
   and dialogue condition central: keep MEPS answers, assistance and free chat
   separate rather than pooling them into a longer document.
5. [Park, Choe and Veitch, 2024](https://arxiv.org/abs/2311.03658) distinguishes
   linear representation from geometry used for measurement/intervention.
   [Circuit Tracing, 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
   distinguishes ordinary output fidelity from counterfactual fidelity. Therefore
   expand the functional-action construction, not just generic embedding cosine.
6. [Gray, Gray and Wegner, 2007](https://dtg.sites.fas.harvard.edu/DANWEGNER/pub/Gray%2C%20Gray%2C%20%26%20Wegner%20%282007%29.pdf)
   motivates agency/experience as distinct queries, without validating our adapted
   item splits. [Stiles' SEQ](https://wbstiles.net/session_evaluation_questionnaire.htm)
   motivates session-level depth/smoothness outcomes rather than trait naming.
7. [Benjamini and Hochberg, 1995](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1995.tb02031.x)
   motivates reporting complete p/q tables with declared comparison families.
   Effect sizes and useful construction remain the primary discovery outputs.

## Bounded execution and acceptance

M27: CPU only, no new model fits. Freeze this plan/catalog before numeric scale
inspection; compare stored and recomputed scores, ranges, completeness and joins.
Build complete within-source matrices for each observer/arm, plus paired cue
contrasts. Fisher r intervals; paired author bootstrap for principal contrasts.
No author-level records in tracked reports. M27 completion is a usable catalog
and actual matrix report, not just a list of proposed analyses.

M28: start with a finite all-nine panel on the existing 24 S1 authors/two views.
Construct functions on declared reference authors and compare the remaining
authors. Checkpoint each query/author; freeze query wording, model, layer and edit
definitions. Old E1/E2 were chosen for three queries: label them legacy common
edits, never claim they target all nine. Add a neutral/query-balanced edit packet
only if legacy coverage prevents interpreting a new query. Per packet cap: 20
minutes neural work, two CPU threads; continue completed units when useful.

M29/M30: save and hash predictions before joining human outcomes. MEPS starts
with all usable free-chat authors and a separate direct-answer view. Primary
pre-session relational scores, secondary post-session ratings and changes remain
separate families. No n=46 supervised calibration. Record text amount, opportunity,
language and empty views; no threshold tuned against scales. Query/suffix-free
function outputs are kept separate from full observer outputs. Missing valid
fixed functions become a recorded next construction task, not fabricated scores.

M31: do not fit a universal shared scalar because M25 did not support it. Build
the conditional signed covariance atlas and explicit per-factor function cards.
Stop expanding a costly packet when remaining units do not change the next
mechanistic decision; record a resumable value-based pause. No blind long queue.
Discovery -> development -> focused verification; arithmetic, privacy and joins
are the minimum correctness floor. Local commits per packet, no automatic push.

## Current execution

Finite goal delivery COMPLETE. M27 completed (492 source links; all46 Study2 participants and22
scores audited). M28 completed: nine functions, held level r=.776-.991 and
natural-response r=.550-.976, no human fitting. See
[M28 report](../reports/grain_vector_mechanisms/ALL_FACTOR_FUNCTIONS_REPORT.md).
M29 completed all1321 source query units on277 authors, with3132 new prefix
calls in2806.81seconds over three bounded invocations. The fixed Big5 functions
have mean human r=.1406 versus cached native .1671; MBTI function AUCs are
EI/SN/TF/JP=.497/.563/.620/.532 on64 authors. All paired differences retain
their uncertainty and observer provenance. [Source report](../reports/grain_vector_mechanisms/ALL_FACTOR_SOURCE_CONNECTION_REPORT.md).
M30 both46-author/
22-query views and their paired comparison are COMPLETE. Free-chat external
AI trust r=.324, attachment anxiety r=.308, overall session evaluation r=.319;
phase-family BH minimum q=.149. The full report retains every target and both
common/specific covariance contributions:
[M30 report](../reports/grain_vector_mechanisms/COUNSELING_ALL_FACTOR_EXTERNAL_REPORT.md).
Both views fit without text clipping; no human-outcome fitting was performed.
M30C completed276 calls: free-chat adapted state r=.427/.536/.518, all endpoint
BH q<=.0031. [State report](../reports/grain_vector_mechanisms/COUNSELING_STATE_EXTERNAL_REPORT.md).
M30B's25 narrower derivative functions and their2300-state transport are COMPLETE.
Free-chat state functions retain external r=.422/.564/.555; all25 functions'
median native fidelity=.899 in free chat and.895 in direct answers. M31 includes
1700 complete narrow query/criterion links and9600 signed cross-path terms.
[Function report](../reports/grain_vector_mechanisms/NARROW_VECTOR_FUNCTION_REPORT.md).
M31's34-function equation index and coefficient/hash ledger are COMPLETE,
covering43 human criterion occasions across disjoint sources. No collector or
automatic grid remains.809 GRAIN tests passed (nonlinear_readout excluded for
the documented Mac native-library conflict). [Complete round and next questions](../reports/grain_vector_mechanisms/ALL_FACTOR_ROUND_DELIVERY_REPORT.md).
This closes the requested finite all-factor construction, not a claim that all
targets are significant or the entire personality theory is universally proven.
