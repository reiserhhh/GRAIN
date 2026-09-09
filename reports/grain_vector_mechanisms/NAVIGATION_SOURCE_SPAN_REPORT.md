# GRAIN source-span extension and affine compilation

2026-09-09. This packet is complete. It answers the source-coverage question
from the frozen [plan](../../docs/GRAIN_NAVIGATION_SOURCE_SPAN_EXTENSION_PLAN_V1.md).
The NAV goal ledger, not that historical plan header, records current execution.

## Findings and mainline decision

Adding twelve unlabelled reference authors improves natural-function recovery:
low state r=.1272 to .4149 and E r=.0901 to .1909. Absolute outside-span state
error falls74%-82%. Approximately one third of that reduction is attributable
to recentering, and two thirds to the added residual directions. This is actual
source-space coverage, not merely a change in the displayed mean score.

Coverage does not automatically repair conditional responses. The new low-state
endorsement-edit correlation is -.7815, despite improved level correlation.
Its real free-chat external r=.2607 improves on the old state constructor's
.1880 but does not replace the earlier four-block projection's .3506. T/F
recovers from reversed ordering to near-random external classification, not to
the useful token-only .7047 result. All candidates and directions remain intact.

The linear-kernel constructor also admits **exact global affine compilation**
in its early-state input. Its retained numerical parameters shrink from289.33MB
to7.11MB, without changing any score. This both simplifies runtime and identifies
the remaining model-class limitation: the correction has a constant derivative
with respect to the early query state. Text-to-early-state remains nonlinear.

## Construction, data and execution

- Same48 synthetic construction snapshots per query, never held scenes.
- Twelve separate PANDORA reference authors from NAV08; two text views averaged
  into one paired early/upper state per author, giving60 construction rows/query.
- Four extracted blocks, centered linear kernel, alpha1, same synthetic reader.
- No hyperparameter search, human-label fitting, MEPS construction or sign flipping.
- The twelve natural evaluation authors are disjoint from the reference authors.
- All277 external source authors are disjoint from these twelve construction
  authors; the overlap audit excluded zero source units. All46 MEPS participants
  are external-only. Previously exposed endpoints remain exploratory connections.

Seventy-two new four-block construction passes took24.47seconds. Evaluation on
worlds and natural edits reused existing early states; upper teacher states were
joined only after predictions froze. External scoring executed923 four-block
views/369 author-query units in193.62seconds, then joined questionnaires.

The residual rank expands from47 to59. Effective fitted degrees of freedom are
8.81/9.07/9.65 for E/T/F/state, not59 independently identified personality factors.

## State and score results

Each natural r below describes96 edited observations from only12 authors.
Paired edits use author-level uncertainty, not96 independent people.

| Target | Natural function r, old | New | Natural MAE, old | New | Held-world r, old | New |
|---|---:|---:|---:|---:|---:|---:|
| E | .0901 | .1909 | .1650 | .0572 | .7374 | .7250 |
| T/F | -.3228 | -.2787 | .1119 | .1014 | .7150 | .7563 |
| Low state | .1272 | .4149 | .0693 | .0670 | .8298 | .8884 |

Held-world correlations use24 renderings of three held scene skeletons. Low-state
world MAE changes .1183 to .1228: better ordering is not lower absolute error.

| Natural target | Outside-state MSE, old | New | Total-state MSE, old | New |
|---|---:|---:|---:|---:|
| E | .05089 | .01116 | .05683 | .01344 |
| T/F | .07608 | .01361 | .09083 | .01732 |
| Low state | .06288 | .01618 | .07055 | .01993 |

The outside fraction can remain high while absolute error decreases substantially.
Do not compare fractions alone or equate Euclidean error reduction with recovered
psychological variance. Orthogonal error accounting closes within1.67e-9.

### Conditional-response recovery remains distinct

Correlation with the matched native edit delta across12 authors:

| Target | Actor, old/new | Endorsement, old/new | Time, old/new | Removal, old/new |
|---|---|---|---|---|
| E | -.259/.231 | -.115/-.291 | .075/.041 | -.088/-.486 |
| T/F | .150/.295 | .324/.357 | .349/.386 | -.176/.040 |
| Low state | -.199/-.174 | -.538/-.782 | -.405/-.385 | .029/.011 |

The low-state endorsement mean absolute response is about.001, versus.007
for the native model. Improved level r does not mean the correct edit rule was
learned. The pre-fixed model-only gate triggered E/state and therefore all three
declared external targets were evaluated, not only favorable ones.

## Real-author connections

| Endpoint | n | New human metric [95% bootstrap CI] | Full-function fidelity r |
|---|---:|---|---:|
| PANDORA E |213| r=.0778 [-.0439,.2021] | .1990 |
| PANDORA T/F |64| AUC=.4745 [.3221,.6237] | .4895 |
| Free-chat low state |46| r=.2607 [-.0050,.4840] | .4468 |
| Task-answer low state |46| r=.1470 [-.0889,.4153] | .4063 |

Same-member paired changes against the old synthetic-only state constructor:
E delta-r=.0350[-.0080,.0806]; T/F delta-AUC=.1702[-.0167,.3360]; free-chat
delta-r=.0727[-.0147,.1634]; task-answer delta-r=-.0183[-.1139,.0835].
The large T/F repair should not be called a significant AUC gain; its AUC remains
.2302 below the token-only comparator (paired CI[-.4148,-.0594]).

Use the new operator as a source-coverage mechanism result, not a universal
predictor replacement. External scores did not select any parameters.

## Center versus new directions

Let r be the actual upper-minus-early state, mu the construction residual mean,
and P the orthogonal projector onto the centered construction residual span:

~~~text
e_old    = (I-P_old)(r-mu_old)
e_center = (I-P_old)(r-mu_new)
e_new    = (I-P_new)(r-mu_new)
gain_center = ||e_old||^2 - ||e_center||^2
gain_span   = ||e_center||^2 - ||e_new||^2
~~~

Because the expanded affine span contains the old one, the second contribution
is nonnegative within numerical tolerance. The first contribution need not be
nonnegative in general. Both are teacher-oracle **diagnostics**, not predictors.

| Target | Relative outside-energy reduction | Fraction of gain: center | Added span | Reader-direction outside RMSE, old/new |
|---|---:|---:|---:|---:|
| E | .7808 | .3441 | .6559 | .08035/.04389 |
| T/F | .8211 | .3416 | .6584 | .09724/.09254 |
| Low state | .7427 | .3432 | .6568 | .07538/.04139 |

T/F exposes the crucial distinction: large Euclidean coverage gain, little gain
in the reader-relevant direction. Euclidean reconstruction is not the objective
function for every useful projection.

## Exact affine program

Use row vectors. Let U be normalized construction inputs, x0 the raw-input mean,
s their scalar normalization, W the linear-kernel coefficients, m the residual
mean, c the kernel column means, and g its grand mean. Let L and b produce the
three fixed reader components from an upper state: parts(y)=yL+b.

~~~text
D = (U-mean(U)).T @ (W@L) / s
C = L + D
d = b + m@L - x0@D + (-c+g)@(W@L)
parts_hat(x) = x@C + d
score_hat(x) = reader_anchor + sum(parts_hat(x))
Delta score_hat = (x_b-x_a) @ C @ ones(3)
~~~

The identity does not assume exactly zero stored kernel-centering moments.
Tests include shifted anchors/moments and a finite-difference derivative check.
The new-versus-old scalar change decomposes into an offset at the old
construction origin and slope action, closing to2.37e-15. A change of reference
center can alter the fitted slope; it is not necessarily a scalar offset.

All576 stored world/natural score-component replays match within6.51e-15.
The independent raw-text bundle passes15 source-denied replays within6.33e-15.
No full P/Q is materialized at scoring time; kernel anchors are absent. Four
extracted Transformer blocks and their841MB weights remain necessary. The7.11MB
figure concerns numerical operators, not the entire runtime.

This is an algebraic simplification of our approximation, not a theorem that
the original LLM or human personality is globally affine. Data-driven reduced
operator construction has established precedents; see
[Kramer, Peherstorfer and Willcox, 2024](https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-121021-025220).
We borrow structured snapshot-based reduction, not fluid-equation conservation
or extrapolation guarantees.

## Artifacts, reproduction and next scientific question

All directories are under ignored results/grain_vector_mechanisms/navigation_v1/.

| Artifact | Location |
|---|---|
| Source IDs, paired states, coefficients | NAV-02/source_span_r1/construction_freeze.json and private_* |
| Worlds/natural function and edit results | NAV-02/source_span_r1/{worlds,natural}/analysis/ |
| Center/direction diagnostic | NAV-02/source_span_r1/center_span_analysis/ |
| Human comparisons and intervals | NAV-02/source_span_r1/external/analysis/ |
| Compiled local raw-text bundle | NAV-11/source_affine_r1/bundle/ |
| Exact compilation/raw replay | NAV-11/source_affine_r1/affine_compilation_replay.csv; replay_source_affine/ |

Entry scripts: run_grain_source_span_extension.py (prepare/construct/evaluate/
analyze), evaluate_grain_source_span_external.py (collect/analyze),
analyze_grain_source_center_span.py, export_grain_source_affine.py (prepare/replay).
Use fresh construction/export directories; preserve frozen code and byte hashes.
The original external collector retains a single Ruff SIM117 nested-with style
exception because its executed source hash is frozen. This is not a test failure.

The next mechanism is **conditional response construction**, not another source
size or alpha search. A bounded future candidate should fit both values and
paired edit responses using separate source-reference authors, then test the
existing twelve evaluation authors and external endpoints without refitting.
[Sobolev training](https://arxiv.org/abs/1706.04859) supplies a concrete precedent
for supervising derivatives as well as values; finite text edits would be our
secant-response adaptation, not an exact derivative or a replication of that
paper. This question is retained separately from completing the NAV execution
cards and the outstanding reader-source A/B comparison.
