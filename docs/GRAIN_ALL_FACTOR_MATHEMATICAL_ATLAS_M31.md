# All-factor mathematical atlas: construction and evidence layers

2026-09-09. COMPLETE M31 synthesis. All source-function, adapted-state and
34-function equation packets are complete. This document distinguishes
exact mathematical identities, implemented estimators, and empirical transport.
It does not declare that a finite collection of correlations proves every
personality-inference use case.

## 1. What the common object is

Use a target specification, not just a personality word:

```
q = (construct, referent, occasion, wording, answer coding)
c = (material opportunity, task/view, corpus, language)
x_uc -> H_theta(x_uc,q,c) -> F_theta,q,c(H) -> s_uqc.
```

Theta is frozen. Human questionnaire Y is external to all these computations.
In the current instrument, s is a candidate-conditioned expected digit or binary
axis probability. The query itself shapes the state carrying the author's text.
Consequently common hidden dimension does not mean every query uses the same
direction. Target availability and target utilization are different parts of the
chain, following [Funder's realistic accuracy model](https://pubmed.ncbi.nlm.nih.gov/7480467/).

This resolves an important scope issue:9 Big5/MBTI functions,22 relational/session
queries, and3 adapted mood queries are distinct target specifications. All34 now
have frozen affine coefficient arrays and measured native transport, following
M28 and M30B. The narrower functions also have external MEPS associations; the
all-nine PANDORA source connection is also complete. The arrays share a
construction, not interchangeable coordinates or uniform psychological validity.

## 2. From an exact path to a reusable function

For a differentiable score F and a piecewise smooth hidden-state path gamma:

```
F(H_1)-F(H_0) = integral_0^1 <grad F(gamma(t)), gamma'(t)> dt.
```

This is the chain rule/fundamental theorem of calculus, not a new GRAIN theorem.
[Integrated gradients](https://proceedings.mlr.press/v70/sundararajan17a.html)
provides a related operational attribution method. It motivates integrating
derivatives over an explicitly chosen path; it does not make that path unique or
prove that an averaged field transports to another author.

The implemented nine-query family freezes derivative summaries from two
independent reference authors. P is the ordered question-region token matrix;
Q is the final-token state. Other context state is held at each declared
reference during partial differentiation. The reusable function is

```
f_q(P,Q) = b_q + <G_q,P-P_Rq>_F + g_q' (Q-Q_Rq).
epsilon_uq = F_q(H_u)-f_q(P_u,Q_u).
```

G averages the two context orientations of the integrated P paths; g is the
origin's partial Q gradient, with both averaged across the two reference authors.
It is not the exact simultaneous integrated gradient of every future P/Q/E path.
E-dependent interactions, curvature and numerical approximation can remain in
epsilon. Reusability is evaluated on separate authors and natural evidence edits.
M28 obtains level r=.776-.991 and response r=.550-.976; these are **native-model
fidelity**, not human questionnaire correlations. The score still requires the
frozen neural prefix. Removing the suffix is not removing every LLM computation.

## 3. Exact mean/configuration/final-state accounting

Let deltaP=P-P_R, delta_m=mean_tokens(deltaP), and
delta_C=deltaP-1 delta_m'. Let a_q=sum_tokens(G_q) and
G_Cq=G_q-1 mean_tokens(G_q)'. Then exactly:

```
M_q = a_q' delta_m
C_q = <G_Cq,delta_C>_F
Q_q = g_q' (Q-Q_Rq)
f_q = b_q + M_q + C_q + Q_q.
```

The partition distinguishes where each query uses its ordered token evidence.
It is an algebraic partition, not three newly discovered psychological factors.
M28 configuration-only response r ranges .530-.949. Adding final-state Q is
useful for some targets (C, JP) and less useful for others (N, TF). Thus the same
form can support target-specific operators without forcing one universal scalar.

## 4. Connect the function to actual author criteria

On one explicitly matched author population and one material condition:

```
Cov(Y_k,F_q) = Cov(Y_k,M_q) + Cov(Y_k,C_q)
              + Cov(Y_k,Q_q) + Cov(Y_k,epsilon_q).
```

Divide all terms by the SAME sd(Y_k)sd(F_q) to express their signed contribution
to native external r. To account for the function's external r instead, use
sd(Y_k)sd(f_q) and omit epsilon. Never add separately standardized correlations.
Signed contributions can cancel; neither a small term nor a remainder is
automatically noise. M29 measures these terms on213 Big5-only and64 MBTI-only
source authors, without fitting coefficients to human labels. Big5/MBTI source
groups cannot be merged into a same-subject bridge.

The atlas therefore keeps separate columns for function/native fidelity,
evidence-response fidelity, author geometry and human criterion association.
Improving the first column need not improve the fourth. This is a useful
localization result: readable model information and psychological cue use are
distinct research objects, not competing definitions of success.

## 5. Shared and specific structure without throwing information away

For V_q=(M_q,C_q,Q_q,epsilon_q), matched query-score covariance obeys

```
Cov(F_q,F_r) = sum_{a=1}^4 sum_{b=1}^4 Cov(V_qa,V_rb).
```

M28 enumerates36 query pairs and2880 signed terms across declared strata;
maximum accounting error6.51e-19. A strong contribution to one query need not
explain the shared contribution between two queries. Coordinate attribution is
conditional on the chosen reference and decomposition, not a uniquely identified
neural circuit.

M30 adds a different, observable score-level partition before new derivatives
are available. Each query's standardized score z_q is projected onto c_-family,
the mean of other queries excluding its own pre/post semantic family:

```
z_q = lambda_q c_-family + r_q.
Cov(z_q,Y_q) = lambda_q Cov(c_-family,Y_q) + Cov(r_q,Y_q).
```

This is descriptive within the46-person cohort, not an independently normalized
new instrument. AI trust's external r=.324 partitions .257 shared +.067 specific;
session evaluation .319 partitions .141+.179. Both are retained. Native output
sharing suggests a question for the hidden operator atlas; it is not itself proof
of a shared causal psychological path. Complete query/criterion matrices follow
the logic of [multitrait-multimethod analysis](https://pubmed.ncbi.nlm.nih.gov/13634291/)
without claiming a fitted latent MTMM model on46 people.

## 6. Conditions belong inside the mathematical program

```
L_qk(c,R) = Corr_R(s_q(x_uc,c),Y_uk).
```

M30 finds free-chat/session-evaluation r=.319 versus direct-answer r=-.023 on
the same people; general AI trust remains positive in both (.324/.265). Task
opportunity supplies a concrete axis of variation in L, consistent with
[dialogue-condition research](https://arxiv.org/abs/2405.13052). It does not
establish a causal treatment effect because these are different tasks in one
study, not a randomized test of our prompt construction.

M30C now adds actual state links: free-chat endpoint r=.427/.536/.518 and partial
r controlling preceding state=.408/.512/.526. Thus the occasion argument in L
has empirical content, not just notation. There is no need to call these stable
traits. M30B constructed all25 narrower query functions independently of the
participants. State-function external r=.422/.564/.555; median native fidelity
is .899 across all25 free-chat queries. Configuration contributes .217/.329/.375
to the respective state external r under the declared partition.

## 7. Complete source and coefficient ledger

M29 connects all nine fixed functions to source labels: Big5 mean r=.1406 on213
authors; EI/SN/TF/JP AUC=.497/.563/.620/.532 on64 other authors. The cached native
Big5 mean is .1671; paired differences include zero and precision/backend effects
remain in the comparator provenance. Configuration contributes .244 to TF's
external r=.207, offset by -.009 mean and -.028 final-state terms. E/A/O also
have configuration-heavy associations; N/C have different allocations. No human
labels fit any operator. [Source report](../reports/grain_vector_mechanisms/ALL_FACTOR_SOURCE_CONNECTION_REPORT.md).

`all_factor_equation_atlas_m31_v1/factor_equation_ledger.json` indexes all34
functions with actual array keys/shapes, anchors, references, source hashes,
native fidelity and external associations. The companion manifest lists43 human
criteria with occasions. M31's narrow atlas contains1700 cross-target links,
600 matched query pairs and9600 signed covariance terms, max error3.47e-18.
The equation ledger is reproducible with `scripts/build_grain_all_factor_atlas.py`.

## 8. Remaining construction decisions

User pause,2026-09-09: all entries below are PENDING, not a live queue. The next
two mainline routes are [state-producing reconstruction and direct useful-projection construction](GRAIN_TEXT_TO_FUNCTION_TWO_ROUTES_PLAN_V1.md).
Recording them does not launch experiments or change the completed M27-M31 results.

The subsequent planning-only update organizes these questions plus synthetic
world construction into [NAV-01–12 work packages](GRAIN_PENDING_PUZZLE_EXECUTION_PLAN_V2.md),
with [primary-source literature](GRAIN_NAVIGATION_LITERATURE_REVIEW_V1.md).
Both real-reference controls and synthetic-only reference functions are planned.
The table below records the M31 evidence, not additional executed experiments.

| Piece | Evidence now | Next executable question |
|---|---|---|
| Common core formula |34 functions; nine natural-response tests; complete human source links and M/C/Q accounting | Extend evidence-response tests to narrower state/relational targets on independent material |
| Narrow targets |25 independent-reference functions; median native fidelity .899/.895 across two views | Why do configuration versus final-state terms differ across targets? |
| State targets |13 adapted items x4 phases; state-function external r=.422-.564 | Develop the occasion-sensitive operator rather than forcing it into a stable trait |
| Query identity | E and EI edit responses nearly unrelated (.047 native on M28 panel) | Separate construct wording from digit/letter response coding in a bounded all-axis packet |
| Cross-target sharing | Exact signed covariance ledger; no common scalar forced | Determine whether source-linked shared terms recur across evidence opportunities |
| Reference-relative author positions |12-author fixed norm exists for core queries | Keep future scoring independent of the incoming batch; do not reuse descriptive MEPS norms as a deployable reference |
| Shared context computation |All34 currently use the same frozen observer, but separate query-conditioned forwards | Test reusable context prefix/KV plus query-specific transforms; this optimization is proposed, not implemented |

Do not replace these questions with another Ridge/LightGBM accuracy grid. The
mainline is an observable mathematical path from author evidence to a specified
trait judgment, with its common and target-specific parts made computable.
