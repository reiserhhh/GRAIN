# All-factor mathematical atlas: construction and evidence layers

2026-09-09. Working M31 synthesis. The expanded goal remains active until the
source-function and adapted-state packets finish. This document distinguishes
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
queries, and3 adapted mood queries are different levels of completion. Nine have
frozen affine coefficient arrays and measured transport. The others currently
have a native query program plus saved states, not interchangeable coefficient
arrays. The catalog records this instead of treating query expansion as completed
function transport.

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

M30C adds state composites with occasion-matched endpoints rather than pretending
all text-derived quantities are traits. There is no need to force a label before
identifying an observable response function and its behavioral connections.

## 7. Remaining construction decisions

| Piece | Evidence now | Next executable question |
|---|---|---|
| Common core formula | Nine functions, reference coefficients, natural-response transport | M29: which M/C/Q terms retain human source relations? |
| Narrow targets |22 queries x2 views; complete external matrix and shared/specific score split | Build externally referenced G/g maps, then test their saved P/Q states; no46-person label fit |
| State targets |13 adapted items x4 phases audited; M30C prepared | Which actual state endpoints are visible in the appropriate view? |
| Query identity | E and EI edit responses nearly unrelated (.047 native on M28 panel) | Separate construct wording from digit/letter response coding in a bounded all-axis packet |
| Cross-target sharing | Exact signed covariance ledger; no common scalar forced | Determine whether source-linked shared terms recur across evidence opportunities |
| Reference-relative author positions |12-author fixed norm exists for core queries | Keep future scoring independent of the incoming batch; do not reuse descriptive MEPS norms as a deployable reference |

Do not replace these questions with another Ridge/LightGBM accuracy grid. The
mainline is an observable mathematical path from author evidence to a specified
trait judgment, with its common and target-specific parts made computable.
