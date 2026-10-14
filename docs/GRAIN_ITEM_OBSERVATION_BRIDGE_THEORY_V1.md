# Same-evidence behavioral observations and trait-name judgments

2026-10-13, HM255. This document specifies the observation objects before
reading the new item outcomes. It does not replace the frozen plan or method.

## 1. The missing mainline relation

HM248 produced a source-selector by reader-target action matrix. HM251 then
constructed query-free covectors from complete public behavioral item keys.
Positive source C actions did not establish positive named C reader actions.
Neither result is a request to improve prediction accuracy or numerical fit.

The new observation family asks the SAME public behavioral statements used to
construct the covectors of the SAME real comment inputs. Its competing
predictions distinguish:

1. An operational behavioral contrast survives, but its named-trait readout
   does not preserve that contrast.
2. The query-free input projection already fails to supply the corresponding
   operational behavioral contrast.

The comparison is between different model observations. It does not establish
that the model computes a trait-name answer by first computing these items.
No causal mediation or human questionnaire-response equivalence is assumed.

## 2. Input, observation and contrast are different types

Let X_uk be the kth actual eight-comment input for historical author u. There
are twelve authors and sixteen inputs per author. Let h(X) denote a frozen
contextual input representation and g_C(X) the old behavioral SOURCE covector.
The old positive laws p_u+ and p_u- were fixed before named or new item answers.

For each public item j, the NEW observation is

    p_j(X) = expit(z_1(X,j) - z_0(X,j))
    l_j(X) = z_1(X,j) - z_0(X,j).

The item sentence and question are part of the observation contract. The p_j
value is a probability CONDITIONAL ON the fixed binary answer alphabet, not
the model's full-vocabulary probability and not the author's actual response.
Full-vocabulary candidate mass and greedy-token diagnostics are also retained.

For any observation f, define the frozen finite action

    A_u[f] = sum_k (p_u+(k) - p_u-(k)) f(X_uk).

This is a signed expectation contrast over actual texts, not a fitted
predictor, population causal effect or latent trait estimate. The positive
law support and source directions are not changed after observing answers.

## 3. Key meaning and common endorsement must both remain

For each ten-item key, let H and L denote its official positive and negative
items, h_q = mean_H p_j and l_q = mean_L p_j. Define

    B_q = 1/2 + (h_q - l_q)/2
    U_q = (h_q + l_q)/2
    Q_q = (sum_H p_j + sum_L (1-p_j))/10
    a_q = (|H|-|L|)/10
    Q_q = B_q + a_q (U_q - 1/2).

B is an equally pole-weighted behavioral key. U retains common endorsement.
Q is the equal-item keyed aggregate, not a claim of administering IPIP to the
author. Both A and C have six positive and four negative items, so a_q=.2.
Their official key includes a known common-endorsement term.

For log odds, use the same linear operations with offset zero:

    B_q^l = (mean_H l_j - mean_L l_j)/2
    U_q^l = (mean_H l_j + mean_L l_j)/2
    Q_q^l = B_q^l + a_q U_q^l.

Applying A_u preserves these identities because the two laws each sum to one.
All five accounts, both poles, every item and both answer units are saved.
U is not declared noise or deleted. A positive Q need not imply a positive B.
Differences between probability and log-odds contrasts can reflect sigmoid
curvature and response saturation, not a change in the authors.

## 4. Relations and inference scope

The primary registered quantity is mean_u A_u[B_C]. Secondary quantities
include mean_u A_u[B_C-B_A], both pole actions and all item actions.
Whole-author bootstrap intervals preserve dependence among sixteen inputs
and twenty queries. They are pointwise descriptive intervals on this
historical panel, not simultaneous family significance or new human accuracy.

The four binary comment choices also admit a finite Walsh expansion. All
signed orders remain. Its interaction terms describe this input family;
they do not localize attention heads, establish a neural multiplication or
define new psychological factors. Matching event-position margins cancels
separable event-position accounts by design, not all contextual meaning.

Useful outcomes are decisions about where behavioral meaning fails or
survives. A usable item contrast would support developing a source-qualified
behavioral observer. Failure of both item and name contracts would direct
work toward contextual target-use construction, not another metric, layer,
sign, rank, wording or accuracy search.

## 5. Mathematical and psychometric ancestry

[IPIP official broad keys](https://ipip.ori.org/newBigFive5broadKey.htm) specify
the public behavioral statements and key polarity. Keying and common-response
accounts are mature psychometric operations.

[Park et al., 2024](https://proceedings.mlr.press/v235/park24c.html) distinguish
input and output representations and identify a causal inner product under
specified assumptions. A source vector cannot automatically become the
correct measuring covector simply because the arrays have the same size.
This experiment does not tune whitening or assume those causal hypotheses.

[Geng et al., 2026](https://arxiv.org/html/2609.12704v1) already construct
behavior-contrast trait directions and compare their relational geometry
with human personality impressions. That extraction or impression geometry
is not this project's invention. Here the added project ability is a complete
SAME-evidence item-response interface between independently constructed input
geometry and named-trait judgments.

The pinned [OpenAI three-state foundations](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/The-exact-reconstruction-threshold-for-the-three-state-symmetric-channel-September-25-2026/build/sections/02-foundations.tex),
lines98-114/166-210, distinguish a specified observation law from its posterior
and require conditional independence for branch combination. Twenty related
queries on the same eight events do not satisfy that independence by design.
No tree threshold, posterior sufficiency or reconstruction theorem transfers
to human traits here. The manuscript supplies design inspiration, not a new
applicable personality theorem.

## 6. Execution provenance

The original immutable item_observation_bridge_r1 collection failed BEFORE
any family was saved: MPS does not support the attempted float64 tensor
conversion. Its runtime receipt and all primary method seals remain.

A separately sealed mps_cpu64_recovery_r1 changes only where the full-vocabulary
logit normalizer is computed: unchanged float32 logits are copied to CPU before
float64 logsumexp. The original failed 1.6077266 model seconds are charged
against the SAME2200-second budget. No prompt, item, source direction, law,
readout, model or cache criterion changes. Tiny CPU compatibility tests precede
the collection; fixed first-family items0/19 provide a direct/cache guard.

Upstream h and item judgments still use the local neural model. This is not an
LLM-free raw-text measuring system. No private questionnaires, API calls, new
training, Git publication, lockbox access or historical queue recovery occur.
The original18 protocols/100 cards/44 questions remain OPEN.

## 7. Post-observation update

All192 families/3840 queries completed once in the separately sealed companion.
The registered mean probability B_C action is -.0046540047, pointwise95% author
CI[-.0098207828,.0011990281],6/12 positive. The log-odds analogue is -.0190772429,
CI[-.0480913658,.0077160130],5/12 positive. Both probabilities and log odds are
retained; neither gives the registered positive mean bridge. This is not a
significant population-reversal claim.

The probability common C action is -.0009747559 and Q_C=-.0048489559. Thus
6/4 item imbalance and common endorsement alone do not explain the negative
balanced key. Raw high C endorsement decreases while raw low endorsement
increases on average. Negative, positive and near-zero banks/items remain.

Historical named C and item log-odds B_C have similar means, but their author
action correlation is -.0165206 (post-outcome descriptive). Equal means do not
identify equivalent author measurement functions. These results do not support
locating the only break at trait-name wording. Close fixed-anchor/mean/metric
rescue and next inventory context-qualified measuring operators.

For a specified cache type K and fixed item j, a mathematical candidate is

    p_j(K)=sigmoid(v01^T F_j(K))
    omega_j(K)=D F_j(K)^T v01
    D p_j(K)=p_j(K)(1-p_j(K)) omega_j(K).

This is ordinary chain rule, not a new theorem, measured mediation or an
already transportable observer. Different token lengths/addresses require
an explicit correspondence before comparing caches. Inventory existing
HM125/148/154/221/223/232 interfaces before a new finite construction; do not
turn this candidate into an automatic gradient/layer/metric precision grid.

All results, failures, full signed accounts and provenance are documented in
the parent execution/GRAIN_ITEM_OBSERVATION_BRIDGE_RESULT_20261013_ZH.md.
