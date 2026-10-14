# GRAIN: pre-answer evidence selection and author composition

2026-10-10. HM86 finite experiment verified; not global theory completion.

## 1. What the new observation establishes

The fixed local Qwen prefix yields query-selective body reading BEFORE layer17
attention output, upper blocks, final normalization or answer logits run. The
observation requires the frozen neural prefix; it is not an all-text/no-LLM
measurement system. Source behaviour directions come from actor-balanced
query-free value means, not previously obtained personality-target scores.
This moves the current interface from fitting answers to inspecting a local
vector operation. It does not prove this operation mediates the final answer
or uniquely implements psychological evidence selection.

On 96 fresh controlled bodies, body-conditional native reading has positive
matched-query selectivity for all six predeclared targets. A frozen neutral
query seed has it for five. All six native within-event pairing terms have
positive selectivity. Uniform-body reading has zero query selectivity by
construction, not an independently discovered null. Actual full-prefix body
contribution has positive selectivity for only three targets and a negative
average. The two observation units MUST remain separate.

## 2. Two levels of query routing

For token i, head h and target query q, use the effective scaled, RoPE-rotated
query in the ordinary attention formula:

    p_hqi(T) = exp(k_hi(T)^T q_h(T)) / sum_j exp(k_hj(T)^T q_h(T))
    m_hq(T) = sum_{i in body} p_hqi(T)
    r_hq(T) = sum_{i in body} p_hqi(T) v_hi(T) / m_hq(T)
    o_hq_body(T) = m_hq(T) r_hq(T).

m is allocation between the body and the surrounding instructions/question;
r is selection WITHIN the body. Native q already depends on the body through
blocks0-16. The neutral-reference q retains the target query but not the
application body, and is moved to the application endpoint's RoPE position.
Both are explicit interventions in the local numerical attention function,
not complete-network causal mediation experiments.

The actual six-query body K/V arrays agree exactly in this design. Thus the
query comparison changes the reading operation, not the earlier body vectors.
Nevertheless, q can encode contextualized information already computed by the
prefix. Stronger native selectivity does not prove a pure target-only selector.

## 3. Event means are not all of the observation

For each declared event group G:

    m_hqG = sum_G p_hqi
    mu_hG = mean_G v_hi
    rho_hqG = sum_G (p_hqi-m_hqG/|G|) v_hi
    o_hq_body = sum_G m_hqG mu_hG + sum_G rho_hqG.

The event-uniform term is the exact expected reading under independent uniform
permutations of values within the event, with K/q held fixed. rho is what this
operation removes. This is not a rule to delete text or label it as noise.
All original values, allocation and pairing observations are preserved.

At two text endpoints, possibly with unequal token lengths:

    Delta o_body
      = sum_G bar(m_G) Delta mu_G
        + sum_G Delta m_G bar(mu_G)
        + Delta sum_G rho_G.

This is a standard symmetric finite-product identity. The final term changes
with both values and weights; it is not an attribution to attention alone.

The body-conditional native selectivity means are .035269375 for event-uniform
and .031621485 for pairing; they sum to .066890860. They use calibrated vector
units, not Pearson r, psychological strength, accuracy or information fractions.
Per-target native pairing selectivity is positive on all six; event-uniform
selectivity is negative on openness. Therefore an event mean-only observation
loses a specific local distinction even when it retains a strong behaviour
content direction. No post-result head or layer was selected.

## 4. Source calibration and author-level consequence

For target t, define d_t as the unit actor-averaged high-minus-low K1 value
direction from24 public source bodies, without a target query in their executed
inputs. Apply every d_t to every local reading; target labels only evaluate the
fixed cross design, not select an application winner. This is supervised
source calibration using designed behaviours, not unsupervised factor discovery.

    phi_tq(T) = <d_t, r_q(T)>
    Theta_tq(P_u) = E_{T~P_u} phi_tq(T)
    Theta_tq(P_u)-Theta_tq(P_v)
        = integral phi_tq(T) d(P_u-P_v)(T).

This links an event-local numerical operation to a declared author evidence
distribution. It is a standard expectation identity, not identification of
latent personality. It is compatible with GRAIN's reference-population spine,
but HM86 does not integrate the historical token-slice factor bank or estimate
natural authors' distributions, choice supply or personal response B_u.

For the full x/y/g/h cube, two synthetic laws

    P+/- = (1 +/- epsilon*y*g)/16, epsilon=.5

have identical separate x/y/g/h marginals. Their local observation difference
is 2epsilon times the yg coefficient. Native matched-query body-conditional
differences are positive for all six targets, .025853009 to .155405136. In the
equal-name-count h=-g restriction they remain positive, .033434300 to .260809694.
The latter changes the K2/K3 role configuration jointly: it rules out a solely
total-name-count account for these comparisons, not slot/order or context.

Native body-conditional matched behaviour action averages .770577453 for A
and .423202452 for B; uniform reading is .998258869 and .962704874. Pairing's
average is .002094359 for A but -.177059041 for B. Its role-sensitive effect
here is largely a relative reduction of the B action, not an amplification
of the A mean. These means are descriptive across differently calibrated
target directions; not pure person attribution or psychological effect sizes.

## 5. Theoretical update and next decision

Retain three objects instead of one scalar answer: total body allocation,
within-body target selection, and event-local pairing/role composition.
Target-corresponding semantic value direction and actor-sensitive joint
observation can be distinct, even on the same raw text. This is the contribution
of the finite experiment, not a new general attention or group theorem.

The next mainline question is whether a source-built numerical kernel can
retain these paired event/role distinctions on a cold new input without
application contextualized target-query computation. HM86's neutral seed is
an already executable starting point. Do not optimize it to reproduce answers;
first distinguish body-dependent query transport from event-local key/value
binding and test the observation's author-composition consequence. Original
18/100/44, natural-language source learning and psychological meaning stay open.

## 6. Mathematical and methodological provenance

- OpenAI/math pinned adc7f1241b42e322a6451854ab7e4b4c146bf78a,
  metric/connection 06-matrix.tex lines19-109, reread in HM86: a fixed paired
  family and uniform spanning argument inspire fixed observations. Harmonic
  jets, smooth vector bundles and its PDE theorem are NOT established here.
  https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex
- Feng/Steinhardt, entity binding, sections2-3 and generality caveat: independent
  entity/attribute manipulations motivate role controls; HM86 does not recover
  their binding IDs or inherit their model-family conclusions.
  https://arxiv.org/html/2310.17191v2
- Geiger et al., causal abstraction: final-output intervention equivalence is
  distinct from our local QKV numerical comparison.
  https://papers.nips.cc/paper/2021/file/4f5c422f4d49a5a807eda27434231040-Paper.pdf

Canonical results: results/grain_math_response_v1/query_evidence_routing_r1.
606 new prefix captures,76.041358 model seconds; no upper scoring, generation,
API, private data or neural training. All12 heads retained.21 focused tests and
Ruff/format passed.12,568,680 alternate scalar checks max1.021405e-14, same agent
not third-party reconstruction. Final status is verification.json; frozen
primary summary's pending-verification wording is retained as history.
