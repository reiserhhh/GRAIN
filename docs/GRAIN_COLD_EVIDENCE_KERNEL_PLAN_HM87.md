# HM87: body-only evidence kernel and query-context discrimination

Frozen before collecting new material, 2026-10-10. No accuracy target.

## Question and prior-work boundary

NEXT50 already reused source queries on query-conditioned evidence. The old
conditional baseline executed layer17 projections on cached states and fitted
an existing score. HM86 already measured neutral versus native queries on body
KV captured during application query forwards. None of these is claimed new.

This packet removes application target-query neural computation from the primary
observation: encode each fresh body once, then apply all six frozen queries in a
separate numerical-only process. Its scientific question is whether role-sensitive
target selection and author-evidence composition persist in this carrier, and
which action changes when the actual contextual query is allowed to move.
Runtime savings alone are not the finding. No teacher score is fitted or read.

## Fixed source and new material

- Reuse HM86's sealed 24 query-free source calibrations and six neutral queries.
  Source behaviour directions are supervised design calibration, not discovered
  personality factors. Model, code, calibration and query-position gaps sealed
  before applications. No new source fit.
- Six cyclic focal/companion target pairs, complete x/y/g/h cube, two sentence
  orders, one new wording family: 192 fresh bodies. All conditions retained.
  K1 actor A; K2 actor g; K3 actor h with a plain notebook. Query target always A,
  not established real author. Reversing sentence order is a text intervention,
  not a gauge transformation or pure change of mathematical coordinates.
- 16 body-only repeats from HM86 context0 are causal-prefix implementation checks,
  not independent replications. Their body KV are compared against old captures.
- 576 actual native query captures on all 96 forward-order fresh bodies provide
  the contextual-query comparison. Reverse order has cold observations only.
- Total 784 captures, at most 600 model seconds. Stop before layer17 attention
  output/MLP and all upper/final score modules. No API, private data, training,
  generation, Git, Claude, automation or old queue.

## Numerical observation

For body tokens I, fixed raw query q0 and virtual query position p:

    a_i = exp(<K_i, R_p q0>/sqrt(d)) / sum_(j in I) exp(<K_j, R_p q0>/sqrt(d))
    r = sum_i a_i V_i
    r = sum_G m_G mu_G + sum_G sum_(i in G)(a_i-m_G/n_G)V_i.

Queries use frozen per-target suffix gaps and the model's default RoPE frequencies.
Within-body normalization cancels non-body keys from the attention denominator;
it does NOT remove prior context already embedded in body K/V. This packet cannot
compute actual whole-prefix body allocation from body-only data.

The cold process opens only bundle.json and padded carriers.npz, never raw text,
design truth, native queries, source units, model weights or neural packages. All
six queries, directions and heads are computed. Uniform-body null gives zero
query selectivity by construction, not an empirical discovery.

On forward bodies retain native and cold observations. For each high/low action,
cross the two raw native queries with the two body carriers at the recipient's
query position. Symmetric four-corner body and query contributions sum exactly
to the native action. This is standard polarization, not unique causal mediation.

## Author consequences and decisions

After cold output is frozen, join the design only for analysis. P+/-=(1+/-epsilon
yg)/16 have identical individual marginals but different event/actor association.
Compare every target, order and readout; also h=-g equal-name-count restriction,
which changes two roles jointly and is not a pure ownership intervention.

- Cold target selection and yg contrast persist: retain a reusable local evidence
  kernel as a GRAIN observation component; do not call it a full personality model.
- Native advantage is carried by moving queries: preserve that conditional
  processing as a separate mechanism, not a missing coefficient to tune away.
- Order changes sign or composition: record the dependency and investigate its
  carrier; do not select the winning order or tune query gaps/head/layer/rank.
- Failure is retained. No accuracy rescue, extra wording sweep or blanket closure.

## Provenance

RoPE rotations and relative positions are established in
[RoFormer](https://arxiv.org/abs/2104.09864). Entity/attribute binding and position
interventions have prior implementations in
[Feng and Steinhardt](https://arxiv.org/html/2310.17191v2).
[OpenAI/math fixed paired jets, lines19-109](https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex)
motivate a fixed observation family only; no PDE uniqueness, smooth-bundle or
uniform spanning theorem is transferred. Novelty remains a contribution candidate
in the evidence-to-author measurement construction, not these standard identities.
