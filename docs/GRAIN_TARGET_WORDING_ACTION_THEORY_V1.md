# Target-consistent action with wording-conditioned departures

2026-10-11. HM154 finite actual-model result, not a new mathematical theorem.
Builds on HM152 conditional predictive units and HM153 relative diagnostics.

## 1. The new scientific observation

For identical texts, target contrast in conditional lexical action partly
persists across three descriptions of the target. It is not exhausted by a
shared context action or a wording-common action. At the same time, individual
positions and some text/condition summaries favor another pairing. The useful
object is therefore a common target contrast PLUS conditional departures, not
an unconditional universal axis and not a classifier optimized for agreement.

The experiment uses all54 neutral synthetic articles and all6 hypothesis cases
per article. There are864 second-slice native positions and5184 selected-token
accounts, not864 independent people. A common mathematical q0 is used for each
six-field comparison; all12 reset laws enter symmetrically. No decoder forwards,
generation, model fitting, API calls or private-data use occurred in this packet.

## 2. Definition and exact reconstruction

At fixed text/history/native position, for target theta and wording f define

    delta(theta,f) = (hHC-hLC)-(hHR-hLR)
    U(theta,f,v) = W[v].delta(theta,f)
    q0 = softmax(W * mean_(theta,f)((hHR+hLR)/2))
    S(theta,f,v) = U(theta,f,v)-E_q0 U(theta,f)
    a(theta,f) = E_q0 U(theta,f)-DeltaA(theta,f)
    k(theta,f,v) = S(theta,f,v)+a(theta,f).

The pinned tied head has zero bias. q0 is the geometric mean of12 normalized
reset probability laws, renormalized; linearity of W makes the midpoint formula
exact. It is not an actual model run at a midpoint state or a human prior.

Six S fields live in the SAME L2(q0) space. For f in three families define

    m_f = (S_Ef+S_Nf)/2, d_f = (S_Ef-S_Nf)/2
    mbar = mean_f m_f, dbar = mean_f d_f
    omega_f = m_f-mbar, xi_f = d_f-dbar.

With target sign s_E=+1,s_N=-1, the exact observation interface is

    k(theta,f,v) = mbar(v) + s_theta*dbar(v)
                    + omega_f(v) + s_theta*xi_f(v) + a(theta,f).

Retain all five terms. In particular, calibrated scores are NOT obtained by
deleting wording/interaction/offsets because a mean target component exists.
They can be signal-bearing or observation-law-dependent; the experiment does
not identify them as psychological noise.

The corresponding balanced Hilbert energy identity is

    mean_(theta,f)||S(theta,f)||^2 = ||mbar||^2 + ||dbar||^2
                  + mean_f||omega_f||^2 + mean_f||xi_f||^2.

It is a designed-table decomposition, not variance of personality in a human
population. The conditional metric varies with text/history; comparing its
energies does not establish a single fixed global Euclidean personality axis.

## 3. A target discriminator that removes common wording action

For each pair of families f,g, average same-target inner products minus
average cross-target inner products equals2<d_f,d_g>. Thus

    T = (2/3)*sum_(f<g)<d_f,d_g>
      = 2*||dbar||^2-mean_f||xi_f||^2.

A target-common component alone is not evidence: averaging always makes one.
T additionally checks whether it outweighs the cross-family disagreement in
this declared units/metric. Common context and wording-only action have T=0.
Interactions can produce either sign; all pair cosines and magnitudes remain.

Independently swapping E/N within each family sends d_f -> epsilon_f*d_f.
Fix the first sign positive to obtain four distinct balanced pairings. Their
T values average to zero by the elementary finite-character identity. This is
an exact design comparison, not a stochastic null, exchangeability claim,
chance-level classification result or persona p-value.

## 4. Actual support and departures

All-unit mean energies, in squared nats per position:

| Common | Target | Wording | Target x wording | Total |
| ---: | ---: | ---: | ---: | ---: |
| .005928716 | .009358746 | .006635367 | .007424665 | .029347495 |

The target x wording energy remains substantial relative to target energy.
T mean is+.011292827, median+.001331592; negative168/864 positions remain.
Three d-field cosines average+.369394(named/described),+.275534(named/anchored),
+.134918(described/anchored). These are vector-field relations, not trait r.

Intended pairing is first in516/864 positions;348 favor another pairing.
Text-mean T is positive52/54, and text-mean intended pairing is first45/54.
All18task x style groups have positive mean T;17/18 favor the intended pairing.
Email/neutral is the exception: intended+.000072003, the +-+ alternative
+.002782930. Within-cell mean of each position's margin is NOT the margin
between cell-mean orbit scores, since max does not commute with averaging.
Keep these two summaries distinct; positive mean T alone is not winning every
local comparison. All neutral49 generation-cap truncations are retained.

## 5. What becomes possible

This identifies a target-organized component of model-conditioned context
action, rather than only observing generic context dependence. It supplies a
concrete relative-target field to test against qualified textual evidence,
then an interface for source-defined conditional operations. All remaining
terms must be available when reconstructing an actual evidence account.

The six fields concern two intended synthetic persona hypotheses. This is a
single E-versus-N meta-contrast, not a full multi-trait latent basis, a human
personality measurement or an LLM-free raw-text encoder. Hidden prediction
states were previously produced by Qwen. GRAIN author choice, condition
response, state/history, ownership and reference remain distinct channels.

Next: distinguish target-semantic action from prompt-semantic agreement by
binding the field to own/other evidence, polarity and opportunity on matched
public controlled materials. Inventory HM10B2/HM114/HM124-146 and existing
qualification/action mechanisms first. Do not replay ownership accuracy or
fit a predictor to these field angles. Freeze the discriminator and identify
what intervention would change this common component versus its departures.

## 6. Sources and novelty boundary

[Ginsbourger et al.2014 Section2 Eqs1-4](https://arxiv.org/html/1409.6008)
supplies the mature commuting-projection/FANOVA ancestry; the finite
Hilbert-valued extension here is elementary linear algebra. HM30 already
applied balanced context/wording decomposition. HM114 already compared
behavior-clause responses across wording; HM10B2 already used finite response
characters. This is a new six-field diagnostic test in this system, not the
invention of any of those operations.

[OpenAI/math](https://github.com/openai/math), fixed adc7f124, hypergraph
build/fourier.tex: reread product-character basis, Parseval and the complete
small-influence proof. The useful transfer is complete orbit accounting,
with no deletion of inconvenient components. Boolean influence and
hypercontractive bounds have not been applied to the text/model here.

## 7. Reproducible finite verification

Canonical results: results/grain_math_response_v1/target_wording_fields_r1.
54 output-head batches/6048 projected vectors,8.170seconds including load/IO.
Alternate CPU64 audit2.297seconds: all selected tokens/Gram-derived accounts
and text/condition means; six fixed whole-vocabulary spots, not every field.
Max algebra error2.50e-15, selected native error1.01e-5, projection spot
error2.10e-5. Same agent, not blind third-party replication.17 focused tests
and new Python lint/format pass. All handles terminal, no unbounded runner.
Original18/100/44scope remains OPEN; no scientific-failure/result overwrite.
