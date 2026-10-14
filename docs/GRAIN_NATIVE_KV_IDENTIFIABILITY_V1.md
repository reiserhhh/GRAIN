# Native historical K/V identifiability

2026-10-11. HM132 is a finite mechanism experiment, not personality prediction
optimization. This document separates a standard mathematical identity from the
new observations needed to identify the operation in the present reader.

## 1. Scientific object and intervention

For each behavior family and queried target, let b in {-1,+1} denote the
designed behavioral contrast and p the queried person (B=-1,A=+1). Let
K_H^o(b) and V_H^o(b) be the actual post-RoPE keys and values of historical
positions H for acting person o. Capture them at every decoder block from the
frozen local observer, not from a fitted personality direction.

Define N_q(b,k,v,p) as the native readout when suffix attention receives
K_H^k(b) and V_H^v(b). The all-block H-only mask, the prior system, the
physical recipient's prefix computations, and the full suffix program stay
fixed. Suffix states and Q are allowed to evolve recursively. Thus this is
an intervention on two defined primitive streams, not a fixed-Q single-head
formula or a decomposition of a human's true personality.

At one suffix attention operation the relevant form is

    O_P = softmax(Q_P K^T / sqrt(d) + M) V,

with the historical pieces of K and V replaced independently. Other reachable
system/suffix pieces are native. Prefix attention rows use the original stock
operation. Both donor and recipient histories have identical positions,
padding, behavior wording and token bags; ownership differs only in the final
declaration. A matched whole-history swap must reproduce the other natural
owner's H-only output. All branches, including failures, must remain recorded.

K is not assumed to be a pure identity vector, nor V a pure content vector.
Any functional interpretation must follow the measured interchange response.
All-block effects include recursive suffix dynamics and are not unique circuit
shares. This packet leaves the origin of HM131 cross-access cancellation open.

## 2. Exact finite character expansion

Any real-valued function on the complete four-role cube has the unique expansion

    N_q(b,k,v,p) = sum_{S subset {b,k,v,p}} gamma_{S,q} chi_S(b,k,v,p)
    gamma_{S,q} = (1/16) sum_{b,k,v,p} N_q(b,k,v,p) chi_S(b,k,v,p).

This is the established character basis of the group (Z/2Z)^4. It is not a
linear-response approximation: the 16 coefficients exactly represent these
16 finite observations, including all interactions. Outside this discrete
support, no continuity, extrapolation or arbitrary-text sufficiency is implied.

Natural histories only expose the diagonal subgroup k=v=o. With bit order
(b,k,v,p), the restriction is

    chi_{(s_b,s_k,s_v,s_p)}(b,o,o,p)
      = b^s_b o^(s_k XOR s_v) p^s_p.

The full restriction map, in integer bit-mask notation, is

    beta_0 = gamma_0 + gamma_6
    beta_1 = gamma_1 + gamma_7
    beta_2 = gamma_2 + gamma_4
    beta_3 = gamma_3 + gamma_5
    beta_4 = gamma_8 + gamma_14
    beta_5 = gamma_9 + gamma_15
    beta_6 = gamma_10 + gamma_12
    beta_7 = gamma_11 + gamma_13.

In particular,

    beta_bop = gamma_bkp + gamma_bvp.

The bkvp interaction becomes the common bp term, not the person-qualified bop
term. Dropping it would lose an effect rather than remove noise.

## 3. Why improving natural accuracy does not resolve this question

The functions N_K=bkp and N_V=bvp are distinct off the natural diagonal, but
identical at every k=v=o input. Any mixture a*N_K+(1-a)*N_V also gives the
same natural bop behavior. The natural restriction has rank8 and kernel
dimension8. Therefore even exact, noiseless natural readout data cannot
uniquely recover the separate key and value action in this unrestricted
16-dimensional function class.

The missing measurement is an intervention, not another fitting objective.
Observe the two off-diagonal combinations (-1,+1) and (+1,-1) at each b,p.
Then all16 support points are present and character inversion identifies the
chosen finite function. A narrower parametric class could identify a mechanism
only by adding assumptions; this experiment avoids making that assumption.

## 4. Relation to the GRAIN author-evidence account

For a constructed author's evidence law P_u(b,o), retain the qualified
observation at queried person p:

    S_u(q,p) = sum_{b,o} P_u(b,o) N_q(b,o,o,p).

For the controlled laws P_+=(1+eta*bo)/4 and P_-=(1-eta*bo)/4,

    S_+(q,p)-S_-(q,p)
       = 2*eta*(beta_bo,q + p*beta_bop,q)
       = 2*eta*(gamma_bk,q + gamma_bv,q
                   + p*(gamma_bkp,q + gamma_bvp,q)).

Both laws have the same content and owner marginals. Their difference comes
from the ownership/content organization. This connects the intervention to
the measurement-construction problem without inventing new authors, declaring
a human psychological score, or treating one event as a complete trait.
For real evidence, the law must come from qualified events, conditions and
observation opportunities; the controlled laws are a witness, not a substitute.

Separate components may reinforce or cancel. A signed projection on the
natural field can be negative or greater than one. It is not an information
percentage. Record all6x6 behavior/target cells and four different output units;
do not rotate their signs after observing the results.

## 5. Ancestry and contribution accounting

[O'Donnell, Analysis of Boolean Functions, 2021 corrected edition, Sections
1.2-1.4](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf)
provides real-valued cube characters and coefficient extraction. The diagonal
restriction above is proved by explicit substitution. It is established math,
not a new representation-theoretic theorem.

[Feng, Russell and Steinhardt 2024, Sections 4.1-4.2](https://arxiv.org/html/2406.19501v1)
separates content/binding hypotheses and tests binding by interchange.
[Heimersheim and Nanda 2024, Section 2.2](https://arxiv.org/html/2404.15255v1)
distinguishes mediation, direct pathways and cancellation. These are tool and
experimental-design antecedents, not discoveries claimed here.

OpenAI/math pinned adc7f124, signed-tensor 03b-isotypic lines20-133, motivates
retaining multiplicity copies before combining observed sectors. Its signed
tensor/Specht density and operator-angle bounds are not applied here: the
population, assumptions and measured operators differ. It is an inspiration
for the identifiability question, not authority for a personality theorem.

The empirical contribution sought is a reproducible, target-conditioned
key/value ownership action in the present behavioral-evidence program. That
would specify which operations a reusable observer must retain. It does not
by itself remove the pretrained encoder or establish psychological validity.

## 6. Execution interface

Frozen plan: `docs/GRAIN_NATIVE_KV_IDENTITY_PLAN_HM132.md`.
Input and result root: `results/grain_math_response_v1/native_kv_identity_r1`.
Runner: `scripts/run_grain_native_kv_identity_hm132.py`.
The explicit prepare/collect/analyze stages must not overwrite existing roots.
Read-only verification rechecks the raw donor receipts, all28 actual blocks,
body isolation, whole-swap/no-op controls, original source readouts, primitive
logits, full character reconstruction and saved tables.

No predictor fitting, model training, API, private data, model selection, head
search, accuracy grid, Git publication or old-queue resumption is involved.
Original18 protocols/100 cards/44 computational scope remains open.

## 7. HM132 observed finite result

All100 batches/1200 native endpoints completed in384.637613 cumulative model
seconds, below the original600-second cap. No uncertain or failed forwards
were replayed. Old24 H-only readings, four matched/self/whole-owner controls,
historical query-independence and body isolation had maximum errors0.

Across the full6x6 field, the key/value/natural person-qualified RMS in the
score unit is .010953/.000756/.010833 on oldbank0 and
.016187/.001233/.016467 on newbank3. Signed key projections on the natural
field are1.008726/.980341; values-.008726/.019659. In log odds the key
projections are.972751/.953852 and values.027249/.046148. These are signed
finite decompositions, not percentages of psychological information.

Therefore the selected natural person-qualified operation is principally
expressed through the historical-key intervention. Values still matter:
newbank agreeableness has score gamma_bkp=-.00116443 and
gamma_bvp=+.00135122, leaving beta_bop=+.00018679. A near-zero natural
response can conceal opposing identified operations. This is not a new trait
correlation, nor a universal inference that keys are identity and values content.

A post-design cache audit of24 pairs/1344 block/primitive records found exactly
zero owner-dependent cache change before the first edited ownership token.
Earlier behavior-token states cannot acquire a later owner's declaration by
retroactive causal attention. Ownership-dependent historical differences occur
within the declaration/downstream causal support. The interpretation concerns
this autoregressive encoding order, not where human personality resides.

The standard restriction identity was numerically checked to6.66e-16; an
independent sign/logit reconstruction gave maximum3.66e-15. Complete4608
four-role character rows and2304 natural-diagonal rows were verified read-only.
The mechanism decision is to retain target-conditioned historical selection and
qualified evidence, rather than fitting a current scalar to mimic native output.
Generic entity routing versus behavior-conditioned target selection remains the
next question; HM131 cross-access normalization/cancellation also remains open.
