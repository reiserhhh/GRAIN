# Independent evidence encoders do not imply independent measurement

HM111 theory companion, 2026-10-10 JST. This document specifies a constructive
dependency law, not a new general theorem or an identification of psychological
truth. Numerical results belong to the separately sealed experiment.

## 1. The missing distinction in the author-measurement chain

HM110 distinguished summing independent scalar observations from summing final
states before one nonlinear read. Both differ from encoding a joint record. That
does not tell us where joint encoding is necessary. A joint current state can be
formed even when every historical event was encoded independently.

Separate the following objects:

    historical event input x_j
    event carrier h_j
    relation/ownership input r
    contextual collector state s
    fixed target observation F_q(s).

The collector is part of the instrument. Its interaction coefficients are not
the author's psychological response operator B_u. The empirical composition of
qualified events is part of the author's evidence account; it is also not a
psychological ground truth supplied by the instrument.

## 2. An actual block-triangular construction

For layer l, define a token-support relation M and write one head operation as

    p_ij^l = exp(<Q_i^l,K_j^l>) / sum_{k:M_ik=1} exp(<Q_i^l,K_k^l>),
    u_i^l  = sum_{j:M_ij=1} p_ij^l V_j^l.

Q/K include the model's scaling/rotary conventions. GQA, all heads, residual
updates, tokenwise normalization and the MLP are retained in T_l. This is not a
claim that the 17-block model has a finite exact linear path expansion.

With event-to-event and event-to-relay permissions both disabled, the causal
recurrences are

    P^{l+1}   = T_l(P^l; P^l),
    E_j^{l+1} = T_l(E_j^l; P^l,E_j^l),
    R^{l+1}   = T_l(R^l; P^l,R^l),
    C^{l+1}   = T_l(C^l; P^l,E_1^l,...,E_6^l,R^l,C^l).

Within each segment, the original token causality remains. Shared-prefix states
are identical and cannot read later inputs. Invisible padding cannot relay a
message. Per-entry coordinates are fixed so that changing one event's length does
not change another event's positional embedding. The owner declaration remains
present in R and visible to C; it is not discarded as nuisance context.

Induction over layers gives, for independently varied fixed-coordinate continuous
event embeddings and tokenwise normalization,

    d E_j^L / d x_i = 0  (i != j),
    d R^L   / d x_i = 0.

This is a structural statement about the declared architecture and intervention.
It is checked numerically on final K/V across every matching event configuration.
It does not assume that naturally observed human events are independent.

There is no analogous requirement that

    d^2 F_q(C^L) / (d x_i d x_j) = 0.

The shared collector can change its queries/weights and pass joint inputs through
the MLP. Discrete behavior contrasts test this joint operation; they are not
estimates of the infinitesimal derivative above.

## 3. What a reusable measurement operation would have to retain

The positive object is an event bank plus a recursive joint collector:

    s_q = F_q(Collector({E_j^0,...,E_j^L},R,C)).

It is not the independent sum sum_j F_q(E_j^L), nor necessarily a nonlinear read
of the sum of final event vectors. Layer-indexed states and their relation context
can carry different information. A future compiled collector must operate on new
units under these rules instead of memorizing observed behavior-pair coefficients.
HM111 still uses the frozen neural T_l; no distilled or encoder-free new-text
operation is claimed. The new cold processes perform author integration only.

For a qualified joint observation kernel T_q(e_i,e_j,r,c), an author account is

    Theta_uq = integral T_q(e_i,e_j,r,c) d P_u(e_i,e_j,r,c).

Marginals do not determine this integral unless the observation is additive on
the relevant support. With a fixed four-corner binary support, the interaction
term b_ij,q E_u[xy|c] is exact. The general observation need not be a universal
quadratic function; conditional coefficients can change with materials.

## 4. The actual route intervention, not a causal percentage

For c=event contextualization and r=common relay, denote the observations by
F00,F10,F01,F11. The signed construction is

    F11 = F00 + E + R + I,
    E = F10-F00,
    R = F01-F00,
    I = F11-F10-F01+F00.

The same identity holds after integration by any fixed author law. E/R/I can
cancel or reinforce the collector. RMS values are not additive, and signed
alignment is not the fraction of personality information or a unique causal
mediation share. Removing attention permissions also changes denominators.

If F00 retains a joint author contrast, an isolated event bank with a current
collector can construct it: prior event contextualization is not necessary for
that finite instrument response. It does not follow that the full routes are
unimportant, or that the collector reconstructs natural full-context scores.

## 5. Link to GRAIN and the four-W account

* what: behavioral event content, including its joint composition;
* who: actor qualification resolved with the relation/ownership context;
* where: source conditions/materials and the observation opportunity;
* when: remains the empirical sequence/time question, not network layer depth.

No condition mean or choice profile is automatically removed. Designed positive
author laws demonstrate an information-preservation requirement, not human
personality validity. The encoder's associations and clinical/psychological
interpretation still require their own links; instrument-only interaction must
not silently be treated as an author trait.

## 6. Prior art and scope of the contribution

[Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html)
already distinguish Q-, K- and V-composition. Their small attention-only examples
do not locate this GRAIN observation's event/relay/collector mechanism for us.
HM111 uses actual fixed-support cuts, not a claim to invent transformer composition.

OpenAI/math adc7f124's 06-matrix.tex lines1--150 were reread for paired complementary
observation families. The harmonic-frame, smooth jet and elliptic uniqueness
assumptions are not transferred to this token dependency graph. The block-triangular
argument and finite differences here use classical dependency calculus.

The project contribution is the tested distinction and its concrete construction
consequence for qualified author evidence, not a new mathematical identity. It
does not complete original18 protocols/100 cards/44 questions or establish an
independent psychological scale.

## 7. Actual finite result and construction decision

All1168 prefixes completed and the literal17-layer hooks were checked. During
collection5256 final K/V invariance comparisons had maximum error0. Arm00 retains
qualified same-marginal author contrasts RMS .010318353/.012620092 versus full
.014178163/.017490808. The collector can form joint observations from independently
encoded historical entries. It is not a substitute for the full-context instrument:
the00/full effect patterns have result-after cosines .234692/.228287, and route
increments/interaction cancel. All components are retained.

The next construction is a layer-indexed event/relay library captured from only
single/base units, followed by fixed numerical C recurrence on new combinations.
That compiler has not been run here. Source pair states/target values must not enter
its inputs, and use of frozen neural parameters must remain explicit. This is an
operation-construction task, not a new score-fitting or head/rank precision search.
