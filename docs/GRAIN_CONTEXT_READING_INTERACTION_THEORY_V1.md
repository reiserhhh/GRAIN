# Finite context-dependent evidence reading

2026-09-11. NEXT63/pass52. [Observed results](../reports/grain_vector_mechanisms/CONTEXT_READING_INTERACTION_REPORT_V1.md).
This is a local model-mechanism statement at frozen H16 coordinates. It develops
the text/GRAIN/query interface without inventing a new psychological construct.

## 1. The tested object

Write the native suffix's fixed logit contrast as L_q(B,R,c), where B is the BODY
state matrix, R contains every other retained state and layout, and c is the last
H16 state. For each original source i/query q, hold B,R fixed and set

    c'_iq=mean_(j != i) c_jq, delta_c=c'_iq-c_iq.
    g_iq=grad_B L_q(B_i,R_i,c_iq),
    delta_g=grad_B L_q(B_i,R_i,c'_iq)-g_iq.

Actual native calculations use the FP32 replacement; saved displacement accounts
for that cast. No query words, BODY tokens or GRAIN direction D_f were changed.
All8 contexts show nonzero delta_g and all32 direction actions change. Mean
relative gradient changes .104451/.160762 follow mean relative last changes
.053535/.057386. The original replay matches exactly at the tested source.

## 2. The mathematical link to a response operator

Where the suffix is sufficiently differentiable along this finite path,

    delta_g = integral_0^1 H_(B,c) L_q(B,R,c+t delta_c) delta_c dt
    delta_tau_f = <D_f,delta_g>.

This identifies a nonzero action of a path-averaged mixed derivative. It does not
identify every Hessian entry, local curvature at every t, or a unique internal
circuit. A real hidden-state counterfactual is useful at that coordinate scope;
the source-mean replacement need not correspond to a natural utterance.

The previous additive family is

    Lhat=b+a mean(z)+beta c+mean_t f_q(z_t,p_t).

Its mixed BODY-by-last derivative is identically zero. It can reproduce neither
the nonzero finite delta_g nor delta_tau for fixed BODY, whatever its coefficients.
Increasing local feature rank does not remove this algebraic restriction.
H16 itself is contextual: this is not a claim that an entire text encoder or
human word meaning is additive. It concerns our post-H16 substitute function.

## 3. A conditional family that permits the missing term

An explicit alternative is

    Lhat=b+a mean(z)+beta c+mean_t sum_m gamma_m(c) phi_m(z_t,p_t).
    H_(z_t,c) Lhat = (1/T) sum_m grad_z phi_m outer grad_c gamma_m.

It keeps one scalar/gradient program, unlike separately fitted scores and fields.
Scalar features phi can be the existing Gaussian/Hermite token dictionary;
gamma can be bounded source-referenced kernel features of the read-in state.
No free-language decoder is needed for this downstream numeric program. Its
context states still require an encoder; current H16 has not been replaced.

The next experiment uses matched augmented source targets for both additive and
conditional candidates. An added data channel and a changed architecture must
not be conflated. Source-mean interventions depend on all4 sources, so only
all4 training/new8 application is valid for their immediate reuse; the old source
leave-one-out protocol cannot silently inherit targets constructed using its
excluded person's state.

## 4. Field geometry versus GRAIN action geometry

For an orthonormal source basis U (excluding the current source in this analysis),

    delta_g=delta_g_parallel+delta_g_perp,
    ||delta_g||^2=||delta_g_parallel||^2+||delta_g_perp||^2,
    delta_tau=delta_tau_parallel+delta_tau_perp.

The field components are orthogonal; their contracted action arrays need not be.
Rank8 mean energy fractions .212/.252 coexist with signed action projections
1.387/.898. Neuroticism's complement -.387 cancels part of the retained action.
The complement is not discarded noise; nor does low full-field coverage imply
that the basis is useless on a given GRAIN direction. Preserve both geometries.

## 5. Existing foundations and the project-specific contribution

[Vaswani et al., Attention Is All You Need, 2017](https://arxiv.org/html/1706.03762v7)
defines attention through query/key-dependent weights applied to values. It gives
a general reason to expect reading to depend on context, not our measured H16
effect sizes or a proof of human personality inference.

[Perez et al., FiLM, 2018](https://arxiv.org/html/1709.07871v2), equations1–2,
uses conditioning information to scale/shift feature maps. This motivates a
context-modulated feature reader. We propose finite scalar kernel products with
analytic derivatives, not a replication of their vision architecture or evidence
that Qwen implements a FiLM layer internally.

The path-integrated mixed derivative and orthogonal projections are standard
calculus/linear algebra. The empirical contribution here is a bounded, source-
addressed demonstration that the SAME author BODY and GRAIN directions are read
differently when only its final read-in state changes, plus a specific algebraic
failure of the current substitute family. Next develop that missing interaction,
not another source-only audit or psychological naming exercise.
