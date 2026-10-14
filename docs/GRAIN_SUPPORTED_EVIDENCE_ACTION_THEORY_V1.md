# Supported evidence action and GRAIN response marks

2026-10-10. HM121 formal construction; actual numerical outcomes are recorded
separately. No declaration of a natural-text manifold, psychological trait or
completion of the original18/100/44 program.

## Observation object

At history layer l, use a finite positive marked measure

    nu_l = sum_i a_i delta_(p_i,k_li,v_li,g_i).

p is the original rotary position. g is a frozen anonymous GRAIN coordinate
from the same addressed lexical slice. g does not enter the attention reader;
it is an observation overlay, not a causal personality label. Zero-mass
representatives do not enter the measure. Refining one atom into several
identical atoms whose masses sum to the old mass preserves this object.
It is not equivalent to inserting repeated text at new positions.

For fixed current Q, attention on nu plus causal current-token measure kappa is

    att = integral exp(logit(q,k)) v d(nu+kappa)
          / integral exp(logit(q,k)) d(nu+kappa).

This is the established Boltzmann-Gibbs measure representation of attention,
not an invented high-mathematics theorem. Current states recur through the
immutable pretrained maps, so the overall reading F(nu) is nonlinear.

## Chosen path and action

Use the explicit positive mixture nu_t=(1-t)nu_low+t nu_high. Both atom families
retain positions and separate marks. Only their masses vary; K/V need not
cross arbitrary padded values. This is a selected instrument interpolation,
not proof of a plausible intermediate utterance or the model's actual
generation/reasoning trajectory.

In a combined atom representation, dm_i=-a_low,i or +a_high,i. Its local action
is the ordinary functional mass derivative

    d att / d m_i = exp(logit_i) (v_i-att) / Z.

Downstream adjoints transport it through the later computation. Schematically:

    D_i F = sum_(layer,current position,head)
            dot(lambda_lrh, exp(logit_lrhi) (v_lhi-att_lrh)/Z_lrh).

lambda is the derivative of the final chosen reading with respect to that
attention output. It is computed, not fitted to human traits. It includes later
recursion; the current query's indirect earlier-state dependence is handled by
the full chain rule, not dropped in a frozen-query approximation.

Integrate along the declared path:

    A_i = dm_i integral_0^1 D_i F(nu_t) dt
    F(high)-F(low) = sum_i A_i.

The second equality is exact for a differentiable path; four-node integration
is approximate and must be reported separately. It cannot be declared exact
because the identity exists on paper.

## Marked observation and reference

Define mu=sum_i A_i delta_(g_i). Its total mass and first moment are

    mass=sum_i A_i
    first=sum_i A_i g_i.

For tagged body atoms, center marks against the fixed physical endpoint
reference tau=nu_low,body+nu_high,body:

    g_bar=integral g d tau / integral d tau
    connected=sum_(body) A_i(g_i-g_bar).

HM121 uses original token masses1, so this is the arithmetic mean of the
original tagged endpoint atoms. If future serialization splits atoms, keep
their fractional reference masses; do not recompute a count mean of duplicate
storage rows. The reference is not an author-cohort normalization or a removal
of genuine condition/state/choice information.

Under g'=L g+b, first'=L first+b body_mass and connected'=L connected.
This law does not make arbitrary-metric cosine invariant. The frozen metric
and per-endpoint wrong-slot control remain explicit.

## Scientific interpretation

A nonzero action says the chosen instrument is sensitive to supported evidence.
Cross-expression correspondence tests whether that action has shared structure.
A correct GRAIN mark beating wrong-slot marks would add an organized interface
between the two project lines. It would not by itself name a personality trait,
prove author ownership, or replace external/behavioral interpretation.

These distinctions prevent numerical fidelity from masquerading as a mechanism.
If correspondence fails, retain the supported action but do not claim a positive
GRAIN-to-personality connection for this four-coordinate lexical representation.

## HM121 observed result and next mechanism

All144 pairs have support-representative midpoint difference0. Correct connected
marks exceed fixed misalignment in new-instance actor-odd weighted cosines by
.184040/.033839, but four-node completeness fraction.093231 exceeds frozen.05.
Localization remains unresolved; no quadrature refinement. Wrong-slot family
matching ranks25/5 beat correct42/57 despite smaller cosines. Total action alone
already matches strongly; first moments add little. Preserve this contrary detail.
Original full nonfinite failure is retained separately from safe log-mass repair.
HM122 therefore tests finite GRAIN-guided body-mass changes with correct/equal
budget shifted positions, not a more accurate integrated attribution.

## Prior art and limited inspiration

- Vuckovic et al. (2020), measure/Markov-kernel attention:
  https://arxiv.org/html/2007.02876
- Sundararajan et al. (2017), Integrated Gradients:
  https://proceedings.mlr.press/v70/sundararajan17a.html
- Zaher et al. (2024), manifold-aligned attribution paths:
  https://proceedings.mlr.press/v235/zaher24a.html
- OpenAI/math fixed matrix-jets construction:
  https://raw.githubusercontent.com/openai/math/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Determination-of-a-metric-and-a-unitary-connection-from-one-boundary-patch-October-5-2026/build/sections/06-matrix.tex

The last motivates separating admissible observation components. Its PDE
assumptions and inverse theorem have not been established for language.
Positive measure mixtures and adjoint derivatives are mature mathematics.
The project contribution must be judged through the concrete author-evidence
measurement interface and the mechanisms its controlled experiments distinguish.
