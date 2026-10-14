# Source-Supported Query Response: Mathematical Object

2026-10-09. HM34 constructive extension of HM33, not a new psychometric scale
or a claim of a universal personality decoder. The primary protocol and its
results are frozen separately. This document explains the object and its domain.

## Observation and Response Are Different

For a fixed decoder layout, H is the four property-token states, P is the
earlier query suffix, Q is the endpoint, and B contains the remaining body,
system and layout. Let F_q(H,B,P,Q) be the actual scalar target readout and
alpha_q=d_H F_q its property covector. P already depends on the preceding
text: identical query tokens do not imply identical query hidden states.

The paired response has the original physical units:

    d_c=(vec(P0)-vec(P1))/2,
    g_c=mean_(H,B,Q) [(alpha_q(H,B,P0,Q)-alpha_q(H,B,P1,Q))/2].

This is a finite response of a covector to a query-state displacement. It is
neither an author's personality score nor an attention probability. Averaging
over H/B/Q defines a particular equally weighted sixteen-corner estimand; it
does not remove the interactions from the original model.

## Identified Part and Unidentified Part

Source matrices D and G have8 columns per query. The fixed-metric extension is

    L0=G(D^T D)^+D^T,
    Pi_D=D(D^T D)^+D^T,
    d=Pi_D d+(I-Pi_D)d.

On the retained numerical source support, this is the minimum-norm
least-squares extension. If G has inconsistent values on the source nullspace,
the interpolation residual is a real failed linear constraint, not silently
fixed by selecting another rank. For compatible G, all extensions with the
same source observations have the form

    L=L0+N(I-Pi_D).

They agree on D and can disagree on a cold unsupported component. The default
extension assigns that component zero computational contribution; it does
not assert that its true response is zero. The observed unsupported norm is
therefore part of the result, not a denoising target.

The Gram threshold is fixed before results at
100*float64_eps*8*lambda_max(D^T D). Primal SVD uses its square root. This is a
numerical-support convention, not the discovery of8 psychological dimensions.
Changing physical coordinates without transporting the input metric can
change the minimum-norm extension; the protocol uses the original metric.

## What the Operator Must Explain

For a fixed H/B/Q, let g_fixed be its actual P half-difference and delta_H
the exact realized float32 property half-displacement. For native finite
action contrast a, the complete signed error has three terms:

    <L0 d,delta_H>-a
      = <L0 d-g_common,delta_H>
        + <g_common-g_fixed,delta_H>
        + (<g_fixed,delta_H>-a).

These are source-operator error, conditional interaction, and local
derivative-to-finite-action error. Their squared errors include cross terms;
adding the three MSEs alone is wrong. Cancellation must be retained.

The pure P-gradient coefficient is not automatically the pure P raw-action
coefficient: H-specific physical displacements differ. The runner therefore
scores all768 addressed finite-action contrasts, not only the48 average
gradient coefficients. A local gradient read from each evaluated decoder is
a reference, not a cold source predictor.

## Higher-Mathematical Inspiration and Its Boundary

OpenAI/math, fixed commit adc7f1241b42e322a6451854ab7e4b4c146bf78a:
the noncommutative rational-formula manuscript's rank section separates exact
observation access and its costs from universal rank detection. Here the
relevant lesson is to state what the observation family identifies. Its
free-skew-field theorem is not applied to floating transformer states, and
its generator proof has not been verified in this experiment.

[Rank section](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Polynomial-Hitting-Lists-for-Noncommutative-Rational-Formulas-September-24-2026/build/sections/rank.tex)

The formulas above are standard finite-dimensional linear identification.
The research contribution, if supported, is the located text-influenced
query carrier and its target-specific action transport, not inventing the
pseudoinverse or borrowing a distant theorem as psychometric proof.

## Integration With GRAIN

GRAIN still determines author/evidence/condition/observation units. The
operator explains how one target-conditioned state can change the effect
of the same evidence coordinates. It must not turn another person's quoted
experience into the author's own evidence, discard context selection, or
rename these query responses as trait factors.

States are still extracted with the frozen encoder/model prefix. Removing
downstream decoder execution for one response computation is not yet removing
all LLM dependence or creating an absolute author score. B, interactions,
event binding, full-target construction and natural author response functions
remain explicit parts of the mainline, rather than being redefined as noise.
