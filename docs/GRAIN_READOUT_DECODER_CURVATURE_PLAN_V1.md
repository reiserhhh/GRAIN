# HM39: Output-Link Versus Decoder Curvature

Status: planned after verified HM38; no new model calls executed by this plan.
Preserve all frozen HM35-HM38 code, inputs, results and failed historical runs.
This is a bounded mechanism follow-up on previously evaluated material, not
fresh independent text replication or source-trained cold prediction.

## Question and Mathematical Object

HM38 locates substantial instance-conditioned reading differences, even when
conditional pair means are accurate. Before another coefficient law is fitted,
determine whether these differences come from the output numerical scoring link
or from decoder geometry. The distinction can change the next compilation task.

At the same fixed upstream state s=(H,P,B,Q), write the ten selected digit logits
as l(s). Use exactly the existing alphabet0..9 with values0,1/9,...,1, and
the existing conditional softmax score F(s)=m(l(s)). Do not change the readout,
temperature, prompt, layer, normalization or token positions.

For one physical H direction d, the P covector is

    v = d_P<d_H F,d> = v_link + v_decoder,
    v_link = J_P(l)^T Hessian(m) J_H(l)d,
    v_decoder = sum_j (d_j m) d_P<d_H l_j,d>.

This is the standard second-order composition chain rule. For p=softmax(l),
g=p*(values-m), the output score Hessian is

    Hessian(m)=diag(g)-p g^T-g p^T.

It is symmetric and kills the common-logit shift, but is NOT generally PSD.
The score is not a cross-entropy training loss. Do not call this a PSD
Gauss-Newton approximation, replace negative curvature, or erase the decoder
term. The ten-logit space has a nine-dimensional shift quotient; that is not
a discovered factor count or a rank statement for the entire decoder.

## Fixed Finite Design and Budget

Use all96HM35 path addresses, but only the already declared midpoint node1,
t=.5:3queries x2roles x2wordings x8H/B/Q cells. Keep both original H arms.
Source48nodes must be saved and hash-sealed before application48nodes.
U_V remains the frozen per-query HM36 basis; no cross-query concatenation.
The midpoint design has no claim about the complete integral or whole path.

New calls are required because HM35 did not save separate digit-logit
Jacobians/curvature. Inherit its score, H gradient and actual P covectors as
replay references; do not recollect its complete288-node experiment.
Maximum96new local forwards;1056first-gradient calls(10digit Jacobians plus
one detached-link-weighted logit gradient per node);192second-gradient calls
(two physical H directions);900model seconds including failed attempts.
Preparation and numerical verification have a separate120CPU-second limit.
0fits,0API,0new neural training. Stop and save the current address if a cap,
unsupported derivative or numerical mismatch is reached; no automatic retry,
partial-as-full acceptance, model switch or budget expansion.

Only one first-derivative graph needs create_graph=True. The ten digit
Jacobian passes retain the forward graph but can use create_graph=False.
Then hold the actual output gradient g fixed as a detached numerical vector
while differentiating sum_j g_j*l_j twice. This computes v_decoder, not the
whole v; differentiating through g would erroneously include v_link again.
The full v replay reference is inherited from the identical HM35 midpoint.
Check actual implementation before freezing; journal attempted/completed calls.

Implementation clarification before acquisition: the actual probabilities and
output weights are float32. Comparing them to an independent float64 softmax
uses8float32-eps as an arithmetic audit, not as a relaxed mixed-action tolerance.
All original score/H-gradient/P-covector and HM38 scalar replay thresholds remain
1e-8. Save the exact parent-minus-composed covector as a numerical remainder,
not a new scientific component. The three signed components(sum of link,
decoder and numerical remainder) must reconstruct the parent at its own units.
The compensated sum of separate digit H-Jacobians is checked at8float32-eps
times the actual gradient maximum(or1e-8, whichever is larger), because that
changes floating-point reduction order; the actual weighted H-gradient replay
against HM35 remains at1e-8. Component source references are arithmetic means
of the two source wordings, solely for this diagnostic;0prediction-law fits.

## Outputs and Mechanism Tests

For every node keep ten logits, probabilities, actual output gradient, analytic
output Hessian, ten H-direction derivatives per arm, ten P-logit gradients,
decoder/link/total P covectors and the original full covector. Preserve source
and application, both physical arms and all addressed cells. Large arrays
may use checkedNPZ; do not return only the sum or a percentage.

Check v_link+v_decoder against the old actual v at the original1e-8absolute
tolerance. Also check original scalar score, weighted H gradient and patched
state/token addresses. Signed contraction with the original DeltaP must match
the parent midpoint scalar action. Independently rebuild softmax derivatives
and the link contraction; large-array verification uses scalar contractions
as well as geometry, not file existence. Keep relative errors only with an
explicit nonzero denominator, and retain cancellation.

There are192node-arm component triples. Project each component into frozen
U_V and retain its complement, with physical H normalization unchanged.
Across the two wordings, apply HM38's exact decomposition separately to link
and decoder coefficients. Their mean/contrast/coupling terms must sum to the
original supported terms, with all signed cross moments retained. Inherit
the original source full-law coefficient as a diagnostic reference, not as
a separate reference for each component; the component source mean naturally
splits that reference and must replay its sum exactly.

Tests before freezing: affine logit map has zero decoder curvature; linear
output score has zero link curvature; nonlinear logit interaction contributes
decoder curvature; score-link curvature may change sign; constant values give
zero derivatives; common-logit shift invariance; two opposing components
cancel; exact double-autograd composition agrees with the analytic sum;
detached versus non-detached link weights; the original query/token layout,
physical units and call caps. Use explicit small analytic functions/CPU toys,
not fabricated neural results. No full-model double-check calls beyond the cap.

## Decision Branches

If link curvature carries most coefficient contrast, the immediate missing
mechanism is a calibrated output-link/Jacobian composition, not a universal
personality axis. Study how source Jacobian information can be transported
without reusing application derivatives as prediction inputs.

If decoder curvature carries the contrast, continue to a bounded internal
attention/MLP composition analysis at this fixed layer and observable, not a
larger coefficient grid. If both matter or cancel, preserve both functions
and their signed composition. A numerical failure is an implementation/
backend issue, not evidence that either scientific mechanism is absent.

## Literature and Continuing Scope

The second-order modular chain rule is Equation(7) of Dangel, Harmeling and
Hennig(2020). This packet read the introduction, notation and main derivation
through the beginning of3.1/Table1(PDF pages1-5), not all9pages or supplements.
The planned mixed H/P blocks use the exact composition identity, not their
block-diagonal training approximation. Schraudolph(2002) was located, but its
PDF fetch failed; no full-paper reading or implementation is claimed.
[Primary derivation](https://proceedings.mlr.press/v108/dangel20a/dangel20a.pdf).

All original18protocols/100cards/44computational requirements, natural-author
response parameters and psychological interpretation remain open. Reader
curvature is not a person's psychological response. No privateMEPS/raw
questionnaires, API, Claude/automation, commit/push/tag/lockbox or old queue.
Goal remainsACTIVE. Freeze implementation, tests, design and spec before calls.
