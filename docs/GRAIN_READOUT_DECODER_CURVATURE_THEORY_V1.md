# HM39: Readout Composition and Cancellation-Sensitive Observation

2026-10-09 JST. Finite HM39 verified with the numerical-audit companion;
the original ratio-verification failure remains preserved. GoalACTIVE.

## 1. The measured object

Keep the original fixed Qwen reader, layer16 post-block leaf, property spanH,
early-query spanP, remaining bodyB and endpointQ. These are reader states, not
coordinates of a human psychological trait. For the same ten digit logits,

```text
F(s) = m(l(s)),  p_j = exp(l_j)/sum_k exp(l_k)
m(l) = sum_j a_j p_j,  a=(0,1/9,...,1)
g = p*(a-m)
C = diag(g) - p g^T - g p^T

v = d_P <d_H F,d>
  = J_P(l)^T C J_H(l)d + sum_j g_j d_P<d_H l_j,d>
  = v_link + v_decoder.
```

This exact second-order composition rule is established mathematics, not our
new theorem. Dangel/Harmeling/Hennig(2020), Eq.7, provides a modular derivation.
Their convex-loss/PSD approximations are not assumed for this signed scalar.
[Primary derivation](https://proceedings.mlr.press/v108/dangel20a/dangel20a.pdf).

C is symmetric and annihilates common-logit shifts. It can have both positive
and negative eigenvalues. The nine-dimensional shift quotient of ten logits
is a scoring-coordinate fact, not nine discovered personality dimensions.

## 2. Actual computation, not only an identity

All96old paths at t=.5 were observed:3queries, source/application,2wordings,
8H/B/Q corners and2original physical H directions. Source48nodes were saved
and hash-sealed before application48. No new predictor was fitted.

Each forward yields ten logit Jacobians with respect to H/P. The output g is
detached before differentiating sum_j g_j*l_j twice to obtain v_decoder.
Differentiating through g again would mix the output-link term into this
quantity. The analytic contraction computes v_link. Both are saved, together
with original physical displacements, logits, derivatives and audit addresses.

The original HM35 total covector is an independent cached replay reference.
The maximum component-sum error is6.030246461679808e-10; original score and
weighted H-gradient replay error are0. The original1e-8physical gate is intact.
A parent-minus-composed numerical remainder is stored, not called a mechanism.

The frozen source U_V is applied separately to each component. For H normn,
theta=(v^T U_V)/n and eta=n*(U_V^T DeltaP) preserve physical action. HM38's
pair means, contrasts, coupling and transport are reconstructed separately
for link, decoder and roundoff, with component source means summing to the
old full-law reference. All signed cross terms remain in the output tables.

## 3. The mechanism is context- and target-dependent

AI-trust source total-action energy is mainly decoder: decoderMS/totalMS
1.0589/1.0125, linkMS/totalMS.0493/.0100(two H arms). In application,
linkMS/totalMS becomes.8039/.6981, decoder.0334/.0484. The link-action RMS
increases21.67/29.35fold, while the decoder-action RMS changes much less.
These energies are not additive percentages; cross terms close their sum.

Application negative affect has strong compensation: link/decoder/cross
MS terms9.2199/7.9551/-16.1750 in one direction, and4.1608/2.6234/-5.7842
in the other. A small total does not mean weak internal component activity.

Application low-mood instance coefficient contrast has link/decoder cosine
-.99250/-.99759. Their individual RMS values exceed the resulting total;
keeping one and deleting the other would substantially change the observable.

At source, the total-active complement has only roughly1e-10action RMS, but
the two component complements each have1e-5scale and compensate. This is a
consequence worth following: source-visible total activity can hide component
directions that may cease to compensate elsewhere. HM40 is the next bounded
test of this possibility, not an already demonstrated transport solution.

## 4. A useful quotient-space formulation

Let T(v_L,v_D)=v_L+v_D. Its kernel contains every pair(u,-u). Observing only
T cannot distinguish different component pairs on the same fiber. For a
block-diagonal linear transport D=(G_L,G_D), an induced transport on totals
exists exactly when D preserves ker(T), equivalently G_Lu=G_Du on that kernel.
Otherwise a source-zero pair produces(G_L-G_D)u after transport.

This is standard factor-through/quotient reasoning. It does not establish that
our decoder has a single linear transport G_L/G_D or that all source-invisible
directions become informative. It specifies the mechanism question: which
joint observables must be retained for a total-response law to transport?

For GRAIN, the practical implication is to retain text evidence, measurement
unit and reader composition together; a common source average or total-active
projection is not automatically sufficient for an instance in another context.
The benefit is a more precise mechanism target, not another accuracy grid.

## 5. Geometry inspiration and its domain

OpenAI/math's fixed-commit RCD manuscript, calculus.tex and curvature-change.tex,
was read locally; the latter's198lines were reread in full. Its weighted
curvature result needs full-support RCD(K,N), bounded Lipschitz F, bounded
continuous G and a specified weak Hessian/test domain. The metric stays fixed
while the reference measure changes. This motivates distinguishing observation
operations from underlying geometry. It does not make our finite softmax
Hessian Ricci curvature, prove personality transport or verify the manuscript.
[Fixed source](https://github.com/openai/math/blob/adc7f1241b42e322a6451854ab7e4b4c146bf78a/preprints/Weak-Hessian-bounds-along-every-geodesic-in-RCD-spaces-September-24-2026/build/sections/curvature-change.tex).

## 6. Numerical and scientific scope

The first verifier failed on a near-zero-denominator ratio: absolute difference
3.814697265625e-6at a roughly1e10quotient. R2 retains every physical1e-8check
and audits only three dimensionless ratios against exact fractions of saved
binary floats and an explicit summation-roundoff bound.612ratio fields were
checked;126have unresolved denominators at the physical1e-8level and are not
interpreted as scientific percentages. The failed R1 remains unchanged.

R2 verified206outputs/10193ancestry-code hashes and173710284array/table values;
the number of comparisons is not sample size or independent scientific proof.
96forwards/1056first/192second-gradient calls cost258.94/900model seconds.
CPU work including the failed verifier and R2 costs38.72/120seconds.14new
tests;66mathfiles593tests pass. This is same-agent alternate numerical review.

Original18protocols/100cards/44computational requirements, natural author
response parameters and psychological meaning remain open. No new private
data/API/training/Claude/automation/commit/push/tag/lockbox/old-queue operation.
