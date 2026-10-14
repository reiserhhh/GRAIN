# Ordered Observation and Conditional Response

2026-10-08. Mechanism development after HM14; not a finished psychological
instrument. Standard interpolation and Hilbert-space arguments are not
claimed as newly discovered mathematical theorems.

## 1. The empirical advance

HM13 showed that an untagged token multiset cannot preserve all tested
reader actions. HM14 now gives a constructive partial remedy: fixed
position-weighted observations, source-only fitted action, no new forwards.
Application full-property action r increases from .230 to .436 for NA,
.150 to .344 for dejection and .381 to .538 for trust. These are action
replay correlations, not psychological correlations.

The ordered observation is useful, but response law and observation
sufficiency are different questions. Complete position moments in rank2
do not restore the 1534-dimensional hidden complement, and a single
query coefficient need not describe every receiver context.

## 2. Four distinct mathematical objects

1. **Typed observation:** token states H, positions/edges E and event
   identity/mode/actor from the evidence ledger. E is not inferred from
   a large trait score, and movement of H with E fixed is not relabeling.
2. **Vector observation:** O_U(H,E), with an explicit retained subspace U
   and complement. Invertibility along position does not imply full hidden
   information was preserved.
3. **Response:** F_q(H;c) or a compiled source law. The query and receiver
   state determine how the observation is used; scalar action is measured
   at actual addressed endpoints, not guessed from vector distance.
4. **Author measurement:** qualified events, history, opportunity and
   reference population aggregate these observations. Reader sensitivity
   is not automatically a stable author coefficient or a personality name.

This preserves GRAIN's psychometric spine while developing a numerical
route through token states. It does not replace author qualification with
a universal semantic similarity score.

## 3. Observation closure and response closure

For fixed T<=4 and distinct p_t, m_k=T^{-1}sum p_t^k h_t, k=0..T-1,
is invertible along token position. With U^TU=I,

\[
H=HUU^T+H(I-UU^T).
\]

The two terms are orthogonal in input space, not independent or additive
under the nonlinear reader. HM14's unchanging span means under reversal
therefore do not make the native reversal action vanish.

If F_q is differentiable on a stated receiver neighborhood, then

\[
F_q(H+D;c)-F_q(H;c)
=\int_0^1 DF_q(H+sD;c)[D]\,ds.
\]

The derivative is a covector depending on the state along the path.
A source-fixed beta_q is a useful empirical linear approximation, not
an identity for that varying derivative. A local expansion may later
separate a context-conditioned first-order law, finite curvature and
unobserved directions. No remainder bound is asserted without its actual
smoothness/neighborhood conditions.

## 4. Retain full vector space without an enormous inverse

Let d_i be the declared ordered observation difference of a source
intervention, y_i its actual native action, and w_i its family weight.
Use one source-only scalar s^2=sum w_i||d_i||^2 / sum w_i and x_i=d_i/s.
This normalization preserves orthogonal hidden-coordinate changes;
coordinatewise standardization would in general change that geometry.

For fixed alpha>0 consider

\[
\min_b \sum_i w_i(y_i-\langle b,x_i\rangle)^2+\alpha\|b\|^2.
\]

Decompose b into the span of source x_i and its orthogonal complement.
The complement changes no source prediction and only increases the
penalty, hence the optimizer lies in the source span. This does not say
unobserved directions are intrinsically meaningless; it describes a
particular source-supported, minimum-norm estimator.

With K_ij=<x_i,x_j>, W=diag(w_i),

\[
a=(W^{1/2}KW^{1/2}+\alpha I)^{-1}W^{1/2}y,
\quad
\hat y(d)=k(d)^TW^{1/2}a.
\]

The system size is the number of source interventions, not the number
of vector coordinates. Identical-state displacement gives exactly zero,
with no action intercept. This is a standard representer/dual-Ridge
construction, not a claim that kernel mathematics recovers all personality.

## 5. Next discriminating experiment

Compare the same fixed ordered observation in the original rank2 and
untruncated hidden spaces. Keep position degrees, source roles, alpha and
queries fixed. Include existing projection/complement endpoint actions,
not merely natural levels. Reconstruct float32 actual interventions
exactly; ideal orthogonality and rounded near-collisions remain separate.

If full vectors retain substantially more action, the old vector frame
was an observation bottleneck. If both fail similarly, investigate
receiver-conditioned response/nonlinear action with targeted interventions
instead of increasing rank/degree/alpha until a favorable table appears.
Both routes leave psychological identification as a separate author-level
question, with the original research scope intact.

## 6. Reading scope and evidence

HM14 primary `ordered_moment_action_r1`; alternate arithmetic
`ordered_moment_verification_r1`, under `results/grain_math_response_v1/`.
1404 interventions,5616 predictions,2556 checked metric rows;0 new forwards.

[Schölkopf–Herbrich–Smola 2001](https://alex.smola.org/papers/2001/SchHerSmo01.pdf),
section2 and its orthogonal-decomposition proof were read. The source-span
argument above is its finite linear special case. Not all11 pages were
read and no general RKHS transport result is imported.

OpenAI/math pinned `adc7f1241b42e322a6451854ab7e4b4c146bf78a`, metric/connection
`06-matrix.tex`170–340: weighted observations, Taylor remainders and order
of matrix products. Used as a design principle only; no PDE theorem is
asserted for a decoder. Formula sufficiency, reader action and psychological
measurement remain distinct deliverables.
