# Observable Action Geometry: Operators and Their Measurement Domain

2026-10-08. Mathematical companion to HM17; HM18 is planned, not executed.

## 1. The new distinction

HM17 reconstructs a frozen context law without refitting:

\[
L_q(x,c)=b_q^\top x+x^\top A_qc,\qquad
A_q=S_q+H_q,\quad S_q^\top=S_q,\ H_q^\top=-H_q.
\]

The decomposition is unique in the declared Euclidean coordinates, and
\(\langle S,H\rangle_F=0\). That statement is about matrices. It does not
make the two scalar actions independent on the measured text interventions.
Define the evaluation map and its design-dependent seminorm:

\[
\mathcal E_\nu(B)(x,c)=x^\top Bc,\qquad
\langle B,D\rangle_\nu
=\mathbb E_\nu[(x^\top Bc)(x^\top Dc)].
\]

Here nu is a declared distribution of addressed interventions, not a
population of people. Orthogonality in Frobenius geometry need not survive
this observation map. In fact HM17's application full-property S/H actions
correlate .99904/.99974/.99969 across the three queries. Both can be useful
parts of a score without being separately observed psychological mechanisms.

## 2. An exact illustrative counterexample

Let a and c be orthonormal, A=a c^T, and evaluate only x=a, context=c.
Then x^T A c=1, while x^T S c=x^T H c=1/2. Frobenius-orthogonal
matrices produce identical observations. Removing H halves the action;
the resulting prediction loss does not identify an independently varying
rotational mechanism. This is an exact algebraic example, not an assertion
that every real text lies in this degenerate case.

In the frozen data, write direct=x^T A c and transpose=x^T A^T c.
Then S_action=(direct+transpose)/2 and H_action=(direct-transpose)/2.
Application transpose/direct RMS is only 2.685%, 1.273%, 1.543% on full
property interventions. Near-collinearity is therefore mechanically
explained, rather than interpreted as two novel personality dimensions.

## 3. Finite movement versus scalar potential

With z=(h-mu)/s_c, x=(t-r)/s_d, kappa=s_d/s_c, z_t-z_r=kappa*x,
the symmetric potential of the same coefficients is

\[
\Phi_q(z)=\kappa^{-1}
\left(b_q^\top z+\tfrac12z^\top S_qz\right).
\]

Therefore

\[
L_q-(\Phi_q(z_t)-\Phi_q(z_r))
=x^\top H_qz_r-\tfrac\kappa2x^\top S_qx.
\]

Even H=0 leaves a finite-step error when the left endpoint approximates
the whole displacement. Conversely, deleting H from a fitted receiver
law and adding the step term does not reproduce a separately fitted
endpoint law. HM17 shows that the latter retains substantially better
calibration than this algebraic surgery. Model constraints and empirical
operator selection must not be conflated.

For shuffled contexts the start is a virtual shuffled coordinate and its
end is that coordinate+kappa*x. Its telescoping algebra is not an observed
physical or psychological cycle. At a complete common fixed reader state,
differences of a deterministic scalar endpoint function telescope by
definition. Missing coordinates or different reader contexts must be
specified before a loop defect can receive an empirical interpretation.

## 4. Finite source support and observation quotients

For source rows (x_i,c_i), the dyadic feature is D_i=x_i c_i^T. Its Gram is

\[
T_{ij}=\langle D_i,D_j\rangle_F
=(x_i^\top x_j)(c_i^\top c_j).
\]

The symmetric/skew Grams are

\[
T^S_{ij}=\tfrac12(T_{ij}+(x_i^\top c_j)(c_i^\top x_j)),\qquad
T^H_{ij}=\tfrac12(T_{ij}-(x_i^\top c_j)(c_i^\top x_j)).
\]

An operator direction in ker(E_source) has zero source action; source
data cannot choose among extensions in that kernel. Compression to the
common source vector support removes ambient directions unsupported by
the dyads, but does not remove this second observation ambiguity.
HM17 has 187 common source coordinates. An unconstrained operator has
187^2=34,969 coefficients there, but only 360 addressed source states per
query. These are capacity bounds, not personality dimension counts.

HM18 will evaluate this geometry with fixed family weights and machine
tolerances, without selecting a better model from held labels. All existing
linear/symmetric/skew/step terms and failures remain intact.

## 5. Contribution to the mainline

The text-to-trait route needs both vector action and its observation law.
GRAIN's contexts, identity bindings and evidence units specify where the
action is evaluated. A shared operator angle is not itself a shared
personality route; target action, source units and design support must be
carried along. This is a positive construction rule: compare functions
under a declared observation design before naming or combining their axes.

These are standard finite tensor, potential and seminorm arguments applied
to the project's frozen interventions, not a newly proved mathematics
theorem. The mechanism evidence concerns reader-score action; human
psychological interpretation remains a separate link in the full goal.

## Sources and actual use

- Jiang, Lim, Yao & Ye, *Statistical ranking and combinatorial Hodge theory*,
  [arXiv:0811.1067v2](https://arxiv.org/abs/0811.1067v2). Gradient/cycle and
  observation-graph distinctions motivate not equating algebraic symmetry
  with a measured closed loop. No native Hodge cycle experiment was run here.
- Schölkopf, Herbrich & Smola, *A Generalized Representer Theorem*,
  [author manuscript](https://alex.smola.org/papers/2001/SchHerSmo01.pdf).
  Finite source-span representation underpins implicit dyads; no theorem
  guarantees unseen psychological effects are determined by that span.
- [OpenAI/math](https://github.com/openai/math), fixed local commit
  adc7f1241b42e322a6451854ab7e4b4c146bf78a, metric/connection draft,
  sections/06-matrix.tex 515-678 read this turn. Uniform bounds before limits
  and operator-identification support are useful reasoning patterns. Its PDE
  hypotheses/uniqueness result are not transferred to text or personality.
