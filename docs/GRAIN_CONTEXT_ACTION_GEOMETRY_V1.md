# Context Action Geometry: Covectors, Potentials and Finite Steps

2026-10-08. Mathematical companion to verified HM16. This is a standard
finite Hilbert/tensor construction applied to controlled reader responses,
not a new theorem about personality. Do not change frozen packet code.

## 1. Observations and source laws

Keep selected prefix spans, token positions, author/event qualifications and
query readout units separate. Let r,t be full ordered observations. Source
only coordinates are x=(t-r)/s_d, c=(r-mu)/s_c and z_h=(h-mu)/s_c.
Set kappa=s_d/s_c, so z_t-z_r=kappa*x. The midpoint is
m=(z_r+z_t)/2. Coordinates refer to the saved observation, not the entire
decoder context. Question states were retained at the original intervention.

The receiver tensor feature is x direct-sum (c tensor x); its Gram is

\[
k_R=\langle x,a\rangle+\langle x,a\rangle\langle c,b\rangle.
\]

The midpoint potential feature is x direct-sum sym(m tensor x), with
sym(A)=(A+A^T)/2. Its Gram is

\[
k_E=\langle x,a\rangle+
\frac{\langle x,a\rangle\langle m,b\rangle+
\langle m,a\rangle\langle x,b\rangle}{2}.
\]

PSD follows from explicit tensor inner products, including on unseen
observations. Orthogonal changes applied to both coordinates preserve
these inner products. Separate standardization of arbitrary coordinates
would define another metric, not automatically this invariant law.

For each kernel, freeze its source weighted mean diagonal nu and solve

\[
u_q=(W^{1/2}(K/\nu)W^{1/2}+I)^{-1}W^{1/2}y_q,
\quad a_q=W^{1/2}u_q/\nu.
\]

The prediction is the raw source kernel row times a_q. It has no action
intercept and gives exactly zero for zero displacement. The raw source
scales differ by model; no claim of identical raw regularization is made.
Source row span and minimum norm do not identify unobserved truth outside
the source design. Finite representation follows the standard
[representer result](https://alex.smola.org/papers/2001/SchHerSmo01.pdf),
not a claim that source-support complements are psychological noise.

## 2. A receiver-dependent covector is an implicit low-rank operator

Write the already fitted receiver law, with no additional fit, as

\[
L_q(x,c)=b_q^Tx+x^TA_qc,
\quad b_q=\sum_j a_{qj}x_j,
\quad A_q=\sum_j a_{qj}x_jc_j^T.
\]

The context-dependent covector acting on x is b_q+A_q*c. A_q is represented
by at most 360 source dyads; no 18432-by-18432 array is necessary. Its
existence here is a reader-response model, not an author's personal B_u.

Decompose uniquely in the declared common Euclidean coordinates:

\[
S_q=(A_q+A_q^T)/2,\quad H_q=(A_q-A_q^T)/2.
\]

S_q is self-adjoint and H_q is skew-adjoint. This decomposition uses a stated
metric and source coordinates. It does not label either part signal/noise.

For the continuous one-form kappa^{-1}(b_q+A_q*z)^T dz on this coordinate
space, symmetry of A_q is the condition for a quadratic potential.
However, **a finite receiver-only edge is not an exact line integral**.
Even if H_q=0, a left-endpoint approximation omits curvature.

Define the symmetric potential from the same receiver coefficients:

\[
\Phi_q^R(z)=\kappa^{-1}
 (b_q^Tz+\tfrac12 z^TS_qz).
\]

Then the exact algebraic discrepancy is

\[
L_q(x,z_r)-[\Phi_q^R(z_t)-\Phi_q^R(z_r)]
=x^TH_qz_r-\tfrac\kappa2x^TS_qx.
\]

The first term is the skew component of the fitted operator; the second is
a finite-step correction. They cannot be conflated with a genuine native
cycle, missing decoder context, or psychological irreducibility.

## 3. The separately fitted endpoint potential

For midpoint source rows, define

\[
b_q^E=\sum_j a_{qj}^Ex_j,\quad
S_q^E=\sum_j a_{qj}^E\operatorname{sym}(x_jm_j^T).
\]

The fitted endpoint prediction is x^T*b_q^E+x^T*S_q^E*m, equal to

\[
\Phi_q^E(z_t)-\Phi_q^E(z_r),\quad
\Phi_q^E(z)=\kappa^{-1}
((b_q^E)^Tz+\tfrac12 z^TS_q^Ez).
\]

Proof: for symmetric S, z_t^TSz_t-z_r^TSz_r equals
(z_t-z_r)^TS(z_t+z_r)=2*kappa*x^T*S*m. This also proves reversal
and telescoping for any three observations in the frozen coordinates.
Phi^R and Phi^E are different functions; do not silently identify them.

HM16 checked this scalar evaluation against all saved kernel predictions:
maximum difference 1.579e-16. This checks the **model algebra**. It has not
measured three native edges at common complete context.

## 4. Source-operator geometry without allocating the ambient matrix

Let X,C be source row matrices and let D_q=diag(a_q). Then
A_q=X^T*D_q*C. Source Gram products give

\[
\|A_q\|_F^2=a_q^T[(XX^T)\odot(CC^T)]a_q,
\]

\[
\operatorname{tr}(A_q^2)=
a_q^T[(CX^T)\odot(CX^T)^T]a_q.
\]

Consequently ||S_q||_F^2=(||A_q||_F^2+tr(A_q^2))/2 and
||H_q||_F^2=(||A_q||_F^2-tr(A_q^2))/2. Cross-query inner products
use a_q and a_p in the same forms. Norm energy is not a proportion of
psychological information. Held-out directional contributions, rather than
operator norm alone, decide whether a component explains observed actions.

## 5. Graph tools require the right vertices

[Jiang et al., Statistical ranking and combinatorial Hodge theory](https://arxiv.org/html/0811.1067)
Sections 2--5 distinguish endpoint gradients from local and global cyclic
edge components. Their weighted decomposition provides a useful diagnostic
after vertices and observed edges have been defined.

For GRAIN, each vertex must preserve the complete intervention context,
not merely a compressed span vector. A star has no observed cycles; filling
missing edges or merging different query contexts can manufacture curl.
F_q(h_j)-F_q(h_i) at fixed complete context telescopes automatically.
Nonzero cycles after projection diagnose the quotient/conditioning or
approximation, not necessarily a nonconservative psychological mechanism.
Hodge projection by itself is therefore not the next empirical claim.

## 6. Relation to GRAIN measurement

Keep the chain: typed evidence -> ordered vector observations ->
source-calibrated target action -> qualified author/event accounts ->
condition/reference/measurement objects. HM16 strengthens the action stage.
It does not replace qualification by a high score, turn model effects into
personal response parameters, or transfer old external validity to new axes.

HM17 will analyze the already frozen operators and native action failures,
including shared and target-specific structure. No new precision grid,
source refit or native forwards are required for that finite mechanism.
