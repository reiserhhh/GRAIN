# Midpoint Observation Increments and Non-Nested Observation Spaces

2026-10-08. HM19 finite mathematical companion, after verified HM18.
Canonical results: `results/grain_math_response_v1/midpoint_polarization_r1`;
alternate arithmetic: `midpoint_polarization_verification_r1`. Zero new
model calls or target fits. This is an observation/action result, not a
count of psychological traits or a new human prediction score.

## 1. The Same Coefficient Space, Different Observations

Use the inherited source units, with x=(t-r)/delta_scale,
c=(r-origin)/context_scale, kappa=delta_scale/context_scale. Then
m=c+kappa*x/2. In the symmetric matrix coefficient space define

\[
 U_i=\operatorname{sym}(x_ic_i^\top),\qquad
 V_i=\frac{\kappa}{2}x_ix_i^\top,\qquad W_i=U_i+V_i.
\]

For a fixed symmetric coefficient S the observations are Frobenius
contractions <S,U>, <S,V>, <S,W>. The skew dyad is unchanged because
skew(xxT)=0. Both U and W live in the same coefficient space; this does
not imply that the spans of their observation rows are nested.

Let L_U=span{U_i}, L_V=span{V_i}, L_W=span{W_i}. Exactly,

\[
 L_U+L_W=L_U+L_V.
\]

Thus a larger rank of W alone is a net difference, not necessarily an
append-only information gain. Keeping U and W separately can preserve
more than keeping only their sum observation W.

## 2. A Direct Geometric Residual, Not Response Regression

Flatten symmetric matrices with the Frobenius metric. Let rows of U,V
be the flattened observations. Their joint Gram is

\[
 G=\begin{pmatrix}G_{UU}&G_{UV}\\G_{UV}^{\top}&G_{VV}\end{pmatrix},
\quad G_{VV,ij}=\frac{\kappa^2}{4}(x_i^\top x_j)^2,
\quad G_{UV,ij}=\frac{\kappa}{2}(x_i^\top x_j)(c_i^\top x_j).
\]

With the orthogonal source row-span projector P_U=U^T(G_UU)^+U,

\[
 V_\perp=V(I-P_U),\qquad
 G_{V_\perp}=G_{VV}-G_{UV}^{\top}(G_{UU})^+G_{UV}.
\]

Proof: P_U is the symmetric idempotent projector onto the row span of U;
expand V(I-P_U)V^T. Its Gram is PSD. Therefore, at exact rank,
rank[U;V]=rank(U)+rank(V_perp). This is the generalized Schur complement
for a joint observation Gram, not a regression of held responses.

The generalized Schur complement is standard mathematics, not a new
GRAIN theorem. [Carlson, Haynsworth and Markham (1974)](https://doi.org/10.1137/0126013)
provide its pseudoinverse formulation; only their publisher abstract was
read. [Li and Mathias, section 2](https://cklixx.people.wm.edu/schur.pdf)
give extremal characterizations and proofs for a PSD block matrix. Here
the Gram interpretation additionally gives the elementary projection proof.

Numerically HM19 weights source rows by sqrt(w), uses only a fixed machine
null threshold100*eps*max(1,lambda_max)*n, and keeps signed eigenvalues.
There is no ridge, selected dimension or held target fit. Truncated nulls
make orthogonality approximate; its defect is recorded, not set to zero.

## 3. Actual Finite Geometry

| Source design | U rank | V rank | W rank | stacked U,V rank | V outside U rank |
|---|---:|---:|---:|---:|---:|
| All360 source states |182|222|264|396|214|
| Full-property64 source states |64|64|64|128|64|

Uniform and family-balanced measures agree on these numerical ranks.
The implied effective U/W overlap is50 for all states and0 for the
full-property subset, using the frozen numerical-rank convention. These
are finite observation-space dimensions, not exact population ranks.

Consequently HM18's net rankgain82 is more precisely214 newly covered
directions minus132 original directions not covered by W alone. A
full-property design has no net rank gain yet its two observation spaces
still differ. Equal ranks do not imply equal measurable content.

Explicit small-coordinate SVD, without Schur subtraction, reconstructs
the residual Gram within9.22e-13. The source numerical-null projection
has an orthogonality/range defect about2.11e-9 in raw Gram units; full-property
defect is below4e-16. All signed residual eigenvalues remain above the
negative machine threshold. We retain the compact source reconstruction
error and use ambient covectors for primitive action checks.

## 4. Which Frozen Queries Use the Increment?

For each separately fitted, unchanged S_q,

\[
 v_q=\langle S_q,V\rangle
   =\langle S_q,P_U V\rangle+\langle S_q,(I-P_U)V\rangle.
\]

The receiver symmetric coefficient is a linear combination of source U
observations, so its new action is structurally zero up to the recorded
numerical-null error. On application full-property material its RMS is
6.43e-15 to1.76e-14. Its normalized roundoff correlations carry no meaning.

The independently fitted endpoint coefficient lies in span W, not
necessarily span U. It uses the new component: application full-property
RMS3.107e-5/1.039e-5/3.786e-5 for NA/low/trust queries, respectively.
That is1.221%/.696%/.809% of each fixed prediction RMS. Receiver-order
reversal designs yield14.209%/10.988%/6.051%; these reverse token-state
order, not sentences or physical endpoint orientation.

V-supported and V-new action series can be negatively correlated even
though their operator-space components are orthogonal. Their scalar RMS
fractions can exceed one through cancellation; they are not fractions of
psychological variance. This extends HM18's observation-dependent geometry.

## 5. The Remaining Joint Response Is Not Disposable Noise

With native vector Y=P+R, exactly

\[
 \operatorname{Cov}(Y_a,Y_b)=\operatorname{Cov}(P_a,P_b)
 +\operatorname{Cov}(P_a,R_b)+\operatorname{Cov}(R_a,P_b)
 +\operatorname{Cov}(R_a,R_b).
\]

Application full-property NA/low native covariance is+8.2451e-7.
Receiver compilation gives-9.6159e-7; residual covariance+2.2861e-6
and the two negative cross terms produce the observed sign. Independent
endpoint compilation also retains this joint discrepancy. A useful
quadratic component is therefore not the entire missing joint mechanism.
Residual structure is retained as a target of further measurement.

## 6. Next Mathematical Decision

Keep separate observations/components rather than replace U by W. Next
test a finite endpoint-exchange orbit R:(x,c)->(-x,c+kappa*x), an involution.
It yields U^R=-U-2V, V^R=V, W^R=-W. The even/odd projectors on functions
over this balanced orbit isolate -V and W without a rank/accuracy search.

This planned design is inspired by finite-group averaging and reversible
numerical observation, not by treating token order reversal as time reversal.
Algebraic reverse rows are virtual: they add design possibilities, not
independent native responses. A reverse neural intervention requires its
own complete-context address and explicit scientific justification.

Original18protocols/100cards/44questions, natural-author inference, author
response B_u and psychological interpretation retain their independent
scope. This packet does not promote those links or erase earlier results.
