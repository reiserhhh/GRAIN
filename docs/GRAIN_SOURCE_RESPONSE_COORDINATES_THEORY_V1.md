# Response-informed coordinates connect GRAIN directions and query selection

2026-09-11. NEXT75 / finite pass64. This is standard projection and differential
calculus assembled for our same-material task, not a new generic theorem.

## 1. What is now a common mathematical object?

On exactly aligned token states H and fixed support M, source reference S gives
the query anchor q_S and a GRAIN direction D_f. The local reader is

    ell(H) = P_M ell_identity(H;q_S) + P_M Phi_S(H) B_Sq
    a(H) = softmax(ell(H)+M)
    F(H;v) = sum_h E_(a_h(H))[v_h]
    D_H F(H;v)[D_f] = sum_h Cov_(a_h(H))(v_h, D ell_h(H)[D_f]).

Here v_h is a projected native value, held fixed only for the key-channel partial
derivative. GRAIN supplies a comparison direction; the query supplies a reading
measure. They are not automatically two additive personality scores. Positive
attention alone is not a signed factor contribution: covariance also uses content
values, direction sensitivity, and their centering.

The user's proposal to inspect both systems on the same utterance is exactly the
address contract of this equation. A whole-author comparison can aggregate these
local responses, but cannot discard their signs or change token boundaries in
one system without transforming the other.

## 2. Why variance coordinates can miss a useful response

Write A for the source derivative design, Y for its residual target. Weighted
least squares can be reduced to ordinary least squares by square-root weights.
At the declared pseudoinverse tolerance:

    min_B ||A B-Y||_F^2 = ||(I-A A^+)Y||_F^2.
    G = A^T A, C = A^T Y, B_d = G^+ C.

This is a statement about the retained numerical range. NEXT75 computes about
13% represented energy for the old all4 design. The source scalar value loss at
B_d is31000-37000, versus about .58 at the regularized joint solution. A derivative
loss coefficient cannot manufacture missing range; within the range, scalar and
directional targets can prefer very different coefficients.

This motivates a change of input coordinates, not output derivative shrinkage.
With source-only standardization Z and its exact directional derivative DZ:

    C_response = [E Z^T Y_N/s_N, E_f DZ_f^T DY_Nf/t_N,
                  E Z^T Y_C/s_C, E_f DZ_f^T DY_Cf/t_C]
    C_response = U Sigma V^T
    psi(H) = (H-mu_S) diag(sigma_S)^(-1) U_32.

The basis is backscaled before reuse in the existing feature function. One source
value-plus-direction objective learns B; every derivative differentiates this
same scalar. Constants removed by P_M are softmax gauge freedoms, not discarded
author-choice signal. Source standardization is fixed, not recomputed on new8.

The new map lowers source value and direction residual losses. In previously
exposed new8, the native-query contrast RMS .121471 -> .077174 moves toward
.059927. But the source-query target is .017467; lower error against a different
target is not a certificate of correct intermediate recovery. This distinction
prevents composite error compensation being mistaken for identified psychology.

## 3. Mathematical precedent and the actual scope borrowed

[O'Leary-Roseberry et al.,2022](https://doi.org/10.1016/j.cma.2021.114199)
develop derivative-informed input projections and output projections for
high-dimensional parametric PDE maps. The
[full arXiv manuscript](https://arxiv.org/pdf/2011.15110) was read: its input
subspaces use Jacobian second moments, and its bounds require stated probability/
regularity assumptions. Their principle is that an input-variance projection and
an output-response projection need not coincide.

Our four directional observations and1536x48 cross-moment matrix are not a full
Jacobian active-subspace estimator. High retained cross-moment energy is not a
bound on prediction or derivative error. A
[2023 corrigendum](https://doi.org/10.1016/j.cma.2022.115673) exists; the publisher
full text was inaccessible in this pass. No result dependent on that correction
or any PDE error bound is imported here.

[Sobolev training (2017)](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html)
is precedent for fitting function values and derivatives together. NEXT75 adds a
task-specific source coordinate construction to that prior NEXT74 objective.
It is an original local implementation/combination, not an invention of derivative
supervision, SVD or least-squares projection.

## 4. The next dependency is the value, not another attention ranking

When a value function is also computed from H, the actual derivative is

    D F(H)[D_f] = sum_h {Cov_(a_h)(v_h,D ell_h[D_f])
                        + E_(a_h)[D v_h[D_f]]}.

The earlier key-only covariance is one partial channel. It cannot silently become
the complete derivative when native V is replaced by a learned V(H). NEXT76
therefore records both terms and their sum, with no forced nonnegative weighting
and no arbitrary deletion of a derivative term. Native fixed readout g and H16
are separately retained until a later source compilation establishes replacements.

This offers a concrete route for the user's interpretation idea: same-text
GRAIN position -> query-specific signed response -> independently coded behavior/
relationship event -> human-readable interpretation. Operationally defined
psychodynamic/TA codes may supply queries and external observations. They do not
turn model responses into unconscious motives or independent human labels.
See the [literature/interface addendum](GRAIN_RELATIONAL_INTERPRETATION_RESEARCH_ADDENDUM_V1.md).
