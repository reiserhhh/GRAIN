# GRAIN conditional reading: output links versus internal curvature

2026-09-10. NEXT24 connects the conditional-input account to a nonlinear
potential. It does not change the GRAIN release or name psychological factors.

## 1. Exact scalar link account

For the original binary T/F candidate score p=F(M,R;x0,L), p is strictly between
zero and one on all saved paths. Set R(t)=R0+t*d, u(t)=logit(p(t)). Then

    grad_R u = grad_R p / (p*(1-p))
    p'(0) = p0*(1-p0)*b,  b=<grad_R u(0),d>, a=u(0).
    p1-p0-p'(0)
      = [sigmoid(a+b)-sigmoid(a)-p'(0)]
        + [p1-sigmoid(a+b)].

This is an identity for the given initial tangent, not a causal or statistical
variance decomposition. The first bracket is output-link curvature. The second
is the finite remainder after replacing the internal logit by its initial affine
approximation. If u is affine on the path, the second bracket is exactly zero.
It can also be zero by endpoint cancellation; inspect saved interior gradients.

When twice differentiable,

    u(1)-u(0)-u'(0) = integral_0^1 (1-t) d.T Hess_u(R0+t*d) d dt.
    Hess_p = p*(1-p)*Hess_u
             + p*(1-p)*(1-2*p)*grad_u*grad_u.T.

The measured logit remainder isolates curvature not explained by the final
sigmoid. It does not identify which layer, attention head, RMSNorm, or MLP
causes it. The Hessian formula is analytic; no full Hessian was collected.
For other nonbinary query alphabets, do not apply this binary inversion blindly.

## 2. A finite, integrable approximation

At retained role slot r, define e_r=R0_r/||R0_r||, P_r=e_r e_r.T. A candidate
potential with one source mean gradient vbar per body context is

    uhat(d)-uhat(0) = <vbar,d> + 1/2 sum_r d_r.T K_r d_r,
    K_r=kpar_r*P_r+kperp_r*(I-P_r).

Its gradient is vbar+K*d and its Hessian is symmetric, so integration and point
scoring are consistent. No independent vector-field fit that violates this
potential is silently substituted. For each reference path and nonzero saved t,
regress v(t)-v(0) on xpar=t*P*d and xperp=t*(I-P)*d. Orthogonality separates the
two coefficients:

    k_j = sum_i w_i <x_ij,delta_v_i> / [(1+.01)*sum_i w_i ||x_ij||^2].

Each path has equal mass; nodes have equal mass within a path. Zero design
energy gives coefficient0, explicitly an unobserved direction rather than proof
that native curvature is zero. The source reference frame is shared by the24
layouts; four paths remain four directional supports despite208896 coordinates.

This model excludes cross-slot blocks and anisotropic rotations within the
tangent subspace. A two-dimensional witness H=[[0,1],[1,0]] with balanced
directions (+/-1,+/-1) gives nonzero +/-1 quadratic actions, but a pooled
radial/tangential diagonal fit gives all-zero coefficients. Thus failure of
this candidate does not reject quadratic response or prove a particular network
mechanism. Likewise, gradient error orthogonal to d carries no action along d;
both field and actual-action errors are needed.

## 3. Relation to earlier GRAIN/path formulas

- Same-material native T_qk=<J_q,D_k> is a local derivative in body directions.
  This packet instead studies finite R changes while body/PQ are conditioned.
  The two are compatible parts of the full conditional function, not duplicates.
- A constant J is the affine special case; a changing J is naturally represented
  by a state-dependent field or potential. This does not erase the exact Cw
  identity for the earlier fixed E5 affine comment reader.
- To convert delta_u to probability, an origin is required. The source-origin
  approximation is a runnable function of cached R plus reference constants.
  Actual b00/b10 origins are oracle-conditioned diagnostics. Neither removes
  the current block16 input encoder or computes all terms of the full F identity.

The empirical outcome and next decision are in the
[seventeenth-pass report](../reports/grain_vector_mechanisms/RETAINED_LOGIT_CURVATURE_REPORT_V1.md).

## 4. Primary precedents and what is borrowed

[Zhang and Sennrich (2019), RMSNorm](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
normalizes by root mean square without centering. For ideal x/||x|| the Jacobian
separates radial and tangential directions. Learned coordinate gains, epsilon,
residuals and attention mean the complete decoder need not obey that simple
geometry. Our blocks are an inspired candidate, not a derived decoder theorem.

[Fuhrlaender and Schoeps (2023)](https://link.springer.com/article/10.1007/s11081-023-09795-y)
use available derivative observations in least-squares quadratic optimization
models. We borrow derivative-enhanced surrogate construction, not their BOBYQA
optimizer, trust-region machinery or convergence result.

[Bouhlel and Martins](https://arxiv.org/abs/1708.02663) reduce high-dimensional
gradient-enhanced kriging through partial least squares. This supports explicit
complexity reduction as a design consideration; our experiment is not kriging
and does not use their inference formulas or hyperparameter search.

These are standard chain-rule/Taylor/Hermite-surrogate principles with a project-
specific source geometry and comparison design. No mathematical-priority claim.
