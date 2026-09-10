# Context reading has both a selection rule and an output range

2026-09-11. NEXT79 finite result; [report](../reports/grain_vector_mechanisms/CONTEXT_DIFFERENTIAL_READER_REPORT_V1.md).

## 1. The tested function

Let c be the paired H16 last-state mean, and source i supply query readers r_iN,
r_iC. Write m_S=mean_i((r_iN+r_iC)/2), d_i=r_iN-r_iC. With the fixed source
Gaussian weights pi_i(c) from the [primary plan](GRAIN_CONTEXT_DIFFERENTIAL_READER_PLAN_V1.md):

    d_hat(c) = sum_i pi_i(c) d_i
    rhat_N = m_S+d_hat/2; rhat_C = m_S-d_hat/2
    Fhat_q(H_N,H_C) = rhat_q(c)^T phat_q(H_q).

This is normalized vector kernel regression. The unconditional case recovers
the old source mean readers. It does not estimate an operator-valued RKHS
regularizer or inherit a large-sample consistency guarantee from three sources.

For a context direction v:

    D pi_i[c][v] = pi_i (c_i-sum_j pi_j c_j)^T v / h_S^2
    D Fhat = rhat^T Dphat + phat^T Drhat[c][Dc].

The second term vanishes for the declared GRAIN BODY directions because their
last row is zero, not because context is detached. Synthetic nonzero-context
tests retain the product rule; 128 actual BODY finite differences pass. This
defines a scalar and its derivative, rather than an arbitrary patched response.

## 2. An elementary range obstruction

Set dbar=mean_i(d_i), V=span{d_i-dbar}, and P_V its orthogonal projector. For
any normalized weights (including signed ones), d_hat-dbar belongs to V. For
the native current differential reader d:

    ||d_hat-d||^2
      = ||d_hat-dbar-P_V(d-dbar)||^2
        + ||(I-P_V)(d-dbar)||^2.

The second summand is invariant to all such weight choices. This is ordinary
orthogonal projection/Pythagoras, not a newly discovered mathematical theorem.
The empirical result is that it occupies **78.60%** of current conditional
new8 vector error, and **58.83%** in source4LOO. The source range has rank3/2.

This proves a limitation of a finite output dictionary. It does not prove the
same percentage is missing from a scalar response: multiplication by the native
evidence vector induces a different, possibly singular metric, and cross terms
can cancel. We record that projected error separately. In fact the current
kernel improves vector error about1.09% in new8 while differential K+V projected
error worsens about6.38%. Improved representation norm is not improved use.

## 3. What this changes in the theory

NEXT78 separated the evidence difference from the reader difference. NEXT79
adds a distinction *within* a conditional reader: selecting among source
outputs and generating new output directions are not the same capability.
The source weights are near-uniform (effective3.955 of4), but sharper selection
cannot create absent directions. Context-dependent reading remains useful as
a target; this particular smoother is closed without promotion or a grid.

Instead learn a local source response operator J_i, so a reader can move:

    d_hat(c) = sum_i pi_i(c) [d_i + J_i(c-c_i)].

This expands the available output range to include images of J_i. The next
experiment obtains a single directional column per source/reference by paired
context perturbations, then applies one fixed field. Whether those columns
carry useful projected response is empirical, not guaranteed by the formula.
The source context summary might still omit relevant BODY information.

## 4. Relation to prior methods and system logic

Normalized smoothing follows the kernel-regression idea in
[Nadaraya (1964)](https://www.mathnet.ru/eng/tvp356). The proposed derivative
teaching is motivated by [Solak et al., NIPS 2002](https://proceedings.neurips.cc/paper/2002/hash/5b8e4fd39d9786228649a8a8bec4e008-Abstract.html):
their Gaussian-process framework combines function and derivative observations,
including local linearizations from perturbation data. Our next finite secant
field is not their GP posterior, uncertainty model, or consistency theorem.
Its scalar/response compatibility must follow from its own declared formula.

GRAIN supplies author-referenced, same-material directions; the query reader
specifies how such directions enter a model's inference component. Neither a
nonzero reader response nor this vector bound names a psychological construct.
The current useful development is a source-taught reading mechanism, while
H16 extraction and full-L assembly remain separate dependencies.

[NEXT80 source-reader motion plan](GRAIN_SOURCE_READER_MOTION_PLAN_V1.md).
