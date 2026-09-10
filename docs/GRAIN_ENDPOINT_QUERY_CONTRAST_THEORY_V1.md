# Query contrast is both a change in evidence and a change in reading

2026-09-11. NEXT78 / pass67. Bilinear decomposition, centered covariance and a
conditional vector-valued reading function. These algebraic identities are
standard; their observed roles in this GRAIN/query component are the result.

## 1. Symmetric two-query decomposition

For the same author/material, two frozen local readers and two output vectors:

    F_N-F_C = r_mean^T delta_p + delta_r^T p_mean = E+R.

Expanding the right side proves the equality; the cross terms cancel. It is
unchanged under a common orthogonal rotation of both vectors and readers, and
reverses sign when the queries are exchanged. Holding the reader equal forces
R=0; holding the output vectors equal forces E=0. All four properties are tested.

This is not a claim that the two paths are psychologically independent. The
native readers depend on the input, and the present target fixes them at each
evaluation point. The identity decomposes the declared component, not a new
derivative of the entire model. For the source-fixed scalar, both terms have
compatible GRAIN derivatives from the same functions; finite differences agree.

## 2. Means, author variation and opposing modulation

Write E=mean(E)+Ec, R=mean(R)+Rc, Cc=Ec+Rc. Then

    Var(C)=Var(E)+Var(R)+2Cov(E,R)
    Cov(E,C)/Var(C) + Cov(R,C)/Var(C) = 1.

For a fixed factor column the centering is over authors, not over factor labels.
Pooling these centered columns is descriptive multivariate energy accounting;
it does not multiply sample size or establish independent factors. Negative
projections mean opposition to the total, not negative information or noise.

In new8 native scalar contrast, differential reading's centered projection is
.740698. In native K+V contrast it is -.128743, with cross covariance -.005405.
The source-assembled function changes the latter projection to+.329503. Even
holding native Q/K/V, source-reader substitution changes it to+.122543. This
localizes a context-dependent reading mechanism not preserved by a constant
reader. Source4's directional projection is instead+.439777; keep that panel.

The same module may preserve ordering and lose conditional modulation, or fit
the mean without fitting author variation. This is why scalar accuracy alone
does not settle the GRAIN-to-query mechanism. Signed interactions should remain
in the function, not be discarded as measurement error by definition.

## 3. Reference dependence of local scalar interpretation

Under common translation p_q'=p_q+c, E'=E and R'=R+delta_r^T c. To preserve F,
the intercept must transform as b_q'=b_q-r_q^T c. This does not contradict common
rotation invariance; translation needs the affine term. The current local
functional fixes its origin and bias convention. It cannot silently acquire
the interpretive meaning of the complete endpoint L through recentering.

Across new8 all source readers are fixed. Across source4LOO each fold has a
different source reference. Report those panels separately rather than treating
their pooled rankings as one globally calibrated measurement population.

## 4. Develop a reading function, not one reading vector

The next object is rhat_q(c), where c is an available context summary, not an
author ID or an observed personality label. Keep the shared source reader m_S
and model only the source differential field d(c):

    rhat_N(c)=m_S+d(c)/2; rhat_C(c)=m_S-d(c)/2
    Fhat_q(H_N,H_C)=rhat_q(c)^T phat_q(H_q).

The full derivative has TWO terms:

    dFhat_q[D] = rhat_q(c)^T Dphat_q[D]
                + phat_q^T (Drhat_q(c)[Dc[D]]).

For the next controlled construction, c=(H_N,last+H_C,last)/2. Existing GRAIN
BODY directions are exactly zero at both final positions (`body_directions`
checks the boundary); hence Dc[D_BODY]=0. A context-conditioned reader can vary
between author materials while the same scalar's BODY derivative still uses
the familiar K/V term exactly. For a change of the last/context position the
second term must be included, not hidden with stop-gradient. A separate algebra
test demonstrates this difference.

This choice follows the earlier detected BODY-by-last interaction, but now
targets the endpoint reader rather than repeating a complete-score local-kernel
fit. It remains a two-query conditional program, not a query-independent scorer.

## 5. Literature connection and bounded implementation

[Nadaraya (1964), On Estimating Regression](https://www.mathnet.ru/eng/tvp356)
provides the normalized-kernel regression precedent. The next minimal field uses
positive source-context weights for vector-valued reader differences. Its
asymptotic results do not establish accuracy with only three or four sources;
the frozen experiment tests this specific finite construction.

[Kadri et al. (2016), Operator-valued Kernels for Learning from Functional
Response Data](https://jmlr.org/papers/volume17/11-315/11-315.pdf) develops learning
whose output is itself a function, extending vector-valued/RKHS methods. This
supports treating the reading operator as an object to learn, rather than only
regressing its final scalar. Our next normalized smoother is not their
operator-valued regularized solution and does not inherit its guarantees. That
distinction keeps the first implementation small without overclaiming novelty.

See [report](../reports/grain_vector_mechanisms/ENDPOINT_QUERY_CONTRAST_REPORT_V1.md)
and [NEXT79 fixed plan](GRAIN_CONTEXT_DIFFERENTIAL_READER_PLAN_V1.md).
