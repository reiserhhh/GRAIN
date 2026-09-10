# GRAIN reading functions: means, conditional strength and missing conditions

2026-09-10. NEXT25. This extends the conditional function/input theory without
changing GRAIN release scores or adding a named psychological dimension.

## 1. Different functions of the same conditional distribution

Let Z denote a centered candidate-logit interaction of the frozen observer,
X the declared text-vector input, and C the pair/context condition. For an actual
conditional law with finite second moments:

    mu(X,C)=E[Z|X,C]
    M(X,C)=E[Z Z.T|X,C]
    V(X,C)=M-mu*mu.T
    energy(X,C)=E[||Z||^2|X,C]=||mu||^2+trace(V).

Two opposed actions can have zero mean and nonzero strength. A mean-vector
reader alone cannot distinguish all such distributions. This is distinct from
the already proved linear-fit identity (A Z)v=A(Zv): a fixed scalar projection
of a linear vector fit is redundant, but taking a second moment is nonlinear.

For any finite fitted prediction m, a different identity holds:

    ||z||^2-||m||^2=||z-m||^2+2<m,z-m>.

Only suitable conditional expectation/orthogonality removes the cross term in
expectation. Our old OOF vectors have measured positive AND negative cross terms;
the difference of their energy and target energy is not automatically noise,
irreducible uncertainty or human personality variance.

## 2. Exact moments of a finite, approximate law

The new source-only kernel smoother defines explicit masses on training actions
z_i and a zero atom. Squared symmetric-pair kernel similarities are nonnegative;
fixed source-relative alpha supplies the zero mass. Then

    p_hat = pi_0 delta_0 + sum_i pi_i delta_z_i, sum(pi)=1
    mu_hat=sum_i pi_i z_i
    V_hat=sum_i pi_i (z_i-mu_hat)(z_i-mu_hat).T
          + pi_0 mu_hat mu_hat.T.

This covariance is PSD, and the first/second-moment identity is exact. These
facts certify arithmetic, not p_hat=P(Z|X,C). Kernel squaring loses some signed
similarity distinctions, few source authors limit support, and smoothing mixes
different authors. Reference variance must not be relabeled individual emotional
instability. The positive mixture is a strength candidate; its signed mean does
not replace the existing signed Ridge function.

The empirical result is instructive: distant-pair energy MSE improves relative
to the norm of a predicted mean, but not relative to a simple source pair-energy
mean. The extra moment mostly restores a missing group magnitude. The existing
adjacent signed reader already carries useful individual energy information.
See the [finite report](../reports/grain_vector_mechanisms/CONDITIONAL_ENERGY_REPORT_V1.md).

## 3. The role of conditions: deterministic observer, uncertain reduced input

For fixed weights, numerical protocol and complete state H, Z=F(H) is a
deterministic numerical object. In the ideal model Var(Z|H)=0. If X is only a
projection/aggregation of H, Var(Z|X) can be nonzero without any random biological
process: different complete states share the same reduced input. Finite empirical
mixtures also contain approximation error even if all true inputs were supplied.

For a genuine additional condition A, total conditional variance gives

    E trace Var(Z|X)
      = E trace Var(Z|X,A)
        + E ||E[Z|X,A]-E[Z|X]||^2.

Thus more information cannot increase the optimal expected squared-error risk,
but a particular refined subgroup can have greater variance than the coarse
group. Nor does every finite fitted model improve when given more variables.
The supplied counterexample tests both statements. This links GRAIN's condition-
relative measurement with the current missing-input problem: retain informative
conditions; do not assume that all remaining spread is noise to delete.

These are standard expectation identities, not proof that measured human
personality becomes deterministic or error-free at a social scale. Z here is
the model's numerical interaction, not a person's latent personality.

## 4. Nonlinear scoring does not automatically exploit the extra moment

For the alphabet score S and a declared additive logit context z_add:

    E S(z_add+Z) = S(z_add+mu)
                   + 1/2 trace(Hess_S(z_add+mu) V) + higher_order.

The leading correction can be computed without another model call. In this
finite panel it is only about1.1e-6 for adjacent and5.7e-8 for distant pairs,
versus actual interaction RMS6.79e-4/1.68e-4. We can accurately calculate the
correction, but it does not recover the main missing response. z_add itself is
still borrowed from current native endpoints. NEXT26 replaces that dependency,
rather than finding ever more precise corrections to the last softmax.

## 5. Literature alignment

[Song, Huang, Smola and Fukumizu (2009)](https://icml.cc/2009/papers/538.pdf)
embed conditional distributions so different expectations can be evaluated in a
common representation. Their operator estimator uses assumptions about the
underlying function spaces. We borrow the conditional-functional viewpoint, not
their inverse-covariance estimator or a universal distribution-identification claim.

[Park and Muandet (2020)](https://proceedings.neurips.cc/paper/2020/hash/f340f1b1f65b6df5b5e3f94d95b11daf-Abstract.html)
formalize conditional mean embeddings as conditional expectations and distinguish
that definition from stronger operator assumptions. Our finite positive mixture
has ordinary computable moments; consistency/calibration of this particular
small-panel smoother is not imported from their convergence results.
