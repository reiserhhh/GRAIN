# NEXT25: conditional action mean versus conditional action energy

2026-09-10. Freeze before new cache statistics/construction. Current mainline:
real author material -> vector evidence -> usable personality-query mechanism.
This packet asks whether an additional second-moment readout is useful; it is
not another mean-prediction hyperparameter search or a new clinical trait.

## Fixed material and decision

Use the existing24 neuroticism source authors, four author-disjoint folds,
adjacent(0,1)/distant(0,11) comment pairs and both fixed Fourier bag/position
inputs. Targets are the previously measured centered mixed candidate logits
y=z_ij-z_i-z_j+z_0. Every pair and input family is retained. Original projection,
fold-local geometry, pair sizes and old readouts must replay before new scoring.
No new model calls, raw texts, human questionnaire reads or new participants.

First account for the old vector outputs m using the exact finite identity:

    ||y||^2-||m||^2 = ||y-m||^2 + 2 <m,y-m>.

The cross term need not vanish for a fitted OOF predictor. Do not call every
energy gap conditional variance. Report totals and within-fold correlations
separately; original negative distant-pair correlations remain recorded.

## One distribution-valued candidate

Reuse the existing symmetric pair kernel k and its training-only scale. Construct
a nonnegative kernel-weighted empirical action distribution. For each held fold,
input family and pair type, use only the18 training authors of that pair:

    h_i(x) = (k(x,x_i)/scale)^2
    alpha = .01 * mean_train(sum_j h_j(x_train))
    pi_i(x) = h_i(x)/(alpha+sum_j h_j(x)), pi_0=alpha/(alpha+sum_j h_j(x)).

The remaining mass pi_0 is a declared zero-action atom, preserving exact zero
when either comment contrast is zero. Positive masses give an actual finite
distribution, unlike unconstrained signed kernel Ridge weights. This is a
kernel smoother candidate, not a calibrated posterior or the exact conditional
embedding theorem. No bandwidth/penalty/kernel selection.

Compute from the SAME weights:

    mu=sum_i pi_i*y_i
    M=sum_i pi_i*y_i*y_i.T
    V=M-mu*mu.T
    e=trace(M)=||mu||^2+trace(V).

Retain aligned and orthogonal energies using the old fold-local pair direction.
V is PSD by construction including the zero atom; this algebra does not prove
it describes individual irreducible randomness. A squared similarity can erase
sign distinctions. The mean of this candidate is a diagnostic, not a proposed
replacement for the signed Ridge reader. The candidate target is action energy.

Compare e to ||mu||^2, the old Ridge norm-squared, the pair training mean-squared
and the pair training mean of squared norms. Report energy MSE, MAE, r, within-
fold r and paired author bootstrap intervals (2000 reps, seed2501). Finite pairs
stay attached to authors. Freeze new moments before held-target analysis.

For the original score link S, also compare S(z_add+mu) with
sum_i pi_i*S(z_add+y_i)+pi_0*S(z_add), including a second-order covariance
approximation. Label z_add oracle-dependent. This tests the actual consequence
of discarding the second moment, not a new fully independent score or human
accuracy. Preserve that the old measured output-link curvature was small.

## Tests, outputs and next branch

- Synthetic opposite-action mixture: zero mean, nonzero energy; mean/second-
  moment identity; arbitrary-predictor cross term; PSD and zero/swap properties.
- Score Hessian analytic/finite differences; mixture/Jensen approximation;
  train-only reference usage, unique author/fold pairing and exact old replay.
-16 finite smoother constructions,0 optimization/model training,0 neural calls.
  Publish aggregates only; private matrices/IDs/targets remain ignored.
- If energy transport improves, retain a separate strength coordinate, not a
  rescaled signed prediction. If it fails, preserve the mathematical distinction
  and do not add another energy grid. Either way proceed to NEXT26: explicitly
  choose the target and inputs for removing a currently borrowed native endpoint.
- Do not combine old E5 and native block16 results into a fictional runtime.

## Primary sources

[Song et al. (2009)](https://icml.cc/2009/papers/538.pdf) represents conditional
distributions by embeddings and supports computing different functionals from
one conditional representation. We use a simpler positive finite mixture, not
their inverse covariance estimator or a characteristic-kernel claim.
[Park and Muandet (2020)](https://proceedings.neurips.cc/paper/2020/hash/f340f1b1f65b6df5b5e3f94d95b11daf-Abstract.html)
distinguishes conditional-expectation definitions from strong operator assumptions.
The moment identities here are exact for our finite mixture; population accuracy
is an empirical question on the inherited source panel.
