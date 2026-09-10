# Complete conditional interface: information, reading and assembly

2026-09-10, nineteenth finite logic pass, NEXT26. Plan `bf21c94`, implementation
`6f09daf`, native evaluation after prediction freeze. This develops a complete
conditional subfunction, not a claim that the whole GRAIN/personality theory is done.

## 1. Three different meanings of complete

1. **Runtime complete:** numerical inputs and source assets suffice to return a
   score without opening a current author's native endpoint. Achieved and tested
   by a separate interpreter with all undeclared file reads denied.
2. **Information complete:** no target-relevant distinction has been collapsed.
   M/R/layout determine the mean-null state with the fixed PQ anchor, up to the
   original floating-point implementation. The finite Fourier contrasts for
   individual comments remain reductions, not invertible full token states.
3. **Functionally faithful:** the fitted numerical function preserves the native
   response across held authors. This specific complete candidate does not.

These are independent properties. Removing a decoder call is an engineering
achievement; it does not by itself show that the decoder's reading has transferred.

## 2. Fixed conditional target and source-only formula

For centered candidate logits and the original ten-point N answer alphabet:

    z_ij = z_0 + d_i + d_j + m_ij,
    d_i = z_i-z_0,
    m_ij = z_ij-z_i-z_j+z_0,
    S(z) = sum_a v_a exp(z_a) / sum_a exp(z_a).

The zero state collapses each of twelve comments to its own mean, retains all
template states and fixes P/Q to the inherited reference. Only comments i,j
are restored. This is not the untouched full author text or a human self-report.

Let c=(M,R,L). For each nonconstant block b, source-only normalization is

    psi_b(c) = (c_b - mean_source(c_b))/sqrt(mean_source ||c_b-mean_source||^2),
    k_C(c,c') = mean_active_b <psi_b(c),psi_b(c')>.

There are 18 source authors per fold. The base is vector Ridge with source target
mean as unpenalized intercept. Singleton increments share a product feature:

    k_d((c,phi),(c',phi')) = (1+k_C(c,c')) <phi,phi'>,
    feature = phi direct_sum (psi(c) tensor phi).

Thus zero contrast has zero singleton action exactly. The unchanged symmetric
pair kernel gives zero mixed action if either contrast is zero, and exact swap
symmetry. The source penalty is n*.01 times the training mean kernel diagonal.

This is a standard regularized finite kernel expansion, not a newly claimed
representer theorem. The source-span form and its representer-theorem foundation
follow [Scholkopf, Herbrich & Smola (2001)](https://alex.smola.org/papers/2001/SchHerSmo01.pdf).
The experimental contribution is the conditional input/endpoint decomposition
and the measured differences between reading local action and a complete score.

## 3. What this experiment discovered

- The runtime is genuinely endpoint-independent: four folds, twelve new small
  heads, original mixed heads untouched, exact numerical denied-file replay.
- With true additive endpoints, mixed-action assembly has score r about .9995
  or higher. With all endpoints predicted, pooled r is -.107 (bag) and .0006
  (position). This isolates the importance of the previously borrowed input;
  it does not erase the earlier measured local mixed-action relationship.
- Baseline reading itself loses individual variation: base r=-.164, within-fold
  r=.010, RMSE .01480 versus source-mean .01420. The native baseline SD is .01379.
  This loss appears even though the declared M/R/layout information is retained.
- Supplying the true mixed vector to predicted additive logits barely changes
  total performance. Further interaction or second-moment calibration is not
  the next high-value repair.
- Full logit MSE looks tiny relative to total candidate-logit energy (.006-.007),
  while score error is around all of the between-author score variance. A common
  candidate preference profile is not recovery of individual variation.

## 4. Error is not a bag of positive shares

Writing e=e0+ei+ej+em yields the exact sample identity

    E||e||^2 = sum_a E||ea||^2 + 2 sum_{a<b} E<ea,eb>.

Base error is the largest squared component, but several cross terms are negative.
For position/adjacent, base-j contributes about -.002445 to logit MSE. Replacing
one term by truth can remove compensating error and slightly worsen the whole.
That is why the distant true-mixed assembly is marginally worse than learned
mixed assembly; it is not evidence that a wrong interaction is scientifically better.

The next question is therefore **which directions a query reads**, not whether
all token dimensions are present somewhere in an input vector. Euclidean context
similarity is an empirical function-family choice. With only18 source authors it
can miss relevant direction/curvature, even if the target is deterministic given
the complete hidden input. This experiment does not distinguish lack of source
coverage from an unsuitable kernel, or identify a unique attention circuit.

## 5. Connection to GRAIN and interpretation

GRAIN supplies group-referenced expression coordinates on the same material;
query-conditioned sensitivity supplies the action assigned to directions there.
Prior same-material coupling and native differential results remain intact.
The new interface shows exactly where a usable local connection must become a
learned complete response. It neither changes the released GRAIN scorer nor fits
Big5/MBTI labels. Clinical/projective interpretation still uses the separate
evidence-bound renderer contract, not a diagnostic claim from this N model score.

Next: [query-directed baseline reading](GRAIN_QUERY_DIRECTED_BASELINE_PLAN_V1.md).
