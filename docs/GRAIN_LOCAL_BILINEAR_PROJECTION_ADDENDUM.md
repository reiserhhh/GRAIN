# Projection Control for Local Bilinear Construction

2026-09-13. Added during Essays encoding; neither domain has target-head model
artifacts yet. The original producer/plan remain unchanged and hash-bound.

The keyed semantic mean is a fixed linear projection of normalized unit-mean
E5: sbar=hbar D. Our caches are already L2-normalized up to float rounding.
Thus adding sbar to PCA64(hbar) can restore directions omitted by the PCA budget;
it does not create new information relative to full hbar. The covariance term
also has the identity Cov(g,s)=Cov(g,h)D. These are algebraic properties of the
declared representation, not a claim about the whole language-model mechanism.

Add one non-PCA E5/Ridge reference in both domains, source standardization only,
same training/validation/held authors and alpha1/10/100/1000. No new embeddings,
labels, hyperparameter grid extension or changes to the paired models.
Report its held metrics and paired-bootstrap1000 differences for paired and
additive relative to this reference. All five traits retained. This comparison
tests whether a co-location gain goes beyond that particular PCA compression.

Also numerically verify sbar=hbar D (within float normalization tolerance), and
report each domain's source-PCA retained directional energy ||P64 D||^2/||D||^2
per facet. Directional energy is geometric, not criterion importance. Do not
choose or reweight facets using it or the held results.

This addendum is independently executable after the frozen main comparison.
It does not restart the active encoder. The result is a reference/control,
not permission to extend a prediction-accuracy search indefinitely.
