# GRAIN V6 Ordered-Transition Stage J3: Protocol

## Why J3 exists

J1 showed that an author's natural selection, expression, and pair marginals
are technically reproducible. It did **not** show that temporal ordering adds
information: a pair mean can stay stable even after order is erased. J3 asks
the narrower mathematical question that remains.

## Object

For a frozen event map `phi(Z)` and an author `u` in technical view `v`, define
the centred ordered-transition operator:

\[
C_{u,v}=\frac{1}{m}\sum_{t=1}^{m}\phi(Z_t)\phi(Z_{t+1})^\top
- \bar\phi^-_{u,v}\bar\phi^{+\top}_{u,v}.
\]

Unlike a first-order pair mean, `C` removes the source and successor marginal
means. The primary J3 operator further standardizes each entry by its source
and successor marginal standard deviations,
\(R_{jk}=C_{jk}/(s^-_j s^+_k)\), so a person's overall path amplitude cannot
masquerade as an author-specific ordering law. `R` is zero in expectation when
paired source and successor features are independent under the specified
sampling process. J3 uses a fixed random
Johnson-Lindenstrauss sketch of text plus the observed selection field only to
make this finite operator computationally manageable; it does not learn a
factor or a target prediction head.

## Technical replication and null

Each author has two disjoint sets of evenly spaced, chronological three-event
blocks. Real `C` uses their observed within-block order. The null averages
nonidentity permutations **inside each same block**, preserving author,
selection/text event multiset, block membership, and observation count while
destroying only local order. The primary contrast is:

\[
\Delta = \mathrm{AUC}_{\text{real }C} - \mathrm{AUC}_{\text{block-order null }C}.
\]

The per-author matched-versus-stranger margin difference is bootstrapped and
tested with a paired sign-flip randomization. Neither test is a personality or
causal claim.

## Frozen gates

Before PANDORA endpoint extraction, synthetic null, weak, and moderate
autoregressive worlds choose the smallest per-view event count. The calibration
does not use an arbitrary absolute delta: its moderate-effect lower 5th
percentile must exceed the null upper 95th percentile, and weak-effect median
must exceed the null median. This ensures that a realistic variance-heterogeneous
null cannot be accepted merely because the operator is noisy. On confirmation
authors, both independent word and character representations must meet the
frozen real-AUC, delta-AUC, and sign-flip conditions. A failure is evidence that
this particular finite ordering estimator is not supported; it is not evidence
that people lack dynamics.

## Nonclaims

Passing J3 does not identify a latent personality axis, external criterion,
clinical change, causal sequence effect, cross-language invariant, or a
person-specific transition law. It only supports a reproducible, centred local
ordering operator for the frozen representation and corpus.
