# HM151: conditional response relations, not interpolation accuracy

2026-10-11. Result-after question prompted by HM150; this plan precedes the new
rank/relation computations. Original18/100/44 scope remains OPEN.

## Scientific question and competitors

HM126 already compares isometry and anisotropic deformation; HM127 already
compares functional similarity and author-law observability. Do not repeat them.
HM150 supplies actual joint persona/style observations but an unconstrained
linear transformation in a high-dimensional space may explain ANY finite set.
Does that explanation impose observable relations in the actual design, and
does the frozen GRAIN tensor lift remove those relations?

Competitor1: a common linear style map preserves the linear relations among
the neutral persona response vectors. Competitor2: condition-specific responses
break these relations. Competitor3: the representation is saturated on this
finite design, so both stories fit with no observable constraint. Stochastic
generation can also break finite relations without breaking population means.

The response is high-minus-low, so an affine common offset cancels. This is a
hypothesis on response vectors, NOT a general theorem that texts are linear.

## Exact observable relation

Let X,Y have aligned rows: neutral and styled persona contrasts. A common
linear map Y=X B exists exactly when ker(X.T) is contained in ker(Y.T).
With Q the orthogonal projector onto ker(X.T), test QY=0. This follows by
defining the map on the quotient of coefficient vectors; it is established
linear algebra, not a new psychological theorem.

If rank(X)=n_rows, Q=0 and ANY Y admits a linear map. If both aligned matrices
have full row rank in the same ambient dimension, an invertible extension also
exists. An apparent perfect fit in that case is not evidence for transport.

For nonempty source kernel, choose a unit coefficient witness using only Q:
project the coordinate basis with the largest projector diagonal, first-index
ties. Save its neutral closure and styled displacement. Do not choose the
witness by its target effect. All coordinates/channels remain in the account.

## Fixed inputs and comparisons

Use all702 actual HM150 r2 observations, authenticated against their seals.
No raw private texts, new forwards, prompts, fitting, training or API calls.
Three Qwen generation seeds are replicates, not authors. Four fixed views:

1. Qwen input token mean,1536 dimensions.
2. Qwen block16 token mean,1536 dimensions.
3. Frozen source-standardized GRAIN4x4,16 dimensions.
4. The existing full625-coordinate GRAIN tensor Phi, without a new fit.

For each view and style newsletter/spelling, report every original scope:
discovery18 response cells, development6, evaluation12 and all36. These are
descriptive design groups, not a newly selected train/test partition.
Primary finite matrices use means of seeds11/23/37. Supplement uses seed11
alone to construct Q/c, then applies the SAME c to seeds23/37 in both neutral
and styled views. Retain their signed cross-products and neutral instability.
Do not claim the three-seed means or a finite violation identify a population
mechanism; there is no significance gate or accuracy metric.

Use eps*max(shape)*largest singular value as the fixed arithmetic rank rule.
Report all singular values, rather than sweeping thresholds. No unit/rank/
metric/polarity or target selection. Tensor saturation and raw saturation are
possible outcomes, not acceptance targets.

## Mathematical and literature support

Khemakhem etal2020 separates observed fit from identifiable latent recovery:
https://proceedings.mlr.press/v108/khemakhem20a.html . Its injective mixing,
conditional factorized prior and variability assumptions are NOT established
for GRAIN. Context modifying the expression map is not automatically its
auxiliary-variable construction.

Gresele etal2020 supplies multi-view identifiability under a specified noisy
view model: https://proceedings.mlr.press/v115/gresele20a.html . Two arbitrary
encoders of the same text are not automatically independent noisy views.

OpenAI/math pinned adc7f124 Universal-Tensor-Squares preliminaries21--100 uses
dual component tests under group-representation hypotheses. This inspires
looking at relation-sensitive tests instead of total fit. No Specht module,
generic basis theorem or psychological identifiability is imported.

## Execution and decision

New module,runner,tests,plan; new conditional_relation_circuits_r1 only. Prepare
seals all input/code addresses before execution. CPU cap120s; no model calls.
SVD projector verified by pivoted QR and direct relation arithmetic. Preserve
all old failures and signed outcomes. One finite run, no accuracy rescue.

Report whether each space actually constrains common linear transport, how
lifting changes that constraint, and which finite relations survive. If all
high-dimensional spaces saturate, do not fit a larger operator; define added
observation identities or independently justified transport structure first.
If GRAIN relations fail only at the scale of neutral replicate variation,
retain realization uncertainty rather than declaring target-specific biology.
