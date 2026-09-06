# Information, readability and native model use

2026-09-07, round32. Mathematical interpretation of completed source results,
not a new universal personality theorem or permission to replace the frozen
experiment grid. The personality-interpreter objective stays intact.

## 1. What a layer comparison measures

For fixed model parameters, full token states H_l and declared pooling A_l,
write Z_l=A_l(H_l). The current result is the held-out score of a restricted
statistical learner applied to Z_l, with finite source data, scaler, PCA and
regularization. It is not a direct estimate of I(Y;H_l).

Under deterministic consecutive layers, complete states and fixed/conditioned
inputs Q, data processing gives

```
Y -> H_l -> H_(l+1), conditional on Q
I(Y;H_(l+1) | Q) <= I(Y;H_l | Q).
```

This statement requires all state needed for the transition to be included;
it is not a theorem about a single token, mean/std summary or selected subset
of the computational graph. Generally Z_(l+1) is not a function of Z_l alone,
so the analogous order between the two pooled representations does not follow.
For example, let X be a fair sign and full states change from `(X,-X)` to
`(X,X)`. Both contain X, but their means change from0 to X. This constructed
example proves that improved pooled readability need not mean new input
information was created; it is not a model of human personality.

Even without pooling, information and restricted readability differ. For
independent fair signs X1,X2 and Y=X1*X2, Y is determined by the pair, but its
covariance with either input is zero. A population squared-loss linear head
with intercept therefore predicts zero. Supplying the product coordinate
makes the same target linearly readable. The improvement is computational
re-expression of existing information, not creation of a new human fact.

Pimentel et al. formalize probing via information bounds and data processing;
their conclusions depend on the object/assumptions being probed. Their
idealized uniqueness argument must not be imported as a guarantee for our
finite-precision, pooled and compressed caches. [ACL2020 paper](https://aclanthology.org/2020.acl-main.420/)
The finite-data distinction between total information and ease of extraction
is developed further by Pimentel and Cotterell. [EMNLP2021 paper](https://aclanthology.org/2021.emnlp-main.229/)

## 2. What the current results actually add

- All9 no-PCA/PCA comparisons have intervals crossing0. This does not establish
  that projection was the main bottleneck or that discarded coordinates carry
  no signal. It changes compression and the fitted regularization problem.
- The complete three-arm fixed-layer profiles vary by outcome, with no common depth winner.
  This is compatible with reorganization of accessibility. It does not identify
  an EI layer, a TF circuit, or total psychological information at each depth.
- All36 primary MBTI heads improve proper probability losses over fold-trained
  prevalence priors, even when their0.5 decisions are almost all majority.
  This establishes modest source predictive differentiation, not a native LLM
  disposition rule or a calibrated psychological score.

See [projection](../reports/grain_vector_mechanisms/VM05_PROJECTION_SENSITIVITY_REPORT.md),
[layers](../reports/grain_vector_mechanisms/VM05_FIXED_LAYER_REPORT.md) and
[proper scores](../reports/grain_vector_mechanisms/VM05_PRIMARY_PROPER_SCORE_REPORT.md).

## 3. Why a trained head cannot explain native inference by itself

Let g_D be a readout trained on source outcomes D, and let g_native be the
fixed model's own decoder operating after question q. Generally

```
g_D(A_l(H_l(x))) != g_native(H_last(x,q)).
```

The first function can exploit correlations learned from our labels that the
second never uses. Conversely, the native computation might make a relation
accessible that an averaged body vector plus linear head misses. This is a
possibility, not yet a demonstrated advantage of our1.5B observer. Probing's
disconnect from native behavior is a recognized limitation.
[Belinkov2022](https://aclanthology.org/2022.cl-1.7/)

Probe capacity and source memorization need controls. Hewitt and Liang's
control-task argument motivates careful selectivity checks, but our author
permutation and discussion controls are adaptations, not reproductions of
their word-type control task. [Hewitt and Liang2019](https://aclanthology.org/D19-1275/)
MDL probing also considers the cost/ease of extracting labels rather than
only final accuracy; no MDL result has been run here, and it is not silently
substituted for the fixed source-head protocol.
[Voita and Titov2020](https://aclanthology.org/2020.emnlp-main.14/)

## 4. Questions the existing queue can resolve

1. Fixed layers/masks are complete: the descriptive profile persists across
   the three arms. Carry this forward without selecting a new primary layer;
   masks on the same authors do not count as independent replications.
2. Compare task-conditioned and body channels on the same source authors:
   does the fixed question expose more readable outcome information, or just
   inject common prompt/position variance? Two social-initiation prompts do
   not cover all9 direct trait tasks.
3. Compare own/matched/no-history with the same current text: which changes
   reflect relevant evidence access rather than lexical continuity or donors?
4. Compare the direct candidate probabilities and functional controls with
   source-trained heads. Strong output prediction, code fidelity and model
   intervention evidence must be linked, not assumed interchangeable.
5. Compare GRAIN increments on exactly the same eligible authors, then freeze
   source decisions before independent bridge outcomes are read.

These are already required work, not new outcome-driven layers or training.
The core mechanism question becomes: **which computation turns available
author evidence into a task-usable representation, under which readout and
source conditions?** If no native advantage appears, report that limit for
the tested observer rather than declaring that all LLMs lack such a mechanism.
