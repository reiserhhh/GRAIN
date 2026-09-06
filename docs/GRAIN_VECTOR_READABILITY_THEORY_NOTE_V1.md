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

## 5. A hybrid linear probe does not test arbitrary complementarity

Round33 completed the raw same-ready comparison. With a frozen fitted model,
standardization and unwhitened PCA are affine maps. Ridge's output and the
binary logistic head's log odds therefore have the form

```
z = D^(-1)([g,h] - m)
eta = b + beta^T V(z - m_pca)
    = b_star + a^T g + c^T h.
```

Consequently the mixed rectangular difference is exactly zero on this scale:

```
eta(g+u,h+v) - eta(g+u,h) - eta(g,h+v) + eta(g,h) = 0.
```

This is a standard algebraic limitation, not a new psychometric theorem. PCA
mixes coordinates but cannot create products between the two input blocks.
Logistic probabilities can show nonzero mixed differences because the sigmoid
is curved; that alone is not an interaction term in the log-odds model. Each
upstream embedding/profile may itself be nonlinear in text: the limitation is
specifically the fitted head's cross-block function, not all text processing.

The actual60 saved hybrid heads (two representations x five tasks x five folds
plus full-source head) pass480 synthetic rectangles, maximum4.44e-16, and exact
coefficient pullback within4.43e-16. No real feature rows, new target tables,
refits or neural forwards are used. These artificial inputs verify the saved
implementation against the algebra; they are not plausible-person interventions.
Packet: vm05_raw_ready_additivity_audit_r1,
identityd156388e913a7c5b1327f726135fd69186bfe4b3db575876c902db7dd10a184b.

Thus the observed19/20 zero-crossing GRAIN-increment intervals narrow the claim
to **weak or uncertain additive usefulness under this projection/cohort**. They
do not refute nonlinear complementarity. Conversely, the capacity limitation
does not demonstrate that a useful nonlinear interaction exists in these data.
Source-trained nonlinear probes would still not prove native LLM causal use.

The current task/history/direct queues remain the next actions, not an expanded
search for a favorable source score. If interaction becomes a targeted follow-up,
a declared low-rank bilinear term `g^T U V^T h` is one bounded alternative, with
rank/regularization chosen only inside source-inner folds, matched additive and
label-permuted controls, and no bridge-driven selection. That follow-up is not
run, not retrospectively part of the completed comparison and not required to
call this additive instrument audit complete.

## 6. Native output convention is part of the measured function

Round34's complete invented semantic panel separates correct behavioral ordering
from invariant score meaning. The standard wording orders17/18 pairs correctly.
Correctly recoded Big5 scale inversion gives0/10 positive pairs, while MBTI
preference/answer-list reversal retains7/8 but produces large absolute shifts.
Candidate mass remains above.9924.

Write `s_e(x)=v_e^T softmax_Ae(logits_theta(x,question_e))`, where e identifies
the response convention and v_e its declared decoding. A measurement-like
invariance would preserve s_e after legitimate recoding. It fails in this panel.
Evidence sensitivity, coding compliance, numerical interpretation and natural-text
external validity therefore need separate status fields. Neither candidate mass
nor readable source labels settle the other properties.

This does not imply absent relevant hidden-state distinctions: MBTI ordering can
persist with a shifted output prior. Nor does the panel localize a reasoning
circuit or establish a general LLM limitation. Keep the unchanged direct source
control and planned history/task comparisons to separate contextual representation
from native answer behavior.
[Full results](../reports/grain_vector_mechanisms/VM05_DIRECT_SEMANTICS_REPORT.md).

Round35 adds a bounded post-hoc necessary-condition check. Align candidate logits
by declared semantic score, then subtract the common logit mean to fix only the
softmax gauge. Under `u_e(x)=b_e+g(x)`, the mixed contrast
`[u_c(high)-u_p(high)]-[u_c(low)-u_p(low)]` is zero and independent of the
no-evidence reference. Actual recorded contrasts are nonzero, so a fixed vector
bias cannot restore exact semantic response invariance on these authored cases.
This is not proof that calibration cannot improve rankings, nor a new personality
law or localization of the responsible computation. Probability-scale mixed
differences alone would be insufficient because the sigmoid is nonlinear.
See [the complete diagnostic](../reports/grain_vector_mechanisms/VM05_ENCODING_INTERACTION_REPORT.md).
