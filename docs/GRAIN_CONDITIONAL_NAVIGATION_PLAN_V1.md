# GRAIN conditional navigation: finite experiment plan V1

2026-09-09. FROZEN DESIGN for the next research goal. Current execution status
belongs in GRAIN_CONDITIONAL_NAVIGATION_GOAL_V7.md, not by rewriting this plan.
Based on the actual two-round [seminar](../reports/grain_vector_mechanisms/NAVIGATION_SEMINAR_20260909.md).

## Objective and theory target

Construct the part of a fixed observer's text-to-trait function that changes
when the same evidence is denied, reassigned, retimed or removed. Preserve the
achieved B0 T/F and B4 state programs. Do not search for a higher human AUC/r.

For text T and query q, x_q(T) denotes the early query/end coordinates, y_q(T)
the upper coordinates, and f_q(T)=b_q+y_q(T)L_q the fixed functional components.
The sum of the three components plus the separate anchor is the scalar score.
The approximation uses early inputs only:

```
f_hat_q(T) = early_reader_q(x_q(T)) + h_q(x_q(T))
Delta_e f_hat = f_hat_q(T_e) - f_hat_q(T)
```

Candidate h is a centered linear or fixed linear-plus-quadratic kernel
operator. No independent edit-specific offset is supplied at inference: both
scores must be evaluations of the same function. Query coordinates are not
pooled across questions of different token length.

The factorized loss on source construction points is:

```
R = teacher_components - early_components
L_value = ||R - h(X)||_F^2 / n
L_edge = ||D(R - h(X))||_F^2 / m
L = L_value + lambda L_edge + alpha ||h||_H^2
```

D is the incidence matrix of the four declared within-author edit pairs.
Differences of existing rows add supervision weight, not new information or
linear span. New endpoints extend support. Linear full-state regression and
projected regression commute for identical fitting conditions; changing only
the output representation is not an experimental treatment.

## Fixed scope and data

- Queries: extraversion, TF_cont (F positive), adapted_dejected5.
- Same frozen Qwen observer, FP32/MPS; weights/readers unchanged.
- Construction source:12 already-declared PANDORA reference authors, view0;
  each supplies original/self_report/other_report/endorsed/denied/today/past/remove.
  All12 passed the same first-person proposition eligibility rule before scoring.
- Eight endpoints ×12 authors ×3 queries =288 full-model units. One forward can
  save block3 early states, block16 states and the native readout with scoped hooks.
- Existing48 synthetic construction rows and12 natural author-mean rows remain
  available as the fixed support anchor. Never add held-scene/evaluation rows.
- Natural evaluation: existing disjoint12 authors/96counterfactual worlds;
  already exposed, not a new lockbox. Separate external source277 and MEPS46.
- MEPS is external-only: no construction, fit, source selection or outcome tuning.
  No raw private text/API upload, LoRA, new dependencies, old lockbox access or push.

## CON-00: support and response diagnosis

Zero neural calls. Read verified frozen aggregates and numeric caches only.

1. Tabulate explicit intervention support versus evaluation conditions.
2. For each query/edit/program, report signed r, through-origin slope,
   mean-absolute amplitude ratio, sign agreement and MSE/zero-response MSE.
3. Decompose native-program error into native-fixed and fixed-program terms,
   including twice their cross product. Retain per-author rows privately.
4. Apply the existing residual-span projector to paired residual differences.
   Report outside, inside and reader-space cross terms. Center terms cancel.
5. A per-author scalar-gain oracle may describe leftover shape, explicitly
   labeled unusable at inference. No fitted evaluation gain enters a program.

Deliverables: support_inventory.csv, response_metrics.csv, error_accounting.csv,
response_geometry.csv, summary.json and a public aggregate report.

## CON-01: supported conditional construction

Collect288 source units; pause at a20-minute or process-error boundary with
resumable per-unit output. Save fixed block3 evidence summaries during the same
forward to avoid recollecting if CON02 is needed. Evidence tokens are selected
by exact excerpt spans, not the question/instruction scaffold. Summary is one
unweighted mean plus one query-mean cosine-weighted mean (temperature1, no fit).

Four matched variants on the identical augmented data:

| Variant | Function family | Objective |
|---|---|---|
| linear_value | linear kernel | values |
| linear_secant | linear kernel | values + paired finite differences |
| quadratic_value | k=z'z'+0.5(z'z')^2 | values |
| quadratic_secant | same quadratic kernel | values + paired finite differences |

Construct input centering/scaling in each source fit only. Alpha=1 in the
sum-of-squared-values convention; lambda is fixed by the ratio of value-kernel
trace per value row to edge-kernel trace per edge row in construction only.
Clamp a degenerate denominator by reporting an inactive edge term, not arbitrary
huge weight. Record the actual lambda. No alpha/depth/rank/human-label search.

Source comparisons: leave-one-reference-author-out and two declared condition
holdouts (denied, past). A held endpoint and every incident edge must leave the
fit together. Keep old48 synthetic and other12-mean rows only when their author
is not the held author; otherwise remove the corresponding mean anchor too.
Do not combine author and condition holdout scores as independent replication.

After source diagnostics, fit the four declared programs once on all permitted
source endpoints. Freeze coefficient/config/source hashes before natural targets
are joined. Held evaluation inputs can reuse precomputed early states. Evaluate
all variants, not just a favorable target, against A_source, B0, B4 and full f.

Decision: retain a candidate for a target when model-only source author holdout
has positive response association, lower paired-response error than zero response,
and the natural exposed panel improves at least one specified response axis
without a sign reversal on that axis. Report paired uncertainty and all axes;
do not turn p=.05 or two-target success into a blanket development barrier.
Condition holdouts classify interpolation versus rule transport, not an
automatic rejection of a useful conditions-supported program.

## CON-02: one conditional input augmentation, if needed

Trigger: query-slot-only CON01 gives no usable response improvement for a query.
Append the two fixed evidence summaries saved at block3. Repeat the same four
model variants on the same source endpoints, with source-only normalization.
No learned low-rank softmax attention/MLP grid. This tests whether the selected
all-token summaries add useful information; failure does not prove information
absent from the whole hidden sequence. At most288 four-block evaluation passes
if evaluation summaries are unavailable. No new full-model collection.

Stop this branch after one comparison; record the located limitation. Do not
continue adding features until held performance rises.

## CON-03: independent execution and external linkage

Choose candidates by the preceding model-only rule, not external labels.
Export raw-text scoring with four extracted blocks; source-only fixed arithmetic
must reproduce cached scores. Test source-denied import/runtime behavior and
score differences from one common score function. Existing token-only remains
the only zero-Transformer selector unless a new one is actually implemented.

One external pass, at most923 four-block views, shared by all selected numerical
programs. Freeze predictions before joining labels. Compare same-member B0/B4,
native and fixed-function controls, report n, r/AUC, paired CI and dependence on
material type. No candidate is relabeled or sign-flipped from these endpoints.
If no candidate qualifies, this packet closes as a documented non-execution
decision, not a successful new runtime or an excuse for another fitting loop.

## CON-04: bounded theory delivery

Publish the exact formulas, source-support map, objective×capacity results,
conditional-input outcome, portable dependency ledger and external results if
triggered. Separate an algebraic identity, model-computation result and human
association. Keep finite counterexamples. Test critical functions and current
regressions; preserve unrelated worktree changes; make local commits.

Goal closes when CON00/01 and triggered CON02/03 are executed and interpreted,
and CON04 records an operational construction or a specifically localized
failure. It does not require every candidate to win or all errors to be zero.

## Next-phase pieces, not an automatic expansion

1. Query family: extend independent construction beyond3 queries to9 core and
   then selected narrow targets; test shared versus query-specific operators.
2. Persona write/read: whether manipulating a trait-related state has the
   corresponding effect on emitted evidence and subsequent inference.
3. Portability: lexical/scenario/model changes; retain explicit reference norms
   and distinguish independent human association from model-fidelity transport.

We can deliver a finite local theory without claiming universality. These three
questions remain valuable research, not hidden unfinished cards from NAV01-12.
