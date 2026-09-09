# GRAIN conditional navigation: formula, construction and remaining mechanism

2026-09-09. Finite CON00–CON04 delivery. Research goal revision V7, not a product release.
[Results](../reports/grain_vector_mechanisms/CONDITIONAL_NAVIGATION_DELIVERY_REPORT.md),
[frozen design](GRAIN_CONDITIONAL_NAVIGATION_PLAN_V1.md),
[seminar](../reports/grain_vector_mechanisms/NAVIGATION_SEMINAR_20260909.md).

## 1. Exact input-to-output dependency chain

For a complete excerpt list T and a fixed trait/state query q:

1. Render the frozen query-specific prompt; tokenize without silent truncation.
2. Execute the first four frozen Qwen2.5-1.5B Transformer blocks. No generation,
   final LM head, upper-layer activations or author labels are used at inference.
3. Pack the ordered early question slots and the separate final slot:
   `x_q(T) = vec(P_early) concatenated with Q_early`.
4. Evaluate a frozen early reader plus a source-fitted residual program.
5. Average view scores within an author when the evaluation declares multiple
   views. No current-batch normalization or author ID enters the score function.

The E-only input experiment appended two excerpt-only block3 summaries. It did
not qualify for external use; the delivered selected programs use P/Q only.
All retain a four-block neural front end. The historical B0 remains the actual
zero-Transformer program. Neither is an assertion that LLM construction was unnecessary.

## 2. The scalar function and source objective

Let x be the early state and y the frozen upper block16 state. With reader matrix
L_q (three columns: mean/configuration/final), bias b_q and anchor a_q,

```
f_q(T)       = a_q + 1' (y_q(T) L_q + b_q)
R_i          = (y_i - x_i) L_q
f_hat_q(T)   = a_q + 1' (x_q(T) L_q + b_q + h_q(x_q(T)))
Delta_e_hat  = f_hat_q(T_e) - f_hat_q(T)
```

This f is the fixed functional readout, distinct from the full model's native
score. Source construction fits f; both f and native responses are then measured.
The same f_hat computes both endpoints; no condition-specific offset or known
edit label is supplied at inference.

Source-only center/scale produces `z=(x-mean_X)/s`, with
`s^2=mean_i ||x_i-mean_X||^2`. Candidate kernels are
`k1(z,z')=z^T z'` (inner product) and `k2=k1+0.5*k1^2`; kernels are centered again
on source rows after the quadratic map. All moments exclude held rows.

In unambiguous sum-of-squares convention the implemented objective is

```
J(h) = ||R-h(X)||_F^2
       + lambda * n/m * ||D(R-h(X))||_F^2
       + ||h||_H^2
```

Alpha=1 here. Dividing this expression by n makes the RKHS coefficient 1/n,
not1. The frozen plan's normalized symbolic alpha must be interpreted with its
explicit sum-convention clause. The implementation did not change conventions
between variants.

D contains four within-author endpoint differences. Define centered source K,
`lambda = (tr(K)/n) / (tr(D K D')/m)` for secant variants; lambda=0 for value
variants or a degenerate edge denominator. Then

```
M    = I + lambda * n/m * D'D
C    = (K + M^{-1})^{-1} (R - mean_R)
h(x) = mean_R + k_centered(x,X) C
```

M1=1, so the ordinary mean intercept is retained. Numerical tests compare the
linear case to the weighted primal solution. All fitting used156 source rows,
alpha1, and the same fixed degree1/2 factorial; no rank/alpha/depth search.

Two useful identities:
- Fixed output projection commutes with a linear target-separable estimator:
  fitting Y then multiplying by L equals fitting YL under identical conditions.
  A quotient alone is not a performance-improving treatment.
- Existing differences DY add row constraints/weight, not new independent
  endpoints or span. CON01 added new conditional endpoints as a separate step.

Linear h has a constant derivative in x, but x(T) is contextual and nonlinear.
This is not a claim that a constant coefficient ignores textual conditions.
Discrete text edits are finite differences, not derivatives with respect to words.

## 3. What the edge graph does and does not identify

The actual design has60 isolated anchor rows plus48 disjoint endpoint pairs.
Consequently `rank(D)=48`, `dim ker(D)=108`, and each nonzero eigenvalue of D'D
is2. Secant direction weights are `1+2*lambda*n/m`: about448 for linear T/F
and385 for linear state. These numbers describe the source row metric, not
human-trait importance or degrees of freedom of the hidden model.

For any author-constant vector c, Dc=0. Therefore

```
D(f+c) = Df
```

while the between-author order of f+c may differ arbitrarily. A verified
synthetic example has exactly identical within-author differences and zero
between-author level correlation. This is an algebraic non-identification of
levels by responses alone. It is not non-identification of our entire objective,
which also contains source value information and RKHS regularization.

The empirical result fits this distinction: improved natural edit responses
do not entail improved human-criterion ordering. The graph fact explains why
one implication cannot be assumed; it does not uniquely diagnose the learned
model's transfer errors. Query-specific evidence, coverage and fixed-reader
approximations can all contribute.

## 4. Finite evidence and decision ledger

| Package | Evidence | Decision |
|---|---|---|
| CON00 |192 metric /576 geometry rows; exact cross-term accounting; conditional support audit | Level span coverage does not ensure response coverage |
| CON01 |288 source forwards; four models/query;12 author and2 condition holdouts/query; frozen natural replay | Retain T/F and state conditional computation gains |
| CON02 |168 four-block calls; same156 source rows; all four E variants; geometry and paired intervals | Stop this E augmentation, do not extend its search |
| CON03 |48 raw-text score replays, source access denied;284 shared external forwards;468 frozen scores | Keep new response programs, do not replace B0/B4 external-link programs |
| CON04 |This formula, aggregate report, scripts/tests/dependency ledger/local commits | Finite plan complete with useful operators and localized negative results |

Construction12 and natural-evaluation12 authors are disjoint; construction
also excludes external source authors. MEPS46 is external-only. No parameter,
query sign or model selector is learned from human criteria. The external
panels were previously exposed, so this is model transfer on existing material,
not a new participant replication.

Source holdouts exclude each held author's mean anchor and all incident edges.
Condition holdouts omit denied/past endpoints; their counterpart is still
allowed in training. They test endpoint-rule transport, not two-unseen-endpoint
generalization. Source folds and exposed natural tests are separate evidence.

All actual runtime/source/report hashes are in the private closure summary.
Raw source support, full/native predictions and author-derived anchor arrays
are not public artifacts. Copied runtime bundle is for authorized local use;
it is not cleared for GitHub publication.

## 5. Rebuild and inspect

Run from the independent repository. Existing r1 outputs are immutable; the
constructors refuse destructive overwriting and collectors resume committed
units. Use new named output directories for intentional reproductions and
propagate their paths in a separately frozen plan, not by modifying old fits.

```bash
PY='../project persona/.venv/bin/python'

# CON02: one E augmentation (defaults refer to the delivered local packet)
"$PY" scripts/run_grain_conditional_augmentation.py prepare
"$PY" scripts/run_grain_conditional_augmentation.py collect --seconds 600
"$PY" scripts/run_grain_conditional_augmentation.py fit

# CON03: select by model responses, export, deny source access, evaluate once
"$PY" scripts/export_grain_conditional_navigation.py prepare
"$PY" scripts/export_grain_conditional_navigation.py replay
"$PY" scripts/export_grain_conditional_navigation.py collect --seconds 600
"$PY" scripts/export_grain_conditional_navigation.py analyze

# CON04: generated tables and graph/summary-geometry evidence
"$PY" scripts/summarize_grain_conditional_delivery.py

# Raw-text use of the completed private bundle; input fields: id/texts/query
"$PY" results/grain_vector_mechanisms/conditional_navigation_v1/CON-03/runtime_external_r1/bundle/run.py \
  --bundle results/grain_vector_mechanisms/conditional_navigation_v1/CON-03/runtime_external_r1/bundle \
  --input /absolute/path/to/authorized_inputs.jsonl \
  --output /absolute/path/to/new_output --program conditional --device mps
```

These are build recipes, not instructions to rerun an already complete packet.
For raw use the selected queries are `TF_cont` (F-positive) and `adapted_dejected5`.
All three candidates are returned without a new selector. Output is an unclipped
mechanism score, not a ready-to-use clinical measure or questionnaire probability.

Local artifacts below `results/grain_vector_mechanisms/conditional_navigation_v1/`:
- CON-00/diagnosis_r1: original support and error decomposition.
- CON-01/source_endpoints_r1 and fit_r1: source snapshots, coefficients, all comparisons.
- CON-02/evidence_e_r1/fit: failed augmentation and frozen predictions.
- CON-03/runtime_external_r1: bundle, selection proof, source-denied replay, external analysis.
- CON-04/closure_r1: all response metrics, augmentation uncertainty, row-metric spectrum,
  input-geometry result and a hash-indexed delivery summary.

## 6. Next research, without pretending to know a completion percentage

The present finite construction works; the universal theoretical target is
larger. Three next-phase families were already declared, and this round locates
one connection problem that should precede their expansion:

1. **Reference position plus conditional response.** Preserve the existing B0/B4
   level function and learn a source-only, reference-anchored response correction.
   Compare whether one raw-text function can retain both level ordering and edit
   response. Use source-original constraints and held-author level/response tests;
   do not concatenate scores or tune a blend on human outcomes. First compare one
   fixed constrained construction with the existing programs, not a new grid.
2. **Query-family construction.** Once that join is useful on at least one query,
   extend the same constructor to Big5 and four MBTI axes, then narrow queries.
   Ask which components are shared and which query-specific; reuse M28/M31's
   already-built upper function atlas rather than rebuild all reader functions.
3. **Persona write/read relation.** One paired write/read intervention packet:
   does a selected state manipulation alter emitted evidence and its later
   inference in corresponding directions? This requires new causal evidence,
   not similarity of vector directions alone.
4. **Transport.** One lexical/scenario/model change at a time, holding the
   reference norm explicit. Cross-model mechanism transfer and human linkage
   have separate outcomes. No universal LLM-free nine/34-query claim is inferred
   from B0's useful T/F result.

These are ordered new questions, not unfinished CON cards or permission for
an automatic unbounded continuation after this goal. Their exact experimental
design should be frozen before new execution.

## 7. Relation to established mathematics

This uses ordinary RKHS regression, finite-difference supervision and a graph
Laplacian row metric, not a newly discovered linear-algebra theorem. Related
learning designs include [Sobolev training](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html)
and [Jacobian matching](https://proceedings.mlr.press/v80/srinivas18a.html).
[Causal abstraction](https://jmlr.org/papers/v26/23-0058.html) motivates comparing
interventions across programs, but our finite experiment is not an identification
proof of an entire reasoning circuit. The research contribution is the measured
query-conditioned construction/decomposition and its executable, testable limits.
