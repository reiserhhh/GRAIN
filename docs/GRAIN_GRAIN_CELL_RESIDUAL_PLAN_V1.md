# NEXT91: nonlinear GRAIN patterns versus within-pattern token configuration

2026-09-11. **PLANNED, not executed.** NEXT90/pass79 found useful current-moment
alignment and active query-shared/specific residuals. Preserve NEXT89's common
reading gains and NEXT90's .451/.327 action coverage. Do not return to an
accuracy, source-count or kernel grid.

## Decision

Does the remaining native query action mainly require nonlinear combinations of
the existing GRAIN patterns, or differentiation among token vectors with the same
GRAIN coordinates? The answer chooses a new input mechanism before another fit.
This is the same-material interface, not a redefinition of factors as diagnoses.

## Frozen inputs

Reuse NEXT90 original4/reused8,24full-L H16 BODY gradients, unchanged frozen
GRAIN values/positions/directions and current-map summaries. No extra source12
captures, text, query/factor choice, labels, MEPS, native calls or model training.
Use the corrected NEXT90 audit and all file/model/query binding evidence.

## Three nested maps and four components

1. Current A1=[1,centered g1..g4]. Reproduce its results exactly.
2. A2 adds every gi*gj with1<=i<=j<=4, giving at most15columns. Centered g uses
   the same per-material means; intercept/linear terms retained. One complete
   degree2 family, no polynomial degree/grid/feature selection after outcomes.
3. Saturated cell map: group exact equal four-dimensional frozen g rows using
   `np.unique(axis=0)`, no tolerance rounding or bin search. The projector is
   within-cell mean, implemented by grouped sums/counts without a dense T matrix.

Use thin SVD for A1/A2 with fixed machine-precision rank rule. Normalize nonzero
polynomial columns by their norm before rank detection to avoid arbitrary units;
the exact span is unchanged. Verify numerical nesting against the cell projector.
Record ranks, source/recipient condition numbers and structural degeneracy.

For every query N/C/mean/difference and every original GRAIN direction:

```
J = P1 J + (P2-P1) J + (Pcell-P2) J + (I-Pcell) J
tau = <J,D> = sum_k <J_k,D>
```

The four vector components are orthogonal when nesting holds. Their scalar
contractions can cancel, so retain all signed cross terms and all factors.
Report gradient energy separately from response alignment, as NEXT90 requires.

The saturated map is a permissive finite-material upper bound on row weighting
by arbitrary functions of g, not a source-fitted model. Singleton cells make it
trivial; report cell count, singleton token share, multi-token sizes and effective
within-cell degrees of freedom. If nearly all tokens are singletons, report the
bound as uninformative; do not coarsen g after inspecting gradients.

## Conditional vector alignment

For each cell, the within-cell action is exactly

```
tau_within = sum_cells |cell| * trace Cov_cell(J, D)
```

using denominator|cell| and the declared coordinate pairing. Verify against the
projector contraction. This does not assume token power or GRAIN score sign
predicts the action. Keep N/C shared and differential within-cell geometry,
and source4/new8 differences. No naming of raw tokens, author identities or
psychological constructs in public outputs.

## Implementation, checks and limits

Suggested `grain_cell_residual.py`, runner
`scripts/run_grain_grain_cell_residual.py`, synthetic tests. Private
`results/grain_vector_mechanisms/logic_grain_cell_residual_v1/`; public
`GRAIN_CELL_RESIDUAL_REPORT_V1.md` with cohort-only tables:
cell/rank geometry, four-component gradient account, signed factor responses,
query-shared/difference account, conditional covariance identity and provenance.

Freeze code/plan/material before analysis. Separate prepare/analyze/replay/audit
processes; preserve completed author units.60numericCPU including replay,
0fits/0native/API/Y. Two-author time-only cost gate, no outcome-based stopping.
Reuse verified AppleDouble handling and preserve literal categorical labels in
CSV parsing; do not modify old result packages for filesystem convenience.

Tests: nesting, degenerate/singleton/repeated cells, conditional covariance
identity, shared/opposing queries, signed cancellation, exact old-map parity,
orthogonal hidden-coordinate changes and complete cohort aggregation.

## Next branch

If nonlinear/between-cell terms carry useful action, develop one source-only
GRAIN interaction feature using the observed mechanism, not the evaluation
target as an input. If within-cell terms remain active, develop query-conditioned
token-vector alignment or local interactions; another g-only weight is inadequate
for those local directions. Both mechanisms may coexist and should be retained.
If saturation is trivial, state why and use fixed source-defined vector summaries
rather than an outcome-selected grouping threshold.

The target remains full-L local sensitivity; the whole compiled program has
additional evidence inputs and is not declared blind merely because A1 is.
H16 remains neural-derived. Finite mechanism development, not global theory
exhaustion. Research objective remains open; runtime goal is PAUSED at NEXT90
closeout. This plan has not been launched.
