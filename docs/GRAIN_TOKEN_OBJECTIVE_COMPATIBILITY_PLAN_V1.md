# NEXT61: source value/field compatibility, before another extractor

2026-09-11. Planned, not executed. The same-material GRAIN/query interface has
useful configuration response, but NEXT60's source density loss is already >1.
One decision: is the fixed scalar dictionary unable to express the source field,
or is value fitting/regularization suppressing a field it could represent?

## Frozen scope

Reuse NEXT60's ten source/query references, both arms, exact rank8 descriptors,
centers, bandwidths, author weights and original source targets. No application
labels/gradients, new8 response-guided choices, neural/API calls, MEPS or lockbox.
No rank/alpha/bandwidth search. Keep NEXT60 predictions/coefficients unchanged.
Budget60s CPU, one numerical replay. This is a source diagnostic, not a new
deployable model. Diagnostic least-squares solves must be counted explicitly.

## Mathematics to calculate

Let V be source scalar design normalized by sqrt(n)*sL, target ell. Let D be
stacked source density design normalized by sqrt(n*T*rank)*sK, target k. The
original objective is ||V theta-ell||²+||D theta-k||²+.01||P theta||².

1. Reconstruct V/D/ell/k independently from source arrays, verify saved Gram/rhs
   and normalization. Keep fixed numerical SVD cutoff1e-10 relative to largest
   singular value, not choose a cutoff from held performance.
2. Project k into col(D). Report orthogonal residual energy, rank, min-norm
   coefficient norm and projected density MSE. Compare with original fitted
   source density loss. This separates dictionary-external residual from failure
   to use the available dictionary. A tiny singular mode with a huge coefficient
   is numerical capacity, not a practical solution.
3. Holding original source values V theta0 fixed, let N span ker(V); project
   e=k-D theta0 into col(DN). Obtain the exact reachable gradient improvement
   without altering those source scores. Report ||V delta|| and ||D delta||,
   coefficient norm, ridge penalty change and complete objective difference.
   Repeated scalar observations are author-level constraints, not independent
   token labels. If feasible low-norm field changes survive the scalar constraint,
   value/gradient conflict is not an intrinsic integrability obstruction.
4. Compute the original normal-equation force vectors separately:
   V.T(Vtheta-ell), D.T(Dtheta-k), .01 P.T P theta. Verify their sum, signed inner
   products and norms. Separate empirical fitting conflict from regularization.
5. Inspect the intercept+last-token-only scalar design rank (these columns have
   zero BODY derivative). Calculate whether the source scalar residual can be
   independently anchored using those columns, and the amplification required.
   Do not extrapolate an exact 3/4-point anchor to new-author accuracy.

Report all20 fits, query/reference groups and source-only capacity vs realized
loss; no selection of a favorable factor. Compute no new8 performance for these
diagnostic projections, which would turn source explanation into another tuning
round. One repeated calculation must reproduce numerical outputs and equations.

## Decision and deliverables

- Mostly dictionary-external field: next change is a better conditional/local
  input dictionary or pairwise relation, not a different loss weight on the same
  functions. Identify which source field component is missing before designing it.
- Large low-norm feasible field improvement with original scalar scores held:
  next develop a compatible fitting/anchoring formulation, without current scores.
- Improvement only at huge coefficient amplification: treat conditioning/source
  support as the bottleneck; do not promote the unregularized projection.
- Mixed result: quantify both terms and select the larger actionable cause, not
  claim all paths fail or launch several grids.

Files: new module/script/test with `token_objective_compatibility` stem; private
`results/grain_vector_mechanisms/logic_token_objective_compatibility_v1/`;
aggregate `TOKEN_OBJECTIVE_COMPATIBILITY_REPORT_V1.md`, theory and ledger updates.
Freeze primary implementation before opening diagnostic outcomes. Source field
projection is a finite linear-algebra answer, not universal theory completion.

Literature context: [Sobolev Training](https://arxiv.org/abs/1706.04859) motivates
value/derivative teaching; [GDML](https://arxiv.org/abs/1611.04678) motivates a
compatible field. These do not guarantee good finite-dictionary conditioning or
resolve the current empirical tradeoff. Orthogonal projection/nullspace accounting
are standard least-squares identities, not a new psychological theorem.
