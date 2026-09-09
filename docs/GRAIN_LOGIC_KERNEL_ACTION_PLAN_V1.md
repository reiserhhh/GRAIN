# NEXT16: a conditional kernel contrast for measured configuration action

2026-09-10. Fixed after NEXT15 localized remaining N evidence-configuration
contribution+.1266[.0248,.2133] outside current f. This is not a new human-label
prediction search. T/F conditional-mean contribution+.0694 remains a separate
target; do not assume one function solves both queries.

## Difference from old constructions

M12/M13 use shallow pooled/query-attention means. M16 fits a high-dimensional
gradient field from four reference authors; M17/M18 restrict action structure.
M19 demonstrates nonzero odd and even configuration effects. M20 uses finite
position/state-even fields, yielding weak actual odd-action recovery. M21 exact
endpoint constraints barely change it. M22/M23 source-score readers and M24
P/Q interventions concern different inputs/targets, not E configuration with
P/Q fixed. All remain valid within their addresses.

This packet changes the construction object: learn a **difference of nonlinear
token-distribution features**, directly from already measured plus/minus endpoint
actions. It preserves comment means and conditions instead of discarding them.
No gradients, new model calls, attention training, or Y are required.

## Fixed finite design

- Reuse all48 M19 authors, two queries,3 material views. Project all cached body
  token states; no token sampling or new eligibility cuts. Existing M16 design
  supplies groups; existing M11 arrays supply states; M19 supplies mirror targets.
-32 fixed Gaussian random directions per query, W~N(0,1/d), seed20260910+query
  index. Projection is label-independent and fitted nowhere. Store projections
  once; all scaling below is training-fold only.
- Four author folds: sorted author IDs, seeded permutation, index modulo4.
  Every author's three views and both mirror endpoints stay in one fold.
- For token t, projected comment mean c_t and deviation v_t=h_t-c_t. Center c
  using training authors' token means; scale both by training authors' RMS v
  per coordinate. Each author/view has equal total weight when fitting geometry.
- Four fixed arms: Taylor degree3 versus full Fourier, each without versus with
  global token-position phase. Position is original index scaled to[-1,1];
  fixed eta~N(0,1) per direction. No degree, bandwidth, rank, seed or alpha search.

Let theta be the standardized mean coordinate, plus eta*position for positioned
arms; u is the standardized deviation. Per direction, pooled cos/sin differences
have the exact parity components (token mean, normalization1/sqrt(32)):

    odd  = mean[-sin(theta)*sin(u), cos(theta)*sin(u)]
    even = mean[ cos(theta)*(cos(u)-1), sin(theta)*(cos(u)-1)].

Taylor control replaces sin(u) by u-u^3/6 and cos(u)-1 by -u^2/2, with exactly
the same directions/geometry. Both vanish at v=0; reflecting deviations changes
odd sign and leaves even unchanged. Conditional means remain in theta.

Fit odd and even action heads separately with zero intercept, training-only RMS
feature scaling, mean-square loss+.01||beta||^2. Equivalently, this minimizes
equal-weight paired plus/minus loss because cross terms cancel exactly. This
is not raw TF/N score calibration. Report plus=odd+even and minus=-odd+even.
All four arms have equal basis dimensions. Cached neuron states remain an input
dependency; this does not yet remove the upstream encoder/LLM.

## Endpoints and comparison rules

- Primary: author-OOF actual odd/even/plus action r, MSE and MSE/zero. Also
  report material-level summaries separately, not as independent author samples.
- Use the two predeclared Fourier-versus-Taylor contrasts at fixed position,
  and positioned-versus-unpositioned Fourier. No criterion-based winner.
- Freeze all predictions before joining already exposed M29 source self-reports.
  Report their links descriptively for all arms; no predictive-promotion claim.
-1999 whole-author paired bootstrap, pointwise, frozen predictions. Old M20 has
  four-reference training; its r is historical context, not a matched competitor
  for this18-train/6-eval fold design.
- Validate zero, reflection parity, direct cosine/sine contrast, paired-loss
  identity, fold leakage and provenance. Publish only aggregates.

## Mathematical rationale and limitations

[Rahimi & Recht (2007)](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
provide finite random features for kernel approximation. Here cosine/sine
differences preserve nonlinear distribution changes invisible to the linear
mean. [Sriperumbudur et al. (2010)](https://www.jmlr.org/papers/v11/sriperumbudur10a.html)
give conditions for injective probability embeddings;32 finite directions do
not inherit an infinite characteristic kernel's injectivity guarantee.

For any differentiable feature phi, phi(C+d)-phi(C) is exactly the integral of
its directional derivative. A learned linear combination is therefore an
explicit candidate action function, not an arbitrary error correction. Its
additivity over tokens remains a restriction: joint token interactions or
unrepresented non-body conditioning can remain outside it. Positioned features
encode token-state/position association, not the whole autoregressive history.

## Resource and delivery boundary

Cached arrays about0.93GB, CPU at most4 BLAS threads, no new dependencies.
Projection budget600seconds; stop resumably if exceeded rather than restart.
Run four fixed arms once, retain negative results, no expansion merely to improve
r. Outputs under `results/grain_vector_mechanisms/logic_kernel_action_v1/`;
aggregate report, tests, logic/goal updates and local commits. No release, API,
MEPS, raw-text upload, model-weight training or lockbox change.
