# NEXT68: a compatible cross-token pooled quadratic reader

2026-09-11. PLANNED, not run. NEXT67 measured finite nonlocal reading on all
source4/two-query contexts, while NEXT66 predicts exactly zero. Develop that
missing dependency now; do not add another native dose/region grid.

## One architecture change

Keep the complete NEXT66 local conditional scalar as the nested feature prefix.
For each token use its frozen original-source U8/scale coordinates z_t and
p_t=2*t/(T-1)-1, t=0..T-1. Define

    b_t = concat(z_t, p_t*z_t)              # 16 dimensions, retains a position moment
    v(H) = mean_t b_t - mean_original_source(mean_t b_t)
    F_joint = F_local + .5 * v.T @ M_q @ v  # symmetric M_q, 136 coefficients

No whitening or extra between-author scaling; use the inherited token scales.
Diagonal coefficient features are .5*v_i²; off-diagonal are v_i*v_j. M is not
constrained PSD: reinforcing and cancelling interactions are both allowed.
The local scalar has1169 coefficients; joint1305. Both refit their coefficients
against identical source views/targets. Frozen NEXT66/original-reader results
remain separately reported. Application takes only H16/layout/frozen GRAIN inputs.

BODY gradients, fullL and all directional effects derive from the SAME scalar.
For disjoint tokens the new mixed derivative is J_j.T M_q J_i / T²; the input
dependence changes, not just the coefficient penalty. Query heads remain separate.
No fixed early/late partition is required by the predictor; it is an experiment
partition only. A sixteen-dimensional summary is a bounded approximation, not
proof that all nonlocal reading is low rank.

## Source material and objective

Same four source authors and two queries. Four views per author: original,
NEXT63 last-shifted, NEXT67 front-plus and front-minus. Reconstruct changed inputs
from frozen design; verify hashes/byte invariance. No new native calls or humanY.
Current/new8 g or L cannot enter fitting or application. Source views are not
independent authors; donor-shifted targets preclude source leave-one-author-out.

For each query, use these fixed source-normalized terms:

1. Equal-author/equal-view score and coordinate-density losses, inherited score/
   density scales. Coordinate density k=T*(g@U)*scale, computed for all BODY rows.
2. NEXT66 original-versus-last-shifted half-difference density loss with its
   unchanged source energy normalization.
3. NEXT67 plus/minus half-difference coordinate density restricted to unchanged
   B, source mean squared target energy normalized to1. Do not include changed A
   in this particular term. Stop as unsupported if this energy <=1e-12 times
   original density_scale²; do not replace it with a winner factor.
4. Ridge alpha=.01 for every non-intercept coefficient, no search.

Every term has coefficient1, known before outcomes. Include the identical B loss
in local control: its prediction there is zero, so its loss is an additive
constant. Independent objective reconstruction and nested Gram prefix checks
must distinguish expressivity from changed fitting/materials.

## Bounded execution and decisions

- Four solves: local/joint x two queries, max180CPU seconds. No MLP training,
  nonlinear optimizer, retraining old controls or alpha/rank sweep.
- Freeze implementation before runs. Save source views, objectives, full scalar
  and gradient predictions before joining any evaluation targets.
- Primary mechanism output: source B half-difference SSE/zero, all four fixed
  GRAIN effects and full-gradient vs coordinate-gradient distinction.
- Development utility: existing exposed new8 fullL RMSE/r and full/allocated
  GRAIN-action error vs matched local and existing original-material reader.
  Source mechanism acquisition is not new8 or questionnaire accuracy.
- Report any nonzero readable B response and its error; improvement over zero
  and matched control advances development. Keep predictive replacement a
  separate decision based on wholeL and all-action errors, not one factor.
- If the term remains ineffective, record the actual representational restriction
  and reconsider nonlinear/normalized selection; do not automatically run a
  sequence of ridge/feature-capacity variants. No fresh author panel to pad a table.

## Files and tests

New module `grain_research/vector_mechanisms/coupled_pooling.py`, staged runner
`scripts/run_grain_coupled_pooling.py`, focused tests, aggregate report/theory.
Private outputs `results/grain_vector_mechanisms/logic_coupled_pooling_v1/`;
public aggregate `reports/grain_vector_mechanisms/coupled_pooling_v1/`.

Tests: scalar/gradient finite difference, mixed derivative symmetry, independent
normal equations, same-view local prefix, fixed source centering, forbidden
target reads; full cached arithmetic replay without new native calls.

This tests the first coupled readout, not an assertion that a small observed
cross-region effect explains previous whole-model failure. H16 still requires
the original encoder. Release/scorer/lockbox/MEPS rules unchanged; goal ACTIVE.
