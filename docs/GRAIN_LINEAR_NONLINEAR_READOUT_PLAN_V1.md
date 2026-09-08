# GRAIN M22: linear, nonlinear, and conditional readout research

2026-09-08. User-requested extension of the delivered reference-function framework.
Mainline: real author text -> vector evidence -> trait judgment -> a reusable
mathematical readout. This is not a new personality benchmark tuning campaign.

## Decision and finite execution

Does the remaining teacher-linked information become accessible by changing the
readout on the SAME vectors, or is richer query-conditioned representation needed?

M22 is a cached CPU comparison, no new LLM inference, API, LoRA, questionnaire,
bridge, MEPS outcome, or lockbox access. Use all existing authorized D1 source
authors: T/F 2328 and neuroticism 213. Reuse the three-view cue-mask E5 mean and
mean-plus-query-weighted pools. Also compare the existing M10 query-state packet
(T/F 64, N 213) with its IDENTICAL-author E5 controls and identical teacher target.
M10's teacher is separately recorded; do not mix it with the earlier D1 observer.

All models share training-only standardization and PCA at most 64 dimensions.
Each author is one row; all three views are averaged. Five outer author folds use
the existing `s4_outer` assignment. Inner three author folds select ONLY teacher
MSE. Inner training refits PCA/scales, including residual-head training. The finite
grid is Ridge alpha 1/10/100 and 100/300 trees, learning rate .04, depth 1 or 3.
No outer-test early stopping, sample selection, sign flips, or label optimization.

| Reader | Question |
|---|---|
| Ridge | Which relationships are affine in these coordinates? |
| LightGBM stumps | Do nonlinear single-coordinate effects help? |
| LightGBM depth 3 | Do coordinate combinations help beyond stumps? |
| Ridge + stump residual | Are missed marginal effects useful? |
| Ridge + depth-3 residual | Are missed joint effects useful? |

The residual learner fits the training residual of the fitted Ridge. Validation
and test residuals are never used for fitting. This is stagewise supervised
prediction, not an orthogonal or uniquely identifiable decomposition of the LLM.
Stump additivity is relative to the training PCA basis, not original token axes.
Tree-versus-stump differences compare finite estimators, not equal-capacity causal
ablations. Do not call a negative tree result proof that no nonlinear signal exists.

Outputs: manifest/input hashes, saved fold models, private OOF before outcomes,
selected configurations, teacher fidelity/MSE, author-paired CI against Ridge,
external r/AUC, exact teacher/reader/remainder label-covariance accounting,
and a concise public aggregate report. Existing outcomes are exposed discovery
data, not fresh confirmation. No method is selected by those outcomes.

Bound: 20 minutes fitting plus cache loading/reporting, two CPU threads, no GPU.
Pause with partial artifacts at fold boundaries if exceeded. One fixed grid only.

## Theory and subsequent tasks

1. **M22 now:** the fixed comparison above; retain linear successes and local
   nonlinear gains by query rather than seeking a universal winning model.
2. **M23 next:** use a useful reader to localize conditional effects. Specify
   `s(x,z)=beta0(z)+beta(z)'x`, where z is measured context or query-state information,
   not the answer label. Compare with additive effects and conditional interactions
   on paired existing material. An improvement decides which vector groups to edit.
3. **M24 then:** compare predicted mixed finite differences with the fixed LLM's
   existing paired edits, followed only if useful by a bounded new intervention.
   Integrated Hessians or finite mixed differences localize the interaction; tree
   feature importance alone is not a neural path.

The full loop remains: discover a readable relationship -> localize its vector
support -> manipulate that support -> retain a mathematical approximation. AUC
alone cannot finish this loop. Avoid extending model grids simply to beat a score.

## Primary literature and what is borrowed

- Park, Choe & Veitch (2024), [linear representation and geometry](https://proceedings.mlr.press/v235/park24c.html):
  concept directions/probing/steering and metric dependence. Linear representation
  can be produced or used by nonlinear computation; personality is our application.
- Hastie & Tibshirani (1993), [varying-coefficient models](https://doi.org/10.1111/j.2517-6161.1993.tb01939.x):
  coefficients depend on other variables; foundation for M23, not proof of our routes.
- Janizek, Sturmfels & Lee (2021), [Integrated Hessians](https://www.jmlr.org/papers/v22/20-1223.html):
  pairwise interactions in differentiable networks; M24 inspiration, not implemented
  by calling a tree learner or computing its gain importance.
- Friedman (2001), [gradient boosting](https://doi.org/10.1214/aos/1013203451):
  stagewise function fitting; squared-loss residual learning in M22.
- Ke et al. (2017), [LightGBM](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html):
  efficient boosted trees. Optional research dependency, not a released GRAIN scorer.

Linear/nonlinear decompositions require a fixed representation, reference,
function class and fitting rule. Coexisting predictive terms are a working
hypothesis about accessible information, not necessarily two independent circuits.
