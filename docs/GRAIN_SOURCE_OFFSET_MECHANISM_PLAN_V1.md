# NEXT56: source intercept versus non-intercept correction in the whole function

2026-09-10. Planned, not executed. One mechanism test following NEXT55; no new
fixed-basis fit family. Goal: explain the whole-function relocation and decide
whether a source-defined ingredient preserves usable response, not chase a local r.

## Why this is the next question

NEXT55 crossed cached gradients locates response loss in state relocation rather
than tangent transport.99.95% of the correction energy is token-constant within
material. That is not necessarily the learned source intercept: non-intercept
features can have a material mean. Test the exact coefficient, not a post-result
current-target mean shift. This is a within-model stitching/functional-compatibility
question, not a proposal to strip common themes from author text.

## Fixed functions and compatible derivatives

Keep source4 leave-one-out/new8 all4, same12 materials/two queries/four GRAIN
directions, H16/layout/observer/c/actual suffix22–end. Use only the frozen NEXT53
joint pointwise source maps. No source refit, additional authors, human Y or query grid.

The map's features are [p,1,s,s^2], with fixed source center/scale and layout.
Bias index is `coef.shape[0]-3`; assert its center=0, scale=1 and feature=1.
With B the frozen coefficient matrix, b=B[bias], define

    V(X)=Z(X)B - 1 b^T
    I(X)=X;                       DI[D]=D
    O(X)=X+1 b^T;                 DO[D]=D
    Vmap(X)=X+V(X);               DVmap[D]=D+DZ[D]B
    F(X)=X+1 b^T+V(X);            DF[D]=D+DZ[D]B.

F is the existing joint map. Reuse frozen NEXT55 I/F scores, gradients and tau,
do not rerun them. Only O and Vmap receive new actual suffix evaluations. The
remainder includes position and content features; do not call it pure personality.
All inputs/derivatives are computed from allowed current H16 and fixed source
assets, never from current original pre22/gradients/score/tau. Convert at the same
FP32 observer boundary as NEXT55. Save full candidate gradients for cached follow-up.

## Execution bound and validation

- 48 new candidate forward/reverse pairs (12x2x2),2 JVP+4 central-FD forwards.
  One gradient contracts all four directions; checks at first source/N/O/F01
  and first new8/N/Vmap/F01. Epsilon=.05 and tolerances match NEXT55.
- 120s measured native cap,60s CPU; loading recorded separately. Stop on cap/error,
  preserve partial output, no repeated model run merely to complete a matrix.
- Prediction-stage target-read denial probes and exact source hashes as NEXT55;
  freeze all candidates before opening evaluation targets. One cache-only replay.
- Test exact F=O+Vmap-I and the corresponding derivative identity, source bias
  index/layout invariants, analytic/finite-difference derivative, and query pairing.
  The **states** add exactly; do not assume their nonlinear suffix outputs add.

## Analyses and decision, all fixed before outputs

Report all source4/new8 query and factor rows, including weak/negative ones:

1. Whole L: r, mean error, SD, slope, RMSE and SSE/source-constant baseline.
2. Whole tau: r, RMS, signs, SSE/zero and per-author raw N-minus-confirmation
   contrasts. Same utility rules as NEXT55; a contrast benefit stays contrast-only.
3. Ingredient differences and interaction for both L and tau:
   offset=(O-I+F-Vmap)/2; remainder=(Vmap-I+F-O)/2;
   interaction=F-O-Vmap+I. Use exact signed error accounts, not causal percentages.
4. Gradient norms/cosines to I/F and tangent norms; source-offset energy,
   remainder energy and signed cross energy. Compare true b to the observed
   material-mean correction without treating them as equal by definition.

If O largely reproduces the full-map gradient shift/loss and Vmap preserves the
identity response, identify the fixed offset as the chief relocation route. If
Vmap or the interaction dominates, retain that result and do not force the offset
story. Quantify the differences with the frozen tables, not a new binary cutoff.
If a candidate passes a score or response utility rule, retain its exact scope;
otherwise keep NEXT53's component but close this whole-state substitute family.
In either case next progress must change whole-function/readout dependency, not
reopen a rank/alpha/layer search on these12 people. No added target residuals.

## Deliverables and boundaries

Implement `scripts/run_grain_source_offset_mechanism.py`, a small reusable
decomposition module/tests, cache auditor, aggregate report/theory and dependency
update. Private `results/grain_vector_mechanisms/logic_source_offset_mechanism_v1/`.
Freeze primary code and this plan before outputs. No change to scorer, release,
old lockbox, MEPS, labels, clinical naming, API settings or remote branch.
No full-LLM-independence claim: both native H16 and suffix remain visible costs.
