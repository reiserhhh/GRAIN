# M13: construct per-material functions before author aggregation

2026-09-08, frozen before M13 fitting. M12's per-author mean reader found TF b but
almost no N b variation. Its large reduction against old b_ref was mostly offset.

Question: Does averaging three views *before* learning hide a usable conditional
function? No new features, labels, participants, neural calls or hyperparameter grid.

- Reuse the exact M12 48-author split and four cached feature families.
- Each outer fold:18 training authors×3 original views,6 test authors×3 views.
- Same train-only standardized, trace-normalized kernel Ridge,alpha1.0.
- Fit measured b per view; predict each test view; then average per author.
- Author labels never enter fitting. Fold assignments are unchanged and author
  overlap prohibited. Do not count the72 views/query as72 independent people.
- Report author and view b fidelity, author within/between target variation, and
  fullscore`mean(b_hat_view)+PQ-b_ref`versusM12's unchanged author-first predictions.
- Freeze outputs before the exposed sourceY join. Keep all methods, primary fixed
  anchored_attention. A gain changes the order-of-operations component of the
  common construction, not a claim about inherent superiority of any trait query.
- No success: stop re-fittingb on this tiny panel; next investigate token
  configuration/conditional interaction rather than rank/alpha or label scoring.

The distinction is mathematical: for nonlinear f, E[f(X)] generally differs from
f(E[X]); learning a linear approximation at one level can change the estimand.
This experiment tests approximation before aggregation, not Jensen's inequality
as proof of personality inference. It preserves the native model target exactly.
