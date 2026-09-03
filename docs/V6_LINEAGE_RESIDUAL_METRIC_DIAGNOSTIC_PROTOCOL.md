# GRAIN V6 Residual Metric Diagnostic: Post-Hoc Protocol

## Trigger and status

The frozen L1 lineage audit observed a discovery-condition residualized
four-coordinate anchor vector with higher same-author AUC than its raw version.
This protocol is written **after observing that result**. It is therefore an
explanatory diagnostic, not a new confirmation or promotion gate.

## Question

Does the residual AUC increase coincide with, or diverge from, the scalar
split-half reliability metric used by the older C2/PRED-1 line?

On the same J1-compatible technical views and the exact same frozen four anchor
coordinates, report for each coordinate `c`:

\[
r^{\mathrm{raw}}_c = r(X^L_c, X^R_c),\qquad
r^{\mathrm{resid}}_c = r(\widetilde X^L_c, \widetilde X^R_c),
\qquad \delta_c = r^{\mathrm{resid}}_c-r^{\mathrm{raw}}_c.
\]

The residual map is exactly the L1 discovery-only population subreddit mean
subtraction. Author bootstrap gives an interval for each `\delta_c`.

## Interpretation

- AUC and scalar reliability may move in different directions because AUC uses
  the geometry of the full four-coordinate vector, while each `r_c` retains
  one coordinate’s population variance.
- This diagnostic cannot establish which object is more psychological. It only
  resolves whether L1’s apparent conflict is a metric/estimand divergence.
- No labels, raw-text artifacts, hyperparameter search, or report-driven model
  changes are permitted.
