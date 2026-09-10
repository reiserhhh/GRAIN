# NEXT21-23: action geometry and complete input dependencies

2026-09-10. Ordered exploratory continuation after NEXT19/20; all source
outcomes have been exposed previously. This is not a new blind benchmark.

## NEXT21: one reference-support calibration

Use exactly the four TF reference authors and twenty other source authors from
NEXT20. Reuse their136-position retained-state displacements, the saved two-body
integrated matrices and actual symmetric allocations. No new model calls.

First contract each reference's own matrix, the shared mean and its deviation
with that reference's actual displacement. Keep integration error separate;
report cross-terms, not disjoint psychological variance shares. Check the
near-zero mean displacement induced by the four-author reference against mean
actual action, since a homogeneous linear reader cannot generally fit a nonzero
mean target when its inputs sum to zero. Small FP32 reference rounding remains
recorded; do not manufacture another useful direction from numerical rank.

Then fit ONE correction centered at the existing full matrix g0:

    g = g0 + D.T @ solve(D@D.T + n*.01*s2*I, phi-D@g0)
    s2 = trace(D@D.T)/n.

Four reference rows only, no intercept, no centering, no per-coordinate scaler,
no rank/alpha selection, no questionnaire Y. Compare original g0, zero and the
reference target mean. Preserve all arms. Report20-author action MSE, signed
response and bootstrap intervals before linking Y. Keep the diagnostic B
assembly's exact baseline/body dependencies explicit.

Report singular spectrum and ridge soft support for evaluation displacements,
not an unqualified full-rank claim. The correction lies in observed reference
row space; orthogonal inputs retain g0. Verify primal/dual equality, zero input,
rotation/unit rescaling, held-target isolation, and normal-equation stationarity.

Theory anchor: [Scholkopf, Herbrich, Smola2001](https://alex.smola.org/papers/2001/SchHerSmo01.pdf),
especially biased regularization (Remark1). This is an instance of a standard
representer/regularized least-squares construction, not an invented theorem.

## NEXT22: direction, amplitude and final output on the fixed N pairs

Reuse the old24-author/four-fold, adjacent/distant endpoints and all four fixed
readouts. No new fit. For each pair type/fold, define v as the unit TRAINING
mean centered-logit interaction. Evaluation endpoints never determine v.

    a = v.T @ mixed;    residual = mixed-a*v.

Report actual energy and prediction error parallel/perpendicular to v, signed
amplitude fidelity, norm fidelity, and cross-fold direction agreement. Verify
the orthogonal energy identities and the old pair-mean control exactly.

For S(z)=softmax(z).values, calculate the analytic gradient at the cached additive
logit z_add. Compare gradient-projected mixed action with the exact S difference,
retaining the Taylor remainder. This locates whether vector errors matter in
the actual answer direction; it is not a new classifier or Y-selected metric.

Freeze these numerical components before source-Y links, and retain both pair
types/arms. Do not promote overall pooled r that only separates pair categories.
Do not refit a projected scalar target when linear-in-target Ridge would give
the same projection of the existing prediction; verify this identity in a test.

## NEXT23: complete conditional dependency contract

Trace current source code and cached exact quartets. Inventory actual input
requirements for raw text, prefix states, P/Q, body mean/configuration, retained
states, layout reference, local/joint action, output map and source reference.
Replay B and full four-cell closure independently from stored endpoints.

Separate GRAIN's population-referenced expression measurements from query
mechanism coordinates, and distinguish model evidence used for construction
from runtime inputs. Mark exact endpoint/oracle dependencies in conditional
assemblies. No release scorer/old lockbox change, no new MEPS or label training.

## Limits and outputs

One small calibration, CPU caches only, no neural/API calls or new dependencies.
All author-linked arrays remain ignored. Publish aggregate metrics, theory and
input dependency table, update31-edge atlas and goal cards, local commits only.
If calibration does not improve, investigate the measured support/conditioning
mechanism rather than start a new tuning grid. Integral approximation inherited
from NEXT20 remains explicit; no extra gradient collection to fix presentation.

[Active subspaces, Constantine et al.](https://arxiv.org/abs/1304.2070) motivate
respecting the sampling measure when relating gradients to functional variation.
Their gradient second-moment matrix and this displacement-weighted error metric
are different objects; neither yields psychological validity by itself.
