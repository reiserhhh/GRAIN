# S19: actual functional gradients reveal a shared response direction

2026-09-08. COMPLETE:24 source N authors, view0,96 forwards and72 backwards,
183.26seconds after loading. No questionnaires or new target fitting.

## Main finding

A mean gradient estimated from other authors reconstructs a held author's
pairing-removal effect at r=.8865, bootstrap95%CI[.7149,.9675], with squared error
.2147 of effect energy. The actual integrated gradient reconstructs the effect
almost exactly. This recovers an operative vector direction rather than merely
a coefficient that predicts another model's response.

| Readout | Effect r | MSE / effect energy | Sign agreement |
|---|---:|---:|---:|
| Author's integrated gradient | .99999977 | .000000453 | 1.0000 |
| Author's native gradient | .99954447 | .000985995 | 1.0000 |
| Other23 authors' mean gradient | .88650597 | .21467808 | .9167 |
| Rank1 gradient diagnostic | .90255331 | .18079294 | .9167 |
| Rank2 gradient diagnostic | .88846612 | .20639566 | .9167 |
| Rank4 gradient diagnostic | .94651989 | .10263708 | .9583 |

The rank diagnostics use the held author's gradient projected onto training-only
subspaces. They are not no-backprop predictors. Only the mean-gradient row
predicts without the held author's gradient. Rank selection was fixed, not tuned.

## Geometry and numerical checks

- First uncentered gradient second-moment direction:95.117% of energy.
- Median between-author gradient cosine:.959254; norm coefficient of variation:.185540.
- Median native/integrated gradient cosine:.9999648.
- Mean leave-author-out gradient projection error: rank1=.05053, rank2=.04303,
  rank4=.02912. This is gradient geometry, not explained human trait variance.
- Maximum path completeness gap:2.32e-6 score units; endpoint replay error0.
- All24 used the frozen three-node Simpson rule; no five-node retry was needed.
- No model weight required gradients or acquired parameter gradients.

The8-author pilot gave mean-gradient r=.9690; the completed24 estimate is.8865.
Retain the full result, not the more favorable resource-check number.
Curvature along this small head intervention is modest, but between-author
gradient differences produce visible reconstruction error.

## Explaining S18's reader-null effect

Contracting the actual integrated gradient with S18 selected/null vectors
reconstructs their observed effects: covariance geometry selected/null
r=.999968/.999999; Euclidean selected/null r=.998985/.9999997.
Thus beta^T d_null=0 and g_bar^T d_null!=0 describe different readouts, not a
contradiction. The learned C reader and final suffix have different covectors.

## Mathematical update and next step

For each context-dependent suffix F_u and its pairing displacement d_u:

E_u=F_u(h_u)-F_u(h_u-d_u)=g_bar_u^T d_u.

With g_ref the training-author mean, the finite transport error is
E_u-g_ref^T d_u=(g_bar_u-g_ref)^T d_u, up to quadrature error.
This uses the path-integral completeness principle of
[Sundararajan, Taly & Yan2017](https://proceedings.mlr.press/v70/sundararajan17a.html).
The gradient second-moment analysis follows the finite algebra underlying
[Constantine & Gleich2015](https://arxiv.org/abs/1408.0545);
the global active-subspace guarantees are not assumed for different F_u.

S20 freezes this mean on24 authors, scores cached vectors for all213, then
checks actual effects on the next24 and reports source associations on the
remaining189. E is the pairing effect, not S9's body-collapse contrast C.
This distinction is the next mainline question, not a reason to discard the
shared functional mechanism. No question-based outcome calibration is planned.

## Reproduction and provenance

Implementation6f22ed9, MPS export fix8b8d5f2. The r1 attempt failed before any
complete author because GPU Float64 conversion preceded CPU transfer. Preserve
r1:2 forwards,1 completed autograd before the export exception,0 successful-export
backward counter. The r2 retry corrected only that order and added an MPS test.

Private artifacts: results/grain_vector_mechanisms/functional_gradient_s19_r2/
contains manifest, pilot_008.json, per-author gradient NPZ/JSON, metrics.csv,
private_predictions.csv, private_s18_contractions.csv, summary.json and run_status.
Code, model, source, prompt and packet hashes are checked. Private data ignored.

Commands using the parent project's Python environment:
- scripts/run_grain_functional_gradient.py prepare
- scripts/run_grain_functional_gradient.py run --max-authors 8
- scripts/run_grain_functional_gradient.py inspect
- scripts/run_grain_functional_gradient.py run --max-authors 24
- scripts/run_grain_functional_gradient.py evaluate

Use a new output directory for a rerun; do not overwrite recorded pilot/analysis.
Focused tests: head-gradient6, transfer4 passed, including actual MPS export,
frozen tiny-Qwen directional finite differences, hook cleanup and author isolation.
The release scorer, T/F control, protected bridge/MEPS and lockboxes are unchanged.
