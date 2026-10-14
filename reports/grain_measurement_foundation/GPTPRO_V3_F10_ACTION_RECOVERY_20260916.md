# F10 action-space recovery: real effects, explicit charts and curvature

## Decision

F10 now has actual cached native effects, source refits for the direct text-contrast
model, and an explicit four-edit quadratic geometry experiment. This replaces the
old placeholder run as evidence for these subparts, not the entire F10 family.
No task card or overall theory-completion claim is promoted by this report.

Three distinct objects remain separate: whole-text score contrast, native donor
intervention effect, and predicted/compiled intervention effect. Recovering a
mathematical chart is not evidence of a native common attention chart.

## Real-cache recovery

Frozen `conditional_operator_v1/r2`: 36 source pairs, 72 application pairs, two
queries, 576 application intervention rows. Source ridge reconstructed with
augmented least squares agrees with cached direct predictions to 8.33e-17.
The frozen source-only shrinkage grid selects c=1 for both queries. It uses
in-sample source loss, not cross-validation. There are no source native
interventions in this cache; whole-text source contrasts cannot substitute for them.

| Query / target | Zero RMSE | Direct or compiled RMSE | Native donor RMSE |
|---|---:|---:|---:|
| E / whole-text contrast, direct | .092853 | .116747 | n/a |
| E / whole-text contrast, compiled | .092853 | .096926 | .083067 |
| E / donor effect, compiled | .033568 | .036537 | n/a |
| N / whole-text contrast, direct | .040504 | .042581 | n/a |
| N / whole-text contrast, compiled | .040504 | .040490 | .041097 |
| N / donor effect, compiled | .007029 | .007840 | n/a |

E donor versus full contrast r=.464280; E compiled versus full contrast r=-.260283.
Source signal transport failed in this old compiled operator, not in all possible
GRAIN methods. Native intervention effects and real-person prediction are different.

Bootstrap uses 2,000 draws per unit: paired persona-pairs and hierarchical
task-then-pair sampling. Direct models refit source scale, ridge and shrinkage
on every draw; native effects have fixed-fit application-only uncertainty.
There are only three source tasks and six application tasks. No human population
or broad task-universe coverage is established by these intervals.

- E compiled-minus-zero donor-target MSE: +.00020814;
  task/pair interval [.00002267, .00047579].
- E donor-minus-zero full-target MSE: -.00172158;
  task/pair interval [-.00669751, -.00001479].
- Direct E full-target difference is +.00500819, but task/pair interval
  [-.00038819, .04536083] is inconclusive. Do not replace this with the narrower
  pair-only interval to imply generalization across tasks.

## PV01 recovered, not silently rescued

All Euclidean/J-weighted summary metrics were reproduced exactly from
`grain_measurement_discovery_v1/pv01/r1/application_effects.csv`.
For E donor effects, weighted RMSE .033991 is smaller than Euclidean .038105,
but still above the zero reference .033568. N has the same ordering:
.007107 weighted, .008049 Euclidean, .007029 zero.

The six derivative checks contain twelve failed scale comparisons. Their old
runner labels every failure `PRECISION_LIMITED` without a discriminating test.
All six are now recorded as UNKNOWN in a separate triage artifact. FP32/MPS
precision is a candidate, not a demonstrated cause. Two step sizes, without
persisted raw +/- scores and gradient arrays, do not distinguish implementation
error, numerical precision and curvature. Original files remain untouched.
No new forward calls were made. Source gradients/fitted PV01 weight arrays were
not persisted here; source-refit uncertainty and a shared native edit chart are
therefore still missing. This is not a reason to block other executable families.

## Explicit four-edit geometry

Coordinates are content pole, subject, modality and audience, in a known synthetic
continuous chart, not an assertion that discrete text edits are continuously
parameterized. At h(e)=h0+A e and f_q(h)=u_q^T h+1/2 h^T B_q h:

    J_q(e) = (u_q + sym(B_q) h(e))^T A
    G(e) = sum_q w_q J_q(e)^T J_q(e)
    G v = lambda (C_source + 1e-6 I) v

Equal query weights, frozen rank two; numerical null rank uses
max(shape)*eps*s_max. A single query gradient produces an outer product, not a
scalar dot product. Invalid weights/covariances are rejected. Projection uses
the declared denominator metric; PCA uses its Euclidean orthogonal projector.

Three seeds, each 300 source / 100 validation / 200 application edit vectors;
source covariance only selects axes. The diagnostic uses matched edit draws
across two explicitly constructed worlds and no application selection. It is a
counterexample/geometry packet, not the complete card-level simulation contract.

| Application world | Zero MSE | PCA rank2 | Action rank2 | Full chart |
|---|---:|---:|---:|---:|
| Linear | .063868 | .064151 | 6.25e-30 | 0 |
| Null-direction curvature | .202569 | .064157 | .134678 | 0 |

The first world deliberately places large variance away from response directions:
variance preservation and action preservation differ. In the second world,
first-order null directions have quadratic response. Action rank2 still retains
all linear response, yet loses the curvature and is worse than PCA on total loss.
Neither geometry is universally superior. The exact example f(e)=e1+e3^2 at zero
has J e3=0 but f(.5 e3)-f(0)=.25. A local nullspace cannot be called irrelevant
over finite changes. This directly supplies the F15-D null-curvature certificate.

## Verification and artifacts

- 5,400 central-difference checks at .01/.001/.0001; maximum error 2.62e-12.
- 14,400 predictions replayed by scalar polynomial sums and metric-whitened
  least squares rather than importing the production scoring function;
  maximum difference 3.22e-15. Aggregate metric difference 1.39e-17.
- Generalized eigen certificates replay to 2.67e-15.
- Numerical audit is an alternate implementation by the same agent, not D02
  independent reconstruction. Bootstrap intervals are computed, not yet given
  a separate full-draw implementation replay.
- Focused tests cover chain rule, PSD/shape validation, rank/nullspace,
  second-order response, source isolation and source ridge reconstruction.

Result root: `results/grain_gptpro_foundation_v3/F10/`:
`cache_recovery_r1`, `uncertainty_r1`, `pv01_audit_r1`, `geometry_r1`,
`geometry_audit_r1`. Each keeps numeric rows and a summary; original r1 evidence
is not overwritten. Code: `f10_geometry.py`, cache/geometry/uncertainty runners,
two audit scripts and two focused test files.

## Remaining execution

1. A bounded predeclared native derivative diagnosis is still executable if
   raw interventions can be reconstructed: identify a precision/formula/curvature
   contrast within the allowed 100-call budget, not a head/rank sweep.
2. Native source shrinkage/refits require actual native source target/gradient
   recovery; current caches alone cannot close them.
3. Discrete four-edit signatures on a common material, full bootstrap replay,
   task-card contracts/hash/DAG integration remain open.
4. Continue F11 MTMM using the actual F08 readers and finite shared-method
   counterexamples; do not turn this packet into endless operator tuning.

No API, private original text, MEPS fitting, new model training, remote job,
lockbox, commit or push. This packet leaves no background experiment running.
