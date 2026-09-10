# NEXT57: one explicit score function and its GRAIN derivatives

2026-09-10. Pass46 COMPLETE; whole-system goal ACTIVE. Primary freeze `99eae77`.
The previous turn made progress through the NEXT56 source-offset mechanism.

## Finding and decision

An explicit H16 scalar can be evaluated without the current model suffix, and
its GRAIN derivatives can come from that exact same function. Source value plus
derivative teaching fits both objects closely. On the eight application authors,
however, the joint function reverses score ordering while retaining a specific
F04 response association (r=.639/.610). This is not an all-or-nothing outcome:
local direction learning and transport of between-author score differences separate.

The fixed pooled affine family closes: neither candidate passes both-query score
or response utility. Keep the value-only N score association and joint F04 response
as descriptive findings, but do not deploy this replacement or tune its penalty
on the same panel. NEXT58 tests the source BODY/final-state compensation geometry
to choose an input/function change, not another gain or rank sweep.

These are original fixed-model logits L and GRAIN derivatives tau, not personality
questionnaire accuracy. N=neuroticism; C=confirmation_seeking, not Conscientiousness.
Source4 leave-author-out and new8 all-source use the same12 reused materials;
new8 is disjoint from construction, not a never-exposed independent replication.

## Exact procedure

For each fixed reference, reuse the source-only NEXT53 pointwise mean mu/PCA32 U.
No new PCA. BODY mean and final-token state give64 contextual features:

    phi = [mean_BODY((X16-mu)U), (X16_last-mu)U]
    dphi_f = [mean_BODY(D_f U), D_f,last U].

The second directional block is exactly zero for these BODY-only perturbations.
Source-only column center/SD and target mean/SD define z and normalized targets.
SD<1e-8 becomes1; target SD is floored at1e-6. All64 columns stay.

    Lhat = ellbar + sL (a + w^T z)
    tauhat_f = sL w^T Dz[D_f].

Both objectives share inputs, normalizations and alpha=.01. Value-only minimizes
mean normalized source score error plus alpha||w||^2; joint adds mean normalized
source directional error, averaged over authors x4, with weight1. Bias is unpenalized.
Five references x2 queries x2 objectives=20 small augmented least-squares solves.
Original NEXT51 source L/tau, not failed NEXT55/56 outputs, are the only teachers.
Prediction cannot numerically read current or source teacher outputs; H16 is still
the original neural prefix. No current endpoint add-back, new neural/API/Y call.

## Full score and response comparison

New8, eight authors in every row. Score ratio=SSE/source-constant SSE;
response ratio=SSE/zero. Frozen utility requires both queries r>0 and ratio<1.

| Objective | N score r | C score r | N score ratio | C score ratio | N response r | C response r | N response ratio | C response ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Value | .300320 | .336404 | .941716 | 1.673011 | -.399989 | -.150845 | 1.337304 | 1.380929 |
| Joint | -.781558 | -.473019 | 5.994831 | 4.735134 | .108394 | .176637 | 2.318372 | 3.513198 |

Value-only N beats its source-constant score error by5.83% at this small-panel
scope. Joint source normalized derivative losses drop from1.778367/.378648 to
.001462/.000108, while source value losses remain7.26e-6/1.03e-6. This accurately
solved source objective does not recover application score ordering.

New8 score SD is .190/.747 for value,1.436/1.104 for joint, vs native .904/1.039.
Mean errors .179/.831 become .265/1.315. The N joint failure is therefore not
merely a lost common score origin: its slope is -1.242 and ranking is negative.

Source4 leave-one-out score r is -.898/-.680 for value, -.902/-.797 for joint.
Source4 response r is .030/-.072 and -.004/.112, respectively. All cells stay in
the tables; fitting diagnostics are not presented as leave-one-out performance.

## What the response improvement contains

All four new8 within-factor correlations are reported, rather than pooling32
directional cells as32 independent authors:

| Factor | N value | N joint | C value | C joint |
|---|---:|---:|---:|---:|
| whole:F01 | -.207948 | -.216693 | .094099 | -.192604 |
| whole:F02 | -.504076 | -.215360 | .512514 | -.308850 |
| whole:F03 | -.143340 | -.310933 | -.360409 | -.051127 |
| whole:F04 | -.251430 | .639420 | -.435630 | .609591 |

F04's positive joint ordering is a post-result localization, not a selected new
head or a named psychological trait. Its SSE/zero remains1.294/2.304 and signs
are5/8 and7/8. No new significance or external-validity claim is attached.

For response change d=joint-value and e=value-native, error accounting gives
DeltaMSE=2E[ed]+E[d^2]. On new8 N/C the linear terms are -.609280/-.337233
(locally useful orientation), but correction energies1.183434/.714697 dominate.
Total errors increase .574154/.377464. Score changes have positive linear terms
1.936938/2.053397 as well, so a common response shrinkage is not a demonstrated
whole-score remedy. No such held-selected correction was applied.

Raw N-minus-C response contrast r=-.458/-.093 for value/joint, SSE/zero1.575/1.701.
NEXT56 identity/non-intercept contrast .774/.811 remains useful **with a suffix**;
it is context, not a substitute teacher or achieved no-suffix result.

## Design information and next mechanism

The actual source design has65 coefficients including bias, score-row rank4,
direction-row rank16, joint data rank20 (LOO:3/12/15). Hence the observations leave
a45-dimensional data nullspace,50 in LOO; the positive penalty chooses a unique
solution but does not make its unobserved continuations uniquely established.
The regularized condition numbers48.5-230.4 and equation checks show this is not
a failed numerical solve. It does not prove that no nonlinear or token-addressed
function can work, or that input data lack personality information.

The last-token view predicts score levels but supplies no BODY-direction
constraints. A candidate mechanism is BODY/last compensation on the source
authors, then a changed relationship on application authors. This is a testable
next question, not a finding of this packet. See the [NEXT58 plan](../../docs/GRAIN_VALUE_TANGENT_COMPENSATION_PLAN_V1.md).

## Artifacts, verification and execution corrections

- 20 fits; measured prepare/fit CPU5.565s, application1.166s, replay1.166s, audit1.755s;
  sum9.652s, well within180s. Process startup, report I/O and tests are additional
  minor wall time, not native model compute. No long-running model remains.
- 32 permitted source target-field reads across isolated reference fits;33 primary
  forbidden-file probes plus5 replay probes, no unexpected denials. Current
  target numerical reads during fit/application=0; human-label reads=0.
- 24 prediction packets,48 scores/192 actions,24 score/96 action contrasts;
  17 CSVs replay exactly.192 same-function finite differences max1.2834e-10;
  normal-equation residual4.7977e-13; independent metric error4.4409e-16.
- 36 focused tests including6 new primary/audit tests pass;18 existing
  torch/Python3.14 warnings. Ruff passes. No full-suite claim.

Initial report assembly encountered macOS `._` AppleDouble files when globbing
status JSONs. Enumeration-only correction `1dd024a` precedes the completed
reports; original manifest/runner and partial report are retained privately.
`enumeration_amendment.json` binds original/current code and unchanged model/
prediction/replay freezes. No fit, prediction or scientific parameter rerun.
The independent auditor first used FP32 source reductions against the estimator's
explicit FP64 reductions; corrected the audit, not its tolerance or the model.
Source normalization now matches exactly; a separate regression test records this.

Private packet: `results/grain_vector_mechanisms/logic_complete_function_extraction_v1/`.
Public [13 aggregate tables and audit](complete_function_tables/inventory.json).
Code: `grain_research/vector_mechanisms/complete_function_extraction.py`,
`scripts/run_grain_complete_function_extraction.py`,
`scripts/audit_grain_complete_function_extraction.py`.

```bash
"../project persona/.venv/bin/python" scripts/audit_grain_complete_function_extraction.py
```

Use the cache auditor for the frozen packet. A fresh reproduction uses a new
`--output` with stages prepare, fit, predict, evaluate, then predict/evaluate
`--folder replay`; do not overwrite the current artifact. No scorer/release/
lockbox/MEPS/private-text export or remote push.

Method precedent: function value and derivative learning in
[Czarnecki et al., 2017](https://arxiv.org/abs/1706.04859). Our fixed affine
implementation tests that principle on this particular material/feature contract;
the paper is not evidence that this64-feature function must generalize.
Read [theory](../../docs/GRAIN_COMPLETE_FUNCTION_EXTRACTION_THEORY_V1.md).
