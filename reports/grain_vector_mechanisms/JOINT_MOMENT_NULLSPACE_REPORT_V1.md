# NEXT90: GRAIN moments recover useful actions but leave query-sensitive configuration

2026-09-11. **COMPLETE / finite pass79.** Primary `d537b2d`; portable auditor
`4017f95`. Zero new fitting, neural inference, human-label or MEPS access.

## 1. Main finding

The current GRAIN moment input is substantially more aligned with the four
measured native query actions than plain BODY mean. On the reused eight-author
panel, its visible signed action projection rises from .054106 to .450520 for
neuroticism and .056654 to .327163 for confirmation-seeking. Thus previous GRAIN
development retained useful information; this is not a reason to discard it.

The remaining configuration is still active, and cannot be restored by making
only a downstream f(phi) more nonlinear. It contains much of the previously
observed shared query gradient, plus a distinct query-difference component.
This is a property of a **particular frozen moment map**, not a verdict against
the whole GRAIN system or the token-resolved compiled-plus-lift program.

## 2. Same material, three nested input maps

Original4 and reused8,24native query contexts, fixed H16, BODY addresses and
GRAIN values. Not the additional eight source authors captured in NEXT88.
Cached full-L BODY gradients are bound to the same model/output contrast as
NEXT59. H16 states/positions/GRAIN/layout match the current NEXT89 input cache.
Reconstructed96 native actions differ from old target tau by at most4.7763e-6,
inside the unchanged atol1e-5/rtol3e-4 contract.

The maps are old BODY-mean PCA32, full BODY mean, and full BODY mean plus four
GRAIN cross-moments. The paired last context is held fixed. In the BODY space,
the current map has A=[1,centered G1,...,centered G4], P=A A+ and kernel A.T D=0.
Every material has rank5; new8 BODY sizes692–1127, median889; source4 sizes
794–1004, median859.5. Current A condition is1.95–2.53, no rank fallback.

The Euclidean/Frobenius metric and frozen GRAIN scores are explicit. This is not
a re-tokenized natural-text edit or a differentiation through the separate GRAIN
scoring system. Cached native gradients are diagnostic outcomes, not deployed
predictor inputs.

## 3. Functional action and gradient energy are different quantities

For native gradient J and original GRAIN direction D, both projector components
are retained. Native tau=<J,D>; visible tau=<PJ,PD>; residual tau=<(I-P)J,(I-P)D>.
The table reports sum(tau_visible*tau)/sum(tau^2), not an explained-variance or
personality-information percentage. Signed values may be negative or exceed1.

| Map, reused8 | N visible projection | Confirmation visible projection | Query-difference visible projection |
|---|---:|---:|---:|
| Old mean-PCA32 | .058362 | .029880 | .057433 |
| Full mean | .054106 | .056654 | .030004 |
| Current GRAIN moments | .450520 | .327163 | .356620 |

More geometric coverage need not improve each signed action: expanding PCA mean
to the full mean adds a negatively aligned component for some responses. With
current moments, residual action projections remain .549480/.672837 for N/C;
their squared-action ratios are .404034/.634493, with nonzero cross terms.

| Current-map gradient account | N retained energy | Confirmation retained energy | Difference retained energy |
|---|---:|---:|---:|
| Reused8 | 3.4181% | 2.2293% | 2.0739% |
| Original4 | 1.6911% | 1.5906% | 1.4906% |

A small gradient-energy projection can carry appreciable action along selected
GRAIN directions. Conversely, 96–98% residual gradient energy is **not** an
estimate of lost personality information. It describes a large conditional
BODY space and a small finite moment map in one coordinate metric.

The original4 visible action projections are only .008442/.050829, versus
.450520/.327163 on new8. Source and recipient alignment differ strongly even
though ranks, lengths and A condition overlap. This is descriptive support/
orientation heterogeneity, not a new population-level causal claim. Keep both
cohorts; do not report new8 as a universal coverage fraction.

## 4. Factor-specific and query-shared structure

Current-map residual signed action projections on new8:

| Factor | N | Confirmation |
|---|---:|---:|
| F01 | .404322 | .489969 |
| F02 | .648798 | .649066 |
| F03 | -.158813 | 1.008876 |
| F04 | .725482 | .807828 |

F03 provides a concrete reason not to use uniform positive weighting. N's visible
projection1.158813 is partly cancelled by residual -.158813; confirmation's
visible -.008876 is almost entirely offset by residual1.008876. The squared
response cross ratios are -1.284354/-1.407817. Orthogonal vector components can
produce opposing scalar effects under the same gradient.

| Paired native-gradient geometry | New8 | Original4 |
|---|---:|---:|
| Original mean author cosine | .657102 | .727653 |
| Residual mean author cosine | .655815 | .727146 |
| Residual paired common-energy fraction | .776751 | .861590 |
| Residual paired difference-energy fraction | .223249 | .138410 |

The original shared-query gradient was already known in NEXT39. This result
locates it relative to the **current** GRAIN moment map, rather than announcing
shared paths for the first time. It also retains a nonzero differential channel.
The common component can include general model processing and shared material;
it is not automatically a common personality trait or scale covariance.

Residual per-token squared sensitivity has only weak average correlations with
the four GRAIN scores on new8 (all views: absolute mean r<=.034575). Signed linear
moments are zero by construction (max norm2.74e-18); squared sensitivity need not
be, as a synthetic witness shows. These results give no support for a simple
"larger GRAIN score -> larger residual attention weight" rule.

## 5. What is mathematically established

Within the fixed map, Dphi[D_null]=0 implies D(f composed with phi)[D_null]=0 for
any differentiable downstream f. Here (I-P)J is nonzero in all24native contexts.
Its normalized direction gives the maximal unit-norm moment-null first-order
response ||(I-P)J||. This is a target-dependent local existence witness, not a
new predictor, finite-step intervention or psychological diagnosis.

The actual current feature implementation returns feature tangents below
2.73e-18 in the checked null direction. The whole NEXT89 assembly also uses
token-dependent compiled evidence outside phi; that program is not f(phi) alone.
The witness therefore does not assert that its full output is invariant.

## 6. Reproducibility, correction and files

- Six aggregate tables (2/24/120/18/32/8 rows),12complete private author units and
  six private numeric tables replay exactly. All aggregates independently rebuilt.
- Signed-action closure4.70e-15; paired energy5.56e-17; current null condition
  1.75e-16. Independent least-squares projection relative discrepancy2.02e-17.
- Old mean-PCA32's96rows reproduce NEXT59: native max5.11e-15, visible2.23e-16,
  residual2.67e-15. The stronger current map, not changed old targets, drives differences.
- Prepare2.145140, analysis2.601171, replay2.690829, canonical total including
  independent audit7.502983numeric CPU seconds. No model process started.
- 23primary focused tests;27including portability checks pass,2known SWIG warnings.
  Ruff/format/whitespace checks pass. One pre-freeze unused-variable lint repair.
- Original auditor rejected50macOS AppleDouble sidecars, and the publisher then
  correctly refused missing audit approval. Added auditor verifies sidecar magic,
  version and counterpart before excluding metadata. Numeric artifacts and the
  frozen primary were not changed or rerun. Failed-stage numeric work was not
  instrumented; rejection occurred before the independent numeric audit.

The CSV category `part=null` is a literal label, not a missing value. Read with
`pd.read_csv(path, keep_default_na=False)`; convert designated numeric columns
explicitly if processing genuinely missing metrics. The portable audit tests this.

Public [publication manifest](joint_moment_nullspace_v1/publication.json).
Private `results/grain_vector_mechanisms/logic_joint_moment_nullspace_v1/`.
Theory [current moment geometry](../../docs/GRAIN_JOINT_MOMENT_NULLSPACE_THEORY_V1.md).

```bash
../project\ persona/.venv/bin/python -m pytest -q \
  tests/test_grain_joint_moment_nullspace.py \
  tests/test_grain_moment_portable_audit.py \
  tests/test_grain_material_reader_field.py \
  tests/test_grain_pooled_feature_null_response.py
```

For a fresh reproduction use a new `--output` folder with the primary runner's
prepare/analyze/replay stages, then call `audit_grain_joint_moment_nullspace.audit`
on that folder. Completed audit/output folders deliberately refuse overwrites.
The old primary audit is retained for provenance but is not portable on this
volume. The publisher targets the canonical audited directory.

NEXT91 compares quadratic GRAIN moments with a saturated equal-GRAIN-cell
projection. This distinguishes insufficient factor interactions from genuinely
within-cell vector configuration before developing another reader. No labels,
factor naming or accuracy grid. [Plan](../../docs/GRAIN_GRAIN_CELL_RESIDUAL_PLAN_V1.md).
Finite pass79 complete; research objective remains open. Runtime goal was PAUSED
at closeout; NEXT91 was not started.
