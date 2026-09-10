# NEXT45: query contrast is amplified, then revised by a negative MLP update

2026-09-10. **Finite experiment complete; whole-system goal ACTIVE.**

## Finding and next action

The query-dependent incoming direction found in NEXT44 is not built by steady
one-way amplification. New8's contrast grows strongly in blocks21-23, then is
partly counteracted, particularly by MLP24. That negative update appears across
all four new8 factors and in both cohorts. Retain it as part of the computation,
not a noise component to remove.

Next study one mechanism: whether MLP24's counteraction comes from transporting
an already different input through a similar operator, or from different local
gate/norm operators under the two query contexts. A tested exact recurrence now
states this distinction explicitly. No broader layer/rank/accuracy search.

## What was analyzed

Same4 source plus8 reader authors, two queries, four GRAIN directions, two fixed
shared readout modes. NEXT44 established equal BODY states/positions/directions.
Reuse NEXT42 initial and first17 signed updates, ending at attention25. MLP25
and later increments were excluded. No new model calls, fitting, labels or people.

The target T is NEXT44's transported query-difference component, not a human
trait score or full native model output. There are32 actions per new8 mode,
from8 people x4 directions. This is an explanatory analysis of explored caches,
not a new blind prediction experiment.

## Fixed-window accumulation

New8, own mode; final T RMS=.162922:

| Cumulative endpoint | Relation to final T, r | RMS | SSE/zero | Signed projected share |
|---|---:|---:|---:|---:|
| After block20 | .663679 | .031172 | .785197 | .125705 |
| After block23 | .913569 | .208040 | .291530 | 1.169512 |
| After attention25 | 1.000000 | .162922 | 6.19e-15 | 1.000000 |

The last row is telescoping parity, not an accuracy discovery. The block21-23
group alone contributes1.043807; the block24-to-attention25 group contributes
-.169512, counteracting the preceding accumulation. Pooled mode closely agrees:
cumulative RMS=.031168/.208662/.163415, middle projected share1.171188.

Source4 also overshoots in aggregate: cumulative RMS=.039124/.134055/.071220,
middle share1.289445. The exact shape and source of the contrast varies by cohort.

## Which operations contribute

| Stage | New8 own projected share | New8 pooled | Source4 own |
|---|---:|---:|---:|
| Attention22 | +.290940 | +.288533 | -.035430 |
| MLP22 | +.354303 | +.355716 | +.425493 |
| MLP24 | -.245628 | -.247120 | -.219469 |

MLP24 new8 own shares by F01-F04: -.442136, -.140542, -.158675, -.286217.
Its RMS=.095267 is the largest single-stage scalar contrast RMS in that group,
yet it counteracts final T. Large vector movement does not necessarily mean a
large positive contribution to the queried reading.

Attention and MLP aggregate own-mode shares are.567159/.432841 in new8, versus
.198129/.801871 in source4. Attention22 is not a universal positive source;
the stable observed MLP sign pattern should not hide this material difference.
All18 stages including zero initial, all factors and both modes are reported.

## Interactions, not independent information percentages

New8 own negative pair cross-energy fractions are -.424472 for MLP22/MLP24
and -.330464 for attention22/MLP24. Both triangles of the full moment/covariance
matrices are retained. Other cross-terms offset these: net cross-energy is
+.000561 with diagonal energy.025983, giving final energy.026543.

Source4 diagonal energy.018986 and cross-energy-.013913 give final.005072.
Signed shares can be negative or above one. They are descriptive projections
of observed derivative contributions, not personality-information percentages,
independent causal layer importance, or predicted effects of deleting a layer.

Adding the previous fixed reader-contrast term R back to reconstructed T retains
NEXT44's compiled-versus-native block25 contrast r=.982463/.976153 in new8
(own/pooled). This links the quantities without silently replacing the target.

## Mathematical development and literature alignment

For supplied full-state local residual Jacobians A, the exact two-query identity is

    delta_d_(s+1) = Abar_s delta_d_s + delta_A_s dbar_s.

The first term carries an existing difference; the second is the operator-contrast
contribution. Unrolling keeps the order of later operators and requires neither
inverses nor commuting matrices. Two synthetic tests verify these cases. This
does not claim that the saved last-position updates identify every full Jacobian.
MLP24 is positionwise, making it a tractable next local test; attention additionally
depends on other positions.

[Theory and derivation](../../docs/GRAIN_QUERY_TRANSPORT_FORMATION_THEORY_V1.md)
align this with [residual-stream contributions](https://transformer-circuits.pub/2021/framework/index.html),
[the distinction between additivity and global linearity](https://www.cs.columbia.edu/~johnhew/residual-stream-not-linear.html),
and the recent [response/Green-operator preprint](https://arxiv.org/html/2605.25225v2).
The latter supplies a useful mathematical viewpoint, not validation of personality
measurement. We keep the exact finite recurrence rather than posit a physical field.

## Execution and verification

- Primary implementation/analysis frozen in `11192be` before preparation/analysis.
-1728 signed stage values; numeric run.244946s and replay.246857s;0 new neural
  calls, scoring, backwards, fits, human labels or APIs. No live experiment.
-24 query vectors match cached pre-MLP25 tangents: max relative7.549e-8,
  max absolute5.048e-7, below unchanged original tolerances.
- All initial final-position tangents are zero. Scalar roundoff max4.735e-8
  stays within the measured vector-rounding bound; no extra unexplained term.
- NEXT44 T replays within9.888e-17;12 CSVs byte-identical on independent rerun.
  Independent stage-share error4.719e-16, moment-sum error2.211e-15;0 later-stage rows.
-93 focused tests pass with20 existing PyTorch/SWIG deprecation warnings; Ruff
  and whitespace checks pass. No full-suite claim, release/scorer/lockbox changes.

## Artifacts and next step

Private root: `results/grain_vector_mechanisms/logic_query_transport_formation_v1/`.
Contains prefix-only vectors, hashes, analysis/replay, numeric checks and audit.
Do not publish raw vectors, private individual rows or original author materials.

Public aggregates: [stages](query_transport_formation_tables_v1/stage_summary.csv),
[factor stages](query_transport_formation_tables_v1/factor_stages.csv),
[windows](query_transport_formation_tables_v1/window_summary.csv),
[factor windows](query_transport_formation_tables_v1/factor_windows.csv),
[operation types](query_transport_formation_tables_v1/kind_summary.csv),
[full covariance](query_transport_formation_tables_v1/covariance.csv),
[accounting](query_transport_formation_tables_v1/accounting.csv),
[target linkage](query_transport_formation_tables_v1/linkage.csv).

[NEXT46](../../docs/GRAIN_QUERY_COUNTERACTION_PLAN_V1.md) is planned, not executed:
24 pre-MLP24 primal forwards plus4 fixed boundary JVP checks, max120s native;
then a fixed2x2 local-operator comparison,60s numeric. No long queue or tuning.

```bash
PY="../project persona/.venv/bin/python"
# Prepare needs a new directory; preserve the completed packet.
"$PY" scripts/run_grain_query_transport_formation.py prepare
"$PY" scripts/run_grain_query_transport_formation.py analyze
"$PY" scripts/run_grain_query_transport_formation.py analyze --analysis-name analysis_replay
"$PY" scripts/run_grain_query_transport_formation.py audit
"$PY" -m pytest -q tests/test_grain_query_transport_formation.py tests/test_grain_query_response_recursion.py
```
