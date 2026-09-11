# NEXT89: joint observations recover a missing common reading channel

2026-09-11. **COMPLETE / local functional progress.** Finite pass78;
primary `44a9b4a`. Twelve source authors, eight reused discovery recipients,
one120-coefficient solve, unchanged upstream programs/input/scales/kernel/weights.
No new neural call, human label, MEPS material or lockbox access.

## 1. What changed

The source teacher now supplies both absolute queries, not only their difference.
Each query has five functional observations: local scalar plus four anchored
GRAIN directional actions. These are not five personality traits. The queries
are neuroticism and confirmation-seeking, not Big5 N and C.

Both absolute local readings improve compared with the NEXT88 compatible12
reader. The old primary contrast is preserved algebraically, rather than being
rediscovered by a different regression. The additional information is the
previously omitted **paired mean observation**. This enables a same-material
interface that distinguishes two queries both reading high from both reading low.

Fixed primary arm `response_coordinates__source_value`, query-specific mode:

| Observable | Native12 error | Compatible12 error | Joint12 error | Joint vs compatible |
|---|---:|---:|---:|---:|
| Neuroticism scalar | .064608 | .027679 | .018768 | 32.19% lower |
| Confirmation scalar | .112405 | .241725 | .114631 | 52.58% lower |
| Scalar query difference | .154112 | .084376 | .084376 | identical |
| Scalar paired mean | .059820 | .060581 | .027153 | 55.18% lower |
| Total-action query difference | 7.276944 | 6.903260 | 6.903260 | identical |
| Total-action paired mean | 4.522091 | 4.530717 | 4.236389 | 6.50% lower |

Error is SSE divided by the same native target's squared energy, not an accuracy,
R-squared, or fraction of personality recovered. The raw paired scalar error
sum falls .801563 -> .439010 (45.23%). The total-action errors remain large;
ratios above1 mean worse squared error than the zero readout despite some strong
correlations. No uniform action-recovery claim follows.

## 2. Mean correction is not improved author discrimination

Neuroticism local scalar r changes .851715 -> .826894; confirmation .035944 ->
.071276; difference stays .493259. Mean scalar r .666241 -> .651307. Scalar
mean RMS .659745 -> .821116, native .823328. Thus the strongest gain is common
level/amplitude, not a new ranking result. Author-centered N MSE .022897 ->
.022389, confirmation .028453 -> .029606, mean .018004 -> .018327.

Native12's confirmation error .112405 remains slightly below joint12 .114631.
Do not select a per-query winner after seeing the evaluation outcomes.
Whole paired reader-vector error moves 1.225232 ->1.309957, even as functional
reading improves. Coordinate-vector resemblance and useful evidence reading
continue to be different objectives.

## 3. GRAIN directions and other compositions

All four factors are retained. In the fixed primary arm, joint teaching shifts
both query observations equally, so their difference is unchanged. The raw paired
action error reductions are:

| Coded observation | Compatible12 error sum | Joint12 error sum | Reduction |
|---|---:|---:|---:|
| F01 total | 1.454802 | 1.458248 | -0.24% |
| F02 total | 2.357877 | 1.876419 | +20.42% |
| F03 total | 1.225424 | 1.053209 | +14.05% |
| F04 total | 4.290360 | 4.441887 | -3.53% |

F02 compatible-coordinate r is .743995 for N, .456240 for confirmation. F03
absolute-coordinate r is only .112674/.123615, despite an improvement in raw
paired error. Confirmation scalar coordinate itself is negatively ordered
(r=-.322548, error1.176498); assembled absolute reading improves through the
combined offset and coordinate, not accurate coordinate recovery everywhere.
All20 coordinate rows,13 tables and22 older comparators remain available.

Total K+V query-difference error across every composition:

| Arm | Compatible12 | Joint12 |
|---|---:|---:|
| identity / identity value | 20.850818 | 20.142477 |
| identity / source value | 19.654383 | 18.984999 |
| response coordinates / identity value | 6.829233 | 6.803680 |
| response coordinates / source value | 6.903260 | 6.903260 |

The exact equality applies to the five constrained primary rows, not to every
other arm, K/V separately, or the shared-reader counterfactual mode.

## 4. Mathematical account, after the primary result

The new ten-output fit uses a scalar kernel times the identity, not a learned
between-query covariance. With fixed source rows, weights and regularizer,
linear target transforms commute with regression. The difference of the new
source targets is the old compatible contrast target. Their fitted difference
and its derivative must therefore remain the same.

`scripts/account_grain_joint_query_innovation.py` reads frozen arrays without
refitting. It verifies source target/fitted/coefficient contrast to8.61e-16,
new8 coordinate contrast to1.46e-16, five primary observation contrasts to
6.89e-15, and diagonal/correction contrast to5.40e-11. Both query observation
shifts agree to6.89e-15. Paired error gains equal the mean-channel gains to4.47e-15.

This is an explicit explanation of the gain, not independent empirical evidence
from multiple reparameterizations. Changing to mean/difference output coordinates
alone cannot improve this fixed regression. Three synthetic algebra witnesses
cover target identity, linear-transform commutation and paired error accounting.

The new common reader depends on material. Its derivative RMS is .00888128;
hardcoding it to zero, as an older constant-common helper did, would be wrong.
The new implementation includes that derivative in both shared and specific modes.

## 5. Execution and checks

- All six stages plus replay complete;13CSV and all private numeric arrays replay
  exactly. All22 older comparator rows retained within1e-12 serialization tolerance.
- One source solve /120coefficients. Source kernel condition31.765864;
  source residual4.68e-16, largest source fitted error2.87e-11.
- Each query frame rank5; source condition maxima19.571085/16.424827;
  recipient maxima19.128718/16.915417. No rank fallback or recipient tuning.
- 128compiled query calls per application,256with replay.64anchored FD max
  5.59e-10;64whole-function FD max3.91e-9, including the moving common channel.
- Canonical numeric CPU42.715075s; zero-fit account .012065s. Hashing, imports,
  tests/publication excluded. Both two-author cost gates passed; no live model job.
- 32primary focused tests before commit; final broader focused suite40passed,
  2known SWIG deprecation warnings. Ruff/format/whitespace checks pass.
- Pre-fit synthetic run exposed an unused-import reexport removal and an
  off-support Gaussian derivative fixture. Fixed before the first fit. A later
  test-file shebang lint issue was repaired; no empirical stage was rerun to
  hide a failed outcome. No empirical stage failure occurred.

## 6. Outputs, reproduction and next decision

Private package: `results/grain_vector_mechanisms/logic_joint_query_functional_reader_v1/`.
Public tables: [publication manifest](joint_query_functional_reader_v1/publication.json).
Post-result account: [account.json](joint_query_innovation_account_v1/account.json).
Theory: [joint observations](../../docs/GRAIN_JOINT_QUERY_FUNCTIONAL_READER_THEORY_V1.md).

```bash
../project\ persona/.venv/bin/python scripts/run_grain_joint_query_functional_reader.py --stage audit
../project\ persona/.venv/bin/python -m pytest -q tests/test_grain_joint_query_reader.py tests/test_grain_joint_query_innovation.py
```

Immutable output folders must not be rerun in place. A full fresh reproduction
uses `--output` with a new private folder, then prepare/teachers/models/predict/
evaluate/replay/replay-evaluate/audit in distinct processes. The publisher and
zero-fit account currently target the canonical result, and deliberately refuse
to overwrite their published directories.

Promote paired absolute observations as an effective same-material interface,
retaining negative factors and original contrasts. Stop output-basis/weight
sweeps. [NEXT90](../../docs/GRAIN_JOINT_MOMENT_NULLSPACE_PLAN_V1.md) inspects what
query-sensitive token configuration is still lost by the **current** BODY mean
plus four GRAIN cross-moments, reusing cached full-L gradients. That is a distinct
input question, not a claim that a local component's score equals the full LLM.
H16 remains neural-derived. Overall logic goal ACTIVE; finite pass78 complete.
