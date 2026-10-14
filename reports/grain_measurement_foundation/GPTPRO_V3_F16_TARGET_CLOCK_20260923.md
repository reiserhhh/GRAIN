# F16 explicit target-clock recovery

## Result and production implication

F16-R01 is implemented and run. Evidence loss no longer silently changes the
requested current-state time or the four future observation times. A new adapter
wraps the existing T2/T3 bundles; the frozen scorers, source fits, norms and previous
results remain unchanged. This closes the numerical target-clock subtask, not
semantic evidence qualification, F16 parent acceptance or the full research goal.

The distinction is consequential even without changing a model. On the six fixed
simulation authors below, removing observations44..47 and propagating to the
requested cutoff47, rather than relabeling the posterior at43, changes T2 by a
mean absolute .307977 response units and T3 by .032130. These are differences
between two target times given identical retained evidence, not improvements in
prediction error or estimates of human effect sizes.

## Formula and inputs

Let t be the last retained observation, T the requested cutoff and d=T-t. For
each source-conditioned personal parameter draw, the declared core state law is
`s_(k+1) = rho*s_k + epsilon_(k+1)`, with innovation variance Q. Its state moments
are propagated without a measurement update:

`m_T = rho^d*m_t`

`V_T = rho^(2d)*V_t + Q*(1-rho^(2d))/(1-rho^2)`.

This happens **before** integrating over personal and source parameters. Moving
only an aggregate score would discard dependence between rho, m and V. T3 then
uses the unchanged (+1,-1,+1,-1) plan at T+1..T+4 and retains shared future
innovation covariance. Empty history uses the declared stationary source prior;
it is marked prior-only, with no invented zero responses or fake last event.

The adapter accepts one author/episode, actual increasing integer event times,
finite observed responses and an explicit nonnegative target_time. Future
observations, duplicate events, mixed episodes, extra truth fields and wrong
reader/world hashes are rejected. Missing events are omitted, not renumbered.
The core schedule remains z in {-1,+1}; unknown real-world time units are not
silently converted into these discrete model steps.

The original complete48 source reference is retained and labeled. We do not
create missingness-specific norms from the application group or inherit the
old full-prefix calibration gate for a new missingness pattern.

## Frozen execution and checks

Protocol: `configs/grain_f16_cutoff_v1.json`, written before this application run.
Six fixed authors are the first/last archived application IDs in each of the
three existing seeds. Each is scored under both T2 and T3 and six masks:
full48, internal12..19 missing, trailing44..47 missing, both masks, all missing,
and only the first observation. Thus72 objects are **not72 independent authors**.

All200 existing source models and200 personal draws are used for each object;
source fitting is not rerun. Source indices0/100/199 and all200 personal draws
receive a separately derived joint-covariance check on every object. That dense
reference does not use the production filtering/propagation formulas.

| Check | Result |
|---|---:|
| Complete result objects in a separate cold process | 72/72 exactly equal |
| Unsupported numerical objects | 0/72 |
| Conditional parameter checks against dense covariance | 43,200 |
| Largest dense mean/variance discrepancy | 6.66e-16 |
| Largest archived complete-prefix score/variance/norm discrepancy | 2.78e-17 |
| All saved mixtures independently re-summed via raw moments | 72 |
| Largest saved-mixture discrepancy | 7.99e-15 |
| Input/result artifacts hashed by the post-run audit | 110 |

The related suite passed89 tests and11 subtests, including44 new explicit-clock
tests. One initial unit assertion required bitwise equality across different
variance-summation wrappers; the .0000000000000000555 discrepancy was checked
under the already-frozen numerical tolerance instead. No scientific threshold,
model or integration count was altered. Ruff passes on all new Python files.

Smoke had six score objects for one fixed author and3,600 dense checks. Full cells
took about16.3-16.5 seconds for T2 and9.1-9.2 seconds for T3 with at most3 CPU
workers. Both runs are terminal. No API, neural training, private text, MEPS,
questionnaire labels, new human data, remote work, lockbox or publication was used.

## Same-evidence clock effects

Each row describes six fixed authors, not a calibrated population estimate.
The variance change compares the requested target with the last-record target
using the same observed history and parameter draws.

| Mask | T2 mean absolute score change | T2 mean variance change | T3 mean absolute score change | T3 mean variance change |
|---|---:|---:|---:|---:|
| Complete | 0 | 0 | 0 | 0 |
| Internal missing, endpoint still47 | 0 | 0 | 0 | 0 |
| Last4 missing | .307977 | +.201589 | .032130 | +.006112 |
| Internal8 and last4 missing | .276371 | +.198362 | .027615 | +.006754 |
| First observation only | .169266 | +.081744 | .031194 | +.072691 |

Zero clock effect for internal missingness does **not** mean missing observations
do not matter: both compared functions already use the same reduced history and
the same last time. All-missing rows have no last-observation comparison and
are reported as null, not zero effect. No outcome truth was opened, so this run
does not reassess accuracy or interval coverage. The previous T1/T2 failed
calibration cells remain failed. The dense audit conditions on the produced
parameter draws; it does not independently refit every MAP/Hessian.

## Artifacts and commands

- Adapter: `grain_research/gptpro_foundation_v3/f16_cutoff.py`.
- Separate dense reference: `grain_research/gptpro_foundation_v3/f16_cutoff_audit.py`.
- Execution: `scripts/run_grain_gptpro_f16_cutoff.py`.
- Post-run reconciliation: `scripts/audit_grain_gptpro_f16_cutoff.py`.
- Full run: `results/grain_gptpro_foundation_v3/F16/cutoff_full_r1/`.
- Audit, paired CSV and hashes: `results/grain_gptpro_foundation_v3/F16/cutoff_audit_r1/`.
- Smoke: `results/grain_gptpro_foundation_v3/F16/cutoff_smoke_r1/`.

From the execution repository, using a new output root for a new revision:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/run_grain_gptpro_f16_cutoff.py --output results/grain_gptpro_foundation_v3/F16/cutoff_full_r2
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. '../project persona/.venv/bin/python' scripts/audit_grain_gptpro_f16_cutoff.py --run results/grain_gptpro_foundation_v3/F16/cutoff_full_r2 --output results/grain_gptpro_foundation_v3/F16/cutoff_audit_r2
```

Add `--smoke` with a new smoke output directory for the reduced case budget.
`--resume` requires the same code/config/input identity; it is not permission to
overwrite frozen results. No rerun is needed just to repeat this accepted check.

## Next original requirements

F16-R02 is next: eight concrete cards for each of T1/T2/T3, plus qualification and
versioned-history processing on the **same observable target**. The adapter only
handles the numerical consequence of absent/disqualified evidence; it does not
decide that a sentence is a quotation, a hypothetical or the author's experience.
F08 binary-polarity probabilities cannot be plugged into continuous T1/T2/T3
observations without a declared matching observation model. Existing history
helpers also assume particular endpoint conditions and event-noise contracts;
reuse their dependency semantics, not an incompatible score function.

Matched F12 transport, uniform construction/schema, F14 corpus support, remaining
F01/F15/F17 issues and original44+6 acceptance remain active. This packet makes
one required interface operational; it does not replace the full objective.
