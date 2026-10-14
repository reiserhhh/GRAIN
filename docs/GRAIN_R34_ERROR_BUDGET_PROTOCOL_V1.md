# R34 cached error-budget reconciliation: protocol V1 (packet20 A)

Frozen 2026-09-26 JST before any reconstruction arithmetic.
- **Authority.** `NEXT_PACKET_20_ERROR_BUDGET_AND_HISTORY.md` Part A; `SUPERVISOR_CHECK_20260926_2143.md`;
  `SEMINAR_FABLE_17_DECISION.md` correction 1.
- **Runner.** `scripts/run_grain_r34_error_budget.py`.
- **Output root.** `results/grain_gptpro_foundation_v3/F16/error_budget_reconciliation_r1/`. It is created in x-mode
  and never overwritten.
- **Budget.** CPU ≤ 5 min for the cached arithmetic. Two cold replays of one person each, one per text scorer, are
  charged to the shared Part B integration budget. One process, single-thread BLAS. 0 neural calls.

## Estimand and what is being checked

Every archived readout is a finite equal-weight mixture over K components (source bootstraps, or paired
source/reader draws). For a scalar target with per-component mean m_k and within-component variance V_k:

    score = mean_k m_k,   total = mean_k V_k + Var_k(m_k)   (population variance, ddof = 0)

The within term V_k is split further only by the fields each ACTUAL scorer saved. The audit checks this identity
against the saved fields for the implemented mixture. It is not a psychological posterior and not a new
integration. QMC or Monte Carlo variation is numerical error, never a psychological component.

## Inventory and alias resolution (decided before reading numbers)

### Canonical historical original-world ensembles (decomposed)

| id | route | files | saved component fields | reconstruction |
|---|---|---|---|---|
| H_T1 | numeric T1, F04 L | `t1_source_full_r1/{seed}/cold_scores.jsonl` | per bootstrap: mean, conditional_variance (200); totals: personal_conditional, source | full per-component: E[cv], Var(mean), score |
| H_T2 | numeric T2, F05 core | `t2_cold_full_r1/{seed}/cold_scores.jsonl` | per bootstrap: mean, event_state, personal_parameter (200); totals: event_state, personal_parameter, source | full per-component |
| H_T3 | numeric T3 forecast mean | `t3_application_full_r1/{seed}/cold_scores.jsonl` | per bootstrap: mean, event_state = terminal_state + future_innovation + future_observation, personal_parameter; totals incl. source and total | full per-component, plus the state-carried vs future-innovation split; standardized variance = total / reference_sigma² from `bundle.json` |
| H_T1P | numeric T1 no-source plug-in | `t1_cold_full_r1/{seed}/scores.jsonl` | total only; `source_uncertainty_included = false` | identity not checkable beyond the interval; labelled single-term plug-in |
| H_X1 | text T1, OLD reader, 4 categorical arms | `t1_text_full_r1/{seed}/arm_scores.json` (hybrid_joint, nli_joint, hybrid_factorized, oracle) | personal_conditional_variance, source_and_reader_variance (COMBINED, paired b) | total = sum; per-b parts are not cached, so one cold replay |
| H_X1G | text T1, 2 numeric comparator arms | same file (naive_gaussian, eligible_continuous) | total only (a mixture was computed but its parts were not saved) | named gap |
| H_X2 | dynamic text, OLD reader, 4 categorical arms | `dynamic_text_full_r1/{seed}/{arm}_scores.json` | 5×5 joint components: conditional_state_future, personal_parameter, source_reader (COMBINED); joint mean and covariance; T2/T3 projections | joint covariance = sum; target = projection with TARGETS rows e0 (state) and (0, ¼, ¼, ¼, ¼) (future mean); one cold replay |
| H_X2G | dynamic text, 2 Gaussian reference arms | `dynamic_text_gaussian_r1/{seed}/{arm}_scores.json` (supersedes the numerical integration of these arms in dynamic_text_full_r1, retained) | as H_X2 | as H_X2 |

### Canonical current public serialized bundles (decomposed)

`serialize_full_r1/cold/{name}.scores.jsonl` for the 162 bundles in `bundle_index.json`:
- numeric_t1: 48 bundles, components conditional + pilot + source.
- numeric_dynamic: T2 48 and T3 48, components personal + pilot + source + state_future. The joint T2/T3 covariance
  is saved; its diagonal is checked against the companion bundle's readout for the same author.
- text_t1: 6 bundles, personal_conditional + source_and_reader (combined).
- text_dynamic: T2 6 and T3 6, conditional_state_future + personal_parameter + source_reader. `engine_output` joint
  components are checked as in H_X2.

Numeric mixtures integrate 200 source × 200 pilot calibration draws Cartesian (40,000). Their `plugin` estimator uses
a single calibrator. Text mixtures pair source b with channel draw b (200 draws, not Cartesian). Genuine component
counts are reported from each readout's support field, not assumed.

### Aliases (equality-checked, not decomposed again, never double counted)

| alias | canonical | check |
|---|---|---|
| `t1_text_full_r1/{seed}/application_cold_scores.jsonl` | H_X1 hybrid_joint | score, variance and components equal per author |
| `dynamic_text_full_r1/{seed}/application_cold_scores.jsonl` | H_X2 hybrid_joint | same |
| `dynamic_text_full_r1/{seed}/{naive_gaussian,eligible_continuous}_scores.json` | H_X2G (SUPERSEDED integration) | max difference reported; not expected equal |
| `text_transport_full_r1/scores/{seed}/{T1,dynamic}_changed_pilot_{mixture,plugin}.json` | serialized text bundles | score and components equal per author |
| `t1_transport_full_r1/{seed}/application_scores_before_truth.json` (16 serialized keys) | serialized numeric_t1 | score, variance and components equal |
| `public_contract_full_r1/numeric_T{1,2,3}_{seed}/cold.jsonl` (2-author contract demo) | H_T1 / H_T2 / H_T3 | components equal for matched authors, where matched |
| `*/scores.csv`, `metrics*.csv` | summaries of the above | not read |

### Non-canonical (inventoried, not decomposed)

- Transport comparison arms that were not serialized: text known_pool / uncalibrated / old_matched / oracle; numeric
  uncalibrated / known_inverse. These are study arms, not canonical bundles.
- Evidence-card and source-reference panels (`cards_*`, `source_scores*`). These are not application readouts.

## Outputs

- **inventory.json.**
- **per_person_components.csv.** One row per (set, route, seed, arm or world/mode/estimator, target, author):
  - score, total, and each component with its share of total;
  - reconstruction errors;
  - genuine component count.
- **summaries.json.** Per set/route/seed/arm/target:
  - mean and median share of each component;
  - maximum reconstruction error;
  - number of readouts;
  - interval approximation label.
- **alias_checks.json.**
- **replay_checks.json and replay_components.npz.** Per-b means and within variances from the two cold replays.
- **results.json, runtime.json, decision.md.**

## Tolerances (justified)

- **Saved-total identities.** Sums of ≤ 5 IEEE doubles stored with round-trip repr: absolute error ≤ 1e-12.
- **Per-component reconstruction.** Mean and ddof-0 variance over 200 round-trip doubles: ≤ 1e-12 absolute. The
  replays use the same code, so the same tolerance applies.
- **Aliases.** ≤ 1e-12, except the superseded Gaussian integration (reported, no tolerance).
- **Failures.** A failure is recorded per readout and never dropped.

## Labels

- **T2 current state vs T3 future mean.** These are different targets. T3's within term includes future innovation
  and observation noise, so ratios are never ranked across T2 and T3.
- **Combined terms.** source_and_reader / source_reader stay COMBINED. They are not split into source and channel.
- **Pilot.** The pilot component is calibration-draw variance under the declared transport world. It is not
  personal noise.

## Checksum bindings

| file | sha256 |
|---|---|
| `docs/GRAIN_R34_ERROR_BUDGET_INPUTS_V1.json` | `4109c2f406b07c6b7b1902bec6bbdcaec4a4e363370af482e93b9ce2d37fffa3` |
| `scripts/freeze_grain_r34_error_budget_inputs.py` | `7de620c05c7366dfd1503f680615e2f462b05a372bcff9cb8b247a941f348891` |
