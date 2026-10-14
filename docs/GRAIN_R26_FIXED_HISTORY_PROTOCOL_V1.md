# R26 fixed-history inference and isolated cost control: protocol V1 (packet20 C)

Frozen 2026-09-26 JST before any rescore or new run.
- **Authority.** `NEXT_PACKET_20_ERROR_BUDGET_AND_HISTORY.md` Part C; `SUPERVISOR_CHECK_20260926_2143.md`;
  `SEMINAR_FABLE_17_DECISION.md` correction 4.
- **Runner.** `scripts/run_grain_r26_fixed_history.py`.
- **Output root.** `results/grain_gptpro_foundation_v3/F06/fixed_history_inference_r1/`, created in x-mode.
- **Budget.** CPU ≤ 5 min. One process, single-thread BLAS. 0 neural calls.
- **No rerun.** The 12,800-run packet19 matrix is NOT rerun. The runner imports the frozen packet19 functions (bank,
  grids, priors, likelihoods, planners, `run_person`) unchanged from `scripts/run_grain_r26_reactive_head_to_head.py`,
  whose sha equals packet19 `runtime.json` code_sha256.

## C1. Fixed-history rescore (2,400 history records; no planner calls)

**Histories.** World REACT, mode naive, policies S / O / OC, planning targets T1 / T2, persons u = 0…399. The saved
item sequence `items` comes from `per_person_runs.csv`. The S planner ignores the planning target, so S histories
are identical across targets; this is checked and reported (400 distinct S histories counted twice).

**Response regeneration (data-generating side only).** From `latents_and_stream.npz` (τ, κ, ν, U) and the saved
items, with r_0 = 0:

    y_t = 1[U[u, t] < logistic(a (τ + κ c + r_t) + n ν)]
    r_{t+1} = .6 r_t + .20 art

Exactness check, required for every one of the 2,400 records:
- recompute the ORIGINAL naive posterior (r = 0 in the likelihood) from (items, y);
- est_T1, est_T2, est_CS, lo_T1, hi_T1, lo_T2 and hi_T2 must equal the saved values (tolerance 1e-12 absolute; NaN
  where the original was NaN);
- probe_count, r_sum and r_final must also equal the saved values.

A failure is recorded and that record is excluded from the contrast, never silently fixed.

**Rescore.**
- The same items and the same realized responses are used. The posterior uses the AWARE likelihood with the recorded
  own-action reaction r_t, computed from the planner's own items by the declared law.
- The rescore function receives only (items, y, policy). It gets no τ, κ, ν, U or truth, no future responses at
  time t, and no planner call.
- S keeps its scalar model (κ = 1, no nuisance) and its existing misspecification: est_T2 = mean(τ) + 2 and no T2
  interval (NOT_SUPPORTED).
- The current state CS keeps its definition: naive = mean(τ); aware = mean(τ) + r_8.

**Contrast.** Aware-rescore against the original naive estimate on the same history, metrics computed outside
inference against saved truth. The regime decomposition, exact for RMSE point differences:

    (aware online − naive online) = (aware online − aware rescore) + (aware rescore − naive online)

- **(aware online − aware rescore)** is the history/choice effect under a fixed inference family: the aware planner
  picks different items, so responses differ.
- **(aware rescore − naive online)** is the pure inference effect on fixed histories.

## C2. Isolated reaction-cost control (800 new policy runs)

**New arm.**
- World REACT, mode aware, policy OC, targets T1 / T2, persons 0…399.
- The ORIGINAL costs are .05 × artificiality and .05 × nuisance, with reaction-cost coefficient 0.
- Implementation: the packet19 `op_choose` is wrapped so that `react_cost` is always False, and the unchanged
  `run_person` is called.
- The same exogenous streams (τ, κ, ν, U) are used as the packet19 known-law OC runs (coefficient .10).

**Deterministic pairing check.** The 800 packet19 OC / aware / REACT runs are replayed with the unchanged functions.
Every saved field must match exactly (1e-12 on floats, exact on items).

**Cost-zero rule check.** With coefficient 0, the utility of every candidate item equals the original definition
without the reaction term (gain/scale − .05 art − .05 n).
- Checked item by item at the actual posterior states of the first 20 persons' new runs, against the original
  `op_choose` called with react_cost = False.
- Also checked by replaying the ZERO-world OC aware runs of persons 0…19 (both targets) with the wrapper against the
  saved rows.

**Contrast.** Cost-0 against coefficient-.10, both aware, same law, same inference family. This isolates the
policy-cost regime, not an individual causal state effect. There is no weight grid or search.

## Reported for C1 and C2

- T1/T2 bias, MSE and RMSE for the planning-target estimand; the other estimand is secondary.
- Coverage of the stated 95% posterior quantile interval where it exists (S T2 NOT_SUPPORTED).
- CS bias and RMSE, separate from trait τ. Final τ + r_8 is policy-specific; different policies have different CS
  targets.
- Probe COUNT (number of art = 1 items) and probe SHARE (count / 8), both labelled.
- Cost components: .05 × count, .05 × Σ n, and the reaction accounting .10 × ΣΔr, reported even when unpenalized.
- Paired RMSE differences with the original whole-person bootstrap: `default_rng([19101, 3]).integers(0, 400,
  (2000, 400))`, percentile 95% CI.
- A CI containing 0 is INCONCLUSIVE, not equivalence. No requirement that the aware or cost-0 arm wins. Preferred
  policy depends on the declared utility.

## Checksum bindings

| file | sha256 |
|---|---|
| `scripts/run_grain_r26_reactive_head_to_head.py` | `4a993e8e3bbe84169755bb7d1c244a87434be8f5409c05d822edddd4838408bb` |
| `results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1/per_person_runs.csv` | `c68e2416e552eb0981407e3320ec48787462b2bc72a85c0c9c7f5d38b2f4e932` |
| `results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1/latents_and_stream.npz` | `ba30f3de4a5e77f94ed3796b5b74bb76034d69f12f0b42775b1ed10b941aa46a` |
| `results/grain_gptpro_foundation_v3/F06/reactive_head_to_head_r1/results.json` | `0504fa36aa8556aad34d4e5d6c58305c4fb8ae7a9ba5d44337ec809db23dfcfc` |
| `docs/GRAIN_R26_REACTIVE_HEAD_TO_HEAD_PROTOCOL_V1.md` | `c9ae7870e30d853019029c13b9e5c5c796fa601c5daabe39d5ec04134d522e6c` |
