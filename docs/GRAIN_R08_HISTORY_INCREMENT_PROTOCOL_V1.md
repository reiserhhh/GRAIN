# R08 incremental history information on existing PANDORA caches: protocol V1 (packet22 B)

Frozen 2026-09-26 JST before any new score.
- **Authority.** `NEXT_PACKET_22_HISTORY_INFORMATION.md` Part B; `SUPERVISOR_CHECK_20260926_2347.md`.
- **Original clauses.** P02 pp.17–19; P03 pp.22–23. This experiment addresses only the NATURAL OBSERVATIONAL history
  leg.
- **Code.**
  - Module `grain_research/r08_history_increment.py` (bound).
  - Runner `scripts/run_grain_r08_history_increment.py`, which records its own sha at run time.
- **Output root.** `results/grain_gptpro_foundation_v3/F14/history_increment_r1/`, created in x-mode.
- **Budget.** ≤ 10 min CPU, one process, single-thread BLAS. 0 neural calls. No raw-text scan, no 17.6M-row rescan and
  no MEPS.
- **Privacy.** Per-author and per-event records are saved only as `private_*` files; reports give aggregates only.

## Question

Does the observed history contain useful incremental information beyond personal composition / the prefix mean? Is
that information in ORDERED history, or only in accumulated content? An exact negative in these estimators is a
valid answer. There is no parameter search to rescue it.

## Data, roles, identity (unchanged from packets 15–16)

- **Loader.** The pinned `run_grain_r08_pandora_history.load()`.
- **Roles.** `run_grain_r08_pandora_venue_oom.roles_and_dictionary()`: source-train 403 / source-validation 101 / held
  120 authors, disjoint.
- **Events.** 36 events per author. The prefix is events 0–24; events 25–35 are scored prequentially (1,320 held
  events).
- **Cache values.** 30 IPIP-keyed semantic projections (source-train standardized). They are NOT raw embeddings,
  anonymous GRAIN factors or validated personality scores.
- **Disclosure.** The same 120 held authors were used in packets 15–16. This is a new EXPLORATORY contrast, not an
  untouched confirmatory lockbox.

## Replay gates (tolerance 1e-12; stop on failure)

- Held venue global/static/order-one/combined log losses against `pandora_history_comparison_r1` and
  `pandora_venue_oom_r1`.
- Held OOM log loss, from the SAVED rank-20 source-fit operators (`oom_fit_source.npz`, no refit), against packet 16.
- Source-validation OOM mean log loss against packet 16's `results.json`.
- Held keyed MSE (global/static/PCA5/AR1/old blend) against packet 15.

## B1: one nested venue contrast

- **Baseline.** pB = .8 pStatic + .2 pOrder1, with α = 3 (frozen).
- **OOM.** Saved rank-20 operators with the EXACT packet-16 filter: clipping, floor 1e-4, reset and refusal rules.
- **Mixture.** pMix = (1 − w) pB + w pOOM. On a refused OOM event, pMix := pB (declared nested fallback). Refusal,
  floor and reset counts are reported.
- **Selection.** w ∈ {0, .05, .10, .25, .50, 1.0}, chosen on source-validation mean log loss over all scored
  validation events. Exact ties go to the smaller w. Then w is frozen.
- **Held scoring, once.**
  - pMix vs pB on all events, and original OOM vs pB on identical non-refused events, for log loss and Brier.
  - 2,000 paired whole-author bootstrap draws (`default_rng(20260926)`, as before).
  - Strata: raw-venue novelty measured BEFORE the top-20/OTHER collapse, and outcome class (top-20 vs OTHER). Each
    stratum's event count and contribution to the pooled delta are reported.
- **Scope.** w = 0 is a valid decision. It does not mean all dynamic information is absent. The OTHER class is not
  rare-venue prediction.

## B2: coupled semantic-history readouts

- **Target.** The next event's 30 keyed values, source-train standardized. These are observed text-view values, not
  latent truths or future behaviour.
- **Features.** At event t only events before t enter. m_t is the full prefix mean, last_t the last event, and r_t the
  mean of the last 4 events.
- **Two fixed linear residual predictors of y_t − m_t:**
  1. Coupled-last: x = last_t − m_t (30 dimensions).
  2. Coupled-history: x = (last_t − m_t, r_t − m_t) (60 dimensions).
- **Fit.** All source-train authors' positions 25–35 (same rows for both).
  - Objective: (1/n)‖Y − XB‖²_F + 1·‖B‖²_F. The loss SUMS the 30 output dimensions and AVERAGES rows.
  - Normal equations: (XᵀX + n I) B = XᵀY. No intercept beyond m_t, and no rank, α or window grid.
- **Blend.** Per model, ONE multiplier w ∈ {0, .25, .5, 1} on source-validation keyed MSE (mean over dimensions and
  events). Exact ties go to the smaller w. Both are frozen, and no winner is chosen on held data.
- **Reported.**
  - Both models against old static, AR1 and blend: paired author deltas.
  - The incremental-history contrast, coupled-history − coupled-last.
  - Per-facet held MSE and novelty-stratum contributions.
- **Structural control.**
  - At each prediction, permute ONLY the observed prefix, with seed `[22009, role(1 = validation, 2 = held), author,
    t]`.
  - Recompute last and recent-4 from the permuted prefix and keep m_t EXACTLY unchanged (original mean). Apply the
    same frozen coupled-history readout. No refit or reselection.
  - Validation and held effects are reported separately. If w_hist = 0 the control is exactly redundant, and this is
    recorded.
  - It is not an exchangeable-null p-value or proof of causal dynamics.

## Other checks

- **Leakage.** Changing a held future target must not change its own or earlier predictions (test).
- **Cold replay.** Saved coefficients, source mean/SD and selected weights must replay held predictions without
  refitting (1e-12).
- **Not claimed.** No accuracy gate. No claim of controlled do(C), raw-embedding/GRAIN equivalence or construct
  validity.

## Checksum bindings

| file | sha256 |
|---|---|
| `grain_research/r08_history_increment.py` | `bd50efb40a66f8275ca7f0ec6bfcb66b3ff76bddd8fc376b37ba1fdaa4210590` |
| `scripts/run_grain_r08_pandora_history.py` | `128f5b648cda42b0f71fd2292ccb659de77874f92181b6776c010b7d9f2e9b01` |
| `scripts/run_grain_r08_pandora_venue_oom.py` | `9ef4ef4689545418a8666a919bfca44c9bb1c96de1623952e514957055ce496b` |
| `results/grain_gptpro_foundation_v3/F14/pandora_history_comparison_r1/held_event_scores.csv` | `7e086a0ebfd5a097ea9f63320f3bd7e74207b20524e570598aea7ec357941199` |
| `results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/held_event_rows.csv` | `30149e9abd187cb9bf0e67be7749f7e383d4884928afac5bcabb34b7fe97ae94` |
| `results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/oom_fit_source.npz` | `cf22778cac9556afccfc9e21ff4ce2223c7b4b9c0b1a0271666e29da777bf8ec` |
| `results/grain_gptpro_foundation_v3/F14/pandora_venue_oom_r1/results.json` | `8b12433d5b52839cb495aedc8206c83f1bfd34c634b333c79d9a71e7166e9694` |
| `results/grain_gptpro_foundation_v3/F16/event_addresses_r1/private_pandora_event_addresses.csv` | `61c98ca01fc7acd067ef924611009b2e9ca2cf5d9888a4514a91f6cfb4a10a7e` |
| `results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora/private_units.npz` | `184887cb4962c8872b670557a988fbaf85e0bef696a97466cd9836f4a5d30a89` |
| `results/grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora/private_addresses.csv` | `bb622a49ee48fd282eb085885e1c031e9734e687a659c8fdb5997c500f09cddf` |
