# R14 same-object channel comparison: protocol V1 (frozen before any fit; zero neural forwards)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_08_DECISION.md` Packet A.
- **Runner and output.** `scripts/run_grain_r14_same_object_channels.py`, writing to
  `results/grain_gptpro_foundation_v3/R14/same_object_channels_r1/`.
- **Status.** An exploratory mechanistic comparison on an already-examined finite bank. It is not a blind holdout,
  and not a human sample.
- **Resources.** CPU only, 2 workers, 1 BLAS thread each, 90-minute cap. Zero neural, API or GPU calls.

## Inputs (hashes verified before fitting; STOP on any mismatch)

- **Bank and cached readings.** `F16/text_transport_full_r1`: `reader_scores.csv` (sha fe5e33cb…), `materials.csv`,
  `grain_features.npz` (380×24 features plus raw head outputs).
  - Roles: source 80; pilot fit 60 / validation 40; application 200.
  - Statuses: actual/negated qualify (q=1); quoted/hypothetical do not (q=0).
- **Cached NLI readings.** `pole_readout` (pinned NLI pole) and `nli_qualification` (frozen actual-self hypothesis).
- **Historical GRAIN readings.** `grain_qualification` from the F11 comment-view head
  (`F11/grain_comment_reader_r1/readout_bundle.json`). Its pole output is constant .5 and is recorded but never used as
  a pole reading.
- **Transported lexical reader.** `F08/reader_recovery_r1/reader.pkl`, the canonical F08 lexical reader whose
  scores (sha 92efe929…) the F08 downstream froze. It is applied with `f08.predict_reader`.
- **Histories.** `generate_world` C2 xor/iid, seeds 20260915–17, 48 events, prefix 12; counts source 300,
  validation 100, application 200. The truth is used only by the separate evaluation.

## Readers and channels (p and q kept separately; q = 0 means unavailable evidence)

| Channel | Pole p | Qualification q | Form |
|---|---|---|---|
| oracle | rendered material pole | metadata q | exact |
| nli_soft | NLI pole | NLI qualification | soft |
| nli_hard | NLI pole ≥ .5 | NLI qualification ≥ .5 | hard |
| hybrid_nlipole_grainq | NLI pole ≥ .5 | historical GRAIN qualification ≥ .5 | hard (the actual old F16 route, named as a hybrid) |
| grainpole_grainq | fixed GRAIN-feature pole readout ≥ .5 | historical GRAIN qualification ≥ .5 | hard |
| lex96_transported | F08 lexical pole ≥ .5 | F08 lexical qualification ≥ .5 | hard |
| lex80_refit | lexical pole ≥ .5, refit | lexical qualification ≥ .5, refit | hard |

- **GRAIN pole readout.** Step one recomputes R21's exact equal-feature/opposite-pole collisions on the cached
  features and reports the identifiability bound. Then a single fixed readout is fit on source-80 only:
  StandardScaler plus Ridge(alpha 1). Raw and clipped predictions are kept. A constant or uninformative output is a
  valid failed-channel result.
- **Lexical refit.** The original F08 `fit_reader` (word 1–2 / char 2–4 TF-IDF, max 5,000 each; pole Ridge α1;
  qualification Logistic C1), fit on source-80 only. Vocabulary and every fitted state are source-only, and fit-input
  hashes are logged. Both lexical forms are reported; neither is selected post hoc.

## Rendering (the same object for every channel)

- **Canonical streams.** The F08 downstream stream `SeedSequence([seed, 8, world_index, author_index])` draws
  statuses (0–3), variants (0–3) and ancillary poles in the canonical order. For the complete panel, status is
  actual if the draw is even and negated if odd.
- **Material choice.** The transport bank holds more than 4 materials per stratum, so the material within the
  (role, pole, status) stratum is chosen from a separate declared stream,
  `SeedSequence([seed, 8, world_index, author_index, 1601])`. Draws are uniform over the stratum's material IDs,
  sorted.
- **Role mapping.** Application authors use application materials; validation authors use pilot-validation-40
  materials and are diagnostic only; source authors (secondary axis only) use source-80. Pilot-fit-60 is used only
  to fit the secondary-axis emission law.
- **Shared per event.** Latent history, status, ancillary pole, material ID and time position are identical across
  all channels.

## Downstream (primary)

- **Source fit.** The same canonical oracle-observation source fit (`fit_observed_source`) for every channel, with
  the source initial pair and source mean.
- **Estimators.** The `QualifiedHistory` candidates B1, B2, M, source_mean and personal_mean.
- **Panels.**
  - Complete panel (q = 1 for every channel, the F08 semantics): an oracle-eligibility diagnostic only.
  - Mixed panel: each channel's own q, in filtered mode, with all time positions kept.
- **Metrics per author (t ≥ 12).**
  - Realized-outcome Brier and log loss.
  - **Next-p squared error**, (p̂ − p_true)², with p_true = `c2_next_probability(y_{t−2}, y_{t−1})`. This is
    reported separately; probability MSE is not the Brier score.
- **Canonical replay gate (before the main run).** The same loop, with the canonical F08 lexical reader table and
  F08 rendering (variants 0–3, no extra stream), must reproduce `F08/downstream_full_r2/metrics_by_author.csv`
  exactly for seed 20260915, both worlds, both panels, and the oracle/soft/hard channels. Otherwise STOP.

## Secondary axis (single, declared): measured-emission-law source fit

- Hard channels only, mixed panel.
- Source authors are rendered on source-80 with stream `[seed, 8, 100 + world_index, s_index]` plus the
  material-choice stream `[…, 1601]`, then read by the channel.
- **Emission law.** P(r | y) is fit on **pilot-fit-60 only**: the `mnar_source` render law, with uniform averaging
  over each stratum's materials as the one adapter. It is fed to the `mnar_source` pooled EM and `fitted_from_counts`.
  No application truth and no calibration enter.
- **Disclosed limitation.** The E-step is pooled (not the author-level E-step), and M is known to be limited under a
  pooled E-step.
- **Reporting.** EM-source minus oracle-source per channel, estimator and world. The primary comparison does not
  depend on this axis.

## Analysis

- **Contrasts.** Paired per-author differences, channel − oracle, for each estimator × world × panel, on application
  authors.
  - 2,000 bootstrap draws stratified by seed, with the same draws across channels.
  - Intervals are conditional on the finite bank and the fitted artifacts.
  - World-specific contrasts are kept separately.
- **Layer contrasts.** They are reported where the F08-07 estimands match. They overlap and are not additive.

  | Layer | Contrast |
  |---|---|
  | render | oracle mixed vs complete |
  | qualification | hybrid vs nli_hard (same pole, q replaced) |
  | pole readout | grainpole vs hybrid (same q, pole replaced) |
  | lexical transport | lex96 vs lex80 |
  | source fit | secondary axis |
  | estimator | B1 / B2 / M under oracle |

- **Also reported.**
  - Reader-level accuracy and Brier per role for every channel's p and q (lexical solvability).
  - Evidence cards: one per status, plus matched examples tracing actual updates.
- **Outputs.** Manifests, typed reader table, source-fit bundles, per-author metrics, contrasts, identification
  checks, runtime, tests and common records. The records fail closed on any mismatch.
