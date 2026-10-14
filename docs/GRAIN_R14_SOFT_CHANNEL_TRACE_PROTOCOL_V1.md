# R14 soft-channel and event-trace companion: protocol V1 (frozen before computation)

Date 2026-09-26 JST. Authority: `SEMINAR_FABLE_09_DECISION.md` Packet B.
- **Runner.** `scripts/run_grain_r14_soft_channel_trace.py`, writing to
  `results/grain_gptpro_foundation_v3/R14/soft_channel_trace_r1/`.
- **Scope.** Zero neural, API or GPU calls. CPU, at most 2 workers × 1 BLAS thread, 60-minute cap.
  `same_object_channels_r1` is not modified.

## Reused, frozen

- The same bank and typed reader table (`R14/same_object_channels_r1/typed_reader_table.csv`, sha verified).
- The same rendering streams (`render_transport`), `generate_world` histories, oracle-observation source fits,
  `QualifiedHistory` settings, worlds (xor/iid), seeds (3), panels (complete/mixed) and candidates (B1, B2, M,
  source_mean, personal_mean).
- No new heads or fits.

## Channels (12)

- The 7 archived channels, recomputed for the trace.
- Soft versions of four channels, using the cached raw p and q:
  - hybrid_soft (NLI p, GRAIN q);
  - grainpole_soft (GRAIN-ridge p = .5 constant, GRAIN q);
  - lex96_soft (p, q);
  - lex80_soft (p, q).
- `grain_noinfo_reference`, a declared no-pole-information reference (below).
- Complete panel: q = 1, as in r1.

## The constant-.5 hazard (derivation, recorded)

- **Mixed panel (filtered mode).** A reading (p = .5, q) has likelihood L(y) = q·.5 + (1 − q)/2 = .5 for both
  states. It adds zero likelihood evidence, the pair update is pure prior propagation, and so **B1(p = .5) equals B1
  under prior propagation**.
- **But the archived `observe` still:**
  - adds q × (model-imputed joint) to the B2/M counts, gram and rhs;
  - adds q to the evidence mass;
  - adds q·.5 to the total, so personal_mean moves toward .5.

  This is **prior/model recycling, not newly observed person evidence.**
- **Complete panel (independent mode).** A p = .5 reading is written directly as the lag state, making the pair
  uniform, and adds uniform pseudo-counts. That is not prior propagation.
- **No-information reference.** It advances the event clock and propagates the prior law: pair ← filter_step with
  q = 0, so the likelihood is flat in both panels. It adds nothing to counts, gram, rhs, mass or total, so every
  statistic stays at its prior state.
- **Why a no-information reference is justified here.** The bank's exact balanced GRAIN feature collisions (150
  classes, every one with both poles) mean this channel has no pole information. This is not a general claim for
  arbitrary p = .5 readers.
- **Paired contrasts** (application authors):
  - hard1 − soft.5: thresholding bias in the degenerate channel;
  - soft.5 − no-info: model recycling;
  - no-info − oracle: the estimator's prior-only cost when pole information is absent.
- None of these is labelled "all representational loss".

## Event trace

- **Content.** One row per (world, seed, role, author, panel, channel, t):
  - event_id, material_id, text sha12 and address sha;
  - status, q_meta, material_pole;
  - p_read and q_read;
  - y and p_true (evaluation only);
  - the 5 candidate predictions made before the event;
  - evidence_mass and transition_evidence_mass (after the event), and prediction status.
- **Scope.** Validation and application authors, stored as zstd parquet.

## Checks (tolerances declared now)

1. **Reconstruction.** Author-mean Brier, log loss and next-p rebuilt from the trace equal the archived
   `metrics_by_author.csv`, read with round-trip parsing, within 1e-12 absolute for all 7 archived channels.
   - A mismatch beyond 1e-12 is archived and STOPs the packet before any comparison.
2. **Flatness.** For p = .5 and q ∈ {0, .3, 1}, `filter_step`'s likelihood is equal across states. In the mixed
   panel, B1(grainpole_soft) equals B1(no-info) at every event within 1e-15.
3. **Independent prior law.** No-info B1 predictions for 10 declared authors equal an independent numpy propagation
   of the source transition from the initial pair, within 1e-12.
4. **Render checks.** Application authors read only application materials and validation authors only
   pilot-validation materials. Status counts are reported.

## Analysis

- **Contrasts.** Paired per-author differences for the contrasts above, plus soft − hard for hybrid, lex96 and
  lex80, and soft − oracle.
- **Bootstrap.** 2,000 draws stratified by seed, with the same draws as r1 (seed 20260915). Pooled r is not used;
  the metrics are Brier, log loss and next-p.
- **Recycling mass.** The mean per-author evidence mass and imputed transition mass for grainpole_soft versus
  no-info.
