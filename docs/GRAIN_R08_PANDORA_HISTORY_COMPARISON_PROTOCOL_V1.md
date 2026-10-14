# R08 observational PANDORA history comparison: protocol V1 (amends the packet-14 manifest; frozen before execution)

Date 2026-09-26 JST. Authority: `NEXT_PACKET_15_HISTORY_AND_PRAGMATIC_READER.md` Part C; `SEMINAR_FABLE_13_DECISION.md`.
- **Amends** `F14/sequence_inventory_r1/next_packet_manifest_r08_psr.json` (sha 4ffd0fb7…). The corrections are
  listed at the end.
- **Runner and output.** `scripts/run_grain_r08_pandora_history.py`, writing to
  `results/grain_gptpro_foundation_v3/F14/pandora_history_comparison_r1/`.
- **Scope.** CPU ≤ 5 min, 0 neural calls. No text, labels or questionnaires; no 17.6M-row rescan; no MEPS.
- **What this is.** A bounded natural-process experiment. It does **not** by itself meet P03's controlled-context
  clause: later events are not do(C), and new-venue events are not interventions.

## Inputs and row identity

- **Event addresses:** `F16/event_addresses_r1/private_pandora_event_addresses.csv` (sha 61c98ca0…).
- **Cached keyed scores:** `grain_vector_mechanisms/all_plan_closure_v1/conditional_reference_r2/pandora/private_units.npz`
  (sha 184887cb…). Row r of the scores equals row r of `.../pandora/private_addresses.csv` (sha bb622a49…).
- **Row identity.** Event rows are mapped to score rows by the key pair (author_index, unit). The mapping must be
  one-to-one, with equal start/end offsets and equal venue; the run stops otherwise.
- **What the scores are.** 30 **IPIP-NEO-300 keyed semantic anchor scores**, the facets A1–A6, C1–C6, E1–E6, N1–N6
  and O1–O6.
  - Each is a normalized sentence-encoder vector projected on (mean of keyed+ items − mean of keyed− items).
  - Code: `grain_research/local_bilinear.keyed_directions`; items in
    `predecessor_integration_v1/material/ipip_neo300.json` (sha 0f4b023a…).
  - They are not raw embeddings, anonymous GRAIN factors or personality measurements.

## Roles and prequential scoring

- **Source authors (504, fitting only)**, sorted by author_index. Positions with index % 5 == 0 form the
  source-validation subset (≈ 101 authors), used only to choose α and the blend weights. The rest are source-train.
- **Held authors (120, scoring only).** Each author's 36 events are ordered by timestamp (no ties).
  - Events 1–25 initialize the history; events 26–36 are scored (1,320 events).
  - Each event is predicted **before** it is added to the history.
  - Population parameters are frozen, and every model sees the same history.

## Venue-choice models (categories: top-20 source-train venues plus OTHER)

- **Global:** the source-train marginal.
- **Static:** (n_c + α g_c)/(n + α) over the author's prefix, with α ∈ {1, 3, 10, 30, 100} chosen on
  source-validation.
- **Order-one:** a population transition P(next | last category) from source-train, with +1 smoothing.
- **Combined:** w·order-one + (1 − w)·static, with w ∈ {0, .1, …, 1} chosen on source-validation.
- **Novelty pinned before the collapse.** Whether the raw venue appeared in the author's prefix is recorded before
  the top-20 + OTHER collapse. Losses are split into known and novel raw venues, with denominators.

## Keyed-score models (30 dimensions, standardized by source-train mean and SD)

- **Global:** 0, the standardized source mean.
- **Static:** the author's prefix mean.
- **Static PCA5:** the prefix mean projected onto the top-5 principal components of source-train author means
  (36-event means) and reconstructed.
- **AR(1):** prefix mean + φ ⊙ (last − prefix mean). φ is fitted per dimension on source-train prequential pairs, by
  least squares through the origin, over positions 2–36.
- **Combined:** static + w·(AR − static), with w ∈ {0, .1, …, 1} chosen on source-validation.
- **Label:** bounded keyed-view next-event prediction.

## Outputs

- **Losses.** Venue log loss and Brier; keyed-score MSE averaged over 30 dimensions.
- **Uncertainty.** Paired author bootstrap: 2,000 draws, seed 20260926, deltas against static.
- **Descriptives.** Event-time gaps; losses split by gap tercile.
- **Row-identity check** and support counts.
- **No required win.** An order-one or AR(1) negative is a negative for those order-one baselines only, not a
  refutation of predictive-state theory.

## Corrections to the packet-14 manifest

- **What the columns are.** They are keyed IPIP-NEO-300 facet scores, not raw embeddings or GRAIN factors.
- **Prequential updating is explicit.**
- **The source-validation subset** is used for every choice.
- **Novelty is pinned before the collapse.**
- **Row identity is checked by keys, not by shape.**
