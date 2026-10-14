# Packet-23 acceptance repairs: protocol V1 (packet24 A)

Frozen 2026-09-27 JST before execution.
- **Authority.** `NEXT_PACKET_24_SOURCE_CROSSED_ACTION.md` Part A; `SUPERVISOR_CHECK_20260927_0149.md`.
- **Runner.** `scripts/run_grain_packet23_repairs.py`. Cold scorer: `scripts/cold_score_grain_r08_raw_grain_bundle.py`
  (bound).
- **Output root.** `results/grain_gptpro_foundation_v3/F14/raw_grain_history_repairs_r1/` (x-mode).
- **Not done.** No fitting, no re-estimation, 0 neural calls.

## The three repairs

1. **Machine-readable hash chain.** material manifest → common manifest file sha → preflight manifest identity →
   cohort parent identity. The historical false gate stays unmodified.
2. **Metadata-only GRAIN comment order.**
   - Traverse only `vm05_full_source_cohort_r1/author_views.json` (the preflight source) and
     `big5_only_author_views.json` (the material source).
   - Read only author_id, the ordered comment_id, the view index and the timestamp. Body and all other fields are
     never accessed, printed, scored or exported.
   - Compare all 624 × 36 units with the pinned event addresses: event_key by unit, view_index and timestamp.
   - A mismatch STOPS GRAIN acceptance. Nothing is reordered.
3. **Adjunct bundle and fresh sealed process.**
   - The bundle COPIES the packet-22 keyed mu/SD, the frozen keyed directions, and the packet-23
     PCA/scalers/readouts/feature names, with provenance hashes.
   - A fresh subprocess (cwd /) under an audit hook may open only the bundle plus the declared numeric inputs: raw
     comment vectors, VM05 units, event-address timestamps and the held role list.
   - It recomputes the keyed targets from the raw vectors and scores the 1,320 held events.
   - Its outputs are compared with the packet-23 held losses. The earlier same-process coefficient replay stays
     historical.

## Checksum bindings

| file | sha256 |
|---|---|
| `scripts/cold_score_grain_r08_raw_grain_bundle.py` | `d583dc1fdb7c12889b76d1ce14ec46d5ce9834be505f03600e3444e51d684de2` |
| `results/grain_vector_mechanisms/vm05_full_source_cohort_r1/author_views.json` | `a1f8a221bc8d09ebfe86723b8555b0a13d6aa583bf762ea6f7dd50465596cfc6` |
| `results/grain_vector_mechanisms/vm05_full_source_cohort_r1/big5_only_author_views.json` | `603e620e22fbb2b75823aa7a3fa6405de01cd56fff6329ee95d54ad82727cf62` |
| `results/grain_vector_mechanisms/vm05_full_source_cohort_r1/manifest.json` | `8d9fe8e6599ddf4ed92f8d2f3eb03763b42ccf8f200898fccf2c1d5a939599e7` |
| `results/grain_vector_mechanisms/vm05_preflight_r1/manifest.json` | `721fd18d51ed21ebed6ce17ecd5e0579032330211532f0f14d4a7b581784a582` |
| `results/grain_vector_mechanisms/vm05_preflight_r1/common_material_manifest.json` | `9e0043cb4ca029733d15f4d326cf5dd67fcc4f93412ef659a96e6e616ba3c792` |
| `results/grain_vector_mechanisms/predecessor_integration_v1/material/material_manifest.json` | `73c92b282b3bb10991de329d0400e403333fa06c7f567d58230198e74ab59f80` |
| `results/grain_gptpro_foundation_v3/F14/history_increment_r1/coefficients.npz` | `a1fa08d28ba4109139cc1ebde7446bf2ecd0d549925da22eeec7dcdbc06b9db7` |
| `results/grain_gptpro_foundation_v3/F14/raw_grain_history_r1/private_bundle.npz` | `2e58b0827f7ef5a24aa58983af8254d3cd8a4ff34db33c05f895e6e19d3e5392` |
| `results/grain_gptpro_foundation_v3/F14/raw_grain_history_r1/private_held_event_losses.csv` | `98eebcb6b5c44f8e2e279304008b6119fd883ffb4f97ea2d7327b71bcb5db80b` |
