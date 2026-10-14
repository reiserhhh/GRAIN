# R34 text half-channel sensitivity pilot: protocol V1 (packet20 B)

Frozen 2026-09-26 JST before any half-channel score is computed.
- **Authority.** `NEXT_PACKET_20_ERROR_BUDGET_AND_HISTORY.md` Part B; `SEMINAR_FABLE_17_DECISION.md` correction 1.
- **Runner.** `scripts/run_grain_r34_text_half_channel.py`.
- **Output root.** `results/grain_gptpro_foundation_v3/F16/text_half_channel_sensitivity_r1/`, created in x-mode.
- **Budget.** ≤ 10 min CPU total, shared with Part A's replays. One process, single-thread BLAS. 0 neural calls. No
  new text or reader scoring.

## What this is

A two-core calibration SENSITIVITY of text scores to the reader channel. It is **not** population split-half
reliability, not an additive reader-variance component, and not a production ensemble. Each half arm is an explicit
**one-component diagnostic view** (`DIAGNOSTIC_ONE_COMPONENT_VIEW`). It uses:
- source model b = 0 of the frozen seed-20260915 bundle;
- the b = 0 numerical integration seed, as in production (`SeedSequence([seed, 161, 0, 0])` for T1;
  `[seed, 164, 0, 0]` for dynamic);
- the bundle's own `qmc_power` (T1: 1024 nodes; dynamic: 256). It is never lowered.

Source parameters, priors, units and integration nodes are identical across the half arms. Only the channel differs.

## Fixed design (decided before outcomes)

- **Bundles.** Historical text bundles `F16/t1_text_full_r1/20260915/bundle.json` and
  `F16/dynamic_text_full_r1/20260915/bundle.json`. Their `channel_counts` equal the counts of the 32 role = validation
  materials of the F08/F11 reader bank, checked before this protocol. The same bank is used here.
- **Calibration partition.** The lexicographically first of the 8 whole-core balanced partitions in
  `F16/source_half_reconstruction_r1/results.json`:
  - half A cores (pole, subject_status) = [[0, actual], [0, hypothetical], [1, actual], [1, hypothetical]];
  - half B = the other 4 cores.
  - Whole cores (all 4 variants) stay together, one core per (pole, qualification) stratum per half.
- **Channels.**
  - For each arm (`nli_joint` with key nli_qualification; `hybrid_joint` with key grain_qualification), per half:
    counts = `f16_t1_text.channel_counts(half rows, arm)`.
  - The channel is the smoothed MEAN (counts + .5) / Σ (counts + .5), the Jeffreys/Dirichlet .5 convention of
    `channel_draws`. No 200 copies are made.
  - Reference views: the FULL 32-material smoothed mean, and (for the cold-replay check only) production channel draw
    b = 0.
  - NLI and hybrid are kept separate.
- **Persons.**
  - For each route, the first 20 author_ids in sorted order from that bundle's `application_inputs.jsonl`, selected
    before outcomes.
  - Every event material of every selected person must have role = application. Calibration (validation) materials
    and persons can never be evaluated. A violation aborts the run.
  - The same identities and the same raw cached event records are used in every channel arm.
- **Scoring.**
  - A thin wrapper builds an in-memory view of the bundle with `source_models = [model_0]` and
    `channel_draws[arm] = [channel]`. It calls the unchanged production primitives: `f16_t1_text.score_counts` (T1)
    and `f16_dynamic_text.score_panel` (T2 = state at t = 47; T3 = mean of the 4 plan responses).
  - The original bundle is verified first (`verify_bundle` / `validate_bundle`).
  - No production builder is rewritten, and no source or provenance check is relaxed on a stored bundle.
- **Cold-replay check.** For the pilot person of each route, the view built with production channel draw b = 0 must
  reproduce production component b = 0, captured by wrapping the internal primitive in a full production call
  (tolerance 1e-12).
- **Pilot and stop rule.** Score person 1 on both routes in every arm and record the time. If 20 × that time exceeds
  the remaining budget, stop at the largest complete prefix of the sorted IDs, named in results.json.

## Reported (native units; no standardization)

- **Channels.** Counts and smoothed means per half and arm; TV per (pole, qualification) stratum and overall; the
  split core IDs and material IDs.
- **Per person, per arm, per target** (T1 endpoint contrast; T2 current state; T3 future response mean):
  - mean and within-view variance under half A, half B and FULL;
  - d = A − B and z = d / √((V_A + V_B)/2);
  - |A − FULL| and |B − FULL|.
- **Application support per person and arm:**
  - the counts table (condition × category);
  - the support label;
  - category marginals, i.e. pole-readout high/low and qualification high/low distributions;
  - event addresses (event_id, material_id, t, z).
- **Summaries per route, arm and target:** mean |d|, RMS d, mean |z|, max |z|, and the share of persons with d = 0
  exactly.
- **TV = 0.** If a half pair has TV 0 in an arm, its scores are identical by construction. This is reported as
  channel INSENSITIVITY for that split, not as positive robustness evidence. Negative or zero results are valid.

## Not done

- No crossed source × reader design, no 200-component half ensembles, no other partitions or seeds.
- No precision change and no new reader outputs.

## Checksum bindings

| file | sha256 |
|---|---|
| `results/grain_gptpro_foundation_v3/F16/t1_text_full_r1/20260915/bundle.json` | `0878a225b64b376749662249bf35b73103b05a8fcae8b7ae19e03fb6a397d91d` |
| `results/grain_gptpro_foundation_v3/F16/t1_text_full_r1/20260915/application_inputs.jsonl` | `c897aaf27ca88a35e18c6cece5748f763bdb041b41ebf6ae28dfe65e17d72256` |
| `results/grain_gptpro_foundation_v3/F16/dynamic_text_full_r1/20260915/bundle.json` | `73470c7eab7370c2546a9089fb1372a3b59748df29b3f5d9daa5dc7c16e02091` |
| `results/grain_gptpro_foundation_v3/F16/dynamic_text_full_r1/20260915/application_inputs.jsonl` | `e722d5533e67029d4826d785e5b69930b69b0273786cd967a9f074100fe59546` |
| `results/grain_gptpro_foundation_v3/F08/nli_reader_r1/reader_scores.csv` | `ec280aa5fcbf384563599e326eabdfa5d514ba6728b789b3a0c00304f7a967bf` |
| `results/grain_gptpro_foundation_v3/F11/grain_comment_reader_r1/reader_scores.csv` | `28280d77653a31c19dd3f823f48b2794d9a94234b3d266f3c125cefdfe4be4ad` |
| `results/grain_gptpro_foundation_v3/F16/source_half_reconstruction_r1/results.json` | `83be4435e010e06bbbedc1630e6f88f6bfadcef5dcf817395f874ddaf77e1ea3` |
| `grain_research/gptpro_foundation_v3/f16_t1_text.py` | `32cfddf7233b24cc690fe0c5d096e29dcc5d0f9e49f4cf06304ad6ad7b69bd83` |
| `grain_research/gptpro_foundation_v3/f16_dynamic_text.py` | `bec07462a4f7cfe674c572ea95a466a8d7635781312530301b3a20d821eb4d1d` |
| `grain_research/gptpro_foundation_v3/f16_t1_source.py` | `d2c5e0da84744f2baf458745810a1d5bf2e60fce34ee7397626e9026434f2cd9` |
