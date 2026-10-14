# F14 actual cached-material interface audit

## Result and scope

The existing PANDORA and Essays cached numerical views pass actual finite-value, full material-address and source/application partition checks. This replaces the earlier interface audit's literal finite-score flag with computations. It does not rewrite or promote the old GP05/F14 results.

| Dataset | Authors | Source / held authors | Embedding units | Authors with one unit |
|---|---|---|---|---|
| PANDORA | 624 | 504 / 120 | 22,464 | 0 |
| Essays | 2,467 | 2,213 / 254 | 13,790 | 4 |

All 36,254 units have finite numerical values and matched `(author_index, unit, start, end)` addresses between the local-bilinear and conditional-reference views. Both datasets have zero duplicated material addresses and zero authors without units. Source/held sets are individually unique, disjoint and jointly cover the declared authors.

An embedding unit is not an independent occasion. Multiple Essays chunks do not become repeated conversations, and these counts are not the entire underlying raw corpora.

## Method and verification

- Code: `grain_research/gptpro_foundation_v3/f14_cache_interface.py`.
- Runner: `scripts/audit_grain_gptpro_f14_cache_interface.py`.
- Tests: `tests/test_grain_gptpro_f14_cache_interface.py`; six tests passed, including changed addresses within the same author, NaN, overlapping/incomplete/duplicated partitions and invalid spans. Ruff passed.
- Evidence: `results/grain_gptpro_foundation_v3/F14/cache_interface_audit_r1/summary.json` contains input hashes and separate dataset-level check results.

Only previously authorized numeric caches, material metadata and four numeric address columns were read. No text, questionnaire values or condition-name values were read, and only aggregate checks are reported. Source embeddings/arrays were loaded with `allow_pickle=False`; source normalizers were not reconstructed in this packet.

The inspected roots are `results/grain_vector_mechanisms/all_plan_closure_v1/local_bilinear_r2/` and `conditional_reference_r2/`, each for `pandora` and `essays`. This is an interface check, not newly established psychological validity or calibration.

## Remaining work

1. Locate original metadata and record calendar-time/occasion/opportunity availability. Absence in cached address columns does not establish absence in the corpus.
2. Inventory authorized X and MEPS schemas without adding raw-text/questionnaire access or fitting.
3. Reconstruct source-fixed normalizers and finish adapter contracts and target-support checks.
4. Connect qualified textual observations to a matching F16 target; do not relabel a binary reader output as a continuous latent state.

F14 parent acceptance remains open. No new neural/API calls, training, remote execution, lockbox action, commit or push occurred.
