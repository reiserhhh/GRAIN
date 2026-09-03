# Data Access & Reproduction Inputs

This release ships NO user text and NO user identifiers. To reproduce the
worked example you must obtain the source datasets from their owners:

1. **PANDORA** (Gjurkovic et al., 2021, "PANDORA Talks: Personality and
   Demographics on Reddit"): request access from the dataset authors
   (TakeLab, University of Zagreb). You need `all_comments_since_2015.csv`.
2. **Essays** (Pennebaker & King, 1999 stream-of-consciousness corpus):
   available to researchers via the original distribution channels.
3. X/Twitter market text used in one regime test is proprietary and is NOT
   part of the reproduction path (its numbers appear only in reports).

## Preparation

Prepared-file schemas are documented in VALIDATION_PLAN.md and the prep
scripts (`scripts/prepare_*.py`, `scripts/build_suica_tiers_v2.py`). Tier
membership is derived deterministically (SHA-1 stable-hash rules in code)
from the raw datasets — no membership files are required.

## Expected local layout

The repository commits only an empty `data_sets/` skeleton. Place locally
obtained data under these paths; all real files below `data_sets/` are ignored
by Git:

```text
data_sets/
├── PANDORA_official/
│   └── all_comments_since_2015.csv
├── prepared/
│   ├── pandora_official/
│   │   ├── pandora_official_big5_prepared.csv
│   │   ├── pandora_official_bridge_strict377.csv
│   │   ├── pandora_official_bridge_supplementary393.csv
│   │   └── mbti_axes/
│   │       ├── pandora_official_EI_cont_prepared.csv
│   │       ├── pandora_official_SN_cont_prepared.csv
│   │       ├── pandora_official_TF_cont_prepared.csv
│   │       └── pandora_official_JP_cont_prepared.csv
│   ├── suica_tiers_v2/
│   │   └── <deterministically derived tier files>
│   └── big5/
│       └── essays_original_prepared.csv
└── x_fullmarkettext/
    └── <private local X/Twitter market-text export>
```

## Integrity manifest (SHA-256 of our prepared inputs)

Byte-identical preparation can be verified against:

| Prepared file | SHA-256 |
|---|---|
| `pandora_official_big5_prepared.csv` | `b6d15f01d43f113fcc4a5b23b8079e222cac77e5dbd115aba0058d4dc07d6f88` |
| `pandora_official_bridge_strict377.csv` | `861cf827626fb4a507b13e05da59dd95c9054cc0202c6c780ec3c1ef22fef3ed` |
| `pandora_official_bridge_supplementary393.csv` | `0df662bd510ac393428b72a81a9bba9672d7a4d051a86ce4236debb3a444c9c2` |
| `essays_original_prepared.csv` | `4cf023c733684936a7a5e0e95603184220a809d3bd874408203cbd3ad1a0fccf` |
| `pandora_official_EI_cont_prepared.csv` | `fd9273ea968b163b4f068b5a0c499e4f782732103d09857807ac769705476f0a` |
| `pandora_official_SN_cont_prepared.csv` | `b2b54916d62406e119438312472be58fe049bfd6b4df9075c691d83f56e8a5db` |
| `pandora_official_TF_cont_prepared.csv` | `7514fd6a4c076d69934e9d7253310670472b4fb73e7aa6724cb62e8f5ac82760` |
| `pandora_official_JP_cont_prepared.csv` | `d26d4ca9eeb56e9f8179a259a62a049fe4245290e3de47620a54642c55a2af40` |

## Zero-data verification

Without any dataset you can still verify the estimator layer:

```bash
python scripts/run_grain_synthetic_ground_truth_v2.py   # P0
python scripts/run_grain_p0b_thin_cell_regime_v3.py     # P0-B
python -m pytest -q tests/test_suica.py                 # 39 tests
```
