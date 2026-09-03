# Local Data Directory

This directory is intentionally committed as an empty skeleton only.

The GRAIN release does not redistribute user text, user identifiers, prepared
PANDORA/Essays files, X/Twitter market text, or lockbox data. Put locally
obtained datasets under the paths below when reproducing the full analyses:

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

All files under this directory are ignored by Git except this README and
`.gitkeep` placeholders.
