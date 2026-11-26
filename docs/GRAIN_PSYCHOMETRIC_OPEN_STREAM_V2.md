# GRAIN Psychometric Open-Stream Architecture (V2 / `grain_core.psychometric_open_stream`)

The D89/R24 joint entry is `grain_core.response_profile.measure_calibrated` (see
[contract and CLI](GRAIN_RESPONSE_PROFILE.md)). Confusion rates refer to observed replies.
Passing a per-opportunity card requires explicit `calibration_denominator="among_replies"`.
If the fraction of confirmed replies is p and their corrected bounds are [l,h], the
per-opportunity bounds are [p*l, p*h + unknown_reply_fraction]. Confirmed silence and
unconfirmed exposure have different meanings. Three-way reader DIF on reply contrasts
is not silently reused for contrasts that include differential participation.

The reply-level inverse requires compatibility between the observed positive-rate
interval and the reader calibration envelope. When they are disjoint, the tool
returns `reader_incompatible`, an uninformative bound, and no corrected point or
point-contrast decomposition. This is different from `reader_unidentified`, in
which the reader's sensitivity and false-positive intervals do not establish
positive discrimination. In neither case does clipping a point to zero or one
make it an observation.

This module links [open-stream operators](GRAIN_OPEN_STREAM_OPERATORS_G1_G6.md) and **conditional response profiles (`D88`/`D89`)** in a deterministic scoring layer. Use the [evidence panel](GRAIN_EVIDENCE_PANEL.md) first when the work is joining logs or checking collection coverage; this module assumes that the relevant units and denominators are already defined.

---

## 1. Three-Layer Open-Stream Psychometric Model

| Layer | Core Problem | Deterministic Operator | Output Layer-$M$ Guarantees |
| :--- | :--- | :--- | :--- |
| **Layer 1: Selection & Opportunity** | Spoken-only text mixes topic opportunity with silence; non-response under presence differs from unobserved exposure (`G-3`, V1 Eq. 1) | `measure_irtree_selection_expression` | Separates **Node 1 (Venue Selection / Participation)** with sharp Manski exposure bounds $[\eta_{\min}, 1]$ from **Node 2 (Conditional Expression)**; evaluates contiguous split-half reproducibility against `>= 200` stranger nulls (`G3`/`G4`). |
| **Layer 2: Calibration, DIF & Precision** | Imperfect lexical/LLM readers distort level and contrast via condition-dependent sensitivity/FPR (`D88`); unequal utterance counts $n_{j,t}$ create attenuation & unreliable differences (`G-1`); surface genre shifts slope (`G-5`) | `calibrate_reader_misclassification`, `estimate_bivariate_caps_reliability`, `deattenuate_cross_panel_correlation`, `audit_register_dif_invariance` | • Exact **Rogan-Gladen inversion** & 3-way **Reader-DIF decomposition** (`closure_error <= 1e-15`), refusing `reader_unidentified` when $s_{\text{lo}} \le f_{\text{hi}}$;<br>• Cross-fitted **2D Kelley Empirical-Bayes CAPS estimator** with per-person reliability $\rho_u(\Delta)$, $\text{MDD}_{95,u}$, and unclipped Rouder de-attenuation;<br>• Meredith **DIF hierarchy** (`L0`–`L3` license) via `CrossFittedRegisterCalibrator` + `G11` MTMM check. |
| **Layer 3: Non-Ergodic Dynamics & Transfer** | Group/person raw mean trends mix within-condition change with cohort turnover AND personal condition-mix (`G-4`, `D88`); single noisy utterances trigger `HM281` negative transfer (`G-2`, `G-6`) | `decompose_hierarchical_cohort_person_mix`, `fit_ct_lsto_variance_components` | • Exact **5-term Hierarchical Cohort × Condition-Mix** telescoping identity (`within_condition`, `condition_mix`, `stayer_reweight`, `entry`, `exit`; `closure <= 1e-14`);<br>• Multi-start (`>= 5`) **CT-LSTO semivariogram** splitting $\sigma_{\text{trait}}^2, \sigma_{\text{state}}^2, \sigma_{\text{occ}}^2$ and proving when naive carry strictly hurts ($\sigma_{\text{occ}}^2 > \sigma_{\text{state}}^2$) vs Kalman-shrunk multi-event carry. |

---

## 2. Quick Usage (Python & CLI)

```python
from grain_core import psychometric_open_stream as pos

card = pos.measure_psychometric_card({
    "task": "hierarchical_mix",
    "prev_cells": [
        {"focal": "u1", "condition": "c1", "count": 1, "value": 0.2},
        {"focal": "u1", "condition": "c2", "count": 9, "value": 0.8},
    ],
    "cur_cells": [
        {"focal": "u1", "condition": "c1", "count": 9, "value": 0.3},
        {"focal": "u1", "condition": "c2", "count": 1, "value": 0.9},
    ],
})
assert card["result"]["closure_max_abs_err"] < 1e-14
assert card["result"]["n_stayers_sign_reversed_by_mix"] == 1
```

Command-line bundle replay:

```sh
python -m grain_core.psychometric_open_stream examples/psychometric_open_stream_demo.json
```

---

## 3. Interpretive Guardrails (`G9` / `G12` / `G14` / `G16`)

1. All outputs belong to **Layer $M$** (calibrated behavioral/psychometric quantities relative to declared reference populations and frozen readers), **never** automatic personality traits or clinical diagnoses ($\Theta$).
2. Do not suppress `underpowered_for_individual_reporting`, `no_latent_contrast_variance`, `reader_unidentified`, or `reader_incompatible`. A wide bound or withheld point is a result, not a missing value to fill.
3. Only compare levels across registers when `earned_comparison_license == "L3_scalar_level_comparable"` (`G12`).
