#!/usr/bin/env python3
"""Post-outcome finite readout account;numeric logit corners are not new texts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_context_relation_hm178 as H
from scripts import run_grain_observed_family_hm170 as R


def valuation(x):
    """Use the historical ten-digit ordinal valuation,without recalibration."""
    return softmax(np.asarray(x, float), axis=-1) @ (np.arange(10) / 9)


def response(x, d):
    """Exact finite softmax response on a specified numerical background."""
    return valuation(x + d) - valuation(x)


def main():
    """Describe all old endpoints with fixed action/background/interaction terms."""
    parent = H.DEFAULT
    m = R.check(parent, ("scored_seal.json", "analysis_seal.json"))
    out = parent / "post_outcome_readout_r1"
    out.mkdir(exist_ok=False)
    R.write(
        out / "manifest.json",
        {
            "classification": "POST_OUTCOME_DESCRIPTION_NOT_NEW_NATURAL_INTERVENTION",
            "code_hash": R.digest(Path(__file__)),
            "source_analysis_seal": R.digest(parent / "analysis_seal.json"),
            "source_scored_seal": R.digest(parent / "scored_seal.json"),
            "source_full_context_is_RC": True,
            "order_alternative_retained": True,
            "no_rescore_fit_or_neural_calls": True,
        },
    )
    native = ROOT / m["config"]["native_parent"]
    with np.load(native / "composition_accounts.npz", allow_pickle=False) as a:
        logits, scores = a["native_logits"], a["native_scores"]
    with np.load(parent / "mechanism_accounts.npz", allow_pickle=False) as a:
        components, native_j = a["components"], a["native_ordinal_actions"]
    # These gauge-fixed corners are numeric diagnostics,not observed new inputs.
    logits = logits - logits[..., :1]
    x0, x1 = logits[:, :, 4], logits[:, :, 1]
    d0, d1 = logits[:, :, 3] - x0, logits[:, :, 2] - x1
    corners = np.stack(
        [response(x0, d0), response(x0, d1), response(x1, d0), response(x1, d1)]
    )
    action = corners[1] - corners[0]
    background = corners[2] - corners[0]
    interaction = corners[3] - corners[2] - corners[1] + corners[0]
    account = action + background + interaction
    native_replay = valuation(logits)
    covariance_error = 0.0
    for x, d, target in (
        (x0, d0, corners[0]),
        (x0, d1, corners[1]),
        (x1, d0, corners[2]),
        (x1, d1, corners[3]),
    ):
        p = softmax(x, axis=-1)
        tilt = np.exp(d - d.max(-1, keepdims=True))
        den = np.sum(p * tilt, axis=-1)
        a = np.arange(10) / 9
        alt = (np.sum(p * tilt * a, axis=-1) - valuation(x) * den) / den
        covariance_error = max(covariance_error, float(np.max(np.abs(alt - target))))
    error = float(np.max(np.abs(account - native_j)))
    replay_error = float(np.max(np.abs(native_replay - scores)))
    if max(error, replay_error, covariance_error) > 2e-12:
        raise ValueError("Exact historical readout account failed")
    columns = {
        "actual": native_j,
        "action_at_C": action,
        "background_with_S_minus_C": background,
        "interaction": interaction,
        "action_at_RC": action + interaction,
        "background_with_context_action": background + interaction,
    }
    rows = []
    for u, v in np.ndindex(24, 2):
        row = {"panel_index": u, "view": v}
        row.update({k: float(a[u, v]) for k, a in columns.items()})
        rows.append(row)
    R.table(out / "readout_accounts.csv", rows)
    links = []
    for arm, k, q in np.ndindex(2, 5, 5):
        for endpoint, y in columns.items():
            links.append(
                {
                    "arm": ("cue_mask", "raw")[arm],
                    "component": H.COMPONENTS[k],
                    "target": m["config"]["targets"][q],
                    "readout_term": endpoint,
                    **H.C.signed_agreement(
                        components[arm, :, :, k, q].mean(1), y.mean(1)
                    ),
                }
            )
    R.table(out / "term_links.csv", links)
    summary = {
        "classification": "POST_OUTCOME_NO_NEW_MODEL_OR_PREDICTIVE_FIT",
        "actual_context_action_rms": float(np.sqrt(np.mean(native_j**2))),
        "term_rms": {k: float(np.sqrt(np.mean(a * a))) for k, a in columns.items()},
        "max_account_error": error,
        "native_probability_replay_error": replay_error,
        "alternate_tilt_covariance_error": covariance_error,
        "all_signed_terms_retained": True,
        "numeric_corners_not_natural_inputs": True,
        "neural_calls": 0,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "description_seal.json",
        ("readout_accounts.csv", "term_links.csv", "summary.json"),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
