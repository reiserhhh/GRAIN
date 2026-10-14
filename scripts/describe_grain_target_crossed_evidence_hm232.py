#!/usr/bin/env python3
"""Separate source discrimination from nonlinear averaging in frozen HM232 data."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_target_crossed_evidence_hm232 as H

OUT = H.DEFAULT / "averaging_action_accounts_r1"


def accounts(matrix, common):
    """Retain criterion differences and the mean-action/action-of-mean gap."""
    t, c = np.asarray(matrix, float), np.asarray(common, float)
    if t.ndim != 3 or t.shape[1:] != (5, 5) or c.shape != t.shape[:2]:
        raise ValueError("Complete edge x5reader x5source and edge x5common")
    if not np.isfinite(t).all() or not np.isfinite(c).all():
        raise ValueError("Finite actions required")
    mean_action = t.mean(2)
    variation = t - mean_action[:, :, None]
    gap = mean_action - c
    source = variation.mean(1, keepdims=True)
    interaction = variation - source
    departure = t - c[:, :, None]
    reconstructed = source + interaction + gap[:, :, None]
    energy = {
        "source": float(np.mean(source**2)),
        "interaction": float(np.mean(interaction**2)),
        "averaging_action_gap": float(np.mean(gap**2)),
        "departure": float(np.mean(departure**2)),
        "source_discrimination": float(np.mean(variation**2)),
    }
    return {
        "source_discrimination": variation,
        "averaging_action_gap": gap,
        "source_main": source,
        "interaction": interaction,
        "departure": departure,
        "energy": energy,
        "reconstruction_error": float(np.max(abs(departure - reconstructed))),
        "energy_account_error": abs(
            energy["departure"]
            - energy["source"]
            - energy["interaction"]
            - energy["averaging_action_gap"]
        ),
    }


def main():
    """Save a new descriptor without refitting, replacing, or rerunning science."""
    if OUT.exists():
        raise FileExistsError("No replacement of an uncertain companion")
    H.R.check(
        H.DEFAULT,
        ("captured_seal.json", "cold_seal.json", "analyzed_seal.json"),
    )
    data = H.arrays(H.DEFAULT / "analysis.npz")
    t, c = data["matrix"], data["common_action"]
    a = accounts(t, c)
    # Direct pair differences do not use the original two-way decomposition.
    pair = t[:, :, :, None] - t[:, :, None, :]
    alternate = pair.mean(-1)
    alternate_error = float(np.max(abs(alternate - a["source_discrimination"])))
    original_error = max(
        float(np.max(abs(data["departure"] - a["departure"]))),
        float(np.max(abs(data["interaction"] - a["interaction"]))),
        float(np.max(abs(data["source"] - a["source_main"]))),
    )
    summary = {
        "source_discrimination_RMS": H.rms(a["source_discrimination"]),
        "source_main_RMS": H.rms(a["source_main"]),
        "interaction_RMS": H.rms(a["interaction"]),
        "averaging_action_gap_RMS": H.rms(a["averaging_action_gap"]),
        "source_discrimination_RMS_by_reader": [
            H.rms(a["source_discrimination"][:, q]) for q in range(5)
        ],
        "averaging_action_gap_RMS_by_reader": [
            H.rms(a["averaging_action_gap"][:, q]) for q in range(5)
        ],
        "energy": a["energy"],
        "reconstruction_error": a["reconstruction_error"],
        "energy_account_error": a["energy_account_error"],
        "alternate_pair_difference_error": alternate_error,
        "original_account_error": original_error,
        "same_agent_not_blind": True,
        "no_new_model_calls_fit_or_accuracy_claim": True,
    }
    if (
        max(
            alternate_error,
            original_error,
            a["reconstruction_error"],
            a["energy_account_error"],
        )
        >= 1e-12
    ):
        raise ValueError("Companion accounts disagree; do not rescue precision")
    OUT.mkdir(exist_ok=False)
    R.write(
        OUT / "manifest.json",
        {
            "code": {str(Path(__file__).relative_to(ROOT)): R.digest(Path(__file__))},
            "inputs": {
                str((H.DEFAULT / name).relative_to(ROOT)): R.digest(H.DEFAULT / name)
                for name in ("analysis.npz", "analyzed_seal.json", "manifest.json")
            },
            "timing": "descriptor equations written during source capture,before cold",
        },
    )
    rows = []
    for e in range(len(t)):
        for q in range(5):
            for r in range(5):
                rows.append(
                    {
                        "edge": e,
                        "reader": q,
                        "source": r,
                        "source_discrimination": float(
                            a["source_discrimination"][e, q, r]
                        ),
                        "source_main": float(a["source_main"][e, 0, r]),
                        "interaction": float(a["interaction"][e, q, r]),
                        "averaging_action_gap": float(a["averaging_action_gap"][e, q]),
                        "original_departure": float(a["departure"][e, q, r]),
                    }
                )
    R.table(OUT / "accounts.csv", rows)
    R.write(OUT / "summary.json", summary)
    R.seal(OUT, "descriptor_seal.json", ("accounts.csv", "summary.json"))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
