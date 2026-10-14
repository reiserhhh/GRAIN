#!/usr/bin/env python3
"""Describe sealed source-bank updates after outcomes, without new fitting."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_likelihood_transport_hm240 as H
from scripts import run_grain_observed_family_hm170 as R


def describe():
    """Retain every source-bank direction and expose the dimensional reference."""
    parent = H.DEFAULT
    H.check(parent)
    out = parent / "post_outcome_description_r1"
    out.mkdir(exist_ok=False)
    inputs = ("addresses.csv", "accounts.npz", "summary.json", "verification_seal.json")
    R.write(
        out / "manifest.json",
        {
            "post_outcome": True,
            "primary_unchanged": True,
            "new_model_fit_or_neural_calls": 0,
            "inputs": {name: R.digest(parent / name) for name in inputs},
            "code": {str(Path(__file__).relative_to(ROOT)): R.digest(__file__)},
            "analysis": "all inherited matched bank updates;no winning masks/authors selected",
        },
    )
    addresses = pd.read_csv(parent / "addresses.csv", dtype={"author": str})
    with np.load(parent / "accounts.npz", allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    groups = {
        "full_relation": addresses.primary,
        "all_relation_frames": addresses.operation == "relation",
        "paired_evidence": (addresses.operation == "evidence")
        & (addresses.law == "paired"),
        "product_evidence": (addresses.operation == "evidence")
        & (addresses.law == "product"),
    }
    summaries, rows = {}, []
    for name, group in groups.items():
        first = (group & (addresses.bank == 0)).to_numpy()
        second = (group & (addresses.bank == 1)).to_numpy()
        keys = ["author", "operation", "law", "frame"]
        a = addresses.loc[first, keys].reset_index(drop=True)
        b = addresses.loc[second, keys].reset_index(drop=True)
        if not a.equals(b):
            raise ValueError("Source-bank operation addresses disagree")
        summaries[name] = {}
        for space in ("delta", "character_delta", "character_joint_delta"):
            x, y = result[space][first], result[space][second]
            inner = (x * y).sum(1)
            denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
            if np.any(denominator == 0):
                raise ValueError("Undefined alignment must not be assigned zero")
            cosine = inner / denominator
            summaries[name][space] = {
                "paired_accounts": len(a),
                "positive": int((inner > 0).sum()),
                "negative": int((inner < 0).sum()),
                "zero": int((inner == 0).sum()),
                "pooled_cosine": float(
                    inner.sum() / np.sqrt((x * x).sum() * (y * y).sum())
                ),
                "mean_account_cosine": float(cosine.mean()),
                "range": [float(cosine.min()), float(cosine.max())],
                "scope": "source-instrument agreement,not psychological r or causality",
            }
            for n, address in enumerate(a.to_dict("records")):
                rows.append(
                    {
                        **address,
                        "group": name,
                        "space": space,
                        "signed_inner": float(inner[n]),
                        "cosine": float(cosine[n]),
                    }
                )
    primary = R.read(parent / "summary.json")["groups"]["full_relation"]
    summaries["primary_frozen_joint_to_full_mean_RMS_ratio"] = (
        np.asarray(primary["frozen_joint_tilt_target_delta_RMS"])
        / np.asarray(primary["target_delta_RMS"])
    ).tolist()
    summaries["dimension_reference"] = {
        "uniform_isotropic_only": True,
        "single_statistic_dimensions": 5,
        "joint_residual_dimensions": 26,
        "joint_expected_dimension_ratio": 26 / 31,
        "not_fitted_null_or_personality_information_fraction": True,
    }
    summaries["scope"] = (
        "24 historical authors;dependent bank/frame accounts,post-outcome description"
    )
    R.table(out / "all_signed_alignment_accounts.csv", rows)
    R.write(out / "summary.json", summaries)
    R.seal(
        out,
        "description_seal.json",
        ("all_signed_alignment_accounts.csv", "summary.json"),
    )
    print(json.dumps(summaries))


if __name__ == "__main__":
    describe()
