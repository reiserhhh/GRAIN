#!/usr/bin/env python3
"""Post-outcome description of fixed readings, not another experiment or fit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_relational_diagnostic_update import BITS
from scripts import run_grain_author_pair_criterion_hm277 as X
from scripts import run_grain_observed_family_hm170 as R


def describe(parent):
    """Describe covariance accounting without changing primary tests or frozen scores."""
    R.check(parent, ("computed_seal.json", "verification_seal.json"))
    out = parent / "post_outcome_description_r1"
    out.mkdir(exist_ok=False)
    value = X.load(parent / "readings.npz")
    scores = value["series"][:, :, 2].copy()
    labels = BITS[value["codes"]].copy()
    for bank in (0, 1):
        ix = value["banks"] == bank
        scores[:, ix] -= scores[:, ix].mean(1, keepdims=True)
        labels[ix] -= labels[ix].mean(0)
    covariance = np.einsum("ansq,nr->asqr", scores, labels) / len(labels)
    sd = np.sqrt(np.mean(scores**2, 1))
    label_sd = np.sqrt(np.mean(labels**2, 0))
    association = covariance / (sd[..., None] * label_sd)
    reconstruction = float(
        np.max(abs(covariance[:, 0] - covariance[:, 1] - covariance[:, 4]))
    )
    R.archive(
        out / "accounts.npz",
        covariance=covariance,
        score_sd=sd,
        label_sd=label_sd,
        cross_target_correlation=association,
    )
    R.write(
        out / "summary.json",
        {
            "post_outcome_only_no_new_fit_or_primary_test": True,
            "covariance_reconstruction_max": reconstruction,
            "GRAIN_N_F_J_pair_covariances": covariance[1, [0, 1, 4], 1, 1].tolist(),
            "GRAIN_N_pair_to_J_score_sd_ratio": float(sd[1, 4, 1] / sd[1, 1, 1]),
            "all25_cross_target_relations_descriptive_not_new_confirmatory_family": True,
        },
    )
    R.write(
        out / "manifest.json",
        {
            "script_sha256": R.digest(Path(__file__)),
            "input_sha256": R.digest(parent / "readings.npz"),
            "outputs": {p: R.digest(out / p) for p in ("accounts.npz", "summary.json")},
        },
    )
    print(R.read(out / "summary.json"), flush=True)


def main():
    """Create one separately identified description without rewriting original results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=X.DEFAULT)
    a = p.parse_args()
    describe(a.parent)


if __name__ == "__main__":
    main()
