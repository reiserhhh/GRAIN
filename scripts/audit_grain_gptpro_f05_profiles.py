#!/usr/bin/env python3
"""Replay all GLS-profile objectives using scalar Kalman likelihoods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f05 import eta_from_rho, grouped
from grain_research.gptpro_foundation_v3.f05_recovery import kalman_loglik


def main():
    """Verify source IDs, fifty-point grids and conditional objective arithmetic."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve audits")
    args.output.mkdir(parents=True)
    hashes = 0
    for directory in (args.source, args.result):
        for item in json.loads((directory / "output_manifest.json").read_text())[
            "files"
        ]:
            assert sha256(directory / item["path"]) == item["sha256"]
            hashes += 1
    selection = json.loads((args.result / "fixed_author_selection.json").read_text())
    data = pd.read_csv(args.result / "likelihood_profiles.csv")
    largest = 0.0
    checked = 0
    for selected in selection["selections"]:
        world, seed = selected["world"], selected["seed"]
        root = args.source / f"{world}_{seed}"
        people = grouped(
            [
                json.loads(line)
                for line in (root / "source.jsonl").read_text().splitlines()
            ]
        )
        assert selected["author_ids"] == sorted(people)[: len(selected["author_ids"])]
        model = json.loads((root / "source_model.json").read_text())
        precision = np.linalg.inv(model["prior"]["Sigma"])
        mu = np.asarray(model["prior"]["mu"])
        subset = data[(data.world == world) & (data.seed == seed)]
        assert set(subset.author_id) == set(selected["author_ids"])
        for (author, n, mode), group in subset.groupby(["author_id", "n", "mode"]):
            np.testing.assert_allclose(group.rho, np.arange(50) * 0.02, atol=1e-15)
            objectives = []
            for row in group.itertuples(index=False):
                psi = np.array([row.a, row.b, eta_from_rho(row.rho)])
                nll = -kalman_loglik(psi, people[author][:n], model["log_sigma"])[
                    "loglik"
                ]
                delta = psi - mu
                penalty = (
                    0.5 * delta @ precision @ delta if mode == "posterior" else 0.0
                )
                np.testing.assert_allclose(
                    [row.negative_loglik, row.prior_penalty],
                    [nll, penalty],
                    atol=1e-10,
                    rtol=1e-8,
                )
                largest = max(
                    largest,
                    abs(row.negative_loglik - nll),
                    abs(row.prior_penalty - penalty),
                )
                objectives.append(nll + penalty)
                checked += 1
            drop = 2 * (np.asarray(objectives) - min(objectives))
            np.testing.assert_allclose(
                group.twice_objective_drop, drop, atol=1e-10, rtol=1e-8
            )
    report = {
        "status": "PROFILE_REPLAY_PASS",
        "profiles_replayed": checked,
        "hashes": hashes,
        "max_objective_difference": largest,
        "scope": "source-only IDs and full grid likelihood/penalty replay via alternative Kalman implementation; finite-difference Hessian remains approximate",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == "__main__":
    main()
