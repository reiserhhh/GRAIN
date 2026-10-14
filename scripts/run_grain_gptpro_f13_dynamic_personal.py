#!/usr/bin/env python3
"""One frozen source bootstrap fit -> observed-prefix Laplace parameter mixtures."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3.f05 import grouped
from grain_research.gptpro_foundation_v3.f13_dynamic_parameter import (
    laplace_person,
    parameter_mixture,
)
from scripts.run_grain_gptpro_f13_dynamic_source_bootstrap import dump


def main():
    """Never read outcomes/latent truth; future conditions are supplied design only."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fit", required=True, type=Path)
    parser.add_argument("--application", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source_fit.read_text())
    if source["status"] != "FITTED":
        raise ValueError("source fit unavailable")
    people = grouped([json.loads(s) for s in args.application.read_text().splitlines()])
    if set(people) & set(source["source_ids"]):
        raise ValueError("source/application overlap")
    if any(r["role"] != "application" for p in people.values() for r in p):
        raise ValueError("application-only observations required")
    ids = sorted(people)[:6] if args.smoke else sorted(people)
    budgets = [16] if args.smoke else [16, 32, 48]
    draws = 10 if args.smoke else 200
    paths = [
        args.source_fit,
        args.application,
        Path(__file__),
        Path(laplace_person.__code__.co_filename),
    ]
    protocol = {
        "source_fit": str(args.source_fit.resolve()),
        "source_bootstrap": source["bootstrap"],
        "world": source["world"],
        "seed": source["seed"],
        "budgets": budgets,
        "draws": draws,
        "approximation": "Gaussian Laplace in a,b,eta; central Hessian step1e-4; no eigenvalue repair; nonconvergence/curvature rejected",
        "decomposition": "conditional state/event covariance plus across-personal-parameter covariance; source mixture is subsequent layer",
        "files": [
            {
                "path": str(p.resolve()),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in paths
        ],
    }
    dump(args.output / "protocol_before_run.json", protocol)
    records = []
    started = time.monotonic()
    failures = 0
    wi = ["core", "near_root", "shared_shock"].index(source["world"])
    for ni, n in enumerate(budgets):
        for i, author in enumerate(ids):
            rows = people[author]
            prefix = rows[:n]
            # Fixed future z schedule, no future response value enters the estimator.
            future = [r["z"] for r in rows[n : n + 4]]
            posterior = laplace_person(prefix, source["fit"])
            record = {"author_id": author, "budget": n, "posterior": posterior}
            if posterior["status"] == "LAPLACE_APPROXIMATION":
                recipe = [source["seed"], 13, 5, wi, source["bootstrap"], ni, i, 1]
                normal = np.random.default_rng(np.random.SeedSequence(recipe)).normal(
                    size=(draws, 3)
                )
                parameters = (
                    np.asarray(posterior["fit"]["psi"])
                    + normal @ np.linalg.cholesky(posterior["parameter_covariance"]).T
                )
                mixed = parameter_mixture(
                    parameters, prefix, source["fit"]["log_sigma"], future
                )
                record.update(
                    rng_recipe=recipe, mixture={k: v.tolist() for k, v in mixed.items()}
                )
            else:
                failures += 1
                record["mixture"] = None
            records.append(record)
            if len(records) % 10 == 0:
                dump(
                    args.output / "checkpoint.json",
                    {
                        "completed": len(records),
                        "unsupported": failures,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                dump(args.output / "personal_mixtures.json", records)
    dump(args.output / "personal_mixtures.json", records)
    summary = {
        "status": "PERSONAL_PARAMETER_MIXTURES_COMPLETE_SOURCE_AGGREGATION_PENDING",
        "world": source["world"],
        "seed": source["seed"],
        "source_bootstrap": source["bootstrap"],
        "prefixes": len(records),
        "unsupported": failures,
        "parameter_draws": (len(records) - failures) * draws,
        "elapsed_seconds": time.monotonic() - started,
    }
    dump(args.output / "summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
