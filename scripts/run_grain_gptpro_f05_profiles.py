#!/usr/bin/env python3
"""Source-fixed twenty-author F05 likelihood and prior-curvature diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f05 import grouped
from grain_research.gptpro_foundation_v3.f05_profiles import (
    central_hessian,
    profile_point,
)
from grain_research.gptpro_foundation_v3.f05_recovery import fit_person, kalman_loglik
from scripts.run_grain_gptpro_f04_application_recovery import dump, table


def main():
    """Freeze lexical source-ID selection before calculations, never use truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve earlier diagnostics")
    args.output.mkdir(parents=True)
    for item in json.loads((args.source / "output_manifest.json").read_text())["files"]:
        assert sha256(args.source / item["path"]) == item["sha256"]
    source = json.loads((args.source / "summary.json").read_text())
    if bool(source["smoke"]) != args.smoke:
        raise ValueError("smoke/full mismatch")
    selected = []
    for outcome in source["outcomes"]:
        world, seed = outcome["world"], outcome["seed"]
        path = args.source / f"{world}_{seed}" / "source.jsonl"
        ids = sorted(
            {json.loads(line)["author_id"] for line in path.read_text().splitlines()}
        )[: 2 if args.smoke else 20]
        selected.append({"world": world, "seed": seed, "author_ids": ids})
    dump(
        args.output / "fixed_author_selection.json",
        {
            "role": "source",
            "rule": "first20 opaque IDs in lexical order per world/seed, fixed before profile evaluation",
            "selections": selected,
        },
    )
    profiles, curvatures, summaries = [], [], []
    for selection in selected:
        world, seed = selection["world"], selection["seed"]
        root = args.source / f"{world}_{seed}"
        people = grouped(
            [
                json.loads(line)
                for line in (root / "source.jsonl").read_text().splitlines()
            ]
        )
        model = json.loads((root / "source_model.json").read_text())
        prior = model["prior"]
        precision = np.linalg.inv(prior["Sigma"])
        for author in selection["author_ids"]:
            for n in [8] if args.smoke else [16, 32, 48]:
                prefix = people[author][:n]
                fit = fit_person(prefix, model["log_sigma"], prior)
                psi = np.asarray(fit["psi"])

                def likelihood(x, prefix=prefix, model=model):
                    return -kalman_loglik(x, prefix, model["log_sigma"])["loglik"]

                def posterior(
                    x, likelihood=likelihood, prior=prior, precision=precision
                ):
                    delta = x - np.asarray(prior["mu"])
                    return likelihood(x) + 0.5 * delta @ precision @ delta

                hd = central_hessian(likelihood, psi, 1e-4)
                hp = central_hessian(posterior, psi, 1e-4)
                error = float(np.max(np.abs(hp - hd - precision)))
                # Central differences amplify cancellation; verify at its scale,
                # not the exact covariance-identity tolerance.
                if error > 1e-4:
                    raise AssertionError("prior Hessian addition mismatch")
                curvatures.append(
                    {
                        "world": world,
                        "seed": seed,
                        "author_id": author,
                        "n": n,
                        "psi": psi.tolist(),
                        "fit_success": fit["success"],
                        "attempts": fit["attempts"],
                        "data_hessian": hd.tolist(),
                        "prior_precision": precision.tolist(),
                        "posterior_hessian": hp.tolist(),
                        "data_eigenvalues": np.linalg.eigvalsh(hd).tolist(),
                        "posterior_eigenvalues": np.linalg.eigvalsh(hp).tolist(),
                        "addition_error": error,
                    }
                )
                for mode in ("likelihood", "posterior"):
                    values = [
                        profile_point(
                            prefix,
                            float(rho),
                            model["log_sigma"],
                            prior if mode == "posterior" else None,
                        )
                        for rho in np.arange(50) * 0.02
                    ]
                    objectives = np.array([v["objective"] for v in values])
                    minimum = objectives.min()
                    drop = 2 * (objectives - minimum)
                    for value, difference in zip(values, drop):
                        profiles.append(
                            {
                                "world": world,
                                "seed": seed,
                                "author_id": author,
                                "n": n,
                                "mode": mode,
                                "rho": value["rho"],
                                "a": value["psi"][0],
                                "b": value["psi"][1],
                                "negative_loglik": value["negative_loglik"],
                                "prior_penalty": value["prior_penalty"],
                                "twice_objective_drop": float(difference),
                            }
                        )
                    near = np.array([v["rho"] for v, d in zip(values, drop) if d <= 2])
                    summaries.append(
                        {
                            "world": world,
                            "seed": seed,
                            "author_id": author,
                            "n": n,
                            "mode": mode,
                            "grid_best_rho": values[int(np.argmin(objectives))]["rho"],
                            "diagnostic_drop2_width": float(near.max() - near.min()),
                            "continuous_MAP_rho": float(0.99 * np.tanh(psi[2])),
                            "MAP_outside_positive_grid": bool(psi[2] < 0),
                            "data_hessian_min_eigenvalue": float(
                                np.linalg.eigvalsh(hd).min()
                            ),
                            "fit_success": fit["success"],
                        }
                    )
        print(
            json.dumps({"world": world, "seed": seed, "profile_rows": len(profiles)}),
            flush=True,
        )
    table(args.output / "likelihood_profiles.csv", profiles)
    table(args.output / "profile_summary.csv", summaries)
    dump(args.output / "hessian_diagnostics.json", curvatures)
    dump(
        args.output / "summary.json",
        {
            "status": "SOURCE_DIAGNOSTICS_EXECUTED",
            "profile_rows": len(profiles),
            "hessians": len(curvatures),
            "source_refitted": False,
            "selection": "20 source IDs per world/seed, before diagnostic evaluation",
            "interpretation": "conditional on learned shared noise; positive rho grid; drop2 widths descriptive, not confidence intervals; prior and data curvature separated",
        },
    )
    paths = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f05_profiles.py",
        ROOT / "grain_research/gptpro_foundation_v3/f05_recovery.py",
    ]
    (args.output / "source_archive").mkdir()
    for path in paths:
        (args.output / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
        )
    dump(
        args.output / "input_manifest.json",
        {
            "source_manifest_sha256": sha256(args.source / "output_manifest.json"),
            "files": [{"path": str(p), "sha256": sha256(p)} for p in paths],
        },
    )
    dump(
        args.output / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(args.output)), "sha256": sha256(p)}
                for p in sorted(args.output.rglob("*"))
                if p.is_file() and not p.name.startswith("._")
            ]
        },
    )


if __name__ == "__main__":
    main()
