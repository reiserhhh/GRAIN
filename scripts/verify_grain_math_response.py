#!/usr/bin/env python3
"""Independent arithmetic/cache replay, not an independent human replication."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from run_grain_math_response import (
    DEFAULT_OUT,
    MODELS,
    ROOT,
    M,
    checkpoint,
    sha,
    write_csv,
    write_json,
)


def verify(out):
    """Fail on substantive contract/numerical inconsistency and save evidence."""
    dest = out / "05"
    dest.mkdir(parents=True, exist_ok=True)
    contract = json.loads((out / "00/input_contract.json").read_text())
    for path, digest in contract["historical_inputs"].items():
        p = ROOT / (
            path
            if path.startswith("results/")
            else "results/grain_construction_v1/" + path
        )
        if sha(p) != digest:
            raise ValueError(f"historical input changed: {p}")
    old = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    fresh = [
        json.loads(s)
        for s in (out / "06/00/material_bank.jsonl").read_text().splitlines()
    ]
    bank = old + fresh
    if len(bank) != 768 or len({r["material_id"] for r in bank}) != 768:
        raise ValueError("wrong/duplicate material count")
    groups = np.array([r["family"] for r in bank])
    roles = np.array([r["role"] for r in bank])
    bits = np.array([r["bits"] for r in bank])
    forbidden_terms = (
        "big five",
        "mbti",
        "neuroticism",
        "extraversion",
        "personality score",
    )
    if any(t in r["text"].lower() for r in bank for t in forbidden_terms):
        raise ValueError("psychological label clue in material bank")
    for family in range(48):
        mask = groups == family
        if len(set(roles[mask])) != 1 or mask.sum() != 16:
            raise ValueError("family split leakage or incomplete factorial bank")
        q, _ = M.effect_design(bits[mask])
        np.testing.assert_array_equal(q.T @ q, np.eye(15) * 16)
    # CLI cold replay exercises both linear and composed frozen programs.
    replay_rows = []
    for location in (out / "01B", out / "06"):
        frozen = json.loads((location / "model_freeze.json").read_text())
        for name, digest in frozen["bundles"].items():
            if sha(location / "reader_bundles" / name) != digest:
                raise ValueError(f"frozen model changed: {name}")
    for reader in MODELS:
        for location, arm, expected in (
            (out / "01B", "paired_dual", out / f"01B/scores_{reader}_paired_dual.npy"),
            (
                out / "06",
                "joint_binding",
                out / f"06/scores_{reader}_joint_binding.npy",
            ),
        ):
            cache_root = out if arm == "paired_dual" else out / "06"
            manifest = json.loads(
                (cache_root / f"01B/embeddings/{reader}_manifest.json").read_text()
            )
            if manifest["bank_sha256"] != sha(cache_root / "00/material_bank.jsonl"):
                raise ValueError("embedding material provenance mismatch")
            vectors = cache_root / f"01B/embeddings/{reader}.npy"
            if sha(vectors) != manifest["sha256"]:
                raise ValueError("embedding bytes changed")
            cold = dest / f"cold_{reader}_{arm}.npy"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/score_grain_math_response.py"),
                    "--bundle",
                    str(location / f"reader_bundles/{reader}_{arm}.npz"),
                    "--vectors",
                    str(vectors),
                    "--output",
                    str(cold),
                ],
                check=True,
            )
            expected_values = np.load(expected)
            if arm == "joint_binding":
                expected_values = expected_values[512:]
            delta = float(np.max(np.abs(np.load(cold) - expected_values)))
            if delta > 1e-10:
                raise ValueError(f"cold replay differs: {reader} {arm} {delta}")
            replay_rows.append(
                {
                    "reader": reader,
                    "arm": arm,
                    "max_error": delta,
                    "scored_rows": len(expected_values),
                }
            )
    # Independently reconstruct family metrics from saved predictions.
    reported = json.loads((out / "06/confirmation_metrics.json").read_text())
    max_metric_error = 0.0
    confirm = roles == "confirmation"
    for reader in MODELS:
        for arm, targets in reported[reader].items():
            scores = np.load(out / f"06/scores_{reader}_{arm}.npy")
            for target, metric in enumerate(targets):
                signed = bits[confirm, 0] * (bits[confirm, 3] if target else 1)
                own = bits[confirm, 2] == 1
                gain = float(np.mean(scores[confirm, target][own] * signed[own]) * 2)
                rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                scores[confirm, target]
                                - M.text_targets(bits[confirm])[:, target]
                            )
                            ** 2
                        )
                    )
                )
                max_metric_error = max(
                    max_metric_error,
                    abs(gain - metric["gain"]),
                    abs(rmse - metric["unit_rmse"]),
                )
    if max_metric_error > 1e-10:
        raise ValueError("metric reconstruction differs")
    with (out / "01A/comparisons.csv").open() as stream:
        finite = list(csv.DictReader(stream))
    if len({(r["world"], r["rho"], r["seed"]) for r in finite}) != 24:
        raise ValueError("finite configuration omitted")
    witnesses = json.loads((out / "01A/identifiability_witnesses.json").read_text())
    collision_witnesses = [
        w for key, w in witnesses.items() if key.startswith("collision_")
    ]
    conditional_witnesses = [
        w for key, w in witnesses.items() if key.startswith("conditional_")
    ]
    if not all(
        w["exists"] and w["map_residual"] < 1e-12
        for w in collision_witnesses + conditional_witnesses
    ):
        raise ValueError("missing exact collision certificate")
    # Score/action transport for the repaired program; source combinations only.
    score = {r: np.load(out / f"06/scores_{r}_joint_binding.npy") for r in MODELS}
    withheld = (bits[:, 2] == -1) & (bits[:, 0] == 1)
    source = (roles == "source") & ~withheld
    a = np.linalg.lstsq(
        np.column_stack([np.ones(source.sum()), score["e5"][source]]),
        score["bge"][source],
        rcond=None,
    )[0]
    error = np.column_stack([np.ones(len(bank)), score["e5"]]) @ a - score["bge"]
    transport = []
    for f in np.unique(groups[confirm]):
        mask = groups == f
        e, z = error[mask], bits[mask]
        for target in range(2):
            transport.append(
                {
                    "family": int(f),
                    "target": target,
                    "level_rmse": float(np.sqrt(np.mean(e[:, target] ** 2))),
                    "pole_defect": float(np.mean(e[:, target] * z[:, 0]) * 2),
                    "pole_owner_defect": float(
                        np.mean(e[:, target] * z[:, 0] * z[:, 2]) * 4
                    ),
                    "condition_tagged_pole_defect": float(
                        np.mean(e[:, target] * z[:, 0] * z[:, 3]) * 2
                    ),
                    "condition_tagged_pole_owner_defect": float(
                        np.mean(e[:, target] * z[:, 0] * z[:, 2] * z[:, 3]) * 4
                    ),
                    "held_combination_rmse": float(
                        np.sqrt(np.mean(e[withheld[mask], target] ** 2))
                    ),
                }
            )
    write_csv(dest / "binding_transport_defects.csv", transport)
    write_json(
        dest / "binding_transport_map.json",
        {
            "source_rows": int(source.sum()),
            "map": a.tolist(),
            "scope": "post-result diagnostic, source-only fit; same declared units and wrappers",
        },
    )
    write_csv(dest / "cold_replay.csv", replay_rows)
    write_json(
        dest / "verification.json",
        {
            "material_families": 48,
            "texts": 768,
            "encodings": 1536,
            "finite_configurations": 24,
            "design_author_appearances": 14400,
            "independent_base_random_author_streams": 1800,
            "matched_worlds_and_rho": True,
            "source_validation_application_counts_per_configuration": M.ROLES,
            "finite_validation_role": "generated and reserved; no parameter selection required",
            "collision_witnesses": len(collision_witnesses),
            "conditional_global_witnesses": len(conditional_witnesses),
            "metric_max_error": max_metric_error,
            "cold_replay": replay_rows,
            "independent_human_replication": False,
            "status": "verified_numerical_package_not_global_theory_completion",
        },
    )
    checkpoint(out, "05", "verified", [str(dest / "verification.json")])
    sources = [
        "grain_research/construction/math_response.py",
        "grain_research/construction/hidden_evidence_map.py",
        "scripts/run_grain_math_response.py",
        "scripts/run_grain_math_binding_followup.py",
        "scripts/score_grain_math_response.py",
        "scripts/verify_grain_math_response.py",
        "tests/test_grain_math_response.py",
        "configs/grain_math_material_families_v1.json",
        "configs/grain_math_confirmation_families_v1.json",
    ]
    code_manifest = {}
    for source in sources:
        target = dest / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        code_manifest[source] = sha(target)
    write_json(dest / "code_manifest.json", code_manifest)
    artifact_hashes = {
        str(p.relative_to(out)): sha(p)
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name not in ("artifact_hashes.json", "execution_progress.json")
    }
    write_json(dest / "artifact_hashes.json", artifact_hashes)
    print(
        json.dumps(
            {
                "verification": "passed",
                "metric_max_error": max_metric_error,
                "replays": replay_rows,
            }
        ),
        flush=True,
    )


def main():
    """Read existing experiment artifacts; never start encoders or fit new neural models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    verify(args.out)


if __name__ == "__main__":
    main()
