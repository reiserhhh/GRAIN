#!/usr/bin/env python3
"""Independent arithmetic readback of the frozen HM150 joint experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/grain_math_response_v1/joint_persona_style_r2"
PROGRAM = ROOT / "results/grain_math_response_v1/semantic_anchor_r1/program.npz"


def read_json(path):
    """Read a declared JSON artifact without modifying it."""
    return json.loads(path.read_text())


def digest(path):
    """Authenticate files incrementally, including large dependencies."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def tensor_by_coordinates(values):
    """Reconstruct all625 coordinates by direct products, not runner einsums."""
    affine = np.concatenate((np.ones((len(values), 4, 1)), values), axis=2)
    return np.column_stack(
        [
            np.prod(affine[:, range(4), indices], axis=1)
            for indices in itertools.product(range(5), repeat=4)
        ]
    )


def seed_identity(values):
    """Compute signed three-replicate support by explicit pair enumeration."""
    support = sum(float(values[i] @ values[j]) for i, j in ((0, 1), (0, 2), (1, 2))) / 3
    mean = values.mean(axis=0)
    spread = sum(float((row - mean) @ (row - mean)) for row in values) / 6
    return support, float(mean @ mean), spread


def verify(out):
    """Verify row bindings, hashes, cross-seed accounts and all score accounts."""
    manifest = read_json(out / "manifest.json")
    checks = 0
    for name, expected in manifest["hashes"].items():
        assert digest(ROOT / name) == expected, name
        checks += 1
    assert digest(out / "materials.json") == manifest["materials_sha256"]
    for seal in ("generation_seal.json", "observation_seal.json", "output_seal.json"):
        for name, expected in read_json(out / seal).items():
            assert digest(out / name) == expected, name
            checks += 1
    for stage in ("generation", "observation"):
        assert read_json(out / f"{stage}_status.json")["status"] == "COMPLETE"
    rows = read_json(out / "materials.json")
    records = [
        r
        for p in sorted((out / "batches").glob("[0-9][0-9][0-9].json"))
        for r in read_json(p)
    ]
    assert len(rows) == len(records) == 702
    assert [r["text_id"] for r in rows] == [r["text_id"] for r in records]
    for row, record in zip(rows, records, strict=True):
        assert all(type(v) is int for v in row["prompt_ids"])
        assert all(type(v) is int for v in record["new_ids"])
        assert record["n_new"] == len(record["new_ids"]) <= 192
    address = {r["text_id"]: i for i, r in enumerate(rows)}
    with np.load(out / "numeric_observations.npz", allow_pickle=False) as source:
        values = {
            key: np.asarray(source[key], float) for key in manifest["representations"]
        }
    with np.load(PROGRAM, allow_pickle=False) as program:
        z = (values["grain4_slices"] - program["origin"]) / program["scale"][
            None, :, None
        ]
        weights = program["coefficients"]
    values["grain4_slices"] = z.reshape(702, -1)
    errors = {"seed_support": 0.0, "seed_identity": 0.0, "score_accounts": 0.0}
    support_count = 0
    with (out / "seed_support.csv").open(newline="") as stream:
        for result in csv.DictReader(stream):
            if result["supported"] != "True":
                continue
            contrasts = []
            for seed in (11, 23, 37):
                ids = [
                    address[
                        f"{result['axis']}|{result['family']}|p{pole}|{result['situation']}|s{seed}|{style}"
                    ]
                    for style, pole in (
                        (result["style"], 1),
                        ("neutral", 1),
                        (result["style"], -1),
                        ("neutral", -1),
                    )
                ]
                hs, h0, ls, l0 = values[result["representation"]][ids]
                contrasts.append((hs - h0) - (ls - l0))
            support, mean, spread = seed_identity(np.asarray(contrasts))
            for key, expected in (
                ("cross_seed_support", support),
                ("mean_squared_norm", mean),
                ("within_seed_term", spread),
            ):
                actual = float(result[key])
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
                errors["seed_support"] = max(
                    errors["seed_support"], abs(actual - expected)
                )
            errors["seed_identity"] = max(
                errors["seed_identity"], abs(mean - support - spread)
            )
            support_count += 1
    score_count = 0
    with (out / "score_accounts.csv").open(newline="") as stream:
        for result in csv.DictReader(stream):
            ids = [
                address[
                    f"{result['axis']}|{result['family']}|p{pole}|{result['situation']}|s{result['seed']}|{style}"
                ]
                for style, pole in (
                    (result["style"], 1),
                    ("neutral", 1),
                    (result["style"], -1),
                    ("neutral", -1),
                )
            ]
            hs, h0, ls, l0 = z[ids]
            corner_features = tensor_by_coordinates(z[ids])
            upper = tensor_by_coordinates((h0 + ls - l0)[None])[0]
            lower = tensor_by_coordinates((hs - h0 + l0)[None])[0]
            hs_f, h0_f, ls_f, l0_f = corner_features
            parts = {
                "observed": (hs_f - h0_f) - (ls_f - l0_f),
                "upper_link": (upper - h0_f) - (ls_f - l0_f),
                "upper_action": hs_f - upper,
                "lower_link": (hs_f - h0_f) - (lower - l0_f),
                "lower_action": lower - ls_f,
            }
            arm = ("full", "single", "source_relinked").index(result["arm"])
            target = ("extraversion", "neuroticism").index(result["read_target"])
            for key, part in parts.items():
                expected, actual = (
                    float(part @ weights[arm, target]),
                    float(result[key]),
                )
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
                errors["score_accounts"] = max(
                    errors["score_accounts"], abs(actual - expected)
                )
            score_count += 1
    field_checks = 0
    field_root = out / "response_field_companion"
    if field_root.exists():
        field_seal = read_json(field_root / "seal.json")
        for name, expected in field_seal["codes"].items():
            assert digest(ROOT / name) == expected, name
        for name, expected in field_seal["outputs"].items():
            assert digest(field_root / name) == expected, name
        assert digest(out / "output_seal.json") == field_seal["parent_output_seal"]
        with (field_root / "response_fields.csv").open(newline="") as stream:
            for result in csv.DictReader(stream):
                if result["supported"] != "True":
                    continue
                neutral, styled = [], []
                for seed in (11, 23, 37):
                    for style, bucket in (
                        ("neutral", neutral),
                        (result["style"], styled),
                    ):
                        ids = [
                            address[
                                f"{result['axis']}|{result['family']}|p{pole}|{result['situation']}|s{seed}|{style}"
                            ]
                            for pole in (1, -1)
                        ]
                        bucket.append(
                            values[result["representation"]][ids[0]]
                            - values[result["representation"]][ids[1]]
                        )
                x, y = np.asarray(neutral), np.asarray(styled)
                ux, _, _ = seed_identity(x)
                uy, _, _ = seed_identity(y)
                um, _, _ = seed_identity(y - x)
                alignment = (
                    sum(
                        float(x[i] @ y[j]) for i in range(3) for j in range(3) if i != j
                    )
                    / 6
                )
                for key, expected in (
                    ("neutral_support_per_coordinate", ux / x.shape[1]),
                    ("styled_support_per_coordinate", uy / x.shape[1]),
                    ("alignment_per_coordinate", alignment / x.shape[1]),
                    ("mixed_support_per_coordinate", um / x.shape[1]),
                ):
                    np.testing.assert_allclose(
                        float(result[key]), expected, atol=1e-9, rtol=1e-9
                    )
                np.testing.assert_allclose(
                    um, ux + uy - 2 * alignment, atol=1e-9, rtol=1e-9
                )
                field_checks += 1
    return {
        "verified": True,
        "actual_text_rows": len(rows),
        "authenticated_files": checks,
        "supported_seed_groups": support_count,
        "score_rows": score_count,
        "response_field_groups_verified": field_checks,
        "max_errors": errors,
        "new_model_calls": 0,
        "independence": "same-agent alternate arithmetic,not third-party replication",
    }


def main():
    """Only read scientific artifacts; optionally create a new audit record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    result = verify(args.output)
    if args.audit:
        with args.audit.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
