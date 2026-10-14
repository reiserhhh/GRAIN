#!/usr/bin/env python3
"""Independent algebra/readout replay of frozen HM-04, without new observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_role_moments as M

DEFAULT = ROOT / "results/grain_math_response_v1/local_roles_r1"


def sha(path):
    """Read bytes only to verify the frozen material and program manifests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    """Read public experiment records; no external data sources are opened."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_csv(path):
    """Read the frozen decimal scalar exports."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def model_groups(path):
    """Load numeric arrays without importing the original fitting/scoring code."""
    groups = defaultdict(dict)
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            group, key = name.split("__", 1)
            groups[group][key] = data[name]
    return groups


def affine(model, x):
    """Distribute the frozen affine intercept, unlike the primary centered form."""
    return x @ model["w"] + (model["bias"] - model["mu"] @ model["w"])


def tensor(model, a, b):
    """Expand the product kernel, then apply centering after dual contraction."""
    aa, bb = a @ model["a"].T, b @ model["b"].T
    kernel = 1 + aa + bb + aa * bb
    dual = model["dual"]
    return (
        kernel @ dual
        - kernel.mean(axis=1) * dual.sum()
        - model["column_mean"] @ dual
        + model["grand_mean"] * dual.sum()
        + model["bias"]
    )


def resolve(refs, old, new):
    """Resolve frozen vector rows using only observation indices."""
    return np.array([old[i] if space == "source" else new[i] for space, i in refs])


def main():
    """Verify exact unit identities separately from semantic performance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/local_roles_verification_r1",
    )
    args = parser.parse_args()
    source = ROOT / "results/grain_math_response_v1/binding_r1"
    hash_count = 0
    for root in (source, args.source):
        for name, digest in json.loads(
            (root / "04/artifact_hashes.json").read_text()
        ).items():
            if sha(root / name) != digest:
                raise AssertionError(f"frozen artifact changed: {root / name}")
            hash_count += 1
    bank = read_jsonl(args.source / "00/documents.jsonl")
    by_id = {r["material_id"]: r for r in bank}
    refs = json.loads((args.source / "01/observation_indices.json").read_text())
    slots = np.array(
        json.loads((source / "01/slot_indices.json").read_text())["declared"]
    )
    rows = read_csv(args.source / "02/event_readouts.csv")
    predictions = read_csv(args.source / "02/predictions.csv")
    exported = defaultdict(list)
    for r in rows:
        exported[r["reader"], r["target"], r["material_id"], r["arm"]].append(r)
    coordinates, unit_errors, profile_errors, moment_errors = [], [], [], []
    unresolved = Counter()
    for reader in ("e5", "bge"):
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(args.source / f"01B/embeddings/{reader}.npy").astype(float)
        xdoc = resolve(refs["documents"], old, new)
        for target in sorted({r["target"] for r in bank}):
            groups = model_groups(source / f"02/{reader}_{target}.npz")
            declared = affine(groups["whole_factors"], old[:1536])
            declared[:, 0] = tensor(
                groups["slot_tensor"], old[slots[:, 0]], old[slots[:, 1]]
            )
            whole = affine(groups["whole_factors"], xdoc)
            target_indices = [i for i, r in enumerate(bank) if r["target"] == target]
            flat = [e for i in target_indices for e in refs["automatic_events"][i]]
            auto = affine(
                groups["whole_factors"], resolve([e["readout"] for e in flat], old, new)
            )
            auto[:, 0] = tensor(
                groups["auto_tensor"],
                resolve([e["action"] for e in flat], old, new),
                resolve([e["object"] for e in flat], old, new),
            )
            cursor = 0
            for i in target_indices:
                r, ar = bank[i], refs["automatic_events"][i]
                az = auto[cursor : cursor + len(ar)]
                cursor += len(ar)
                valid = [
                    j
                    for j, e in enumerate(ar)
                    if e["status"] == "ok" and e["event_id"] is not None
                ]
                si = [e["source_index"] for e in r["events"]]
                shared = np.repeat(whole[i : i + 1], len(si), axis=0)
                shared[:, 0] = declared[si, 0]
                ashared = np.repeat(whole[i : i + 1], len(ar), axis=0)
                ashared[:, 0] = az[:, 0]
                broken = declared[si].copy()
                broken[:, 2] = np.roll(broken[:, 2], 1)
                arms = {
                    "whole_document": whole[i : i + 1],
                    "declared_shared": shared,
                    "declared_local": declared[si],
                    "broken_ownership_binding": broken,
                    "automatic_shared": ashared[valid],
                    "automatic_local": az[valid],
                }
                for arm, actual in arms.items():
                    outrows = exported[reader, target, r["material_id"], arm]
                    observed = np.array(
                        [[float(e[f]) for f in ("k", "p", "o", "a")] for e in outrows]
                    )
                    if len(actual):
                        if actual.shape != observed.shape:
                            raise AssertionError("event replay alignment")
                        coordinates.append(float(np.max(abs(actual - observed))))
    for p in predictions:
        r = by_id[p["material_id"]]
        key = p["reader"], p["target"], p["material_id"], p["arm"]
        first = {}
        for event in exported[key]:
            first.setdefault(event["event_id"], event)
        z = np.array(
            [[float(e[f]) for f in ("k", "p", "o", "a")] for e in first.values()]
        )
        expected = {}
        for e in r["events"]:
            k, polarity, owner, actuality = e["bits"]
            expected.setdefault(
                e["event_id"], float(polarity) if k == owner == actuality == 1 else 0.0
            )
        if not np.isclose(
            float(p["expected_unit"]), np.mean(list(expected.values())), atol=1e-14
        ):
            raise AssertionError("truth unit reconstruction")
        if p["complete"] == "True":
            gate = np.prod(
                np.maximum(0, np.minimum(1, (1 + z[:, [0, 2, 3]]) / 2)), axis=1
            )
            values = gate * z[:, 1]
            if p["unit"]:
                unit_errors.append(abs(float(p["unit"]) - float(values.mean())))
            qualified = np.logical_and.reduce((z[:, 0] > 0, z[:, 2] > 0, z[:, 3] > 0))
            if bool(p["prediction"]) != bool(qualified.any()):
                raise AssertionError("absence misreported as an author score")
            if qualified.any():
                profile_errors.append(
                    abs(float(p["prediction"]) - float(values[qualified].mean()))
                )
            clipped = z.copy()
            clipped[:, [0, 2, 3]] = np.clip(clipped[:, [0, 2, 3]], -1, 1)
            m = M.joint_moments(clipped)
            moment_errors.append(
                abs(M.contribution_from_moments(m) - float(values.mean()))
            )
        else:
            if p["prediction"] or p["unit"]:
                raise AssertionError("incomplete observations given a point profile")
            unresolved[p["reader"], p["target"], p["arm"]] += 1
    contrasts = read_csv(args.source / "03/same_word_role_contrasts.csv")
    for c in contrasts:
        a, b = by_id[c["left_id"]]["text"], by_id[c["right_id"]]["text"]
        normalize = lambda s: Counter(
            s.lower().replace(",", "").replace(".", "").split()
        )
        if normalize(a) != normalize(b):
            raise AssertionError("word-matched contrast changed words")
    errors = coordinates + unit_errors + profile_errors + moment_errors
    if max(errors, default=0) > 1e-10:
        raise AssertionError(max(errors))
    report = {
        "status": "PASS",
        "frozen_hashes_verified": hash_count,
        "readout_arrays": len(coordinates),
        "event_rows": len(rows),
        "prediction_rows": len(predictions),
        "word_matched_contrast_rows": len(contrasts),
        "max_readout_error": max(coordinates, default=0),
        "max_unit_error": max(unit_errors, default=0),
        "max_profile_error": max(profile_errors, default=0),
        "max_joint_moment_error": max(moment_errors, default=0),
        "scope": "same-agent alternate formulas; numerical replay is separate from semantic success",
        "new_encodings": 0,
        "new_fits": 0,
        "api_calls": 0,
        "private_reads": 0,
        "source_manifest_sha256": sha(args.source / "04/artifact_hashes.json"),
        "verification_code_sha256": sha(Path(__file__)),
        "moment_code_sha256": sha(
            ROOT / "grain_research/construction/math_role_moments.py"
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
