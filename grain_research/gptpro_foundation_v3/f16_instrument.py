#!/usr/bin/env python3
"""Actual source-frozen T1 construction; no random surrogate instrument scores."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .f13_contract import measurement_result
from .f13_history_contract import conditional_moments
from .f13_uncertainty import validate_covariance

CONDITIONS = (-1.0, -1 / 3, 1 / 3, 1.0)
READER_HASH = hashlib.sha256(b"synthetic_numeric_identity_reader_v1").hexdigest()


def scorer_hash():
    """Bind implementation and imported mathematical dependencies, not just CLI."""
    root = Path(__file__).parent
    files = [
        Path(__file__),
        root / "f13_contract.py",
        root / "f13_history_contract.py",
        root / "f13_uncertainty.py",
        root.parent / "evidence_history.py",
    ]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def raw_t1(records, model):
    """Score endpoint contrast using four-cell source moments and observed data."""
    if records:
        authors = {r["author_id"] for r in records}
        if len(authors) != 1 or len({r["event_id"] for r in records}) != len(records):
            raise ValueError("one author and unique events required")
        if any(r["world"] != "L" or r["z"] not in CONDITIONS for r in records):
            raise ValueError("wrong world or unsupported condition")
    mean = np.asarray(model["mu"])
    prior = validate_covariance(model["Sigma"])
    if (
        mean.shape != (4,)
        or not np.isfinite(mean).all()
        or prior.shape != (4, 4)
        or not np.isfinite(model["sigma2"])
        or model["sigma2"] <= 0
    ):
        raise ValueError("four-cell Gaussian source model required")
    contrast = np.array([-1.0, 0.0, 0.0, 1.0])
    if records:
        x = np.eye(4)[[CONDITIONS.index(r["z"]) for r in records]]
        y = np.array([r["y"] for r in records])
        out = conditional_moments(
            mean, prior, x, model["sigma2"] * np.eye(len(records)), y
        )
        mean, prior = out["mean"], out["covariance"]
    supported = {r["z"] for r in records} >= {-1.0, 1.0}
    return (
        float(contrast @ mean),
        float(contrast @ prior @ contrast),
        "observed" if supported else "model_borrowed",
    )


def build_instrument(brief, source_manifest):
    """Make a self-contained T1 bundle from verified source-only artifacts."""
    if brief != {"target_id": "T1", "world": "L", "target": "m(+1)-m(-1)"}:
        raise ValueError("this constructor only implements the declared T1 brief")
    paths = {}
    for key in ("source", "models"):
        item = source_manifest[key]
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("source hash mismatch")
        paths[key] = path
    records = [json.loads(line) for line in paths["source"].read_text().splitlines()]
    if not records or any(r["role"] != "source" for r in records):
        raise ValueError("reference construction requires source-only authors")
    model = json.loads(paths["models"].read_text())["four_cell"]
    groups = {}
    for r in records:
        groups.setdefault(r["author_id"], []).append(r)
    scores = np.array([raw_t1(rows, model)[0] for rows in groups.values()])
    scale = float(scores.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("nondegenerate source reference required")
    return {
        "target_id": "T1",
        "object_class": "conditional_response_function",
        "invalidating_result": "wrong condition support, dependence or source population invalidates the stated working interval",
        "formula_version": "four_cell_conditional_contrast_v1",
        "observed_role": "synthetic author response",
        "unit": "endpoint response contrast",
        "reference_mu": float(scores.mean()),
        "reference_sigma": scale,
        "source_hash": {k: v["sha256"] for k, v in source_manifest.items()},
        "reader_hash": READER_HASH,
        "world_contract": "F04_L_four_condition",
        "scorer_code_hash": scorer_hash(),
        "support_rule": "both endpoints observed; otherwise explicitly model-borrowed",
        "error_model": "fixed-source Gaussian; source uncertainty and four-cell calibration not certified",
        "explanation_rule": "endpoint contrast with source borrowing; not a named personality trait",
        "model": model,
        "source_reference_authors": len(groups),
    }


def cold_score(raw, bundle):
    """Score without reading source files, norms from current batch, or truth."""
    required = {
        "target_id",
        "object_class",
        "invalidating_result",
        "formula_version",
        "observed_role",
        "unit",
        "reference_mu",
        "reference_sigma",
        "source_hash",
        "reader_hash",
        "world_contract",
        "scorer_code_hash",
        "support_rule",
        "error_model",
        "explanation_rule",
        "model",
    }
    if not required <= bundle.keys():
        raise ValueError("missing bundle fields")
    if bundle["scorer_code_hash"] != scorer_hash():
        raise ValueError("scorer code hash mismatch")
    if (
        bundle["world_contract"] != "F04_L_four_condition"
        or bundle["reader_hash"] != READER_HASH
        or bundle["formula_version"] != "four_cell_conditional_contrast_v1"
        or bundle["object_class"] != "conditional_response_function"
        or not np.isfinite(bundle["reference_mu"])
        or not np.isfinite(bundle["reference_sigma"])
        or bundle["reference_sigma"] <= 0
    ):
        raise ValueError("unsupported bundle contract or reference")
    if bundle["target_id"] != "T1" or raw["world_contract"] != bundle["world_contract"]:
        raise ValueError("wrong target/world")
    if raw["reader_hash"] != bundle["reader_hash"]:
        raise ValueError("wrong reader hash")
    score, variance, support = raw_t1(raw["records"], bundle["model"])
    result = measurement_result(
        score,
        variance,
        {
            "target": "T1",
            "world": bundle["world_contract"],
            "method": bundle["formula_version"],
        },
        support,
    )
    result.update(
        reference_score=(score - bundle["reference_mu"]) / bundle["reference_sigma"],
        reference_variance=variance / bundle["reference_sigma"] ** 2,
        source_uncertainty_included=False,
    )
    return result


def main():
    """Cold CLI consumes only a trusted bundle and declared application inputs."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.bundle.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise ValueError("untrusted bundle hash")
    bundle = json.loads(payload)
    inputs = [json.loads(line) for line in args.input.read_text().splitlines()]
    results = [cold_score(raw, bundle) for raw in inputs]
    args.output.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in results)
    )


if __name__ == "__main__":
    main()
