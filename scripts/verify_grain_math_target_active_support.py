#!/usr/bin/env python3
"""Alternate HM36 QR/SVD, dual contractions and signed half-integral replay."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support as RUN
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import table

VERIFY = ROOT / "results/grain_math_response_v1/target_active_support_verification_r1"


def qr_svd_basis(raw, metadata):
    """Construct the same declared rank from QR then SVD, without a Gram eigensolve."""
    scales = np.linalg.norm(raw, axis=0)
    matrix = raw[:, scales > 0] / scales[scales > 0]
    q, r = np.linalg.qr(matrix, mode="reduced")
    left, singular, _ = np.linalg.svd(r, full_matrices=False)
    selected = singular**2 > metadata["rank_tolerance"]
    if int(selected.sum()) != metadata["rank"]:
        raise ValueError("alternate QR/SVD disagrees with declared source rank")
    return q @ left[:, selected]


def verify(out, destination):
    """Replay all saved numerical values and source ordering within the fixed budget."""
    if destination.exists():
        raise FileExistsError("preserve existing verification")
    started = time.monotonic()
    m = check(out)
    spec, status = m["spec"], read_json(out / "run_status.json")
    if status["status"] != "EVALUATED" or (
        status["learned_bases"],
        status["random_bases"],
        status["application_nodes_loaded"],
    ) != (9, 18, 144):
        raise ValueError("complete actual construction and evaluation required")
    barrier_file = out / "01/source_barrier.json"
    barrier = read_json(barrier_file)
    if (
        barrier["application_nodes_loaded"] != 0
        or status["application_started_at"] <= barrier["frozen_at"]
    ):
        raise ValueError("application read before source barrier")
    if status["source_sha256"] != digest(barrier_file):
        raise ValueError("wrong source barrier")
    hashes = read_json(out / "02/artifact_hashes.json")
    for name, sha in {**barrier["files"], **hashes}.items():
        if digest(out / name) != sha:
            raise ValueError("output changed " + name)
    columns = table(out / "01/source_columns.csv")
    if len(columns) != 288 or any(r["role"] != "source" for r in columns):
        raise ValueError("source-only column provenance required")
    nodes = {
        (int(r["path"]), int(r["node"]), r["arm"], r["space"]): r
        for r in table(out / "02/nodes.csv")
    }
    integrals = {
        (int(r["path"]), r["arm"], r["space"], r["rule"]): r
        for r in table(out / "02/integrals.csv")
    }
    if len(nodes) != 5184 or len(integrals) != 3456:
        raise ValueError("complete node/integral battery required")
    errors = {
        key: 0.0
        for key in (
            "basis_orthogonality",
            "basis_subspace",
            "source_columns",
            "action",
            "geometry",
            "integral",
            "metric",
        )
    }
    parent, design = ROOT / spec["parent"], read_json(out / "00/design.json")

    def bounded():
        if (
            status["analysis_seconds"] + time.monotonic() - started
            > spec["maximum_analysis_seconds"]
        ):
            raise TimeoutError("combined evaluation and verification budget reached")

    for query in spec["queries"]:
        _, rawbank = RUN.load(out / "01/models" / f"{query}_raw.json")
        actual_columns = []
        for path in design:
            if path["query"] != query or path["role"] != "source":
                continue
            for node in range(3):
                _, v, _, _ = RUN.source_node(parent, path, node, "source")
                actual_columns.extend(z.ravel() for z in v)
        rawv = np.stack(actual_columns, axis=1)
        errors["source_columns"] = max(
            errors["source_columns"], float(np.max(abs(rawv - rawbank["V"])))
        )
        models = {}
        for file in sorted((out / "01/models").glob(query + "_*.json")):
            if file.stem.endswith("_raw"):
                continue
            meta, saved = RUN.load(file)
            if meta["random"]:
                gaussian = np.random.default_rng(meta["stream_seed"]).standard_normal(
                    (meta["dimension"], meta["rank"])
                )
                basis = np.linalg.qr(gaussian, mode="reduced")[0]
            else:
                raw = rawbank.get(meta["space"])
                if raw is None:
                    raw = np.column_stack((rawbank["D"], rawbank["V"]))
                basis = qr_svd_basis(raw, meta)
            old = saved["basis"]
            rank = basis.shape[1]
            errors["basis_orthogonality"] = max(
                errors["basis_orthogonality"],
                float(np.max(abs(old.T @ old - np.eye(rank)))),
            )
            # Frobenius angle measure avoids constructing an ambient projector.
            overlap = old.T @ basis
            errors["basis_subspace"] = max(
                errors["basis_subspace"],
                float(np.max(abs(np.eye(rank) - overlap @ overlap.T))),
            )
            models[meta["space"]] = basis
            bounded()
        for path in design:
            if path["query"] != query:
                continue
            accumulated = {space: [] for space in models}
            for node in range(3):
                _, v, delta, _ = RUN.source_node(parent, path, node, path["role"])
                d, vflat = delta.ravel(), v.reshape(2, -1)
                for space, basis in models.items():
                    projection = basis @ (basis.T @ d)
                    full = np.einsum("ij,j->i", vflat, d)
                    supported = np.einsum("ij,j->i", vflat, projection)
                    missing = np.einsum("ij,j->i", vflat, d - projection)
                    values = np.stack((full, supported, missing), axis=1)
                    accumulated[space].append(values)
                    fraction = float(np.sum(projection**2) / np.sum(d**2))
                    for i, arm in enumerate(spec["arms"]):
                        row = nodes[path["path"], node, arm, space]
                        for j, key in enumerate(("full", "supported", "missing")):
                            errors["action"] = max(
                                errors["action"], abs(values[i, j] - float(row[key]))
                            )
                        errors["geometry"] = max(
                            errors["geometry"],
                            abs(
                                fraction - float(row["displacement_supported_fraction"])
                            ),
                        )
                        projected_v = basis @ (basis.T @ vflat[i])
                        vf = float(np.sum(projected_v**2) / np.sum(vflat[i] ** 2))
                        errors["geometry"] = max(
                            errors["geometry"],
                            abs(vf - float(row["covector_supported_fraction"])),
                        )
                bounded()
            for space, values in accumulated.items():
                a = np.stack(values)
                for rule, weights in (("simpson", (1, 4, 1)), ("trapezoid", (1, 2, 1))):
                    denominator = 12 if rule == "simpson" else 8
                    result = np.einsum("n,nij->ij", np.array(weights) / denominator, a)
                    for i, arm in enumerate(spec["arms"]):
                        row = integrals[path["path"], arm, space, rule]
                        for j, key in enumerate(("full", "supported", "missing")):
                            errors["integral"] = max(
                                errors["integral"], abs(result[i, j] - float(row[key]))
                            )
        del rawv, actual_columns, rawbank, models
        print(
            {"verified_query": query, "elapsed": time.monotonic() - started}, flush=True
        )
        bounded()
    rows = list(integrals.values())
    metrics = table(out / "02/metrics.csv")
    for metric in metrics:
        key = tuple(metric[k] for k in ("role", "query", "arm", "space", "rule"))
        chosen = [
            r
            for r in rows
            if tuple(r[k] for k in ("role", "query", "arm", "space", "rule")) == key
        ]
        values = np.asarray(
            [[float(r[k]) for k in ("full", "supported", "missing")] for r in chosen]
        )
        rms = np.linalg.norm(values, axis=0) / np.sqrt(len(values))
        cross = 2 * float(np.dot(values[:, 1], values[:, 2])) / len(values)
        expected = {
            "full_rms": rms[0],
            "supported_rms": rms[1],
            "missing_rms": rms[2],
            "supported_over_full": rms[1] / rms[0],
            "missing_over_full": rms[2] / rms[0],
            "signed_cross_mean": cross,
            "cross_over_full_energy": cross / rms[0] ** 2,
            "identity_error": abs(rms[0] ** 2 - rms[1] ** 2 - rms[2] ** 2 - cross),
        }
        if len(chosen) != 16 or int(metric["count"]) != 16:
            raise ValueError("all sixteen paths per group required")
        for name, value in expected.items():
            errors["metric"] = max(
                errors["metric"], abs(float(value) - float(metric[name]))
            )
    bounded()
    if any(e > spec["absolute_tolerance"] for e in errors.values()):
        raise ValueError(f"fixed alternate numerical tolerance failed: {errors}")
    destination.mkdir(parents=True)
    write_json(
        destination / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM36_NOT_GLOBAL",
            "errors": errors,
            "output_hashes_checked": len(hashes),
            "source_columns": len(columns),
            "node_rows": len(nodes),
            "integrals": len(integrals),
            "metrics": len(metrics),
            "verification_seconds": time.monotonic() - started,
            "combined_analysis_seconds": status["analysis_seconds"]
            + time.monotonic()
            - started,
            "source_sha256": digest(barrier_file),
            "verified_at": now(),
            "new_model_calls": 0,
            "scope": "same-agent alternate numerical replay, not independent human/psychological validation",
        },
    )
    print(read_json(destination / "summary.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RUN.OUT)
    parser.add_argument("--destination", type=Path, default=VERIFY)
    args = parser.parse_args()
    verify(args.out, args.destination)
