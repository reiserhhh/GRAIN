#!/usr/bin/env python3
"""Probe role-space group/metric transport using the actual frozen HM-03 arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_grain_math_binding import (
    ROOT,
    load_model,
    read_bank,
    sha,
    write_csv,
    write_json,
)

from grain_research.construction import math_binding as B
from grain_research.construction import math_binding_geometry as G


def main():
    """Run algebraic transport only: no encoder calls, fitting or private reads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/binding_r1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/binding_geometry_r1",
    )
    args = parser.parse_args()
    manifest = json.loads((args.source / "04/artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        if sha(args.source / name) != digest:
            raise ValueError(f"HM-03 artifact changed: {name}")
    bank = read_bank(args.source / "00/singles.jsonl")
    slots = np.array(
        json.loads((args.source / "01/slot_indices.json").read_text())["declared"]
    )
    code = [
        Path(__file__),
        ROOT / "grain_research/construction/math_binding_geometry.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in code}
    signature["binding_manifest"] = sha(args.source / "04/artifact_hashes.json")
    protocol = args.out / "00/protocol_before_run.json"
    if protocol.exists() and json.loads(protocol.read_text())["signature"] != signature:
        raise ValueError(
            "geometry protocol changed; preserve r1 and use another output"
        )
    for p in code:
        dest = args.out / "00/code" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    write_json(
        protocol,
        {
            "signature": signature,
            "seeds": [8, 108, 208, 308],
            "scope": "same-reader coordinate transport; not between-reader semantic transport",
            "new_encodings": 0,
            "new_fits": 0,
            "private_reads": 0,
            "source_models_frozen": True,
            "application_metrics_do_not_select_a_model": True,
        },
    )
    records, pairing = [], []
    for reader in ("e5", "bge"):
        x = np.load(args.source / f"01B/embeddings/{reader}.npy").astype(float)
        for target in sorted({r["target"] for r in bank}):
            chosen = np.array([i for i, r in enumerate(bank) if r["target"] == target])
            selected = chosen[
                np.array([bank[i]["role"].startswith("unseen") for i in chosen])
            ]
            ai, bi = slots[selected, 0], slots[selected, 1]
            a, b = x[ai], x[bi]
            model = load_model(args.source / f"02/{reader}_{target}.npz")["slot_tensor"]
            reference = B.score_kernel(model, a, b)
            for seed in (8, 108, 208, 308):
                for scaled in (False, True):
                    qa, qb = (
                        G.role_basis(a.shape[1], seed, scaled),
                        G.role_basis(b.shape[1], seed + 1, scaled),
                    )
                    ta, tb = G.transform_role(a, qa), G.transform_role(b, qb)
                    ar, br = (
                        G.transform_role(model["a"], qa),
                        G.transform_role(model["b"], qb),
                    )
                    transported = G.metric_kernel(
                        ta, tb, ar, br, qa["scale"] ** -2, qb["scale"] ** -2
                    )
                    wrong_metric = G.metric_kernel(
                        ta, tb, ar, br, np.ones(a.shape[1]), np.ones(b.shape[1])
                    )
                    wrong_reference = G.metric_kernel(
                        ta,
                        tb,
                        model["a"],
                        model["b"],
                        np.ones(a.shape[1]),
                        np.ones(b.shape[1]),
                    )
                    for arm, k in (
                        ("transported", transported),
                        ("untransported_metric", wrong_metric),
                        ("query_only", wrong_reference),
                    ):
                        values = G.frozen_kernel_score(model, k)
                        records.append(
                            {
                                "reader": reader,
                                "target": target,
                                "seed": seed,
                                "basis": "scaled" if scaled else "orthogonal",
                                "arm": arm,
                                "n": len(values),
                                "rmse_change": float(
                                    np.sqrt(np.mean((values - reference) ** 2))
                                ),
                                "max_abs_change": float(
                                    np.max(abs(values - reference))
                                ),
                            }
                        )
                    if (
                        np.max(
                            abs(G.frozen_kernel_score(model, transported) - reference)
                        )
                        > 1e-10
                    ):
                        raise AssertionError("transported action changed")
                qa, qb = (
                    G.role_basis(a.shape[1], seed),
                    G.role_basis(b.shape[1], seed + 1),
                )
                original = G.hadamard_interaction_gram(a, b)
                for arm, ga, gb in (
                    ("shared_basis", qa, qa),
                    ("independent_bases", qa, qb),
                ):
                    h = G.hadamard_interaction_gram(
                        G.transform_role(a, ga), G.transform_role(b, gb)
                    )
                    pairing.append(
                        {
                            "reader": reader,
                            "target": target,
                            "seed": seed,
                            "arm": arm,
                            "relative_gram_change": float(
                                np.linalg.norm(h - original) / np.linalg.norm(original)
                            ),
                        }
                    )
    write_csv(args.out / "01/coordinate_transport.csv", records)
    write_csv(args.out / "01/hadamard_role_identification.csv", pairing)
    write_json(
        args.out / "summary.json",
        {
            "status": "finite_geometry_probe_verified",
            "transports": sum(r["arm"] == "transported" for r in records),
            "transported_predictions": sum(
                r["n"] for r in records if r["arm"] == "transported"
            ),
            "max_transported_error": max(
                r["max_abs_change"] for r in records if r["arm"] == "transported"
            ),
            "binding_artifacts_verified": len(manifest),
            "new_encodings": 0,
            "new_fits": 0,
            "scope": "algebraic basis transport on frozen actual vectors, not personality validation",
        },
    )
    print(json.dumps(json.loads((args.out / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
