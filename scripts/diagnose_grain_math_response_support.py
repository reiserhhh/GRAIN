#!/usr/bin/env python3
"""Result-after source-equivalent observer witness; not an improved denoiser."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import MODELS, ROOT, sha, write_csv, write_json

from grain_research.construction import math_binding as B

DEFAULT = ROOT / "results/grain_math_response_v1/response_geometry_r1"
DEST = ROOT / "results/grain_math_response_v1/response_support_witness_r1"
TARGETS = ("coordination_action", "verification_action")


def diagnose(out, dest):
    """Keep the original function and exhibit a source-equivalent cold alternative."""
    if (dest / "summary.json").exists():
        raise ValueError("diagnostic frozen")
    manifest = json.loads((out / "04/artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        if sha(out / name) != digest:
            raise ValueError("primary changed: " + name)
    protocol = json.loads((out / "00/protocol_before_run.json").read_text())
    source = ROOT / protocol["spec"]["source_packet"]
    rows = list(csv.DictReader((out / "02/cold_readouts.csv").open()))
    bank = read_bank(out / "00/documents.jsonl")
    indices = json.loads((out / "01/observation_indices.json").read_text())
    events = {
        r["material_id"]: idx["observed"][0]
        for r, idx in zip(bank, indices, strict=True)
        if len(idx["observed"]) == 1 and idx["observed"][0]["status"] == "ok"
    }
    scores, summaries, max_source_error, max_cold_error = [], [], 0.0, 0.0
    for reader in MODELS:
        arrays = np.load(out / f"01A/{reader}_geometry.npz")
        basis, mean = arrays["support"], arrays["objects"].mean(axis=0)
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        source_a = np.repeat(arrays["actions"], len(arrays["objects"]), axis=0)
        source_o = np.tile(arrays["objects"], (len(arrays["actions"]), 1))
        projected_source = mean + ((source_o - mean) @ basis.T) @ basis
        for target in TARGETS:
            model = load_model(source / f"02/{reader}_{target}.npz")["auto_tensor"]
            full_source = B.score_kernel(model, source_a, source_o)
            alternative_source = B.score_kernel(model, source_a, projected_source)
            error = float(np.max(np.abs(full_source - alternative_source)))
            max_source_error = max(max_source_error, error)
            for view in ("strict", "legal_joint"):
                rr = [
                    r
                    for r in rows
                    if r["reader"] == reader
                    and r["target"] == target
                    and r["view"] == view
                ]

                def vector(ref, old=old, new=new):
                    return old[ref[1]] if ref[0] == "source" else new[ref[1]]

                a = np.array(
                    [vector(events[r["material_id"]]["action_ref"]) for r in rr]
                )
                o = np.array(
                    [
                        vector(
                            events[r["material_id"]][
                                "object_ref" if view == "strict" else "legal_ref"
                            ]
                        )
                        for r in rr
                    ]
                )
                proj = mean + ((o - mean) @ basis.T) @ basis
                original = B.score_kernel(model, a, o)
                alternative = B.score_kernel(model, a, proj)
                max_cold_error = max(
                    max_cold_error,
                    float(
                        np.max(np.abs(original - np.array([float(r["k"]) for r in rr])))
                    ),
                )
                for i, r in enumerate(rr):
                    scores.append(
                        {
                            "reader": reader,
                            "target": target,
                            "view": view,
                            "material_id": r["material_id"],
                            "original": float(original[i]),
                            "source_equivalent_alternative": float(alternative[i]),
                            "source_unidentified_term": float(
                                original[i] - alternative[i]
                            ),
                            "original_gate": bool(original[i] > 0),
                            "alternative_gate": bool(alternative[i] > 0),
                            "expected_relation": r["expected_relation"] == "True",
                        }
                    )
                summaries.append(
                    {
                        "reader": reader,
                        "target": target,
                        "view": view,
                        "source_checks": len(source_a),
                        "source_max_error": error,
                        "cold_n": len(rr),
                        "cold_score_difference_rms": float(
                            np.sqrt(np.mean((original - alternative) ** 2))
                        ),
                        "cold_score_difference_max": float(
                            np.max(np.abs(original - alternative))
                        ),
                        "cold_gate_changes": int(
                            np.sum((original > 0) != (alternative > 0))
                        ),
                        "interpretation": "source-equivalent does not mean cold-equivalent; this alternative is not promoted",
                    }
                )
    if max_source_error > 1e-9 or max_cold_error > 1e-9:
        raise ValueError("source equivalence or cold replay failed")
    write_csv(dest / "source_equivalent_cold_scores.csv", scores)
    write_csv(dest / "source_equivalence_summary.csv", summaries)
    write_json(
        dest / "summary.json",
        {
            "result_after": True,
            "primary_manifest_sha256": sha(out / "04/artifact_hashes.json"),
            "source_equivalence_max_error": max_source_error,
            "cold_replay_max_error": max_cold_error,
            "summaries": summaries,
            "source_target_refits": 0,
            "new_encoder_calls": 0,
            "no_alternative_selected_as_best": True,
            "mechanism": "finite source scores do not identify response outside their affine object support",
        },
    )
    write_json(
        dest / "artifact_hashes.json",
        {
            str(p.relative_to(dest)): sha(p)
            for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps(
            {
                "source_error": max_source_error,
                "cold_error": max_cold_error,
                "summaries": summaries,
            }
        ),
        flush=True,
    )


def main():
    """Create a finite result-after witness without changing frozen predictions."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    diagnose(args.out, args.dest)


if __name__ == "__main__":
    main()
