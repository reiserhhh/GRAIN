#!/usr/bin/env python3
"""Result-after final-head quotient diagnostic; no new observer or fitting."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_readout_quotient as Q
from grain_research.vector_mechanisms.navigation import digest, read_json, write_json
from scripts.run_grain_math_response import write_csv


def run(source, out):
    """Project all actual four-corner vectors and replay with a distinct QR basis."""
    if out.exists():
        raise FileExistsError("preserve prior diagnostic")
    hashes = read_json(source / "03/artifact_hashes.json")
    for name, expected in hashes.items():
        if digest(source / name) != expected:
            raise ValueError("primary packet changed")
    m = read_json(source / "00/manifest.json")
    with np.load(source / "01/head.npz", allow_pickle=False) as a:
        head = a["head"].astype(float)
    quotient = Q.quotient_basis(head)
    # Difference rows remove the gauge without using centering or the SVD basis.
    qr, _ = np.linalg.qr((head[:-1] - head[-1]).T)
    if quotient["rank"] != len(head) - 1:
        raise ValueError("QR diagnostic expects full rank modulo one constant gauge")
    states, logits = {}, {}
    for d in read_json(source / "00/design.json"):
        path = source / "01/units" / f"unit_{d['unit']:04d}.npz"
        with np.load(path, allow_pickle=False) as a:
            states[d["text_id"], d["query"]] = a["final_normalized"].astype(float)
            z = a["logits"].astype(float)
            logits[d["text_id"], d["query"]] = z - z.mean()
    cubes = defaultdict(dict)
    for doc in read_json(source / "00/documents.json"):
        for query in m["spec"]["queries"]:
            cubes[doc["cube_id"], query][doc["mask"]] = doc
    rows, error_qr, error_action, error_energy = [], 0.0, 0.0, 0.0
    for (cube, query), cells in cubes.items():
        for incidence in (0, 1):
            part = [cells[i + 4 * incidence] for i in range(4)]
            h = [states[p["text_id"], query] for p in part]
            z = [logits[p["text_id"], query] for p in part]
            delta = h[3] - h[1] - h[2] + h[0]
            actual_logits = z[3] - z[1] - z[2] + z[0]
            split = Q.split_displacement(delta, quotient)
            qr_visible = qr @ (qr.T @ delta)
            error_qr = max(
                error_qr, float(np.max(abs(qr_visible - split["observable"])))
            )
            error_action = max(
                error_action,
                float(
                    np.max(abs(quotient["head"] @ split["observable"] - actual_logits))
                ),
            )
            error_energy = max(error_energy, split["energy_closure_error"])
            row = {
                "cube_id": cube,
                "query": query,
                **{k: part[0][k] for k in ("family", "role", "frame", "actor")},
                "incidence": incidence,
                **{
                    k: split[k]
                    for k in (
                        "full_energy",
                        "observable_energy",
                        "head_null_energy",
                        "null_action_max",
                        "energy_closure_error",
                    )
                },
            }
            row["observable_fraction"] = (
                split["observable_energy"] / split["full_energy"]
                if split["full_energy"] > 1e-20
                else None
            )
            rows.append(row)
    summaries = []
    for query in m["spec"]["queries"]:
        selected = [r for r in rows if r["query"] == query]
        total = sum(r["full_energy"] for r in selected)
        summaries.append(
            {
                "query": query,
                "vectors": len(selected),
                "pooled_observable_energy_fraction": sum(
                    r["observable_energy"] for r in selected
                )
                / total,
                "pooled_head_null_energy_fraction": sum(
                    r["head_null_energy"] for r in selected
                )
                / total,
                "mean_joint_vector_norm": float(
                    np.mean([np.sqrt(r["full_energy"]) for r in selected])
                ),
            }
        )
    out.mkdir(parents=True)
    write_csv(out / "vector_quotients.csv", rows)
    write_csv(out / "summary.csv", summaries)
    report = {
        "status": "PASS"
        if error_qr < 1e-10 and error_action < 1e-3 and error_energy < 1e-8
        else "FAIL",
        "result_after": True,
        "new_model_calls": 0,
        "target_refits": 0,
        "head_rank_modulo_gauge": quotient["rank"],
        "hidden_dimension": head.shape[1],
        "rows": len(rows),
        "head_singular_values": quotient["singular_values"].tolist(),
        "summary": summaries,
        "max_svd_qr_displacement_error": error_qr,
        "max_float32_logit_contrast_error": error_action,
        "max_energy_closure_error": error_energy,
        "head_null_means_psychological_noise": False,
        "normalized_hidden_domain_only": True,
        "not_an_upstream_intervention": True,
        "same_agent_alternate_QR": True,
        "source_manifest_sha256": digest(source / "03/artifact_hashes.json"),
        "code_hashes": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                Path(__file__),
                ROOT / "grain_research/construction/math_readout_quotient.py",
            )
        },
        "primary_hashes_checked": len(hashes),
    }
    write_json(out / "summary.json", report)
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise ValueError("quotient diagnostic verification failed")


def main():
    """Expose one explicit result-after companion without touching the primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/native_joint_r3",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/native_quotient_diagnostic_r1",
    )
    args = parser.parse_args()
    run(args.source, args.out)


if __name__ == "__main__":
    main()
