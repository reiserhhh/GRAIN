#!/usr/bin/env python3
"""Independent-form arithmetic and dependency audit of HM87, same agent."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.navigation import read_json
from scripts.run_grain_author_contrast_hm82 import write
from scripts.run_grain_cold_evidence_hm87 import OUT
from scripts.run_grain_query_evidence_hm86 import check_hashes, unit


def main():
    """Recompute cold means/pairings by grouped normalized sums, not the runner."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    out = parser.parse_args().out
    manifest = read_json(out / "00/manifest.json")
    check_hashes(ROOT, manifest["code"] | manifest["source"])
    check_hashes(out, manifest["inputs"])
    check_hashes(out, read_json(out / "01/raw_seal.json"))
    check_hashes(out, read_json(out / "02/output_seal.json"))
    rows = read_json(out / "00/rows.json")
    for row in rows:
        unit(out, row["unit"])
    bundle = read_json(out / "02/cold/bundle.json")
    audit = read_json(out / "02/cold/cold_audit.json")
    if (
        audit["data_reads"] != ["bundle.json", "carriers.npz"]
        or audit["forbidden_modules"]
    ):
        raise ValueError("Cold dependencies not verified")
    error, checks = 0.0, 0
    with (
        np.load(out / "02/cold/carriers.npz", allow_pickle=False) as data,
        np.load(out / "02/cold/observations.npz", allow_pickle=False) as result,
    ):
        for i, length in enumerate(data["lengths"]):
            k, v = (
                data["keys"][i, :length].astype(float),
                data["values"][i, :length].astype(float),
            )
            ids = data["event_ids"][i, :length]
            for qi, raw in enumerate(bundle["raw_queries"]):
                raw = np.asarray(raw)
                theta = np.float32(
                    int(data["endpoints"][i]) + bundle["gaps"][qi]
                ) * np.asarray(bundle["frequencies"], np.float32)
                d = raw.shape[1] // 2
                rotated = np.empty_like(raw)
                rotated[:, :d] = raw[:, :d] * np.cos(theta) - raw[:, d:] * np.sin(theta)
                rotated[:, d:] = raw[:, d:] * np.cos(theta) + raw[:, :d] * np.sin(theta)
                rotated *= bundle["scaling"]
                expected = np.zeros((4, *raw.shape))
                for head in range(len(raw)):
                    kvh = head // (len(raw) // k.shape[1])
                    logits = np.sum(k[:, kvh] * rotated[head], axis=1)
                    weight = np.exp(logits - logits.max())
                    weight /= weight.sum()
                    for event in range(3):
                        mask = ids == event
                        mass = weight[mask].sum()
                        mu = v[mask, kvh].sum(0) / mask.sum()
                        expected[1, head] += mass * mu
                        expected[2, head] += np.sum(
                            (weight[mask] - mass / mask.sum())[:, None] * v[mask, kvh],
                            axis=0,
                        )
                    expected[0, head] = expected[1, head] + expected[2, head]
                    expected[3, head] = v[:, kvh].sum(0) / len(v)
                error = max(
                    error, float(np.max(np.abs(expected - result["vectors"][i, qi])))
                )
                checks += expected.size
    with (out / "02/polarized_actions.csv").open() as stream:
        actions = list(csv.DictReader(stream))
    action_error = max(
        abs(float(r["body"]) + float(r["query_action"]) - float(r["full"]))
        for r in actions
    )
    with (out / "02/author_composition.csv").open() as stream:
        accounts = list(csv.DictReader(stream))
    with np.load(out / "02/observations.npz", allow_pickle=False) as obs:
        projections, native = obs["projections"], obs["native_projections"]
        points = np.asarray(
            [
                (x, y, g, h)
                for x in (-1, 1)
                for y in (-1, 1)
                for g in (-1, 1)
                for h in (-1, 1)
            ]
        )
        for row in accounts:
            ti, qi = (
                bundle["targets"].index(row["target"]),
                bundle["targets"].index(row["query"]),
            )
            context, order = (ti - 1) % 6, int(row["order"])
            ri = ("conditional", "event_uniform", "pairing", "uniform_body").index(
                row["readout"]
            )
            score = (
                projections[context, order, :, qi, ri, ti]
                if row["view"] == "cold"
                else native[context, :, qi, ri, ti]
            )
            char = points[:, 1] * points[:, 2]
            expected = 2 * manifest["spec"]["epsilon"] * np.mean(char * score)
            error = max(error, abs(expected - float(row["yg_contrast"])))
            mask = points[:, 3] == -points[:, 2]
            expected = (
                2 * manifest["spec"]["epsilon"] * np.mean(char[mask] * score[mask])
            )
            error = max(error, abs(expected - float(row["equal_name_count_contrast"])))
            checks += 2
    summary = read_json(out / "02/summary.json")
    if (
        max(error, action_error, summary["polarization_error"])
        > manifest["spec"]["arithmetic_tolerance"]
    ):
        raise ValueError("Arithmetic audit failed")
    if (
        max(
            summary[n]
            for n in (
                "body_KV_native_error",
                "old_control_KV_error",
                "virtual_query_rotation_error",
            )
        )
        > manifest["spec"]["prefix_tolerance"]
    ):
        raise ValueError("Prefix dependence audit failed")
    write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM87_NOT_GLOBAL",
            "captures": len(rows),
            "alternate_scalar_checks": checks,
            "max_error": error,
            "action_error": action_error,
            "cold_data_reads": audit["data_reads"],
            "cold_neural_packages": [],
            "same_agent_not_independent_rebuilder": True,
            "original_scope_complete": False,
        },
    )
    print("HM87 verified", checks, "alternate scalar checks", error)


if __name__ == "__main__":
    main()
