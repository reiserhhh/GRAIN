#!/usr/bin/env python3
"""HM28 alternate weighted-node means, Vandermonde and exact support witnesses."""

from __future__ import annotations

import argparse
import csv
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_decoder_paths import node_path
from scripts.run_grain_math_position_quotient import OUT, check
from scripts.run_grain_math_response import write_json

DEFAULT = ROOT / "results/grain_math_response_v1/position_quotient_verification_r1"


def verify(packet, out):
    """Replay every source law and cold action without the primary quotient helper."""
    spec = check(packet)["spec"]
    if out.exists():
        raise FileExistsError("preserve verification")
    out.mkdir(parents=True)
    (out / "verify.py").write_bytes(Path(__file__).read_bytes())
    for name, sha in read_json(packet / "02/artifact_hashes.json").items():
        if digest(packet / name) != sha:
            raise ValueError("frozen packet changed " + name)
    parent = ROOT / spec["parent"]
    grand = ROOT / read_json(parent / "00/manifest.json")["spec"]["parent"]
    design = {r["hm24_unit"]: r for r in read_json(packet / "00/design.json")}
    models, errors = {}, {"source": 0.0, "action": 0.0, "metrics": 0.0}

    def compare(kind, a, b):
        error = float(np.max(abs(np.asarray(a) - np.asarray(b)), initial=0))
        errors[kind] = max(errors[kind], error)
        if error > 2e-11:
            raise ValueError(f"{kind}: {error}")

    certificates = {
        (r["query"], r["anchor"]): r
        for r in read_json(packet / "01/source_certificates.json")
    }
    for query, entry in read_json(parent / "01/source_manifest.json").items():
        with np.load(parent / entry["file"]) as data:
            old = {name: data[name].copy() for name in data.files}
        for anchor in spec["anchors"]:
            gs = [old[f"family_{r['family']}_{anchor}"] for r in entry["addresses"]]
            pts = sorted(
                {
                    Fraction(0) if len(g) == 1 else Fraction(2 * j, len(g) - 1) - 1
                    for g in gs
                    for j in range(len(g))
                }
            )
            if pts != list(map(Fraction, (-1, 0, 1))):
                raise ValueError("independent exact position inventory")
            values, weights = [], []
            for point in pts:
                weighted, mass = [], Fraction(0)
                for g in gs:
                    for j, row in enumerate(g):
                        t = (
                            Fraction(0)
                            if len(g) == 1
                            else Fraction(2 * j, len(g) - 1) - 1
                        )
                        if t == point:
                            w = Fraction(1, len(gs) * len(g) ** 3)
                            weighted.append(float(w * len(g)) * row)
                            mass += w
                values.append(sum(weighted) / float(mass))
                weights.append(str(mass))
            values = np.array(values)
            v = np.array([[float(t**k) for k in range(3)] for t in pts])
            coeff = np.vstack(
                (np.linalg.solve(v, values), np.zeros((1, values.shape[1])))
            )
            key = query + "/" + anchor
            record = read_json(packet / "01/source_manifest.json")[key]
            with np.load(packet / record["file"]) as data:
                compare("source", values, data["values"])
                compare("source", coeff, data["coefficients"])
            cert = certificates[query, anchor]
            if (
                cert["vanishing_polynomial"] != ["0", "-1", "0", "1"]
                or cert["node_weights"] != weights
                or (cert["rank"], cert["nullity"]) != (3, 1)
            ):
                raise ValueError("exact polynomial/weight certificate mismatch")
            compare(
                "source",
                float(np.linalg.norm(coeff - old[anchor + "_joint"])),
                cert["ambient_coefficient_difference"],
            )
            if cert["reduction"] != [
                ["1", "0", "0", "0"],
                ["0", "1", "0", "1"],
                ["0", "0", "1", "0"],
            ]:
                raise ValueError("exact t^3=t reduction mismatch")
            models[query, anchor] = (values, coeff)
    actions = list(csv.DictReader((packet / "02/actions.csv").open()))
    for row in actions:
        address = design[int(row["unit"])]
        if (row["query"], int(row["family"]), row["role"]) != (
            address["query"],
            address["family"],
            address["role"],
        ):
            raise ValueError("held address mismatch")
        with np.load(
            node_path(grand, address["hm24_unit"], 0).with_suffix(".npz")
        ) as data:
            d = data["path_delta"].copy()
        values, coeff = models[row["query"], row["anchor"]]
        pts = (
            [Fraction(0)]
            if len(d) == 1
            else [Fraction(2 * j, len(d) - 1) - 1 for j in range(len(d))]
        )
        indices = [(-1, 0, 1).index(t) for t in pts]
        value = sum(
            float(values[i] @ h) / len(d) for i, h in zip(indices, d, strict=True)
        )
        m = np.array([[float(t**k) / len(d) for t in pts] for k in range(4)]) @ d
        poly = float(np.einsum("ij,ij->", coeff, m))
        compare("action", value, float(row["node_action"]))
        compare("action", poly, float(row["polynomial_action"]))
        compare(
            "action",
            value - float(row["joint_action"]),
            float(row["node_joint_difference"]),
        )
        compare("action", np.linalg.norm(m[3] - m[1]), float(row["moment_alias_norm"]))
        compare("action", value - float(row["native_action"]), float(row["error"]))
    metrics = list(csv.DictReader((packet / "02/metrics.csv").open()))
    for row in metrics:
        rs = [
            r
            for r in actions
            if (r["query"], r["anchor"], r["role"])
            == (row["query"], row["anchor"], row["role"])
        ]
        compare(
            "metrics",
            np.sqrt(sum(float(r["error"]) ** 2 for r in rs) / len(rs)),
            float(row["rmse"]),
        )
        compare(
            "metrics",
            max(abs(float(r["node_joint_difference"])) for r in rs),
            float(row["joint_max_difference"]),
        )
    witnesses = list(csv.DictReader((packet / "02/support_witnesses.csv").open()))
    for row in witnesses:
        t = int(row["tokens"])
        pts = (
            [Fraction(0)] if t == 1 else [Fraction(2 * j, t - 1) - 1 for j in range(t)]
        )
        action = sum((p**3 - p) ** 2 for p in pts) / t
        if (
            Fraction(row["null_action"]) != action
            or Fraction(row["paired_difference"]) != 2 * action
            or int(row["augmented_cubic_rank"]) != (4 if t == 4 else 3)
        ):
            raise ValueError("exact support witness mismatch")
    summary = read_json(packet / "02/summary.json")
    for key, expected in (
        ("actions", 96),
        ("metrics", 18),
        ("exact_support_witnesses", 24),
        ("position_laws", 6),
        ("source_rank", 3),
        ("new_model_calls", 0),
    ):
        if summary[key] != expected:
            raise ValueError("finite inventory mismatch " + key)
    write_json(
        out / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM28_NOT_GLOBAL",
            "position_laws": len(models),
            "actions": len(actions),
            "metrics": len(metrics),
            "exact_support_witnesses": len(witnesses),
            "max_errors": errors,
            "new_model_calls": 0,
            "alternate_same_agent_not_human_review": True,
        },
    )
    write_json(
        out / "artifact_hashes.json",
        {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    )
    print(read_json(out / "summary.json"), flush=True)


def main():
    """Run the independent arithmetic check once, preserving previous outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    verify(args.packet, args.out)


if __name__ == "__main__":
    main()
