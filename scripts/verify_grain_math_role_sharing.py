#!/usr/bin/env python3
"""Replay HM-05 and dissect its unit shift through exact seven-term products."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from verify_grain_math_local_roles import (
    affine,
    model_groups,
    read_jsonl,
    resolve,
    sha,
    tensor,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Use distributed-intercept and expanded-kernel formulas, not primary scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/role_sharing_verification_r1",
    )
    args = parser.parse_args()
    base = ROOT / "results/grain_math_response_v1"
    source, app, sharing = (
        base / name for name in ("binding_r1", "local_roles_r1", "role_sharing_r1")
    )
    count = 0
    for root, path in (
        (source, "04/artifact_hashes.json"),
        (app, "04/artifact_hashes.json"),
        (sharing, "02/artifact_hashes.json"),
    ):
        for name, digest in json.loads((root / path).read_text()).items():
            if sha(root / name) != digest:
                raise AssertionError(f"frozen artifact changed: {root / name}")
            count += 1
    bank = read_jsonl(app / "00/documents.jsonl")
    refs = json.loads((app / "01/observation_indices.json").read_text())
    with (sharing / "01/predictions.csv").open() as stream:
        predictions = {
            (r["reader"], r["material_id"], r["arm"]): r for r in csv.DictReader(stream)
        }
    targets = sorted({r["target"] for r in bank})
    errors, components, names = [], [], ("polarity", "ownership", "actuality")
    for reader in ("e5", "bge"):
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(app / f"01B/embeddings/{reader}.npy").astype(float)
        models = {t: model_groups(source / f"02/{reader}_{t}.npz") for t in targets}
        flat = [e for records in refs["automatic_events"] for e in records]
        x = resolve([e["readout"] for e in flat], old, new)
        ax, bx = (resolve([e[k] for e in flat], old, new) for k in ("action", "object"))
        role = {t: affine(models[t]["whole_factors"], x)[:, 1:] for t in targets}
        common = (role[targets[0]] + role[targets[1]]) / 2
        relevance = {
            t: np.clip((1 + tensor(models[t]["auto_tensor"], ax, bx)) / 2, 0, 1)
            for t in targets
        }
        cursor = 0
        for row, events in zip(bank, refs["automatic_events"], strict=True):
            indices = np.arange(cursor, cursor + len(events))
            cursor += len(events)
            native = predictions[reader, row["material_id"], "native_roles"]
            if native["complete"] != "True":
                continue
            target = row["target"]
            other = next(t for t in targets if t != target)
            first = {}
            for j, event in enumerate(events):
                if event["status"] != "ok":
                    raise AssertionError("complete row has unresolved events")
                first.setdefault(event["event_id"], indices[j])
            selected = np.array(list(first.values()))
            values = {}
            for arm, raw in (
                ("native_roles", role[target]),
                ("cross_target_roles", role[other]),
                ("equal_shared_roles", common),
            ):
                factors = raw[selected].copy()
                factors[:, 1:] = np.clip((1 + factors[:, 1:]) / 2, 0, 1)
                gate = relevance[target][selected]
                scalar = gate * factors[:, 0] * factors[:, 1] * factors[:, 2]
                unit = float(scalar.mean())
                previous = predictions[reader, row["material_id"], arm]
                errors.append(abs(unit - float(previous["unit"])))
                hard = (gate > 0.5) & (raw[selected, 1] > 0) & (raw[selected, 2] > 0)
                if previous["prediction"]:
                    if not hard.any():
                        raise AssertionError(
                            "predicted profile lacks observed eligibility"
                        )
                    errors.append(
                        abs(float(scalar[hard].mean()) - float(previous["prediction"]))
                    )
                elif hard.any():
                    raise AssertionError("eligible profile omitted")
                values[arm] = factors
            a, c = values["native_roles"], values["equal_shared_roles"]
            d = a - c
            expanded = []
            for mask in range(1, 8):
                term = relevance[target][selected].copy()
                for j in range(3):
                    term *= d[:, j] if mask & (1 << j) else c[:, j]
                expanded.append(term)
                if row["kind"] == "mixture":
                    components.append(
                        {
                            "reader": reader,
                            "target": target,
                            "material_id": row["material_id"],
                            "component": "_".join(
                                names[j] for j in range(3) if mask & (1 << j)
                            ),
                            "unit_delta_native_minus_shared": float(term.mean()),
                        }
                    )
            direct = relevance[target][selected] * (
                np.prod(a, axis=1) - np.prod(c, axis=1)
            )
            errors.append(float(np.max(abs(np.sum(expanded, axis=0) - direct))))
    groups = defaultdict(list)
    for r in components:
        groups[r["reader"], r["target"], r["component"]].append(
            r["unit_delta_native_minus_shared"]
        )
    summary = [
        {
            "reader": reader,
            "target": target,
            "component": name,
            "mean_signed_delta": float(np.mean(rows)),
            "mean_abs_delta": float(np.mean(np.abs(rows))),
        }
        for (reader, target, name), rows in groups.items()
    ]
    if max(errors) > 1e-10:
        raise AssertionError(max(errors))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("unit_shift_components.csv", components),
        ("component_summary.csv", summary),
    ):
        with (args.out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "status": "PASS",
        "hashes_verified": count,
        "comparisons": len(errors),
        "max_error": max(errors),
        "mixture_component_rows": len(components),
        "component_summary": summary,
        "new_encodings": 0,
        "new_fits": 0,
        "api_calls": 0,
        "private_reads": 0,
        "code_sha256": sha(Path(__file__)),
        "scope": "exact seven-term calibration decomposition; overlapping algebraic terms, not unique causal/psychological mechanisms",
    }
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "component_summary"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
