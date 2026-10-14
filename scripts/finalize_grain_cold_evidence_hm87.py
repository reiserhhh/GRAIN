#!/usr/bin/env python3
"""Finalize existing HM87 tables after a companion array-export failure.

All captures, cold outputs, failed gates and partial companion files remain.
Only numerical reconstruction/verification runs here, with zero model calls.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cold_evidence as C
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_author_contrast_hm82 import write
from scripts.run_grain_cold_evidence_hm87 import OUT, READOUTS, carrier
from scripts.run_grain_query_evidence_hm86 import check_hashes, seal, unit


def main():
    """Reconstruct all fixed projections and audit every existing table cell."""
    out = OUT
    manifest, rows = (
        read_json(out / "00/manifest.json"),
        read_json(out / "00/rows.json"),
    )
    check_hashes(ROOT, manifest["code"] | manifest["source"])
    check_hashes(out, manifest["inputs"])
    check_hashes(out, read_json(out / "01/raw_seal.json"))
    check_hashes(out, read_json(out / "02/cold/seal.json"))
    failed = out / "03_numerical_reconciliation"
    tables = {}
    for name in ("target_selection", "author_composition", "polarized_actions"):
        with (failed / (name + ".csv")).open() as stream:
            tables[name] = list(csv.DictReader(stream))
    final = out / "04_verified_completion"
    if final.exists():
        raise FileExistsError("Preserve previous completion")
    final.mkdir()
    write(
        final / "manifest.json",
        {
            "code": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
            "partial_tables": {
                str(p.relative_to(out)): digest(p) for p in sorted(failed.glob("*.csv"))
            },
            "primary_raw_seal": digest(out / "01/raw_seal.json"),
            "cold_seal": digest(out / "02/cold/seal.json"),
            "failures_preserved": [
                "primary absolute-prefix gate",
                "companion duplicate NPZ keyword export",
            ],
            "new_neural_calls": 0,
        },
    )
    bundle = read_json(out / "00/bundle.json")
    dirs = np.asarray(bundle["directions"])
    body_rows = [r for r in rows if r["stage"] == "body"]
    data = {r["unit"]: carrier(unit(out, r["unit"]), r) for r in body_rows}
    fields = {
        "cold": np.zeros((6, 2, 16, 6, 4, 6)),
        "native": np.zeros((6, 1, 16, 6, 4, 6)),
        "cold_native_KV_realization": np.zeros((6, 1, 16, 6, 4, 6)),
    }
    with np.load(out / "02/cold/observations.npz", allow_pickle=False) as archive:
        frozen = archive["vectors"].copy()
    errors = {
        "cold_arithmetic": 0.0,
        "old_KV": 0.0,
        "native_KV": 0.0,
        "rotation": 0.0,
        "cold_projected_KV": 0.0,
    }
    checks = 0
    for row, cold in zip(body_rows, frozen, strict=True):
        k, v, ids, endpoint = data[row["unit"]]
        fields["cold"][row["context"], row["order"], row["corner"]] = np.einsum(
            "qrhd,thd->qrt", cold, dirs
        )
        for qi, raw in enumerate(bundle["raw_queries"]):
            q = C.rotate(
                raw,
                endpoint + bundle["gaps"][qi],
                bundle["frequencies"],
                bundle["scaling"],
            )
            expected = np.zeros_like(cold[qi])
            for h in range(12):
                logits = np.sum(k[:, h // 6].astype(float) * q[h], axis=1)
                weights = np.exp(logits - logits.max())
                weights /= weights.sum()
                values = v[:, h // 6].astype(float)
                expected[0, h] = np.sum(weights[:, None] * values, axis=0)
                for event in range(3):
                    ix = ids == event
                    mass = weights[ix].sum()
                    expected[1, h] += mass * values[ix].sum(0) / ix.sum()
                    expected[2, h] += np.sum(
                        (weights[ix] - mass / ix.sum())[:, None] * values[ix], axis=0
                    )
                expected[3, h] = values.sum(0) / len(values)
            errors["cold_arithmetic"] = max(
                errors["cold_arithmetic"], float(np.max(np.abs(expected - cold[qi])))
            )
            checks += expected.size
    for row in rows:
        raw = unit(out, row["unit"])
        if row["stage"] == "control":
            old = unit(ROOT / manifest["spec"]["source"], row["old_unit"])
            ix = np.concatenate(row["groups"])
            errors["old_KV"] = max(
                errors["old_KV"],
                *(float(np.max(np.abs(raw[n][ix] - old[n][ix]))) for n in ("k", "v")),
            )
        if row["stage"] != "native":
            continue
        body = rows[row["body_unit"]]
        k, v, ids, endpoint = data[body["unit"]]
        fk, fv, _, _ = carrier(raw, {**row, "event_order": [0, 1, 2]})
        errors["native_KV"] = max(
            errors["native_KV"],
            float(np.max(np.abs(k - fk))),
            float(np.max(np.abs(v - fv))),
        )
        qi = row["query_index"]
        position = endpoint + bundle["gaps"][qi]
        q = C.rotate(
            raw["raw_query"], position, bundle["frequencies"], bundle["scaling"]
        )
        errors["rotation"] = max(
            errors["rotation"], float(np.max(np.abs(q - raw["q"])))
        )
        native = C.reading(k, v, q, ids)
        ref = C.rotate(
            bundle["raw_queries"][qi],
            position,
            bundle["frequencies"],
            bundle["scaling"],
        )
        other = C.reading(fk, fv, ref, ids)
        index = (body["context"], 0, body["corner"], qi)
        fields["native"][index] = np.einsum("rhd,thd->rt", native, dirs)
        fields["cold_native_KV_realization"][index] = np.einsum(
            "rhd,thd->rt", other, dirs
        )
        errors["cold_projected_KV"] = max(
            errors["cold_projected_KV"],
            float(
                np.max(
                    np.abs(
                        fields["cold"][index]
                        - fields["cold_native_KV_realization"][index]
                    )
                )
            ),
        )
    points = np.asarray(
        [
            (x, y, g, h)
            for x in (-1, 1)
            for y in (-1, 1)
            for g in (-1, 1)
            for h in (-1, 1)
        ]
    )
    table_error = 0.0
    for row in tables["target_selection"]:
        ti = bundle["targets"].index(row["target"])
        score = fields[row["view"]][
            (ti - 1) % 6, int(row["order"]), :, :, READOUTS.index(row["readout"]), ti
        ]
        own = score[(points[:, 1] == 1) & (points[:, 2] == 1)].mean(0) - score[
            (points[:, 1] == -1) & (points[:, 2] == 1)
        ].mean(0)
        foreign = score[(points[:, 1] == 1) & (points[:, 2] == -1)].mean(0) - score[
            (points[:, 1] == -1) & (points[:, 2] == -1)
        ].mean(0)
        values = (own[ti], foreign[ti], own[ti] - np.delete(own, ti).mean())
        table_error = max(
            table_error,
            *(
                abs(float(row[n]) - value)
                for n, value in zip(
                    ("own", "foreign", "selectivity"), values, strict=True
                )
            ),
        )
        checks += 3
    for row in tables["author_composition"]:
        ti, qi = (
            bundle["targets"].index(row["target"]),
            bundle["targets"].index(row["query"]),
        )
        score = fields[row["view"]][
            (ti - 1) % 6, int(row["order"]), :, qi, READOUTS.index(row["readout"]), ti
        ]
        char = points[:, 1] * points[:, 2]
        mask = points[:, 3] == -points[:, 2]
        epsilon = manifest["spec"]["epsilon"]
        table_error = max(
            table_error,
            abs(float(row["yg_contrast"]) - 2 * epsilon * np.mean(char * score)),
            abs(
                float(row["equal_name_count_contrast"])
                - 2 * epsilon * np.mean(char[mask] * score[mask])
            ),
        )
        checks += 2
    action_error = max(
        abs(float(r["body"]) + float(r["query_action"]) - float(r["full"]))
        for r in tables["polarized_actions"]
    )
    if (
        max(errors["cold_arithmetic"], table_error, action_error)
        > manifest["spec"]["arithmetic_tolerance"]
    ):
        raise ValueError("Numerical observation/table reconciliation failed")
    audit = read_json(out / "02/cold/cold_audit.json")
    if (
        audit["data_reads"] != ["bundle.json", "carriers.npz"]
        or audit["forbidden_modules"]
    ):
        raise ValueError("Cold dependency audit failed")
    conditional = [
        r for r in tables["target_selection"] if r["readout"] == "conditional"
    ]
    matched_accounts = [
        r
        for r in tables["author_composition"]
        if r["readout"] == "conditional" and r["target"] == r["query"]
    ]
    by_view = {
        view: {
            r["target"]: float(r["selectivity"])
            for r in conditional
            if r["view"] == view and int(r["order"]) == 0
        }
        for view in fields
    }
    summary = {
        "status": "VERIFIED_FINITE_HM87_WITH_DECLARED_PREFIX_GATE_FAILURE",
        "frozen_absolute_prefix_gate": "FAIL_preserved_threshold1e-5",
        "failures_preserved": True,
        "captures": len(rows),
        "model_seconds": read_json(out / "01/status.json")["model_seconds"],
        "errors": errors,
        "table_error": table_error,
        "action_error": action_error,
        "alternate_scalar_checks": checks + len(tables["polarized_actions"]),
        "conditional_selectivity": conditional,
        "matched_author_composition": matched_accounts,
        "forward_signs_agree_under_both_KV_realizations": all(
            (by_view["cold"][t] > 0) == (by_view["cold_native_KV_realization"][t] > 0)
            for t in bundle["targets"]
        ),
        "observed_selectivity_realization_difference_bound": 4
        * errors["cold_projected_KV"],
        "cold_data_reads": audit["data_reads"],
        "cold_neural_packages": [],
        "new_reconciliation_model_calls": 0,
        "same_agent_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    with (final / "projections.npz").open("xb") as stream:
        np.savez_compressed(stream, **fields)
    write(final / "summary.json", summary)
    write(final / "output_seal.json", seal(out, final))
    write(
        out / "verification.json",
        {
            "status": summary["status"],
            "companion": "04_verified_completion/summary.json",
            "frozen_prefix_gate": "FAIL",
            "cold_arithmetic_and_tables": "PASS",
            "original_scope_complete": False,
        },
    )
    print(
        "HM87 verified finite observations with prefix gate failure retained",
        errors,
        table_error,
        flush=True,
    )


if __name__ == "__main__":
    main()
