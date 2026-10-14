#!/usr/bin/env python3
"""Audit constructed fields without overriding HM119's failed compatibility gate."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_expression_response as M
from grain_research.construction import math_recursive_collector as C
from grain_research.construction.math_cold_evidence import rotate
from scripts import run_grain_author_evidence_merge_hm102 as H


def scope_status(numeric_passed, compatibility_passed):
    """Keep numerical coherence distinct from the original scientific protocol gate."""
    if not numeric_passed:
        return "NUMERIC_AUDIT_FAILED"
    return (
        "NUMERIC_AUDIT_PASSED_COMPATIBILITY_PASSED"
        if compatibility_passed
        else "NUMERIC_AUDIT_PASSED_ORIGINAL_COMPATIBILITY_FAILED_NOT_FULL"
    )


def readback(out):
    """Recompute all readouts, sectors, matchings and exports with scope preserved."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(out, m["inputs"])
    for name in (
        "01/raw_seal.json",
        "02_corrected/construction_seal.json",
        "03/output_seal.json",
    ):
        H.verify_hashes(out, H.load(out / name))
    correction = H.load(out / "02_corrected/input.json")
    for path, expected in correction["hashes"].items():
        if C.stream_sha(path) != expected:
            raise ValueError("Changed preserved failure/adapter input")
    folder = out / "numeric_readback"
    folder.mkdir()
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    with np.load(out / "02_corrected/constructed.npz", allow_pickle=False) as z:
        raw = {name: z[name] for name in z.files}
    n = np.empty((3, 2, 6, 2, 2, 2, 2, 2, 6))
    errors = {"reading": 0.0, "fields": 0.0, "permutation": 0.0, "exports": 0.0}
    for i, row in enumerate(rows):
        receipt = H.load(out / f"01/units/unit_{i:04d}.json")
        trace = receipt["trace"]
        if (
            trace["completed_blocks"] != list(range(17))
            or trace["masked_blocks"] != list(range(17))
            or trace["current_tokens_executed"]
            or trace["forbidden_calls"]
        ):
            raise ValueError("Not a history-only source capture")
        index = tuple(row[name] for name in ("variant", "wording", "family")) + tuple(
            (row[name] + 1) // 2 for name in ("condition", "polarity", "owner")
        )
        n[index] = raw["readings"][i]
        for focus, q in np.ndindex(2, 6):
            k, v = raw["k"][i, focus].astype(float), raw["v"][i, focus].astype(float)
            query = rotate(
                np.asarray(reader["raw_queries"])[q],
                max(row["current"]) + reader["gaps"][q],
                reader["frequencies"],
                reader["scaling"],
            )
            readings = np.zeros(2)
            for head, direction in enumerate(np.asarray(reader["directions"])[q]):
                ratio = len(query) // len(k[0])
                ell = np.array([token[head // ratio] @ query[head] for token in k])
                p = np.exp(ell - ell.max())
                p /= p.sum()
                content = np.array([token[head // ratio] @ direction for token in v])
                readings[0] += sum(content) / len(content)
                readings[1] += sum(a * b for a, b in zip(p, content, strict=True))
            errors["reading"] = max(
                errors["reading"],
                float(np.max(abs(readings - raw["readings"][i, focus, :, q]))),
            )
    fields = M.response_fields(n)
    identities = M.verify_response_identities(n, fields)
    with np.load(out / "03/fields.npz", allow_pickle=False) as z:
        errors["fields"] = max(
            float(np.max(abs(value - z[name]))) for name, value in fields.items()
        )
    details = H.load(out / "03/correspondence_details.json")
    for key, saved in details.items():
        sector, mode, variant, wording = key.split(":")
        mode, variant, wording = int(mode[1:]), int(variant[1:]), int(wording[1:])
        source, application = (
            fields[sector][0, 0, ..., mode, :],
            fields[sector][variant, wording, ..., mode, :],
        )
        cross = np.empty((6, 6, 2))
        for a, b, c in np.ndindex(6, 6, 2):
            x, y = source[a, c], application[b, c]
            cross[a, b, c] = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
        values = np.array(
            [
                sum(cross[a, permutation[a], c] for a, c in np.ndindex(6, 2)) / 12
                for permutation in itertools.permutations(range(6))
            ]
        )
        errors["permutation"] = max(
            errors["permutation"],
            float(np.max(abs(values - np.asarray(saved["permutation_values"])))),
        )
    targets, csv_rows = m["spec"]["targets"], {}
    for name in ("conditional_response_vectors", "qualified_author_pairing"):
        seen = set()
        with (out / f"03/{name}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                v, w, f, q = (
                    int(row["variant"]),
                    int(row["wording"]),
                    targets.index(row["family"]),
                    targets.index(row["query"]),
                )
                mode = ("uniform", "weighted").index(row["mode"])
                if name == "conditional_response_vectors":
                    c, o = (
                        (int(row["condition"]) + 1) // 2,
                        ("B", "A").index(row["actor"]),
                    )
                    address = (v, w, f, c, o, mode, q)
                    expected = {
                        "constructed_Gamma": fields["gamma"][address],
                        "source_Gamma": fields["gamma"][0, 0, f, c, o, mode, q],
                    }
                else:
                    address = (v, w, f, mode, q)
                    expected = {
                        "qualified_pairing_contrast": fields[
                            "qualified_family_pairing"
                        ][address],
                        "source_pairing_contrast": fields["qualified_family_pairing"][
                            0, 0, f, mode, q
                        ],
                        "other_actor_diagnostic_NOT_self": fields[
                            "other_actor_pairing_diagnostic"
                        ][address],
                    }
                if address in seen:
                    raise ValueError("Duplicate output address")
                seen.add(address)
                errors["exports"] = max(
                    errors["exports"],
                    *(
                        abs(float(row[field]) - value)
                        for field, value in expected.items()
                    ),
                )
        csv_rows[name] = len(seen)
    compatible = H.load(out / "03/compatibility.json")["passed"]
    numeric = max((*errors.values(), *identities.values())) <= 1e-10 and csv_rows == {
        "conditional_response_vectors": 1728,
        "qualified_author_pairing": 432,
    }
    result = {
        "status": scope_status(numeric, compatible),
        "numeric_passed": numeric,
        "original_protocol_compatibility_passed": compatible,
        "original_protocol_passed": False,
        "errors": errors,
        "identities": identities,
        "csv_rows": csv_rows,
        "scalar_readings_checked": int(raw["readings"].size),
        "full_matchings_checked": len(details) * 720,
        "new_model_calls_or_fits": [0, 0],
        "torch_or_transformers_loaded": any(
            name in sys.modules for name in ("torch", "transformers")
        ),
        "same_agent_alternate_formula_not_third_party": True,
        "original_scope_complete": False,
        "inputs": {
            str(p.relative_to(out)): H.sha(p)
            for p in (
                out / "00/manifest.json",
                out / "02_corrected/construction_seal.json",
                out / "03/output_seal.json",
            )
        },
        "readback_code": H.sha(Path(__file__)),
    }
    H.write(folder / "audit.json", result)
    print("HM119 scoped audit", result, flush=True)
    if not numeric:
        raise ValueError("Numerical construction audit failed; no promotion")


def main():
    """Run a one-time numerical audit without restarting or accepting the protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    readback(args.output)


if __name__ == "__main__":
    main()
