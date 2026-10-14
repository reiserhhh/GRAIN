#!/usr/bin/env python3
"""Alternate polynomial/report-unit audit, including all declared controls."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import MODELS, ROOT, sha, write_csv, write_json

TARGETS = ("coordination_action", "verification_action")
DEFAULT = ROOT / "results/grain_math_response_v1/state_exposure_verification_r1"
PACKETS = ("state_exposure_r1", "state_exposure_confirmation_r1")


def polynomial(model, a, o):
    """Expand constant/linear/bilinear terms instead of centered kernel scoring."""
    d = model["dual"]
    e = d - d.mean()
    u, v = a @ model["a"].T, o @ model["b"].T
    constant = (
        e.sum()
        - model["column_mean"] @ d
        + model["grand_mean"] * d.sum()
        + model["bias"]
    )
    return constant + (u + v + u * v) @ e


def number(row, key):
    """Keep absent measurements absent when reading CSV."""
    return float(row[key]) if row.get(key, "") != "" else None


def table(path):
    """Read the actual numeric output table."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_packet(out, source):
    """Recompute all condition/whole/report/document readouts independently."""
    manifest = json.loads((out / "04/artifact_hashes.json").read_text())
    for name, digest in manifest.items():
        if sha(out / name) != digest:
            raise ValueError("primary changed: " + name)
    idx = json.loads((out / "01/observation_indices.json").read_text())
    bank = read_bank(out / "00/documents.jsonl")
    address = {r["material_id"]: i for i, r in enumerate(bank)}
    conditions = table(out / "03/condition_scores.csv")
    readouts = table(out / "03/report_readouts.csv")
    profiles = table(out / "03/document_profiles.csv")
    errors = defaultdict(float)
    calculated, truth = {}, {}
    all_declared, contrasts = [], []

    def check(key, actual, expected):
        if (actual is None) != (expected is None):
            raise ValueError("absent measurement changed: " + key)
        if actual is not None:
            errors[key] = max(errors[key], abs(actual - expected))

    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        models = [load_model(source / f"02/{reader}_{t}.npz") for t in TARGETS]

        def vec(ref, old=old, new=new):
            return old[ref[1]] if ref[0] == "source" else new[ref[1]]

        rr = [r for r in conditions if r["reader"] == reader]
        items = [
            idx[address[r["material_id"]]][r["arm"]][int(r["report_index"])][
                "triggers"
            ][int(r["trigger_index"])]
            for r in rr
        ]
        a, o = (
            np.array([vec(t["action_ref"]) for t in items]),
            np.array([vec(t["object_ref"]) for t in items]),
        )
        z = np.column_stack([polynomial(m["auto_tensor"], a, o) for m in models])
        for j, r in enumerate(rr):
            for qi, key in enumerate(("coordination_k", "verification_k")):
                check("condition_polynomial", number(r, key), float(z[j, qi]))
            truth[
                r["material_id"],
                r["arm"],
                int(r["report_index"]),
                int(r["trigger_index"]),
            ] = z[j]

        for r in [r for r in readouts if r["reader"] == reader]:
            i, ri, arm = address[r["material_id"]], int(r["report_index"]), r["arm"]
            report = idx[i][arm][ri]
            x = np.array(
                [
                    truth[r["material_id"], arm, ri, ti]
                    for ti in range(len(report["triggers"]))
                ]
            ).reshape(-1, 2)
            if r["control"] == "whole_sentence":
                vector = vec(report["whole_ref"])
                w = [m["whole_factors"] for m in models]
                whole = np.array(
                    [
                        vector @ m["w"][:, 0] + m["bias"][0] - m["mu"] @ m["w"][:, 0]
                        for m in w
                    ]
                )
                x = np.tile(whole, (len(x), 1))
            gate, hard = (
                np.minimum(1, np.maximum(0, (x + 1) / 2)),
                (x > 0).astype(float),
            )
            if r["control"] == "self_trigger_only":
                mask = np.array([t["actor"] == "self" for t in report["triggers"]])[
                    :, None
                ]
                gate, hard = gate * mask, hard * mask
            eligible = report["status"] == "ok" and report["experiencer"] == "self"
            check("report_mass", number(r, "report_mass"), int(eligible))
            result = {}
            for qi, target in enumerate(TARGETS):
                maximum = max(gate[:, qi], default=0)
                fields = {
                    "coverage": maximum,
                    "hard_coverage": max(hard[:, qi], default=0),
                    "signed_expression": report["state_polarity"] * maximum,
                    "flat_sum": sum(gate[:, qi]),
                    "flat_mean": np.mean(gate[:, qi]) if len(gate) else None,
                    "noisy_or_control": 1 - np.prod(1 - gate[:, qi]),
                }
                for field, value in fields.items():
                    value = float(value) if eligible and value is not None else None
                    check("report_" + field, number(r, target + "_" + field), value)
                    result[target + "_" + field] = value
            for field, value in (
                ("coexposure", min(gate.max(0)) if len(gate) else 0),
                ("hard_coexposure", min(hard.max(0)) if len(hard) else 0),
                (
                    "same_trigger_coexposure",
                    max(np.min(gate, axis=1)) if len(gate) else None,
                ),
            ):
                value = float(value) if eligible and value is not None else None
                check("report_" + field, number(r, field), value)
                result[field] = value
            calculated[r["material_id"], arm, r["control"], ri] = result

        for r in [r for r in profiles if r["reader"] == reader]:
            local = [
                v
                for (mid, arm, control, _), v in calculated.items()
                if (mid, arm, control) == (r["material_id"], r["arm"], r["control"])
                and v["coexposure"] is not None
            ]
            for field in (
                "coexposure",
                "hard_coexposure",
                *(
                    t + "_" + f
                    for t in TARGETS
                    for f in (
                        "coverage",
                        "hard_coverage",
                        "flat_sum",
                        "flat_mean",
                        "noisy_or_control",
                    )
                ),
            ):
                check(
                    "document_profile",
                    number(r, field),
                    float(np.mean([v[field] for v in local])) if local else None,
                )
        own = [
            r
            for r in profiles
            if r["reader"] == reader
            and r["arm"] == "declared"
            and r["control"] == "relational"
            and int(r["report_mass"]) > 0
        ]
        all_declared.append(
            {
                "packet": out.name,
                "reader": reader,
                "own_documents": len(own),
                "target_error_cells": sum(
                    abs(number(r, t + "_hard_coverage") - number(r, t + "_expected"))
                    > 1e-12
                    for r in own
                    for t in TARGETS
                ),
                "joint_error_documents": sum(
                    abs(number(r, "hard_coexposure") - number(r, "expected_coexposure"))
                    > 1e-12
                    for r in own
                ),
            }
        )
        lookup = {r["material_id"]: r for r in own}
        groups = defaultdict(dict)
        for r in bank:
            if r["kind"] in {"single", "duplicate"}:
                groups["duplication", r["pair_key"]][r["kind"]] = r
            if r["kind"] == "bundle":
                groups["incidence", r["bundle_key"]][r["bundled"]] = r
        for (kind, key), pair in groups.items():
            left, right = (
                (pair["single"], pair["duplicate"])
                if kind == "duplication"
                else (pair[False], pair[True])
            )
            if left["material_id"] not in lookup:
                continue
            a, b = lookup[left["material_id"]], lookup[right["material_id"]]
            contrasts.append(
                {
                    "packet": out.name,
                    "reader": reader,
                    "kind": kind,
                    "pair_key": key,
                    "same_word_bag": Counter(re.findall(r"\w+", left["text"].lower()))
                    == Counter(re.findall(r"\w+", right["text"].lower())),
                    "coverage_delta_max": max(
                        abs(number(b, t + "_coverage") - number(a, t + "_coverage"))
                        for t in TARGETS
                    ),
                    "flat_sum_delta_max": max(
                        abs(number(b, t + "_flat_sum") - number(a, t + "_flat_sum"))
                        for t in TARGETS
                    ),
                    "coexposure_delta": number(b, "coexposure")
                    - number(a, "coexposure"),
                    "hard_coexposure_delta": number(b, "hard_coexposure")
                    - number(a, "hard_coexposure"),
                }
            )
    if max(errors.values()) > 1e-8:
        raise ValueError("alternate numerical check failed: " + repr(dict(errors)))
    return (
        {
            "packet": out.name,
            "verified_hashes": len(manifest),
            "condition_rows": len(conditions),
            "report_rows": len(readouts),
            "document_rows": len(profiles),
            "errors": dict(errors),
        },
        all_declared,
        contrasts,
    )


def main():
    """Audit both frozen packets without new parsing, encoding or refitting."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT)
    args = p.parse_args()
    if (args.output / "verification.json").exists():
        raise ValueError("verification frozen")
    source = ROOT / "results/grain_math_response_v1/binding_r1"
    manifests = json.loads((source / "04/artifact_hashes.json").read_text())
    for name, digest in manifests.items():
        if sha(source / name) != digest:
            raise ValueError("source changed")
    reports, declared, contrasts = [], [], []
    for packet in PACKETS:
        report, d, c = verify_packet(
            ROOT / "results/grain_math_response_v1" / packet, source
        )
        reports.append(report)
        declared.extend(d)
        contrasts.extend(c)
    write_csv(args.output / "all_declared_summary.csv", declared)
    write_csv(args.output / "all_declared_pairs.csv", contrasts)
    summary = {
        "packets": reports,
        "source_hashes": len(manifests),
        "all_declared": declared,
        "same_agent_alternate_formula": True,
        "independent_human_review": False,
        "new_neural_calls": 0,
        "source_refits": 0,
    }
    write_json(args.output / "verification.json", summary)
    write_json(
        args.output / "artifact_hashes.json",
        {
            str(f.relative_to(args.output)): sha(f)
            for f in args.output.rglob("*")
            if f.is_file() and f.name != "artifact_hashes.json"
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
