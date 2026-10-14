#!/usr/bin/env python3
"""HM-06 alternate formulas, replay order, coverage and artifact verification."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import ROOT, sha, write_csv, write_json

DEFAULT = ROOT / "results/grain_math_response_v1/purpose_accounts_r1"
OUT = ROOT / "results/grain_math_response_v1/purpose_accounts_verification_r1"


def table(path):
    """Read saved rectangular tables without invoking the original scorer."""
    with path.open() as stream:
        return list(csv.DictReader(stream))


def numeric(p):
    """Preserve missing values instead of converting them to artificial zeros."""
    return None if p == "" else float(p)


def alternative_coordinates(events, model, old, new):
    """Distribute affine intercept and expand product-kernel centering."""

    def take(key):
        return np.array(
            [(old if e[key][0] == "source" else new)[e[key][1]] for e in events]
        )

    if not events:
        return np.empty((0, 4))
    m = model["whole_factors"]
    z = take("readout") @ m["w"] + (m["bias"] - m["mu"] @ m["w"])
    m = model["auto_tensor"]
    ka, kb = take("action_ref") @ m["a"].T, take("object_ref") @ m["b"].T
    k = 1 + ka + kb + ka * kb
    z[:, 0] = (
        k @ m["dual"]
        - k.mean(axis=1) * m["dual"].sum()
        - m["column_mean"] @ m["dual"]
        + m["grand_mean"] * m["dual"].sum()
        + m["bias"]
    )
    return z


def scalar(z, mode, account, arm):
    """Expand one purpose-indexed test function independently."""
    k, p, o, a = map(float, z)
    gk, go = min(1, max(0, (k + 1) / 2)), min(1, max(0, (o + 1) / 2))
    if arm.startswith("typed"):
        gm, eligible = float(mode == account), mode == account
    elif arm == "universal_actual":
        gm, eligible = min(1, max(0, (a + 1) / 2)), a > 0
    elif account in ("actual", "hypothetical"):
        a = a if account == "actual" else -a
        gm, eligible = min(1, max(0, (a + 1) / 2)), a > 0
    else:
        gm, eligible = 0, False
    return (
        gk * go * gm * (1 if account == "state" else p),
        gk * go * gm,
        bool(k > 0 and o > 0 and eligible),
    )


def main():
    """Verify frozen vector arithmetic and disclose all excluded records."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=Path, default=DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out, dest = args.packet, args.out
    protocol = json.loads((out / "00/protocol_before_run.json").read_text())
    source = ROOT / protocol["spec"]["source_packet"]
    hash_count = 0
    for root in (source, out):
        for name, digest in json.loads(
            (root / "04/artifact_hashes.json").read_text()
        ).items():
            if sha(root / name) != digest:
                raise AssertionError(f"changed artifact: {root / name}")
            hash_count += 1
    bank = read_bank(out / "00/documents.jsonl")
    by_id = {r["material_id"]: r for r in bank}
    indices = json.loads((out / "01/observation_indices.json").read_text())
    coords, all_events = {}, {}
    for reader in ("e5", "bge"):
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in sorted({r["target"] for r in bank}):
            model = load_model(source / f"02/{reader}_{target}.npz")
            for i, row in enumerate(bank):
                if row["target"] != target:
                    continue
                for arm in (
                    "typed_declared",
                    "typed_observed",
                    "universal_actual",
                    "binary_mirror",
                ):
                    events = indices[
                        "declared" if arm == "typed_declared" else "automatic"
                    ][i]
                    key = reader, row["material_id"], arm
                    coords[key] = alternative_coordinates(events, model, old, new)
                    all_events[key] = events
    event_table = table(out / "02/event_readouts.csv")
    row_counters, aggregate = Counter(), defaultdict(list)
    coordinate_error, contribution_error, eligibility_error = 0.0, 0.0, 0
    for r in event_table:
        key = r["reader"], r["material_id"], r["arm"]
        account = r["account"]
        j = row_counters[key, account]
        row_counters[key, account] += 1
        z = coords[key][j]
        coordinate_error = max(
            coordinate_error,
            float(np.max(np.abs(z - [float(r[f]) for f in ("k", "p", "o", "a")]))),
        )
        value, mass, eligible = scalar(z, all_events[key][j]["mode"], account, r["arm"])
        contribution_error = max(
            contribution_error,
            abs(value - float(r["contribution"])),
            abs(mass - float(r["mass"])),
        )
        eligibility_error += eligible != (r["eligible"] == "True")
        aggregate[key, account].append((value, mass, eligible))
    prediction_error, coverage_errors, count_errors = 0.0, 0, 0
    predictions = table(out / "02/predictions.csv")
    pred_lookup = {}
    for r in predictions:
        key = r["reader"], r["material_id"], r["arm"]
        account = r["account"]
        values = aggregate[key, account]
        events = all_events[key]
        expected_n = len(by_id[r["material_id"]]["events"])
        counts = Counter(e["record_index"] for e in events if e["status"] == "ok")
        complete = all(counts[i] == 1 for i in range(expected_n)) and all(
            e["status"] == "ok" and e["record_index"] is not None for e in events
        )
        coverage_errors += complete != (r["complete"] == "True")
        num, mass = sum(v[0] for v in values), sum(v[1] for v in values)
        prediction_error = max(
            prediction_error,
            abs(num - float(r["numerator"])),
            abs(mass - float(r["soft_mass"])),
        )
        if complete:
            prediction_error = max(
                prediction_error, abs(num / expected_n - float(r["unit"]))
            )
            count_errors += sum(v[2] for v in values) != int(r["observed_events"])
            selected = [v[0] for v in values if v[2]]
            if selected:
                prediction_error = max(
                    prediction_error,
                    abs(float(np.mean(selected)) - float(r["profile"])),
                )
            elif r["profile"] != "":
                raise AssertionError("absence changed into a neutral profile")
        elif r["profile"] != "" or r["unit"] != "":
            raise AssertionError("incomplete document reported as a complete score")
        pred_lookup[r["reader"], r["material_id"], r["arm"], account] = r
    coverage, failures = [], []
    for row, events in zip(bank, indices["automatic"], strict=True):
        counter = Counter(e["record_index"] for e in events)
        issue = [i for i in range(len(row["events"])) if counter[i] != 1]
        coverage.append(
            {
                "material_id": row["material_id"],
                "target": row["target"],
                "kind": row["kind"],
                "expected_events": len(row["events"]),
                "observed_heads": len(events),
                "duplicated_or_missing_records": issue,
                "unresolved_heads": sum(e["status"] != "ok" for e in events),
            }
        )
        if issue or any(e["status"] != "ok" for e in events):
            failures.append(
                {
                    "material_id": row["material_id"],
                    "kind": row["kind"],
                    "text": row["text"],
                    "expected_events": len(row["events"]),
                    "observed": [
                        {
                            "action": e["action"],
                            "mode": e["mode"],
                            "status": e["status"],
                            "flags": e["flags"],
                            "action_span": e["action_span"],
                        }
                        for e in events
                    ],
                }
            )
    version_errors = 0
    programs = {
        r["program_id"]: r
        for r in json.loads((out / "00/version_programs.json").read_text())
    }
    versions = table(out / "03/version_updates.csv")
    for r in versions:
        rows = programs[r["program_id"]]["records"]
        active = {}
        for q in rows:
            if q["operation"] in ("new", "revise"):
                active[q["event_id"]] = q
            elif q["operation"] == "retract":
                active.pop(q["event_id"])
        selected = [
            pred_lookup[r["reader"], a["material_id"], "typed_observed", r["account"]]
            for a in active.values()
        ]
        expected = sum(int(s["observed_events"] or 0) for s in selected)
        version_errors += int(r["after_events"]) != expected
        version_errors += json.loads(r["active_records"]) != [
            a["record_id"] for a in active.values()
        ]
    max_error = max(coordinate_error, contribution_error, prediction_error)
    if (
        max_error > 1e-10
        or eligibility_error
        or coverage_errors
        or count_errors
        or version_errors
    ):
        raise AssertionError(
            (
                max_error,
                eligibility_error,
                coverage_errors,
                count_errors,
                version_errors,
            )
        )
    write_csv(dest / "record_coverage.csv", coverage)
    write_json(dest / "failure_examples.json", failures)
    write_json(
        dest / "verification.json",
        {
            "status": "PASS",
            "source_application_hashes_verified": hash_count,
            "event_rows": len(event_table),
            "predictions": len(predictions),
            "version_readouts": len(versions),
            "max_coordinate_error": coordinate_error,
            "max_contribution_error": contribution_error,
            "max_prediction_error": prediction_error,
            "coverage_errors": coverage_errors,
            "eligibility_errors": eligibility_error,
            "version_errors": version_errors,
            "primary_documents_with_coverage_failure": sum(
                bool(r["duplicated_or_missing_records"] or r["unresolved_heads"])
                for r in coverage
                if r["kind"] != "unsupported_scope"
            ),
            "scope_challenge_documents": sum(
                r["kind"] == "unsupported_scope" for r in coverage
            ),
            "same_agent_alternate_formula": True,
            "human_independent_review": False,
            "no_refit_or_new_encoding": True,
        },
    )
    print(json.dumps(json.loads((dest / "verification.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
