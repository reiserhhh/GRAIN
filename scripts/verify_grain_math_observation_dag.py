#!/usr/bin/env python3
"""HM-07 alternate coordinate/scalar/coverage/version verification, no new fit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_response import ROOT, sha, write_json
from verify_grain_math_purpose_accounts import alternative_coordinates, table


def value(z, e, account, arm):
    """Expand typed test functions independently of the production DAG scorer."""
    k, p, o, _ = map(float, z)
    effective = "desire" if account == "desire_about_other" else account
    if arm in {"dag_typed", "declared_typed"}:
        if account == "desire_about_other":
            own = e.get("conceiver") == "self" and e.get("actor") == "other"
        elif account == "state":
            own = e.get("conceiver") == "self"
        else:
            own = e.get("actor") == "self" and (
                e.get("conceiver") == "self" or e["mode"] in {"actual", "hypothetical"}
            )
        o = 1 if own else -1
    gm = float(e["mode"] == effective)
    gk, go = np.clip((k + 1) / 2, 0, 1), np.clip((o + 1) / 2, 0, 1)
    mass = float(gm * gk * go)
    return mass * (1 if account == "state" else p), mass, k > 0 and o > 0 and gm > 0


def coverage(events, n):
    """Independently account for unknown, duplicate and missing supplied records."""
    counts, bad = Counter(), 0
    for e in events:
        j = e["record_index"]
        if e["status"] != "ok" or j is None or not 0 <= j < n:
            bad += 1
        else:
            counts[j] += 1
    return bad + sum(abs(counts[j] - 1) for j in range(n))


def main():
    """Replay saved rows with alternate formulas and raw-address checks."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--packet",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/observation_dag_r2",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/observation_dag_verification_r2",
    )
    args = ap.parse_args()
    out = args.packet
    source = (
        ROOT
        / json.loads((out / "00/protocol_before_run.json").read_text())["spec"][
            "source_packet"
        ]
    )
    hash_count = 0
    for root in (source, out):
        for name, digest in json.loads(
            (root / "04/artifact_hashes.json").read_text()
        ).items():
            if sha(root / name) != digest:
                raise AssertionError("changed frozen artifact: " + str(root / name))
            hash_count += 1
    bank = read_bank(out / "00/documents.jsonl")
    indices = json.loads((out / "01/observation_indices.json").read_text())
    coords, events_by_key, raw_nodes = {}, {}, 0
    for i, row in enumerate(bank):
        d = json.loads((out / f"01/parses/{i:04d}.json").read_text())
        if (
            d["text_sha256"]
            != __import__("hashlib").sha256(row["text"].encode()).hexdigest()
        ):
            raise AssertionError("raw text mismatch")
        if len({n["node_id"] for n in d["graph"]["nodes"]}) != len(d["graph"]["nodes"]):
            raise AssertionError("duplicate node")
        for n in d["graph"]["nodes"]:
            words = {w["id"]: w for w in d["words"][n["sentence_index"]]}
            path, cur = [], n["ud_path"][0]
            while cur:
                path.append(cur)
                cur = words[cur]["head"]
            if path != n["ud_path"]:
                raise AssertionError("manufactured dependency path")
            w = words[path[0]]
            if (
                n["action_span"] != [w["start"], w["end"]]
                or n["action"] != w["lemma"]
                or n["root_event_identity_inherited"]
            ):
                raise AssertionError("node source mismatch")
            raw_nodes += 1
    for reader in ("e5", "bge"):
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        models = {
            t: load_model(source / f"02/{reader}_{t}.npz")
            for t in {r["target"] for r in bank}
        }
        for i, row in enumerate(bank):
            for arm in ("flat", "dag_vector", "dag_typed", "declared_typed"):
                e = indices[
                    "flat"
                    if arm == "flat"
                    else "declared"
                    if arm == "declared_typed"
                    else "dag"
                ][i]
                key = reader, row["material_id"], arm
                coords[key] = alternative_coordinates(
                    e, models[row["target"]], old, new
                )
                events_by_key[key] = e
    event_rows = table(out / "02/event_readouts.csv")
    predictions = table(out / "02/predictions.csv")
    counters, aggregate = Counter(), defaultdict(list)
    coordinate_error, scalar_error, eligibility_errors = 0.0, 0.0, 0
    for row in event_rows:
        key = row["reader"], row["material_id"], row["arm"]
        counter = (*key, row["account"])
        j = counters[counter]
        counters[counter] += 1
        z, e = coords[key][j], events_by_key[key][j]
        coordinate_error = max(
            coordinate_error,
            float(np.max(np.abs(z - [float(row[f]) for f in ("k", "p", "o", "a")]))),
        )
        v, m, q = value(z, e, row["account"], row["arm"])
        scalar_error = max(
            scalar_error,
            abs(v - float(row["contribution"])),
            abs(m - float(row["mass"])),
        )
        eligibility_errors += bool(q) != (row["eligible"] == "True")
        aggregate[counter].append((v, m, q))
    by_id = {r["material_id"]: r for r in bank}
    prediction_error, coverage_errors = 0.0, 0
    for p in predictions:
        key = p["reader"], p["material_id"], p["arm"]
        declared = by_id[p["material_id"]]
        n = len(declared["events"])
        missing = coverage(events_by_key[key], n)
        coverage_errors += (missing == 0) != (p["complete"] == "True")
        rows = aggregate[*key, p["account"]]
        v, m = sum(r[0] for r in rows), sum(r[1] for r in rows)
        prediction_error = max(
            prediction_error,
            abs(v - float(p["numerator"])),
            abs(m - float(p["soft_mass"])),
        )
        if missing:
            if p["unit"] or p["profile"] or p["observed_events"]:
                raise AssertionError("fake full score for incomplete record")
        else:
            prediction_error = max(prediction_error, abs(v / n - float(p["unit"])))
            if int(p["observed_events"]) != sum(r[2] for r in rows):
                raise AssertionError("eligibility aggregate")
            good = [r[0] for r in rows if r[2]]
            if good:
                prediction_error = max(
                    prediction_error, abs(float(np.mean(good)) - float(p["profile"]))
                )
            elif p["profile"]:
                raise AssertionError("fake empty profile")
    version_rows = table(out / "03/version_dependencies.csv")
    version_errors = 0
    for r in version_rows:
        expected = (
            ["r1"]
            if r["change"] == "alias"
            else ["r2"]
            if r["change"] == "revise"
            else []
        )
        version_errors += json.loads(r["active_records"]) != expected
        ids = json.loads(r["node_ids"])
        version_errors += (
            any(not nid.startswith(expected[0] + "/") for nid in ids)
            if expected
            else bool(ids)
        )
        if r["change"] == "revise":
            version_errors += int(r["actual_main"]) != 0
    if (
        max(coordinate_error, scalar_error, prediction_error) > 1e-10
        or coverage_errors
        or eligibility_errors
        or version_errors
    ):
        raise AssertionError("HM-07 alternate verification discrepancy")
    write_json(
        args.out / "verification.json",
        {
            "status": "PASS",
            "source_application_hashes_verified": hash_count,
            "raw_node_paths_verified": raw_nodes,
            "event_rows": len(event_rows),
            "predictions": len(predictions),
            "version_programs": len(version_rows),
            "max_coordinate_error": coordinate_error,
            "max_scalar_error": scalar_error,
            "max_prediction_error": prediction_error,
            "coverage_errors": coverage_errors,
            "eligibility_errors": eligibility_errors,
            "version_errors": version_errors,
            "same_agent_alternate_formula": True,
            "human_independent_review": False,
            "new_encoding_or_fitting": 0,
        },
    )
    print(
        f"HM-07 verified {len(predictions)} readouts, {raw_nodes} raw node paths",
        flush=True,
    )


if __name__ == "__main__":
    main()
