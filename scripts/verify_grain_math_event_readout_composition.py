#!/usr/bin/env python3
"""Alternate scalar, event and convex-witness replay of HM56; zero neural calls."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import write
from scripts.run_grain_math_event_readout_composition import OUT, check, observation
from scripts.run_grain_math_passive_event_composition import token_proofs


def table(out, name):
    """Read a result table without invoking its production analyzer."""
    with (out / f"03/{name}.csv").open() as stream:
        return list(csv.DictReader(stream))


def verify(out):
    """Recalculate every reading, profile and separation identity from raw logits."""
    from transformers import AutoTokenizer

    m, status = check(out), read_json(out / "run_status.json")
    if (
        status["status"] != "COLLECTED"
        or status["in_flight"] is not None
        or any(
            status[k] != 1080
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        )
        or status["reused_units"] != 216
    ):
        raise ValueError("Complete 1080 new/216 original readings required")
    for name, sha in read_json(out / "03/artifact_hashes.json").items():
        if digest(out / name) != sha:
            raise ValueError("Analysis drift " + name)
    rows = read_json(out / "00/rows.json")
    targets = m["targets"]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    proofs = [
        dict(query=q, **p)
        for q in targets
        for p in token_proofs([r for r in rows if r["query"] == q], tokenizer)
    ]
    if proofs != read_json(out / "00/token_proofs.json") or len(proofs) != 288:
        raise ValueError("Actual token proofs differ")
    scalars, atoms, maximum = {}, {}, 0.0
    by_unit = {r["unit"]: r for r in rows}
    for r in rows:
        raw, endpoint = observation(out, r, m)
        if endpoint.shape != (read_json(out / "01/observer.json")["hidden_size"],):
            raise ValueError("Incomplete endpoint")
        logits = np.asarray(raw["candidate_logits"], float)
        probability = np.exp(logits - logsumexp(logits))
        score = float(sum(probability[j] * j / 9 for j in range(10)))
        log_odds = float(logsumexp(logits[5:]) - logsumexp(logits[:5]))
        scalars[r["unit"]] = {"score": score, "log_odds": log_odds}
        maximum = max(maximum, abs(score - raw["conditional_score"]))
        if r["role"] == "atom":
            atoms[r["content"], r["family"], r["polarity"], r["query"]] = score
    native = table(out, "native")
    if len(native) != 1296:
        raise ValueError("Incomplete readout table")
    for r in native:
        unit = int(r["unit"])
        for k in ("score", "log_odds"):
            maximum = max(maximum, abs(float(r[k]) - scalars[unit][k]))
    profiles = table(out, "profiles")
    if len(profiles) != 1152:
        raise ValueError("Incomplete account profiles")
    for p in profiles:
        r = by_unit[int(p["unit"])]
        events, mentions = {}, []
        for event in r["history"]:
            if event["actor"] != "A":
                continue
            value = atoms[r["content"], event["family"], event["polarity"], r["query"]]
            if event["kind"] == "new":
                events[event["event_id"]] = value
            elif event["kind"] != "alias":
                raise ValueError("Unexpected event operation")
            mentions.append(value)
        for key, values in (
            ("event_profile", list(events.values())),
            ("mention_profile", mentions),
        ):
            maximum = max(maximum, abs(float(p[key]) - math.fsum(values) / len(values)))
        if int(p["own_events"]) != len(events) or int(p["own_mentions"]) != len(
            mentions
        ):
            raise ValueError("Event/mention counts differ")
    comp = table(out, "components")
    grouped = {}
    for r in comp:
        grouped.setdefault(int(r["parent_unit"]), {})[r["query"]] = r
    certificates = table(out, "certificates")
    if len(certificates) != 192 or len(comp) != 1152 or len(grouped) != 192:
        raise ValueError("Incomplete six-query certificates")
    vectors = {}
    for cert in certificates:
        group = grouped[int(cert["parent_unit"])]
        x, y, z = [
            np.array([float(group[q][k]) for q in targets])
            for k in ("focal_atom", "companion_atom", "native")
        ]
        direction = x - y
        length = np.linalg.norm(direction)
        u = direction / length if length else np.zeros_like(direction)
        coordinate = float(u @ (z - y))
        affine = y + coordinate * u
        projection = y + np.clip(coordinate, 0, length) * u
        witness = z - projection
        gap = float(witness @ z - max(witness @ x, witness @ y))
        box = np.maximum(np.minimum(x, y), np.minimum(z, np.maximum(x, y)))
        for key, v in (
            ("segment_distance", np.linalg.norm(witness)),
            ("affine_distance", np.linalg.norm(z - affine)),
            ("box_distance", np.linalg.norm(z - box)),
            ("dual_gap", gap),
        ):
            maximum = max(maximum, abs(float(cert[key]) - float(v)))
        maximum = max(maximum, abs(gap - float(witness @ witness)))
        for j, q in enumerate(targets):
            for key, values in (
                ("segment_projection", projection),
                ("affine_projection", affine),
                ("box_projection", box),
                ("dual_witness", witness),
            ):
                maximum = max(maximum, abs(float(group[q][key]) - float(values[j])))
        # Both endpoints bound every convex combination by linearity.
        if max(witness @ (x - projection), witness @ (y - projection)) > 1e-10:
            raise ValueError("Witness is not separating")
        vectors[int(cert["parent_unit"])] = {"z": z, "x": x, "y": y, "u": u}
    updates = table(out, "updates")
    detail = table(out, "update_components")
    if len(updates) != 144 or len(detail) != 864:
        raise ValueError("Incomplete event-update tables")
    update_index = {}
    for r in updates:
        a, b = vectors[int(r["hi"])], vectors[int(r["lo"])]
        change = a["z"] - b["z"]
        along = a["u"] * float(a["u"] @ change)
        remainder = change - along
        for key, v in (
            ("change_norm", np.linalg.norm(change)),
            ("remainder_norm", np.linalg.norm(remainder)),
        ):
            maximum = max(maximum, abs(float(r[key]) - float(v)))
        update_index[int(r["hi"]), int(r["lo"])] = (change, along, remainder)
    for r in detail:
        v = update_index[int(r["hi"]), int(r["lo"])]
        j = targets.index(r["query"])
        for key, a in zip(
            ("native_change", "along_weight", "context_remainder"), v, strict=True
        ):
            maximum = max(maximum, abs(float(r[key]) - a[j]))
    if maximum > m["spec"]["arithmetic_tolerance"]:
        raise ValueError("Alternate replay differs")
    result = {
        "status": "VERIFIED_FINITE_HM56_NOT_GLOBAL",
        "new_forward_artifacts": 1080,
        "exact_reused_artifacts": 216,
        "scalar_replays": 1296,
        "convex_certificates": 192,
        "update_vectors": 144,
        "occasion_token_proofs": 288,
        "maximum_alternate_error": maximum,
        "new_verifier_forwards": 0,
        "independent_method_replication": False,
        "original_scope_complete": False,
    }
    write(out / "verification.json", result)
    print(result)


def main():
    """Verify existing captures without any model loading or private data access."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
