#!/usr/bin/env python3
"""Read-only independent numerical checks for the frozen HM128 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    """Read a JSON artifact without invoking the experimental implementation."""
    return json.loads(path.read_text())


def verify(root, mapping):
    """Check exactly the declared immutable files, not mutable runtime records."""
    for name, expected in mapping.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen artifact changed: {root / name}")
    return len(mapping)


def norm(vector):
    """Compute a scalar Euclidean norm independently of routing_accounts."""
    return float(sum(float(v) ** 2 for v in vector) ** 0.5)


def response(base, changed, natural):
    """Recompute all person, common and untouched responses using scalar loops."""
    d = [
        float(changed[0][j] - base[0][j] - changed[1][j] + base[1][j]) for j in range(6)
    ]
    n = [
        float(natural[0][j] - base[0][j] - natural[1][j] + base[1][j]) for j in range(6)
    ]
    common = [
        float((changed[0][j] + changed[1][j] - base[0][j] - base[1][j]) / 2)
        for j in range(6)
    ]
    third = [float(changed[2][j] - base[2][j]) for j in range(6)]
    energy = sum(v * v for v in n)
    dot = sum(v * w for v, w in zip(d, n, strict=True))
    return {
        "effect": d,
        "natural": n,
        "towards_natural_fraction": dot / energy if energy else None,
        "differential_norm": norm(d),
        "common_norm": norm(common),
        "untouched_norm": norm(third),
    }


def audit(out):
    """Verify outputs and retain calibration weakness separately from arithmetic."""
    status, manifest = read(out / "status.json"), read(out / "00/manifest.json")
    if status["state"] != "COLLECTED":
        raise ValueError("Full audit requires actual completed collection")
    hashes = verify(ROOT, manifest["code"]) + verify(out, manifest["inputs"])
    for phase in ("01/source_seal.json", "02/output_seal.json", "03/seal.json"):
        hashes += verify(out, read(out / phase))
    worlds = read(out / "00/application.json")
    source = read(out / "01/source_summary.json")
    summary = read(out / "03/summary.json")
    spec = manifest["spec"]
    if len(worlds) != 28 or Counter(w["kind"] for w in worlds) != {
        "fact": 4,
        "behaviour": 24,
    }:
        raise ValueError("Original application scope changed")
    with np.load(out / "01/operator.npz", allow_pickle=False) as arrays:
        op = {k: arrays[k] for k in arrays.files}
    u, omega, q, y, t = (
        op[k] for k in ("entity_basis", "omega", "q", "h_omega", "ht_q")
    )
    transpose = float(np.max(abs(q.T @ y - t.T @ omega)))
    if abs(transpose - source["transpose_product_identity_max_error"]) > 1e-12:
        raise ValueError("Recorded transpose identity inconsistent")
    orthogonality = float(np.max(abs(u.T @ u - np.eye(spec["rank"]))))
    primary = {}
    with (out / "03/behavioural_routing.csv").open() as stream:
        for r in csv.DictReader(stream):
            primary[int(r["row"]), r["arm"]] = r
    records, full_rows = {}, []
    body_query_max, edit_norm_max, numerical_max = 0.0, 0.0, 0.0
    body_query_failures = []
    candidate_mass = []
    for w in worlds:
        for arm in spec["arms"]:
            r = read(out / f"02/a{w['id']:02d}_{arm}.json")
            records[w["id"], arm] = r
            expected = 3 if w["kind"] == "fact" else 18
            if len(r["readings"]) != expected:
                raise ValueError("Missing actual person/target outputs")
            candidate_mass.extend(v["candidate_mass"] for v in r["readings"])
        with np.load(out / f"02/a{w['id']:02d}_slots.npz", allow_pickle=False) as a:
            slots = a["slots"]
            body_query_max = max(body_query_max, float(np.max(abs(slots - slots[0]))))
            if not np.allclose(slots, slots[0], atol=1e-5):
                body_query_failures.append(w["id"])
            projected = (
                (slots[:, 1].astype(float) - slots[:, 0].astype(float)) @ u
            ) @ u.T
            declared = records[w["id"], "binding"]["edit_norms"]
            control = records[w["id"], "random_matched"]["edit_norms"]
            for d, a, b in zip(projected, declared, control, strict=True):
                edit_norm_max = max(edit_norm_max, abs(norm(d) - a), abs(a - b))
        if w["kind"] != "behaviour":
            continue
        partner = next(
            v
            for v in worlds
            if v["kind"] == "behaviour"
            and v["family"] == w["family"]
            and v["wording"] == w["wording"]
            and v["assignment"] != w["assignment"]
        )
        if Counter(w["body"].split()) != Counter(partner["body"].split()):
            raise ValueError("Natural assignment control changed lexical composition")

        def scores(row, arm):
            return np.array(
                [v["conditional_score"] for v in records[row, arm]["readings"]]
            ).reshape(3, 6)

        baseline = scores(w["id"], "baseline")
        # The partner baseline may occur later; its immutable file is read directly.
        natural = np.array(
            [
                v["conditional_score"]
                for v in read(out / f"02/a{partner['id']:02d}_baseline.json")[
                    "readings"
                ]
            ]
        ).reshape(3, 6)
        for arm in spec["arms"]:
            accounts = response(baseline, scores(w["id"], arm), natural)
            expected = primary[w["id"], arm]
            for key in (
                "towards_natural_fraction",
                "differential_norm",
                "common_norm",
                "untouched_norm",
            ):
                if accounts[key] is None:
                    raise ValueError("Undefined natural assignment reference")
                numerical_max = max(
                    numerical_max, abs(accounts[key] - float(expected[key]))
                )
            for j, target in enumerate(spec["targets"]):
                full_rows.append(
                    {
                        "row": w["id"],
                        "family": w["family"],
                        "wording": w["wording"],
                        "assignment": w["assignment"],
                        "arm": arm,
                        "target": target,
                        "differential_effect": accounts["effect"][j],
                        "natural_effect": accounts["natural"][j],
                    }
                )
    factual_max = 0.0
    with (out / "03/factual_interchange.csv").open() as stream:
        for r in csv.DictReader(stream):
            rid, arm = int(r["row"]), r["arm"]
            base = [
                v["conditional_probabilities"]
                for v in records[rid, "baseline"]["readings"]
            ]
            changed = [
                v["conditional_probabilities"] for v in records[rid, arm]["readings"]
            ]
            gain = ((changed[0][1] - base[0][1]) + (changed[1][0] - base[1][0])) / 2
            factual_max = max(factual_max, abs(gain - float(r["swapped_pair_gain"])))
    if numerical_max > 1e-9 or factual_max > 1e-9 or edit_norm_max > 1e-6:
        raise ValueError("Numerical reconstruction inconsistent")
    audit_dir = out / "audit_readonly_r1"
    audit_dir.mkdir(exist_ok=False)
    with (audit_dir / "responses_all_targets.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(full_rows[0]))
        writer.writeheader()
        writer.writerows(full_rows)
    result = {
        "arithmetic_audit": "PASS",
        "verified_hashes": hashes,
        "application_worlds": len(worlds),
        "all_target_response_rows": len(full_rows),
        "routing_readback_max": numerical_max,
        "factual_readback_max": factual_max,
        "random_edit_norm_match_max": edit_norm_max,
        "body_prefix_future_query_max": body_query_max,
        "body_prefix_gate": "FAIL" if body_query_failures else "PASS",
        "body_prefix_failed_rows": body_query_failures,
        "body_prefix_frozen_tolerance_not_relaxed": True,
        "full_protocol_audit": "FAIL" if body_query_failures else "PASS",
        "transpose_identity_max": transpose,
        "entity_basis_orthogonality_max": orthogonality,
        "source_original_probability": source["original_correct_probability_mean"],
        "source_midpoint_probability": source["midpoint_correct_probability_mean"],
        "source_midpoint_probability_drop": source["original_correct_probability_mean"]
        - source["midpoint_correct_probability_mean"],
        "singular_values": op["singular_values"].tolist(),
        "candidate_mass_min": min(candidate_mass),
        "candidate_mass_max": max(candidate_mass),
        "primary_summary": summary["arms"],
        "scientific_success_not_decided_by_arithmetic": True,
        "same_agent_alternate_formula_not_independent_blind_review": True,
        "new_model_calls": 0,
        "full_scope_complete": False,
    }
    with (audit_dir / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


def main():
    """Run only numerical audit; never start or resume the model collector."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    audit(args.root)


if __name__ == "__main__":
    main()
