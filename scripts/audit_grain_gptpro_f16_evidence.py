#!/usr/bin/env python3
"""Audit numeric cards by independent case selection and observation-space solves."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3 import f16_evidence_ledger as ledger


def read(path):
    """Load actual saved outputs, not completion flags."""
    return json.loads(Path(path).read_text())


def digest(path):
    """Identify immutable inputs/outputs exactly."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_rows(original, name, config):
    """Derive final observations directly from the case, bypassing the ledger."""
    rows = [dict(r["observation"]) for r in original["history"]]
    if name == "full_history":
        rows[-1]["y"] += config["correction_delta"]
    elif name == "late_offset":
        for row in rows[-config["tail_count"] :]:
            row["y"] += config["late_offset_delta"]
    elif name in ("missing_plus", "missing_minus"):
        value = 1 if name == "missing_plus" else -1
        rows = [r for r in rows if r["z"] != value]
    elif name == "selection_high":
        rows = [r for r in rows if r["y"] >= config["selection_threshold"]]
    elif name == "quoted_tail":
        rows = rows[: -config["tail_count"]]
    elif name == "hypothetical_all":
        rows = []
    elif name not in ("full_plain", "baseline"):
        raise ValueError("unknown predetermined case")
    return rows


def dense_t1(rows, base):
    """Condition full observation covariance; do not use cell sufficient stats."""
    conditions = [-1.0, -1 / 3, 1 / 3, 1.0]
    c = np.array([-1.0, 0.0, 0.0, 1.0])
    x = np.eye(4)[[conditions.index(r["z"]) for r in rows]]
    y = np.array([r["y"] for r in rows])
    parts = []
    for b, model in enumerate(base["source_models"]):
        mu = np.asarray(model["mu"])
        cov = np.asarray(model["Sigma"])
        if rows:
            obs = x @ cov @ x.T + model["sigma2"] * np.eye(len(rows))
            mean = mu + cov @ x.T @ np.linalg.solve(obs, y - x @ mu)
            variance = cov - cov @ x.T @ np.linalg.solve(obs, x @ cov)
        else:
            mean, variance = mu, cov
        parts.append(
            {
                "mean": float(c @ mean),
                "variance": float(c @ variance @ c),
                "bootstrap": b,
            }
        )
    return parts


def check_mixture(output, base, config):
    """Re-sum all source moments and intervals via raw second moments."""
    result = output["result"]
    parts = (
        result["source_components"]
        if output["target_id"] == "T1"
        else result["per_source"]
    )
    if [p["bootstrap"] for p in parts] != list(range(config["source_bootstraps"])):
        raise ValueError("missing or duplicated source fits")
    if output["target_id"] != "T1" and not result["supported"]:
        if result["score"] is not None or result["variance"] is not None:
            raise ValueError("unsupported source was silently dropped")
        return None
    n = len(parts)
    mean = sum(p["mean"] for p in parts) / n
    source = sum(p["mean"] ** 2 for p in parts) / n - mean**2
    within = (
        sum(p["conditional_variance"] for p in parts) / n
        if output["target_id"] == "T1"
        else sum(p["event_state"] + p["personal_parameter"] for p in parts) / n
    )
    variance = within + source
    interval = (
        [result["lower"], result["upper"]]
        if output["target_id"] == "T1"
        else result["interval"]
    )
    actual = [
        result["score"],
        result["variance"],
        result["components"]["source"],
        result["reference_score"],
        result["reference_variance"],
        *interval,
    ]
    expected = [
        mean,
        variance,
        source,
        (mean - base["reference_mu"]) / base["reference_sigma"],
        variance / base["reference_sigma"] ** 2,
        mean - 1.96 * np.sqrt(variance),
        mean + 1.96 * np.sqrt(variance),
    ]
    np.testing.assert_allclose(
        actual, expected, atol=config["atol"], rtol=config["rtol"]
    )
    return float(np.max(np.abs(np.array(actual) - expected)))


def audit(root):
    """Verify all original rows, cards, history effects, cold objects and mixtures."""
    protocol = read(root / "protocol.json")
    config = protocol["config"]
    if protocol["ledger_hash"] != ledger.code_hash():
        raise ValueError("ledger code changed after run")
    rows_out, errors, t1_errors, dynamic_errors, legacy_errors = [], [], [], [], []
    dense_count = 0
    t1_count = 0
    objects = 0
    identities = set()
    artifacts = {}
    for target in config["targets"]:
        cell = root / target
        cp = read(cell / "protocol.json")
        if (
            cp["config"] != config
            or cp["runner_hash"] != protocol["runner_sha256"]
            or cp["ledger_hash"] != protocol["ledger_hash"]
        ):
            raise ValueError("cell/root protocol mismatch")
        manifest = cp["inputs"]
        for field in (
            "base_bundle",
            "application",
            "application_manifest",
            "bundle_manifest",
        ):
            hashfield = (
                "manifest_sha256"
                if field == "application_manifest"
                else field + "_sha256"
            )
            file = manifest[field]
            if digest(file) != manifest[hashfield]:
                raise ValueError("archived input changed")
            artifacts[file] = digest(file)
        bundle = read(cell / "bundle.json")
        ledger.verify_bundle(bundle)
        base = bundle["scorer"] if target == "T1" else bundle["scorer"]["base_bundle"]
        if base != read(manifest["base_bundle"]):
            raise ValueError("embedded scorer differs from archived source")
        lines = Path(manifest["application"]).read_text().splitlines()
        originals = {}
        indices = sorted({c["author_index"] for c in config["cases"]})
        order = read(manifest["application_manifest"])["author_order"]
        for i in indices:
            original = read(cell / "inputs" / f"before_{i:03d}.json")
            originals[i] = original
            if original["author_id"] != order[i]:
                raise ValueError("author selection changed")
            identities.add((manifest["application"], original["author_id"]))
            if len(original["history"]) != config["prefix_by_target"][target]:
                raise ValueError("original observation budget changed")
            for r in original["history"]:
                path, number = r["material_ref"].rsplit("#L", 1)
                if (
                    path != manifest["application"]
                    or json.loads(lines[int(number) - 1]) != r["observation"]
                ):
                    raise ValueError("material address or original value mismatch")
                if (
                    r["qualification"] != "actual"
                    or r["op"] != "new"
                    or r["recorded_at"] != r["observation"]["t"]
                ):
                    raise ValueError("baseline history fabricated")
        keys = [f"before_{i:03d}" for i in indices] + [
            c["name"] for c in config["cases"]
        ]
        cold = [
            json.loads(s) for s in (cell / "cold_scores.jsonl").read_text().splitlines()
        ]
        if len(keys) != len(cold):
            raise ValueError("cold process incomplete")
        outputs = {}
        for key, cold_value in zip(keys, cold, strict=True):
            raw = read(cell / "inputs" / f"{key}.json")
            entry = read(cell / "scores" / f"{key}.json")
            if entry["output"] != cold_value or entry[
                "input_hash"
            ] != ledger.clock.object_hash(raw):
                raise ValueError("actual cold replay mismatch")
            index = (
                int(key.split("_")[1])
                if key.startswith("before_")
                else next(
                    c["author_index"] for c in config["cases"] if c["name"] == key
                )
            )
            original = originals[index]
            expected = expected_rows(
                original, "baseline" if key.startswith("before_") else key, config
            )
            output = entry["output"]
            stored_ledger = output["ledger"]
            if (
                stored_ledger != ledger.replay(raw, bundle)
                or stored_ledger["records"] != expected
            ):
                raise ValueError("history kept the wrong observations")
            if stored_ledger["reported_events"] != len(
                original["history"]
            ) or stored_ledger["eligible_events"] != len(expected):
                raise ValueError("history/event denominator changed")
            r = output["result"]
            if (
                r["prior_only"] != (not expected)
                or r["observed_count"] != len(expected)
                or output["measured_self_evidence_present"] != bool(expected)
            ):
                raise ValueError("borrowed score confused with measured evidence")
            error = check_mixture(output, base, config)
            if error is not None:
                errors.append(error)
            if target == "T1":
                direct = dense_t1(expected, base)
                for p, q in zip(r["source_components"], direct, strict=True):
                    actual = [p["mean"], p["conditional_variance"]]
                    alternative = [q["mean"], q["variance"]]
                    np.testing.assert_allclose(
                        actual, alternative, atol=config["atol"], rtol=config["rtol"]
                    )
                    t1_errors.append(
                        float(np.max(np.abs(np.array(actual) - alternative)))
                    )
                t1_count += len(direct)
            else:
                if r["target_time"] != original["target_time"] or r[
                    "last_observed_time"
                ] != (expected[-1]["t"] if expected else None):
                    raise ValueError("target clock lost after eligibility change")
                if r["future_times"] != (
                    list(
                        range(original["target_time"] + 1, original["target_time"] + 5)
                    )
                    if target == "T3"
                    else None
                ):
                    raise ValueError("future target changed")
                if r["supported"]:
                    if [c["bootstrap"] for c in entry["dense_checks"]] != config[
                        "dense_audit_source_indices"
                    ]:
                        raise ValueError("missing actual dynamic covariance checks")
                    for check in entry["dense_checks"]:
                        if check["draws"] != config["personal_draws"]:
                            raise ValueError("dynamic check budget reduced")
                        dense_count += check["draws"]
                        dynamic_errors += [
                            check["max_mean_error"],
                            check["max_variance_error"],
                        ]
            if key.startswith("before_"):
                oldroot = Path(manifest["base_bundle"]).parent
                if target == "T1":
                    archived = json.loads(
                        (oldroot / "cold_scores.jsonl").read_text().splitlines()[index]
                    )
                else:
                    archived = read(oldroot / "warm" / f"{index:03d}.json")
                expected_legacy = (
                    [
                        archived[k]
                        for k in (
                            "score",
                            "variance",
                            "reference_score",
                            "reference_variance",
                        )
                    ]
                    if target != "T3"
                    else [
                        archived["raw_score"],
                        archived["variance_components"]["total"],
                        archived["source_standardized_score"],
                        archived["source_standardized_variance"],
                    ]
                )
                actual = [
                    r[k]
                    for k in (
                        "score",
                        "variance",
                        "reference_score",
                        "reference_variance",
                    )
                ]
                np.testing.assert_allclose(
                    actual, expected_legacy, atol=config["atol"], rtol=config["rtol"]
                )
                legacy_errors.append(
                    float(np.max(np.abs(np.array(actual) - expected_legacy)))
                )
            outputs[key] = output
            objects += 1
        for case in config["cases"]:
            key = case["name"]
            card = read(cell / "cards" / f"{key}.json")
            before = outputs[f"before_{case['author_index']:03d}"]
            after = outputs[key]
            if (
                card["before"] != before
                or card["after"] != after
                or card["evidence_type"] != "SIM"
            ):
                raise ValueError("card did not use actual numeric scores")
            if card["source_bundle_sha256"] != digest(cell / "bundle.json"):
                raise ValueError("card linked wrong bundle")
            if read(card["before_input"]) != originals[case["author_index"]] or read(
                card["after_input"]
            ) != read(cell / "inputs" / f"{key}.json"):
                raise ValueError("broken card input link")
            if card["original_materials"] != [
                r["material_ref"] for r in originals[case["author_index"]]["history"]
            ]:
                raise ValueError("card omitted original opportunity addresses")
            effect = card["effect"]
            delta = (
                after["result"]["score"] - before["result"]["score"]
                if after["result"]["score"] is not None
                else None
            )
            if effect["score_delta"] != delta:
                raise ValueError("wrong finite edit effect")
            if key == "full_plain" and before["result"] != after["result"]:
                raise ValueError("unedited input changed")
            if key == "full_history" and len(after["ledger"]["stale_derived"]) != 2:
                raise ValueError("summary/alias of old version not marked stale")
            if key == "hypothetical_all" and after["measured_self_evidence_present"]:
                raise ValueError("hypothetical history treated as self evidence")
            rows_out.append(effect)
    if len(rows_out) != config["expected_cards"]:
        raise ValueError("missing cards")
    for target in config["targets"]:
        if not protocol["smoke"] and Counter(
            r["category"] for r in rows_out if r["target"] == target
        ) != {"full": 2, "missing": 2, "state_selection": 2, "ineligible": 2}:
            raise ValueError("original F16 card categories missing")
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("._"):
            artifacts[str(path)] = digest(path)
    return (
        rows_out,
        {
            "scope": "All cards,original addresses,expected observation selection,cold outputs and mixture moments. Dense T1 full observation solves;dynamic selected-draw covariance checks during warm execution. Not independent parameter refits or language inference.",
            "cards": len(rows_out),
            "numeric_score_objects": objects,
            "distinct_simulated_authors": len(identities),
            "t1_full_observation_checks": t1_count,
            "dynamic_parameter_checks": dense_count,
            "max_t1_observation_error": max(t1_errors, default=0.0),
            "max_dynamic_dense_error": max(dynamic_errors, default=0.0),
            "max_saved_mixture_error": max(errors, default=0.0),
            "max_legacy_full_input_error": max(legacy_errors, default=0.0),
            "source_refits": 0,
            "truth_used": False,
            "F16_parent_complete": False,
            "smoke": protocol["smoke"],
            "artifacts_hashed": len(artifacts),
        },
        artifacts,
    )


def main():
    """Save a separate audit without modifying the scored run."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, summary, hashes = audit(args.run.resolve())
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    with (args.output / "effects.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
