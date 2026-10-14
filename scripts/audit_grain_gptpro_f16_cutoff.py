#!/usr/bin/env python3
"""Reconcile saved cutoff scores, masks, source mixtures and artifact identities."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from grain_research.gptpro_foundation_v3 import f16_cutoff as adapter


def digest(path):
    """Hash the complete recorded artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    """Check every saved mixture with scalar raw moments, not the scorer helper."""
    protocol = json.loads((root / "protocol.json").read_text())
    config = protocol["config"]
    all_rows, errors, dense_count = [], [], 0
    for target in config["targets"]:
        for seed in config["seeds"]:
            cell = root / f"{target}_{seed}"
            bundle = json.loads((cell / "bundle.json").read_text())
            adapter.verify_adapter(bundle)
            base = bundle["base_bundle"]
            declared = json.loads((cell / "protocol.json").read_text())
            if (
                declared["config"] != config
                or declared["adapter_hash"] != protocol["adapter_hash"]
            ):
                raise ValueError("cell protocol differs from root")
            cold = [
                json.loads(s)
                for s in (cell / "cold_scores.jsonl").read_text().splitlines()
            ]
            expected_count = len(config["author_indices"]) * len(config["cases"])
            if len(cold) != expected_count:
                raise ValueError("incomplete cold scores")
            index = 0
            for author_index in config["author_indices"]:
                full = json.loads(
                    (cell / "scores" / f"{author_index:03d}_full.json").read_text()
                )
                full_rows = full["input"]["records"]
                for case in config["cases"]:
                    entry = json.loads(
                        (
                            cell / "scores" / f"{author_index:03d}_{case}.json"
                        ).read_text()
                    )
                    raw, result = entry["input"], entry["score"]
                    if (
                        result != cold[index]
                        or adapter.object_hash(raw) != entry["input_hash"]
                    ):
                        raise ValueError("cold/input identity mismatch")
                    index += 1
                    excluded = set()
                    if case in ("internal_missing", "combined_missing"):
                        excluded |= set(config["internal_excluded_times"])
                    if case in ("trailing_missing", "combined_missing"):
                        excluded |= set(config["trailing_excluded_times"])
                    expected_rows = [r for r in full_rows if r["t"] not in excluded]
                    if case == "all_missing":
                        expected_rows = []
                    elif case == "only_first":
                        expected_rows = full_rows[:1]
                    if raw["records"] != expected_rows:
                        raise ValueError("mask/reindex/imputation mismatch")
                    cutoff = config["target_time"]
                    last = expected_rows[-1]["t"] if expected_rows else None
                    if (
                        result["target_time"],
                        result["last_observed_time"],
                        result["observed_count"],
                    ) != (cutoff, last, len(expected_rows)):
                        raise ValueError("target time or evidence count changed")
                    if result["future_times"] != (
                        list(range(cutoff + 1, cutoff + 5)) if target == "T3" else None
                    ):
                        raise ValueError("future clock changed")
                    if result["prior_only"] != (not expected_rows):
                        raise ValueError("prior-only status wrong")
                    parts = result["per_source"]
                    if [p["bootstrap"] for p in parts] != list(
                        range(config["source_bootstraps"])
                    ):
                        raise ValueError("source ensemble incomplete")
                    if result["supported"] != all(p["supported"] for p in parts):
                        raise ValueError("failed source silently omitted")
                    n = len(parts)
                    if result["supported"]:
                        mean = sum(p["mean"] for p in parts) / n
                        source = sum(p["mean"] ** 2 for p in parts) / n - mean**2
                        within = sum(p["event_state"] for p in parts) / n
                        personal = sum(p["personal_parameter"] for p in parts) / n
                        variance = source + within + personal
                        expected = [
                            mean,
                            variance,
                            source,
                            within,
                            personal,
                            (mean - base["reference_mu"]) / base["reference_sigma"],
                            variance / base["reference_sigma"] ** 2,
                            mean - 1.96 * np.sqrt(variance),
                            mean + 1.96 * np.sqrt(variance),
                        ]
                        actual = [
                            result["score"],
                            result["variance"],
                            result["components"]["source"],
                            result["components"]["event_state"],
                            result["components"]["personal_parameter"],
                            result["reference_score"],
                            result["reference_variance"],
                            *result["interval"],
                        ]
                        np.testing.assert_allclose(
                            actual, expected, atol=config["atol"], rtol=config["rtol"]
                        )
                        errors.append(
                            float(np.max(np.abs(np.array(actual) - expected)))
                        )
                        if [a["bootstrap"] for a in entry["dense_audit"]] != config[
                            "dense_audit_source_indices"
                        ]:
                            raise ValueError("predetermined dense checks missing")
                        if any(
                            a["parameter_draws"] != config["personal_draws"]
                            for a in entry["dense_audit"]
                        ):
                            raise ValueError("dense parameter budget changed")
                    elif result["score"] is not None or result["variance"] is not None:
                        raise ValueError("unsupported numerical score emitted")
                    dense_count += sum(
                        a["parameter_draws"] for a in entry["dense_audit"]
                    )
                    old = result["old_last_observation_target"]
                    all_rows.append(
                        {
                            "target": target,
                            "seed": seed,
                            "author_index": author_index,
                            "case": case,
                            "observations": len(expected_rows),
                            "target_time": cutoff,
                            "last_observed_time": last,
                            "score": result["score"],
                            "variance": result["variance"],
                            "old_clock_score": old["score"] if old else None,
                            "old_clock_variance": old["variance"] if old else None,
                            "same_evidence_clock_delta": result["score"] - old["score"]
                            if old
                            else None,
                            "same_evidence_variance_delta": result["variance"]
                            - old["variance"]
                            if old
                            else None,
                            "supported": result["supported"],
                            "prior_only": result["prior_only"],
                        }
                    )
    grouped = []
    for target in config["targets"]:
        for case in config["cases"]:
            rows = [r for r in all_rows if r["target"] == target and r["case"] == case]
            supported = [r for r in rows if r["same_evidence_clock_delta"] is not None]
            grouped.append(
                {
                    "target": target,
                    "case": case,
                    "objects": len(rows),
                    "mean_abs_clock_delta": float(
                        np.mean(
                            [abs(r["same_evidence_clock_delta"]) for r in supported]
                        )
                    )
                    if supported
                    else None,
                    "mean_variance_delta": float(
                        np.mean([r["same_evidence_variance_delta"] for r in supported])
                    )
                    if supported
                    else None,
                }
            )
    return all_rows, {
        "objects": len(all_rows),
        "dense_checks": dense_count,
        "max_mixture_error": max(errors, default=0.0),
        "unsupported": sum(not r["supported"] for r in all_rows),
        "by_target_case": grouped,
        "scope": "All saved numeric mixtures/masks/clock metadata and cold objects; dense checks reused from separately derived covariance during execution. Not an independent MAP/Hessian refit or calibration study.",
    }


def main():
    """Export replayable paired clock effects without loading any target truth."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, summary = audit(args.run)
    with (args.output / "clock_effects.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        str(p): digest(p)
        for p in sorted(args.run.rglob("*"))
        if p.is_file() and not p.name.startswith("._")
    }
    summary["input_artifacts_hashed"] = len(manifest)
    for name, value in (("summary.json", summary), ("input_hashes.json", manifest)):
        (args.output / name).write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
