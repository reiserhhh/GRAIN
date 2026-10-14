#!/usr/bin/env python3
"""Verify restored F03 artifacts and reconnect the old two-condition readout."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.choice_design import anchored_cells
from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f03 import grouped
from grain_research.gptpro_foundation_v3.f03_identification import population_view


def load(path):
    """Read only declared role observations; no latent targets are required."""
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_counterexamples():
    """Construct identical observed laws with distinct unobserved response means.

    This witnesses the unrestricted-selection bound, not two members of the frozen
    four logistic worlds under a known gamma. It therefore cannot be used to claim
    nonidentification after imposing a known selection law.
    """
    examples = []
    for r in (0.1, 0.2, 0.5, 0.8, 0.9):
        q1, q0 = 0.1 / r, 0.1 / (1 - r)
        law = [1 - r * q1 - (1 - r) * q0, (1 - r) * q0, r * q1]
        np.testing.assert_allclose(law, [0.8, 0.1, 0.1], atol=1e-12)
        assert 0 <= q1 <= 1 + 1e-12 and 0 <= q0 <= 1 + 1e-12
        examples.append(
            {
                "response_mean": r,
                "selection_given_Y1": q1,
                "selection_given_Y0": q0,
                "observed_law": law,
            }
        )
    return {
        "observed_categories": ["S0", "S1_Y0", "S1_Y1"],
        "sharp_population_bound": [0.1, 0.9],
        "examples": examples,
        "scope": "unspecified response-dependent selection, not known MAR or known gamma",
        "construction": "For any r in [g,g+1-f], q1=g/r and q0=(f-g)/(1-r) reproduce the same f,g, with endpoint limits if a denominator is zero.",
    }


def additional_targets(root):
    """Terminal truth scoring of pi, q, publication and two-condition contrast."""
    truth = {
        row["author_id"]: row for row in json.loads((root / "truth.json").read_text())
    }
    with (root / "predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    output = []
    pairs = {}
    for row in predictions:
        latent = truth[row["author_id"]]
        z = int(row["z"])
        key = "-" if z == -1 else "+"
        law = population_view(
            latent["r"][key],
            latent["pi"][key],
            latent["kappa"],
            latent["lambda"],
            latent["gamma"],
            z,
            latent["design"],
        )
        base = {
            "run": root.name,
            "author_id": row["author_id"],
            "role": row["role"],
            "prefix": row["prefix"],
            "view": row["view"],
            "z": z,
        }
        for field, target in [
            ("pi", law["pi"]),
            ("q", law["f"] / law["pi"]),
            ("f", law["f"]),
        ]:
            estimate = float(row[field]) if row[field] != "" else None
            output.append(
                {
                    **base,
                    "target": field,
                    "method": "observable_count",
                    "true_value": target,
                    "estimate": estimate,
                    "squared_error": (estimate - target) ** 2
                    if estimate is not None
                    else None,
                }
            )
        if row["q_bound_low"] != "":
            low, high = float(row["q_bound_low"]), float(row["q_bound_high"])
            target = law["f"] / law["pi"]
            output.append(
                {
                    **base,
                    "target": "q",
                    "method": "sample_plugin_bound",
                    "true_value": target,
                    "bound_low": low,
                    "bound_high": high,
                    "contains_target": int(low <= target <= high),
                }
            )
        group = (row["author_id"], row["role"], row["prefix"], row["view"])
        pairs.setdefault(group, {})[z] = row
    for (author, role, prefix, view), cells in pairs.items():
        target = truth[author]["r"]["+"] - truth[author]["r"]["-"]
        for method in ("naive", "ipw", "fixed_beta"):
            valid = all(z in cells and cells[z][method] != "" for z in (-1, 1))
            estimate = (
                float(cells[1][method]) - float(cells[-1][method]) if valid else None
            )
            output.append(
                {
                    "run": root.name,
                    "author_id": author,
                    "role": role,
                    "prefix": prefix,
                    "view": view,
                    "target": "delta_response",
                    "method": method,
                    "true_value": target,
                    "estimate": estimate,
                    "squared_error": (estimate - target) ** 2
                    if estimate is not None
                    else None,
                }
            )
    return output


def main():
    """Audit same-author designs, immutable inputs and exact legacy transforms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve old audit revisions")
    args.output.mkdir(parents=True)
    (args.output / "same_observation_counterexample.json").write_text(
        json.dumps(selection_counterexamples(), indent=2) + "\n"
    )
    manifest = json.loads((args.input / "output_manifest.json").read_text())
    for entry in manifest["files"]:
        if sha256(args.input / entry["path"]) != entry["sha256"]:
            raise ValueError(f"hash mismatch: {entry['path']}")
    protocol = json.loads((args.input / "protocol.json").read_text())
    summary = json.loads((args.input / "summary.json").read_text())
    roots = sorted(args.input.glob("W3-*_free"))
    target_rows = []
    rows = []
    paired_observations = 0
    fit_count = 0
    not_converged = []
    for free in roots:
        stem = free.name.removesuffix("_free")
        for design in ("free", "fixed", "common"):
            path = args.input / f"{stem}_{design}"
            fit = json.loads((path / "fit_manifest.json").read_text())
            assert fit["source_hash"] == sha256(path / "source.jsonl.gz")
            assert fit["model_hash"] == sha256(path / "model.json")
            if fit["model_status"] == "NOT_CONVERGED":
                not_converged.append(path.name)
            fit_count += 1
            target_rows.extend(additional_targets(path))
        for role in ("source", "validation", "application"):
            fixed = load(args.input / f"{stem}_fixed" / f"{role}.jsonl.gz")
            for design in ("free", "common"):
                actual = load(args.input / f"{stem}_{design}" / f"{role}.jsonl.gz")
                assert len(actual) == len(fixed)
                for a, b in zip(actual, fixed):
                    assert (a["author_id"], a["t"], a["condition"]) == (
                        b["author_id"],
                        b["t"],
                        b["condition"],
                    )
                    if a["selected"]:
                        assert a["observation"] == b["observation"]
                    paired_observations += 1
            if role == "source":
                continue
            for design in ("free", "fixed", "common"):
                root = args.input / f"{stem}_{design}"
                with (root / "predictions.csv").open(newline="") as handle:
                    new = {
                        (r["author_id"], int(r["prefix"]), int(r["z"])): r
                        for r in csv.DictReader(handle)
                        if r["role"] == role and r["view"] == "all_opportunities"
                    }
                for author, records in grouped(load(root / f"{role}.jsonl.gz")).items():
                    for prefix in protocol["prefixes"]:
                        selected = [r for r in records[:prefix] if r["selected"]]
                        z = [r["condition"]["z"] for r in selected]
                        y = [r["observation"] for r in selected]
                        old, counts = anchored_cells(z, y)
                        cells = [new[(author, prefix, z)]["naive"] for z in (-1, 1)]
                        if all(value != "" for value in cells):
                            minus, plus = map(float, cells)
                            mapped = np.array([(minus + plus) / 2, (plus - minus) / 2])
                            error = float(np.max(np.abs(old - mapped)))
                            assert error < 1e-12
                        else:
                            error = None
                            assert np.isnan(old).all()
                        rows.append(
                            {
                                "run": root.name,
                                "role": role,
                                "author_id": author,
                                "prefix": prefix,
                                "old_level": float(old[0])
                                if np.isfinite(old[0])
                                else None,
                                "old_half_contrast": float(old[1])
                                if np.isfinite(old[1])
                                else None,
                                "count_minus": int(counts[0]),
                                "count_plus": int(counts[1]),
                                "transform_error": error,
                            }
                        )
    with (args.output / "legacy_target_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "additional_target_losses.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for row in target_rows for k in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(target_rows)
    result = {
        "status": "PASS"
        if not not_converged
        else "NUMERICALLY_VERIFIED_WITH_SOURCE_NONCONVERGENCE",
        "input_summary": summary,
        "verified_files": len(manifest["files"]),
        "fit_hashes_checked": fit_count,
        "paired_observations_checked": paired_observations,
        "legacy_readouts_checked": len(rows),
        "additional_target_rows": len(target_rows),
        "max_transform_error": max(
            r["transform_error"] for r in rows if r["transform_error"] is not None
        ),
        "source_nonconvergence": not_converged,
        "legacy_code_sha256": sha256(ROOT / "grain_research/choice_design.py"),
        "interpretation": "Old level/half-contrast is a coordinate change of selected conditional means, not a selection adjustment. IPW changes the observation target under assumptions; no numerical equality to old scores is required.",
        "parent_cards_promoted": [],
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
