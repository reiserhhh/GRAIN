#!/usr/bin/env python3
"""Independently replay content-token masks, selected readouts and SVD partials."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.domain_context import user_parts
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import file_hash


def brute_coverage(body, context, n):
    """Compare all sliding windows directly instead of set-based membership."""
    result = np.zeros(len(body), bool)
    if min(len(body), len(context)) < n:
        return result
    bc = np.lib.stride_tricks.sliding_window_view(np.asarray(body), n)
    cc = np.lib.stride_tricks.sliding_window_view(np.asarray(context), n)
    matches = (bc[:, None, :] == cc[None, :, :]).all(2).any(1)
    for index in np.flatnonzero(matches):
        result[index : index + n] = True
    return result


def svd_partial(x, y, z):
    """Project with the design's orthonormal column basis, not least squares."""
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    rank = int(np.sum(s > s[0] * max(z.shape) * np.finfo(float).eps))
    h = np.column_stack((x, y))
    residual = h - u[:, :rank] @ (u[:, :rank].T @ h)
    residual -= residual.mean(0)
    norm = np.sqrt((residual**2).sum(0))
    if len(x) - rank < 3 or min(norm) < 1e-10:
        return np.nan
    return float(
        np.clip((residual[:, 0] * residual[:, 1]).sum() / np.prod(norm), -1, 1)
    )


def main():
    """Audit stored effects without opening labels, training or re-extracting models."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("material", "function", "geometry", "model", "run", "output"):
        p.add_argument(f"--{name}-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private ignored output required")
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Fresh audit output required")
    read = lambda p: json.loads(p.read_text())
    summary = read(a.run_dir / "summary.json")
    manifest = read(a.run_dir / "manifest.json")
    material = read(a.material_dir / "manifest.json")
    function = read(a.function_dir / "summary.json")
    geometry = read(a.geometry_dir / "summary.json")
    if (
        summary["status"] != "COMPLETE_EXPLORATORY_DOMAIN_OVERLAP"
        or summary["identity"] != manifest["identity"]
    ):
        raise ValueError("Incomplete overlap packet")
    for record in (manifest, material):
        if record["identity"] != material_hash(
            {k: v for k, v in record.items() if k != "identity"}
        ):
            raise ValueError("Manifest changed")
    if (
        manifest["material_identity"] != material["identity"]
        or manifest["function_identity"] != function["identity"]
        or manifest["geometry_identity"] != geometry["identity"]
    ):
        raise ValueError("Parent identity mismatch")
    for root, packet in (
        (a.run_dir, summary),
        (a.function_dir, function),
        (a.geometry_dir, geometry),
    ):
        for name, digest in packet["outputs_sha256"].items():
            if file_hash(root / name) != digest:
                raise ValueError("Artifact mutation")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(a.model_dir), local_files_only=True, trust_remote_code=False
    )
    cases = read(a.material_dir / "private_cases.json")
    if material_hash(cases) != material["case_material_sha256"]:
        raise ValueError("Material changed")
    lookup = {c["case_code"]: c for c in cases}
    overlap = pd.read_csv(a.run_dir / "private_overlap_by_case.csv").set_index(
        "case_code"
    )
    parts = pd.read_csv(a.run_dir / "private_partitions_by_case.csv").set_index(
        ["case_code", "partition"]
    )
    mask_checks, point_error = 0, 0.0
    for case in cases:
        code = case["case_code"]
        masks = np.load(a.run_dir / "private_masks" / f"{code}.npz", allow_pickle=False)
        original = np.load(
            a.function_dir / "private_cases" / f"{code}.npz", allow_pickle=False
        )
        gap = original["actual__logp"][1:].astype(float) - original["matched__logp"][
            1:
        ].astype(float)
        np.testing.assert_array_equal(gap, masks["gap"])
        for arm, donor in (
            ("actual", case),
            ("matched", lookup[case["matched_case_code"]]),
        ):
            prefix, body, ix, _ = user_parts(
                tokenizer, case["current"], donor["context"]
            )
            np.testing.assert_array_equal(body, original["body"])
            np.testing.assert_array_equal(prefix, original[arm + "__prefix"])
            for n in (1, 4):
                key = f"{arm}_{n}"
                expected = brute_coverage(body, np.asarray(prefix)[ix], n)[1:]
                np.testing.assert_array_equal(expected, masks[key])
                point_error = max(
                    point_error,
                    abs(expected.mean() - overlap.loc[code, key + "_coverage"]),
                )
                mask_checks += 1
        selected = {
            "all_tail": np.ones(len(gap), bool),
            "neither_4": ~masks["actual_4"] & ~masks["matched_4"],
            "both_4": masks["actual_4"] & masks["matched_4"],
            "actual_only_4": masks["actual_4"] & ~masks["matched_4"],
        }
        for key, mask in selected.items():
            saved = parts.loc[(code, key)]
            estimate = (
                float(np.sum(gap[mask]) / mask.sum()) if mask.sum() >= 4 else np.nan
            )
            if (
                int(mask.sum()) != saved.tokens
                or len(mask) != saved.tail_tokens
                or (np.isnan(estimate) != np.isnan(saved.gap))
            ):
                raise ValueError("Partition eligibility mismatch")
            if np.isfinite(estimate):
                point_error = max(point_error, abs(estimate - saved.gap))
    if point_error > 1e-10:
        raise ValueError("Point estimate mismatch")
    partition_error = 0.0
    aggregates = pd.read_csv(a.run_dir / "partition_summary.csv").set_index(
        ["corpus", "language", "partition"]
    )
    for key, g in parts.reset_index().groupby(
        ["corpus", "language", "partition"], sort=True
    ):
        saved = aggregates.loc[key]
        v = g.gap.dropna().to_numpy()
        if len(v) != saved.eligible_cases or len(g) != saved.all_cases:
            raise ValueError("Case denominator mismatch")
        rng = np.random.default_rng(731)
        boot = (
            [np.sum(v[rng.integers(len(v), size=len(v))]) / len(v) for _ in range(1000)]
            if len(v)
            else [np.nan]
        )
        lo, hi = np.quantile(boot, [0.025, 0.975])
        expected = [v.mean(), lo, hi, g.coverage.mean()]
        partition_error = max(
            partition_error,
            float(
                np.nanmax(
                    abs(
                        np.asarray(expected)
                        - saved[["mean", "lower", "upper", "mean_coverage"]].to_numpy(
                            float
                        )
                    )
                )
            ),
        )
    joined = pd.read_csv(a.geometry_dir / "private_geometry_by_case.csv").merge(
        overlap.reset_index(),
        on=["case_code", "corpus", "language", "stratum"],
        validate="many_to_one",
    )
    target = pd.read_csv(a.run_dir / "conditional_association.csv").set_index(
        ["model", "corpus", "language"]
    )
    partial_error = 0.0
    for key, g in joined.groupby(["model", "corpus", "language"], sort=True):
        z = np.column_stack(
            [
                np.ones(len(g)),
                pd.get_dummies(g.stratum, drop_first=True, dtype=float),
                g[
                    [
                        "log_current_tokens",
                        "delta_log_context_length",
                        "delta_coverage_1",
                        "delta_coverage_4",
                    ]
                ],
            ]
        )
        x, y = g.delta_cosine.to_numpy(), g.delta_tail_logp.to_numpy()
        point = svd_partial(x, y, z)
        rng = np.random.default_rng(731)
        values = []
        for _ in range(1000):
            ix = rng.integers(len(g), size=len(g))
            values.append(svd_partial(x[ix], y[ix], z[ix]))
        valid = np.asarray(values)[np.isfinite(values)]
        low, high = np.quantile(valid, [0.025, 0.975])
        saved = target.loc[key]
        if saved.undefined_draws != 1000 - len(valid):
            raise ValueError("Undefined bootstrap count changed")
        partial_error = max(
            partial_error,
            float(
                np.max(
                    abs(
                        np.array([point, low, high])
                        - saved[["partial_r", "lower", "upper"]].to_numpy(float)
                    )
                )
            ),
        )
    if max(partition_error, partial_error) > 1e-9:
        raise ValueError("Independent interval or SVD replay mismatch")
    result = {
        "status": "PASS_EXPLORATORY_DOMAIN_OVERLAP_AUDIT",
        "parent_identity": manifest["identity"],
        "cases": len(cases),
        "mask_checks": mask_checks,
        "partitions": len(parts),
        "bootstrap_draws": 1000 * (len(aggregates) + len(target)),
        "max_point_error": point_error,
        "max_partition_error": partition_error,
        "max_partial_error": partial_error,
        "new_model_forwards": 0,
        "psychological_labels_read": False,
        "prepared_private_text_read": True,
        "audit_code_sha256": file_hash(Path(__file__)),
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
