#!/usr/bin/env python3
"""Bounded, explicitly exploratory lexical and stratum audit of saved domain effects."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.domain_context import user_parts
from grain_research.vector_mechanisms.domain_overlap import (
    ngram_coverage,
    residual_correlation,
    subset_effect,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import (
    EXPECTED_FILES,
    file_hash,
)


def interval(values):
    """Conditional selected-case bootstrap, excluding undefined case estimands."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"eligible_cases": 0, "mean": np.nan, "lower": np.nan, "upper": np.nan}
    rng = np.random.default_rng(731)
    draws = [
        values[rng.integers(len(values), size=len(values))].mean() for _ in range(1000)
    ]
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "eligible_cases": len(values),
        "mean": float(values.mean()),
        "lower": float(low),
        "upper": float(high),
    }


def main():
    """Read allowed original tokens and saved arrays; perform no model forward."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("material", "function", "geometry", "model", "output"):
        p.add_argument(f"--{name}-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private output must remain local and ignored")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh output required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from transformers import AutoTokenizer

    read = lambda p: json.loads(p.read_text())
    material = read(a.material_dir / "manifest.json")
    cases = read(a.material_dir / "private_cases.json")
    function = read(a.function_dir / "summary.json")
    geometry = read(a.geometry_dir / "summary.json")
    if (
        material["identity"]
        != material_hash({k: v for k, v in material.items() if k != "identity"})
        or material_hash(cases) != material["case_material_sha256"]
    ):
        raise ValueError("Unsealed source cases")
    if (
        geometry["material_identity"] != material["identity"]
        or geometry["function_identity"] != function["identity"]
    ):
        raise ValueError("Parent identity mismatch")
    for parent, summary, status in (
        (a.geometry_dir, geometry, "COMPLETE_DOMAIN_GEOMETRY_FUNCTION"),
        (a.function_dir, function, "COMPLETE_DOMAIN_CONTEXT_FUNCTION"),
    ):
        if (
            summary["status"] != status
            or summary["material_identity"] != material["identity"]
        ):
            raise ValueError("Incomplete or different source material")
        for name, digest in summary["outputs_sha256"].items():
            if file_hash(parent / name) != digest:
                raise ValueError("Changed parent artifact")
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        if file_hash(a.model_dir / name) != EXPECTED_FILES[name]:
            raise ValueError("Frozen Qwen tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.model_dir), local_files_only=True, trust_remote_code=False
    )
    paths = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/domain_overlap.py",
        ROOT / "grain_research/vector_mechanisms/domain_context.py",
        ROOT / "docs/GRAIN_VM06_DOMAIN_OVERLAP_V1.md",
        ROOT / "tests/test_grain_domain_overlap.py",
    ]
    manifest = {
        "protocol": "GRAIN_VM06_DOMAIN_OVERLAP_V1",
        "analysis_status": "POST_HOC_EXPLORATORY",
        "material_identity": material["identity"],
        "function_identity": function["identity"],
        "geometry_identity": geometry["identity"],
        "model_forwards": 0,
        "labels_read": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "parents_sha256": {
            "function_summary": file_hash(a.function_dir / "summary.json"),
            "geometry_summary": file_hash(a.geometry_dir / "summary.json"),
        },
        "code": {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    with zipfile.ZipFile(
        out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    lookup = {c["case_code"]: c for c in cases}
    (out / "private_masks").mkdir()
    records, partitions = [], []
    scalar = pd.read_csv(a.function_dir / "private_conditions.csv").pivot(
        index="case_code", columns="condition", values="tail_logp"
    )
    max_rounding = 0.0
    for case in cases:
        code = case["case_code"]
        data = np.load(
            a.function_dir / "private_cases" / f"{code}.npz", allow_pickle=False
        )
        masks = {}
        for arm, donor in (
            ("actual", case),
            ("matched", lookup[case["matched_case_code"]]),
        ):
            prefix, body, ix, _ = user_parts(
                tokenizer, case["current"], donor["context"]
            )
            np.testing.assert_array_equal(prefix, data[arm + "__prefix"])
            np.testing.assert_array_equal(body, data["body"])
            context = np.asarray(prefix)[ix]
            for n in (1, 4):
                masks[f"{arm}_{n}"] = ngram_coverage(body, context, n)[1:]
        gap = data["actual__logp"][1:].astype(float) - data["matched__logp"][1:].astype(
            float
        )
        if gap.shape != masks["actual_4"].shape:
            raise ValueError("Tail index mismatch")
        max_rounding = max(
            max_rounding,
            abs(
                gap.mean() - (scalar.loc[code, "actual"] - scalar.loc[code, "matched"])
            ),
        )
        if max_rounding > 1e-5:
            raise ValueError("Saved full-tail contrast disagrees")
        np.savez_compressed(out / "private_masks" / f"{code}.npz", gap=gap, **masks)
        base = {
            "case_code": code,
            "corpus": case["corpus"],
            "language": case["language"],
            "stratum": case["stratum"],
        }
        records.append(
            base
            | {
                "log_current_tokens": np.log(case["current_tokens"]),
                **{
                    key + "_coverage": float(value.mean())
                    for key, value in masks.items()
                },
                "delta_coverage_1": float(
                    masks["actual_1"].mean() - masks["matched_1"].mean()
                ),
                "delta_coverage_4": float(
                    masks["actual_4"].mean() - masks["matched_4"].mean()
                ),
            }
        )
        selected = {
            "all_tail": np.ones(len(gap), bool),
            "neither_4": ~masks["actual_4"] & ~masks["matched_4"],
            "both_4": masks["actual_4"] & masks["matched_4"],
            "actual_only_4": masks["actual_4"] & ~masks["matched_4"],
        }
        for name, mask in selected.items():
            partitions.append(
                base
                | {
                    "partition": name,
                    "tokens": int(mask.sum()),
                    "tail_tokens": len(gap),
                    "coverage": float(mask.mean()),
                    "gap": subset_effect(gap, mask),
                }
            )
    r, parts = pd.DataFrame(records), pd.DataFrame(partitions)
    r.to_csv(out / "private_overlap_by_case.csv", index=False)
    parts.to_csv(out / "private_partitions_by_case.csv", index=False)
    pooled = []
    for (corpus, language, partition), g in parts.groupby(
        ["corpus", "language", "partition"], sort=True
    ):
        pooled.append(
            {
                "corpus": corpus,
                "language": language,
                "partition": partition,
                "all_cases": len(g),
                "mean_coverage": g.coverage.mean(),
                **interval(g.gap),
            }
        )
    pd.DataFrame(pooled).to_csv(out / "partition_summary.csv", index=False)
    geom = pd.read_csv(a.geometry_dir / "private_geometry_by_case.csv")
    joined = geom.merge(
        r, on=["case_code", "corpus", "language", "stratum"], validate="many_to_one"
    )
    if len(joined) != len(geom):
        raise ValueError("Case join lost observations")
    controlled = []
    for (model, corpus, language), g in joined.groupby(
        ["model", "corpus", "language"], sort=True
    ):
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
        point, rank, dof = residual_correlation(x, y, z)
        rng = np.random.default_rng(731)
        draws = []
        for _ in range(1000):
            ix = rng.integers(len(g), size=len(g))
            draws.append(residual_correlation(x[ix], y[ix], z[ix])[0])
        valid = np.asarray(draws)[np.isfinite(draws)]
        low, high = (
            np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
        )
        controlled.append(
            {
                "model": model,
                "corpus": corpus,
                "language": language,
                "cases": len(g),
                "design_rank": rank,
                "residual_dof": dof,
                "partial_r": point,
                "lower": low,
                "upper": high,
                "undefined_draws": 1000 - len(valid),
            }
        )
    pd.DataFrame(controlled).to_csv(out / "conditional_association.csv", index=False)
    summary = {
        "status": "COMPLETE_EXPLORATORY_DOMAIN_OVERLAP",
        "identity": manifest["identity"],
        "cases": len(cases),
        "case_observer_pairs": len(joined),
        "partitions": len(pooled),
        "max_float64_tail_vs_saved_float32_error": max_rounding,
        "labels_read": False,
        "model_forwards": 0,
        "causal_or_personality_claim": False,
        "outputs_sha256": {
            str(path.relative_to(out)): file_hash(path)
            for path in out.rglob("*")
            if path.is_file() and not path.name.startswith("._")
        },
    }
    write_json(out / "summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "outputs_sha256"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
