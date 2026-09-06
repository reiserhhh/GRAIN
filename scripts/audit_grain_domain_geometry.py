#!/usr/bin/env python3
"""Independently replay frozen domain windows, pooling and functional associations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_meps_preflight import tokenizer_setup
from scripts.run_grain_vector_mechanism_response_context import file_hash


def independent_associations(x, y, length):
    """Use correlation algebra, not the experiment's least-squares residual code."""

    def corr(a, b):
        a = np.asarray(a, dtype=float) - np.mean(a)
        b = np.asarray(b, dtype=float) - np.mean(b)
        den = np.sqrt(np.sum(a * a) * np.sum(b * b))
        return float(np.clip(np.sum(a * b) / den, -1, 1)) if den > 1e-14 else np.nan

    r = corr(x, y)
    rxz, ryz = corr(x, length), corr(y, length)
    if np.ptp(length) == 0:
        partial = r
    else:
        den = np.sqrt(max(0, 1 - rxz**2) * max(0, 1 - ryz**2))
        partial = (r - rxz * ryz) / den if den > 1e-12 else np.nan
    return np.array([r, corr(pd.Series(x).rank(), pd.Series(y).rank()), partial])


def replay_aggregate(frame):
    """Replay all fixed-case intervals with explicit independent draw calculations."""
    rows = []
    for (model, corpus, language), g in frame.groupby(
        ["model", "corpus", "language"], sort=True
    ):
        x, y, length = (
            g.delta_cosine.to_numpy(),
            g.delta_tail_logp.to_numpy(),
            g.delta_log_context_length.to_numpy(),
        )
        point = independent_associations(x, y, length)
        rank = g.within_cell_percentile.to_numpy()
        boot, gap_draws, rank_draws = [], [], []
        rng = np.random.default_rng(731)
        for _ in range(1000):
            ix = rng.integers(len(g), size=len(g))
            boot.append(independent_associations(x[ix], y[ix], length[ix]))
            gap_draws.append(float(np.sum(x[ix]) / len(ix)))
            rank_draws.append(float(np.sum(rank[ix]) / len(ix)))
        boot = np.array(boot)
        low, high = np.quantile(gap_draws, [0.025, 0.975])
        rlow, rhigh = np.quantile(rank_draws, [0.025, 0.975])
        row = {
            "model": model,
            "corpus": corpus,
            "language": language,
            "cases": len(g),
            "delta_cosine": float(x.mean()),
            "delta_cosine_low": low,
            "delta_cosine_high": high,
            "same_source_percentile": float(rank.mean()),
            "same_source_percentile_low": rlow,
            "same_source_percentile_high": rhigh,
        }
        for j, name in enumerate(("pearson", "spearman", "partial_length")):
            valid = boot[:, j][np.isfinite(boot[:, j])]
            low, high = (
                np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
            )
            row.update(
                {
                    name: point[j],
                    f"{name}_low": low,
                    f"{name}_high": high,
                    f"{name}_undefined_draws": int(1000 - len(valid)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def check_manifest(manifest):
    """Reject mutation of self-addressed lineage records."""
    if manifest["identity"] != material_hash(
        {k: v for k, v in manifest.items() if k != "identity"}
    ):
        raise ValueError("Manifest identity mismatch")


def main():
    """Replay stored embeddings and metadata, not independent neural extraction."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("material", "inventory", "function", "run", "output"):
        p.add_argument(f"--{name}-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private audit must remain in ignored results")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh audit directory required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    read = lambda p: json.loads(p.read_text())
    manifest = read(a.run_dir / "manifest.json")
    summary = read(a.run_dir / "summary.json")
    material = read(a.material_dir / "manifest.json")
    inventory = read(a.inventory_dir / "manifest.json")
    function = read(a.function_dir / "summary.json")
    for m in (manifest, material, inventory):
        check_manifest(m)
    if not (
        summary["status"] == "COMPLETE_DOMAIN_GEOMETRY_FUNCTION"
        and summary["identity"] == manifest["identity"]
        and summary["material_identity"]
        == material["identity"]
        == manifest["material_identity"]
        and summary["function_identity"]
        == function["identity"]
        == manifest["function_identity"]
        and manifest["inventory_identity"]
        == material["inventory_identity"]
        == inventory["identity"]
        and manifest["function_summary_sha256"]
        == file_hash(a.function_dir / "summary.json")
        and manifest["inventory_summary_sha256"]
        == file_hash(a.inventory_dir / "summary.json")
    ):
        raise ValueError("Parent lineage mismatch")
    for parent, data in ((a.run_dir, summary), (a.function_dir, function)):
        for name, digest in data["outputs_sha256"].items():
            if file_hash(parent / name) != digest:
                raise ValueError("Saved artifact checksum mismatch")
    tokenizers, specs = tokenizer_setup(Path.home() / ".cache/huggingface/hub")
    if (
        specs != manifest["tokenizers"]
        or specs != read(a.inventory_dir / "summary.json")["tokenizers"]
    ):
        raise ValueError("Frozen tokenizer identity changed")
    cases = read(a.material_dir / "private_cases.json")
    if material_hash(cases) != material["case_material_sha256"]:
        raise ValueError("Private material identity changed")
    lookup = {c["case_code"]: c for c in cases}
    windows = pd.read_csv(a.run_dir / "private_windows.csv")
    if set(windows.case_code) != set(lookup):
        raise ValueError("Window cases missing or added")
    token_checks = 0
    for (code, arm), g in windows.groupby(["case_code", "arm"], sort=False):
        source = lookup[code][arm]
        end = 0
        for row in g.itertuples():
            if row.start != end or row.end <= row.start:
                raise ValueError("Overlapping or missing character coverage")
            piece = source[row.start : row.end]
            if hashlib.sha256(piece.encode()).hexdigest() != row.text_sha256:
                raise ValueError("Window source hash mismatch")
            for name, (tok, prefix) in tokenizers.items():
                count = len(
                    tok(prefix + piece, truncation=False, verbose=False)["input_ids"]
                )
                if count != getattr(row, "tokens_" + name) or count > 512:
                    raise ValueError("Frozen token budget or count changed")
                token_checks += 1
            end = row.end
        if end != len(source):
            raise ValueError("Tail of raw text lost")
    condition = pd.read_csv(a.function_dir / "private_conditions.csv").pivot(
        index="case_code", columns="condition", values="tail_logp"
    )
    rows = []
    pooled_count = 0
    for completed in summary["completed"]:
        name = completed["model"]
        w = pd.read_csv(a.run_dir / f"private_{name}_windows.csv")
        v = np.load(a.run_dir / f"private_{name}_vectors.npy", allow_pickle=False)
        expected = (
            windows
            if name == "multilingual_e5_large"
            else windows[windows.language == "en"]
        )
        if list(w.source_window_row) != list(expected.index) or v.shape != (
            len(w),
            1024,
        ):
            raise ValueError("Observer window lineage or vector shape changed")
        pd.testing.assert_frame_equal(
            w.drop(columns="source_window_row"), expected.reset_index(drop=True)
        )
        if (
            not np.isfinite(v).all()
            or np.max(abs(np.linalg.norm(v, axis=1) - 1)) > 1e-5
        ):
            raise ValueError("Invalid normalized vectors")
        pooled = {}
        for (code, arm), g in w.groupby(["case_code", "arm"], sort=True):
            average = np.average(
                v[g.index].astype(np.float64), axis=0, weights=g.end - g.start
            )
            pooled[code, arm] = average / np.sqrt(np.sum(average**2))
        pooled_count += len(pooled)
        for case in cases:
            code = case["case_code"]
            if (code, "current") not in pooled:
                continue
            similarities = {
                c["case_code"]: float(
                    np.sum(pooled[code, "current"] * pooled[c["case_code"], "context"])
                )
                for c in cases
                if c["stratum"] == case["stratum"]
            }
            actual = similarities[code]
            matched = similarities[case["matched_case_code"]]
            others = np.array(
                [value for key, value in similarities.items() if key != code]
            )
            rows.append(
                {
                    "model": name,
                    "corpus": case["corpus"],
                    "language": case["language"],
                    "case_code": code,
                    "stratum": case["stratum"],
                    "actual_cosine": actual,
                    "matched_cosine": matched,
                    "random_cosine": similarities[case["random_case_code"]],
                    "delta_cosine": actual - matched,
                    "within_cell_percentile": float(
                        ((others < actual).sum() + 0.5 * (others == actual).sum())
                        / len(others)
                    ),
                    "delta_log_context_length": float(
                        np.log(
                            case["context_tokens"]
                            / lookup[case["matched_case_code"]]["context_tokens"]
                        )
                    ),
                    "delta_tail_logp": float(
                        condition.loc[code, "actual"] - condition.loc[code, "matched"]
                    ),
                }
            )
    replay = pd.DataFrame(rows)
    stored = pd.read_csv(a.run_dir / "private_geometry_by_case.csv")
    keys = ["model", "case_code"]
    replay = replay.sort_values(keys).reset_index(drop=True)
    stored = stored.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(replay, stored, check_exact=False, rtol=0, atol=1e-10)
    numeric = replay.select_dtypes(include="number").columns
    point_error = float(
        np.max(abs(replay[numeric].to_numpy() - stored[numeric].to_numpy()))
    )
    aggregates = replay_aggregate(replay)
    target = pd.read_csv(a.run_dir / "domain_geometry_function.csv")
    pd.testing.assert_frame_equal(
        aggregates, target, check_exact=False, rtol=0, atol=1e-9
    )
    numeric = aggregates.select_dtypes(include="number").columns
    aggregate_error = float(
        np.nanmax(abs(aggregates[numeric].to_numpy() - target[numeric].to_numpy()))
    )
    aggregates.to_csv(out / "replayed_domain_geometry_function.csv", index=False)
    report = {
        "status": "PASS_DOMAIN_GEOMETRY_SAVED_PACKET_AUDIT",
        "parent_identity": manifest["identity"],
        "case_observer_pairs": len(replay),
        "pooled_vectors": pooled_count,
        "common_windows": len(windows),
        "tokenizer_checks": token_checks,
        "summary_rows": len(aggregates),
        "bootstrap_case_draws": len(aggregates) * 1000,
        "max_point_error": point_error,
        "max_aggregate_error": aggregate_error,
        "new_model_forwards": 0,
        "psychological_labels_read": False,
        "prepared_private_text_read": True,
        "independent_neural_extraction": False,
        "audit_code_sha256": file_hash(Path(__file__)),
    }
    write_json(out / "summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
