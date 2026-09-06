#!/usr/bin/env python3
"""Independently replay saved MEPS window pooling and primary correspondence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from scripts.run_grain_vector_mechanism_controls import write_json


def sha(path):
    """Stream artifact checksums without loading private source text."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def unit(x):
    """Independent scalar auditor normalization with invalid-vector rejection."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.sqrt(np.sum(x * x))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Invalid stored representation")
    return x / norm


def replay_pooling(window_meta, windows, view_meta):
    """Rebuild every saved view through an independently implemented turn loop."""
    keys = ["participant_code", "view_kind", "condition", "role", "pooling"]
    if view_meta.duplicated(keys).any():
        raise ValueError("Duplicate view keys")
    rebuilt = {}
    for index, row in view_meta[view_meta.view_kind != "direct_all"].iterrows():
        subset = window_meta[
            (window_meta.participant_code == row.participant_code)
            & (window_meta.view_kind == row.view_kind)
            & (window_meta.condition == row.condition)
            & window_meta.role.isin(
                ["user", "assistant"] if row.role == "mixed" else [row.role]
            )
        ]
        by_turn = {}
        for wi, record in subset.iterrows():
            key = (record.role, record.turn_sequence)
            by_turn.setdefault(key, np.zeros(windows.shape[1]))
            by_turn[key] += record.content_tokens * unit(windows[wi])
        parts = [
            unit(v) if row.pooling == "equal_turn" else v for v in by_turn.values()
        ]
        rebuilt[index] = unit(np.sum(parts, axis=0))
        if (
            len(by_turn) != row.turns
            or int(subset.content_tokens.sum()) != row.content_tokens
            or int((subset.end - subset.start).sum()) != row.chars
            or len(subset) != row.windows
        ):
            raise ValueError("View role/weight/coverage metadata mismatch")
    for index, row in view_meta[view_meta.view_kind == "direct_all"].iterrows():
        source = view_meta[
            (view_meta.participant_code == row.participant_code)
            & (view_meta.view_kind == "direct_answer")
            & (view_meta.role == "user")
            & (view_meta.pooling == row.pooling)
        ]
        if len(source) != 3:
            raise ValueError("Missing fixed answer during independent replay")
        rebuilt[index] = unit(np.sum([rebuilt[i] for i in source.index], axis=0))
    return np.stack([rebuilt[i] for i in range(len(view_meta))])


def point(matrix):
    """Explicit participant/negative loops, independent of vectorized estimator."""
    gaps, ranks = [], []
    n = len(matrix)
    for s in (matrix, matrix.T):
        for i in range(n):
            negatives = [s[i, j] for j in range(n) if i != j]
            gaps.append(s[i, i] - np.mean(negatives))
            ranks.append(
                np.mean(
                    [float(s[i, i] > v) + 0.5 * float(s[i, i] == v) for v in negatives]
                )
            )
    return float(np.mean(gaps)), float(np.mean(ranks))


def main():
    """Audit a complete frozen packet without private text or new model forwards."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--windows-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Audit must remain in the private ignored tree")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Use a fresh audit output directory")
    root = a.run_dir
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        summary["status"] != "COMPLETE_FIXED_FREE_COMPARISON"
        or summary["identity"] != manifest["identity"]
        or manifest["identity"]
        != material_hash({k: v for k, v in manifest.items() if k != "identity"})
    ):
        raise ValueError("Incomplete or unsealed packet")
    for name, digest in summary["outputs_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Packet artifact checksum mismatch")
    checks = json.loads((root / "technical_checks.json").read_text())
    if (
        not checks["passed"]
        or not checks["loading_audit"]["nonlearned_arange_verified"]
    ):
        raise ValueError("No verified technical gate")
    meta = pd.read_csv(root / "private_window_metadata.csv")
    original = pd.read_csv(a.windows_dir / "private_window_metadata.csv")
    pd.testing.assert_frame_equal(meta[original.columns], original)
    if len(meta) != 1609 or int((meta.end - meta.start).sum()) != 366385:
        raise ValueError("Original character/window scope mismatch")
    if meta["content_tokens"].min() <= 0:
        raise ValueError("Nonpositive content token weights")
    view_meta = pd.read_csv(root / "private_view_metadata.csv")
    vectors = np.load(root / "private_window_vectors.npy", allow_pickle=False)
    stored = np.load(root / "private_view_vectors.npy", allow_pickle=False)
    if vectors.shape != (1609, 1024) or not np.all(np.isfinite(vectors)):
        raise ValueError("Window vector shape/finiteness mismatch")
    rebuilt = replay_pooling(meta, vectors, view_meta)
    pool_error = float(np.max(np.abs(rebuilt - stored)))
    if pool_error > 1e-12:
        raise ValueError("Independent pooling replay failed")
    pairs = json.loads((root / "private_pair_metadata.json").read_text())
    estimates = pd.read_csv(root / "meps_fixed_free_mechanisms.csv")
    matrices = np.load(root / "private_similarity_matrices.npz", allow_pickle=False)
    point_error, matrix_error, permutation_error = 0.0, 0.0, 0.0
    audit_rows = []
    for row in estimates.itertuples(index=False):
        item = pairs[row.contrast]
        left = view_meta.loc[item["left_rows"]]
        right = view_meta.loc[item["right_rows"]]
        if (
            list(left.participant_code) != item["participants"]
            or list(right.participant_code) != item["participants"]
            or len(set(item["participants"])) != len(item["participants"])
        ):
            raise ValueError("Broken participant correspondence")
        s = rebuilt[item["left_rows"]] @ rebuilt[item["right_rows"]].T
        matrix_error = max(
            matrix_error, float(np.max(np.abs(s - matrices[row.contrast])))
        )
        gap, rank = point(s)
        point_error = max(
            point_error,
            abs(gap - row.cosine_gap),
            abs(rank - row.same_person_percentile),
        )
        if row.primary:
            rng = np.random.default_rng(1731)
            null = []
            for _ in range(1999):
                permuted = s[:, rng.permutation(len(s))]
                diagonal = np.diag(permuted)
                off = permuted[~np.eye(len(s), dtype=bool)]
                null.append(diagonal.mean() - off.mean())
            pvalue = (1 + sum(v >= row.cosine_gap for v in null)) / 2000
            permutation_error = max(
                permutation_error, abs(pvalue - row.gap_permutation_p)
            )
        audit_rows.append(
            {
                "contrast": row.contrast,
                "participants": len(s),
                "replayed_gap": gap,
                "replayed_percentile": rank,
            }
        )
    if max(matrix_error, point_error, permutation_error) > 1e-12:
        raise ValueError("Independent correspondence replay failed")
    primary = estimates[estimates.primary]
    if len(primary) != 7:
        raise ValueError("Primary family changed")
    ordered = primary.sort_values("gap_permutation_p")
    running = 0.0
    for k, row in enumerate(ordered.itertuples(index=False)):
        running = min(1.0, max(running, (7 - k) * row.gap_permutation_p))
        if abs(running - row.gap_holm_p) > 1e-12:
            raise ValueError("Holm family correction replay failed")
    pd.DataFrame(audit_rows).to_csv(out / "replayed_contrasts.csv", index=False)
    result = {
        "status": "PASS_INDEPENDENT_SAVED_VECTOR_AUDIT",
        "parent_identity": manifest["identity"],
        "windows": len(meta),
        "view_vectors": len(view_meta),
        "contrasts": len(audit_rows),
        "primary_contrasts": 7,
        "max_pooling_error": pool_error,
        "max_matrix_error": matrix_error,
        "max_point_error": point_error,
        "max_primary_permutation_error": permutation_error,
        "primary_permutation_draws_replayed": 7 * 1999,
        "bootstrap_replayed": False,
        "independent_model_forwards": 0,
        "raw_text_read": False,
        "labels_read": False,
        "auditor_sha256": sha(Path(__file__)),
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
