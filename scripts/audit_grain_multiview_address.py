#!/usr/bin/env python3
"""Independently verify saved multiview coordinates and raw-character sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.evidence_contracts import digest
from scripts.run_grain_multiview_address import BASE, E1, PREDECESSOR, VIEWS, write
from suica_core.suica import TOKEN_RE, tokenize


def rank_by_sort(left, right):
    """Independent rank calculation by sorted stranger similarities."""
    left = left / np.linalg.norm(left, axis=1)[:, None]
    right = right / np.linalg.norm(right, axis=1)[:, None]
    sim = left @ right.T
    values = []
    for matrix in (sim, sim.T):
        for i in range(len(sim)):
            strangers = np.sort(np.delete(matrix[i], i))
            lo = np.searchsorted(strangers, matrix[i, i] - 1e-12, side="left")
            hi = np.searchsorted(strangers, matrix[i, i] + 1e-12, side="right")
            values.append((lo + hi) / (2 * len(strangers)))
    return float(np.mean(values))


def run(out):
    """Reconstruct all token ranges and independently calculate coverage/ranking."""
    summary = json.loads((out / "summary.json").read_text())
    for name, sha in summary["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed completed artifact: " + name)
    material = PREDECESSOR / "material"
    data = json.loads((material / "private_material.json").read_text())
    raw = [c for b in data["bundles"] for v in b["views"] for c in v]
    ids = np.array([b["author_id"] for b in data["bundles"]])
    tokens = [list(TOKEN_RE.finditer(c["body"])) for c in raw]
    runtime = joblib.load(E1 / "artifacts/multiview_runtime.joblib")
    with np.load(out / "private_configurations.npz", allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(saved["author_ids"], ids)
    coverage = pd.read_csv(out / "private_sentence_coverage.csv")
    max_feature_error, max_rank_error, token_count, segment_count, coverage_count = (
        0.0,
        0.0,
        0,
        0,
        0,
    )
    for panel in ("full", "A", "B"):
        for view in VIEWS:
            units = pd.read_csv(
                out / f"private_units_{panel}_{view}.csv", dtype={"user_id": str}
            ).set_index("unit_id")
            segments = pd.read_csv(
                out / f"private_segments_{panel}_{view}.csv", dtype={"user_id": str}
            )
            if units.index.duplicated().any():
                raise ValueError("Duplicate units")
            unit_tokens = {k: tokenize(v) for k, v in units.text.items()}
            by_comment, union, covered_units = (
                defaultdict(list),
                defaultdict(set),
                defaultdict(set),
            )
            for seg in segments.itertuples():
                c, ts = raw[seg.source_row], tokens[seg.source_row]
                assert str(c["author_id"]) == str(seg.user_id)
                sub = c["body"][seg.char_start : seg.char_end]
                selected = ts[seg.token_start : seg.token_end]
                assert (
                    seg.char_start == selected[0].start()
                    and seg.char_end == selected[-1].end()
                )
                assert tokenize(sub) == [m.group() for m in selected]
                assert (
                    tokenize(sub)
                    == unit_tokens[seg.unit_id][
                        seg.unit_token_start : seg.unit_token_end
                    ]
                )
                current = set(range(seg.token_start, seg.token_end))
                assert not current & union[seg.source_row]
                union[seg.source_row].update(current)
                by_comment[seg.source_row].append((seg.unit_id, current))
                covered_units[seg.unit_id].update(
                    range(seg.unit_token_start, seg.unit_token_end)
                )
                token_count += len(current)
                segment_count += 1
            for uid, ts in unit_tokens.items():
                assert covered_units[uid] == set(range(len(ts)))
            if panel == "full":
                observed = coverage[coverage.view == view].set_index("sentence_index")
                for i, s in enumerate(data["sentences"]):
                    row = s["author_index"] * 36 + s["comment_index"]
                    st = {
                        j
                        for j, m in enumerate(tokens[row])
                        if m.start() < s["end"] and m.end() > s["start"]
                    }
                    actual = observed.loc[i]
                    assert actual.n_tokens == len(st)
                    assert actual.retained_tokens == len(st & union[row])
                    best = max((len(st & t) for _, t in by_comment[row]), default=0)
                    assert abs(actual.one_unit_completeness - best / len(st)) < 1e-12
                    coverage_count += 1
            h = runtime["representation"].transform(units.text.tolist())
            unit_authors = units.user_id.to_numpy()
            raw_features = []
            for author in ids:
                hh = h[unit_authors == author]
                raw_features.append(np.column_stack([hh.mean(0), hh.std(0)]).ravel())
            f = np.asarray(raw_features)
            frame = (
                pd.read_csv(
                    out / f"private_features_{panel}_{view}.csv", dtype={"user_id": str}
                )
                .set_index("user_id")
                .loc[ids]
            )
            expected = frame[runtime["feature_names"]].to_numpy(float)
            max_feature_error = max(
                max_feature_error, float(np.max(np.abs(f - expected)))
            )
            np.testing.assert_allclose(f, expected, atol=1e-10, rtol=1e-10)
            # Scaler attributes are the sealed source mean/std, never current-batch fitted.
            scale = runtime["scalers"][view]
            x = (f - scale.center) / scale.scale
            shared = runtime["final_model"].encoders[view].predict(x)
            remainder = x - shared @ runtime["final_model"].decoders[view]
            for part, actual in (
                ("raw", x),
                ("shared", shared),
                ("remainder", remainder),
            ):
                delta = float(
                    np.max(np.abs(actual - saved[f"{panel}__{view}__{part}"]))
                )
                max_feature_error = max(max_feature_error, delta)
                np.testing.assert_allclose(
                    actual, saved[f"{panel}__{view}__{part}"], rtol=1e-8, atol=1e-9
                )
    rankings = pd.read_csv(out / "source_disjoint_rank.csv")
    for row in rankings.itertuples():
        value = rank_by_sort(
            saved[f"A__{row.view}__{row.part}"], saved[f"B__{row.view}__{row.part}"]
        )
        max_rank_error = max(max_rank_error, abs(value - row.auc))
    assert max_rank_error < 1e-12
    energy = pd.read_csv(out / "reconstruction_accounting.csv")
    error = np.abs(
        energy.raw_energy
        - energy.shared_reconstruction_energy
        - energy.remainder_energy
        - energy.twice_cross_energy
    ).max()
    assert error < 1e-10
    result = {
        "status": "PASS",
        "segments_replayed": segment_count,
        "token_instances_replayed": token_count,
        "sentence_view_rows_checked": coverage_count,
        "feature_max_error": max_feature_error,
        "rank_max_error": max_rank_error,
        "energy_identity_error": float(error),
        "summary_sha256": digest(out / "summary.json"),
        "audit_script_sha256": digest(Path(__file__)),
        "neural_calls": 0,
    }
    write(out / "audit.json", result)
    print(json.dumps(result), flush=True)


def main():
    """Audit existing output without changing its frozen source or results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "all_plan_closure_v1/multiview_address_r1"
    )
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)


if __name__ == "__main__":
    main()
