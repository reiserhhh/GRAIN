#!/usr/bin/env python3
"""Check genuine old readouts, geometric loop orientation and scalar summaries."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_evidence_merge as A
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_response_connection_hm103 as R


def verify(out):
    """Recompute declared field summaries and verify independent loop consequences."""
    H.verify_hashes(out, H.load(out / "output_seal.json"))
    manifest = H.load(out / "manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    source = ROOT / H.load(R.SPEC)["source"]
    reader = H.load(source / "00/reader.json")
    carriers = np.load(source / "00/carriers.npz", allow_pickle=False)
    replay = []
    for row in range(len(carriers["endpoints"])):
        m, _ = R.models(reader, carriers, row, "joint")
        direct = A.observe(m.k, m.v, m.q, m.d)[0].sum(1)
        replay.append(float(np.max(np.abs(direct - m.value_jacobian(m.x)[0]))))
    with (out / "loops.csv").open() as stream:
        loops = list(csv.DictReader(stream))
    groups = {}
    probe_prediction_error = []
    for r in loops:
        if r["mode"] != "joint":
            continue
        key = tuple(int(r[n]) for n in ("row", "i", "j"))
        groups.setdefault(key, {})[float(r["label"])] = r
        probe_prediction_error.append(
            abs(float(r["probe_delta"]) - float(r["probe_prediction"]))
        )
    area_ratios, probe_even_residual = [], []
    for group in groups.values():
        big, small, rev = (group[s] for s in (0.002, 0.001, -0.002))
        area_ratios.append(float(big["drift_norm"]) / float(small["drift_norm"]))
        probe_even_residual.append(
            abs(float(big["probe_delta"]) + float(rev["probe_delta"]))
        )
    w = np.load(out / "loop_witnesses.npz", allow_pickle=False)
    key_mean_drift, vector_reverse_error = [], []
    for row, pair, start, end in zip(
        w["rows"], w["pairs"], w["offsets"][:-1], w["offsets"][1:], strict=True
    ):
        m, _ = R.models(reader, carriers, int(row), "joint")
        delta = w["delta"][start:end]
        key_mean_drift.append(
            float(np.max(np.abs(delta[: m.size].reshape(m.k.shape).mean(0))))
        )
        # Recompute only the reverse finite control, not any neural inference.
        i, j = list(__import__("itertools").combinations(range(6), 2))[pair]
        rev = R.M.rectangle(m, m.x, j, i, 0.002) - m.x
        vector_reverse_error.append(
            float(np.linalg.norm(delta + rev) / np.linalg.norm(delta))
        )
    summary = H.load(out / "summary.json")
    with (out / "brackets.csv").open() as stream:
        brackets = list(csv.DictReader(stream))
    summary_error = []
    for mode in R.MODES:
        rows = [r for r in brackets if r["mode"] == mode]
        for key, col in (("bracket_norm", "norm"), ("probe_action", "probe_action")):
            values = np.array([float(r[col]) for r in rows])
            for name, actual in R.stat(values).items():
                summary_error.append(abs(actual - summary["modes"][mode][key][name]))
    result = {
        "status": "CHECKED_NUMERIC_COMPANION_NOT_GLOBAL",
        "old_attention_readout_replay_max": max(replay),
        "summary_recompute_error_max": max(summary_error),
        "area_ratio_ideal": 4,
        "area_ratio_actual": R.stat(area_ratios),
        "reverse_vector_relative_error": R.stat(vector_reverse_error),
        "probe_prediction_absolute_error_max": max(probe_prediction_error),
        "probe_orientation_even_residual_max": max(probe_even_residual),
        "common_key_mean_drift_max": max(key_mean_drift),
        "new_neural_calls": 0,
        "same_agent_not_third_party": True,
        "code_sha256": H.sha(Path(__file__)),
    }
    if (
        result["old_attention_readout_replay_max"] > 1e-12
        or result["summary_recompute_error_max"] > 1e-12
        or result["common_key_mean_drift_max"] > 1e-12
        or max(abs(v - 4) for v in area_ratios) > 0.02
        or max(vector_reverse_error) > 0.02
    ):
        result["status"] = "FAILED_NUMERIC_COMPANION_NOT_REPAIRED"
    H.write(out / "companion_verification.json", result)
    print(__import__("json").dumps(result, indent=2))
    if result["status"].startswith("FAILED"):
        raise ValueError("Companion control failed; retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / H.load(R.SPEC)["output"])
    verify(parser.parse_args().output)
