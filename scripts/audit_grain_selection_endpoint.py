#!/usr/bin/env python3
"""Independent NEXT51 replay, native chain and source-only component checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.native_material_reader import direction
from grain_research.vector_mechanisms.selection_endpoint import key_vectors
from grain_research.vector_mechanisms.source_query_selection import source_query
from scripts.run_grain_selection_endpoint import (
    ANCHOR,
    BASE,
    NATIVE,
    QUERIES,
    SOURCE,
    arrays,
    frozen,
    read,
    sha,
    verify,
    write,
)


def audit(out):
    """Check frozen numerical artifacts and independently rejoin prior native effects."""
    verify(out)
    for a, b, n in (
        ("vectors", "vectors_replay", 24),
        ("predictions", "predictions_replay", 12),
    ):
        f, r = frozen(out, a), frozen(out, b)
        assert len(f["files"]) == n and f["files"] == r["files"]
        assert f["extra_denials"] == r["extra_denials"] == 0
    p = read(out / "analysis/summary.json")
    assert p == read(out / "analysis_replay/summary.json")
    for name, h in p["files"].items():
        assert sha(out / "analysis" / name) == sha(out / "analysis_replay" / name) == h
    jobs = read(out / "private_jobs.json")
    w = arrays(SOURCE / "private_weights.npz")
    vector_error, local_error, score_error, tau_error = [], [], [], []
    new_readers = []
    for j in jobs:
        predicted = arrays(out / "predictions" / (j["key"] + ".npz"))
        if j["cohort"] == "new8":
            new_readers.append(predicted["readers"])
        ch = arrays(SOURCE / "predictions" / (j["key"] + ".npz"))["channels"]
        local = arrays(ANCHOR / "predictions" / (j["key"] + ".npz"))["actions"]
        readers = arrays(SOURCE / "private_readers" / (j["key"] + ".npz"))["shared"]
        for qi, (q, key) in enumerate(zip(QUERIES, j["query_keys"], strict=True)):
            c = arrays(SOURCE / "private_capture" / (key + ".npz"))
            e = arrays(out / "private_evidence" / (key + ".npz"))
            v = key_vectors(source_query(c["x"][-1], w)[None], e, w["o"])[0]
            vector_error.append(float(abs(v - ch[qi, 1]).max()))
            u = arrays(out / "vectors" / (key + ".npz"))["vectors"]
            local_error.append(
                float(abs(np.einsum("afh,mh->amf", u, readers) - local[qi]).max())
            )
            cap = arrays(
                out
                / ("capture_source" if j["cohort"] == "source4" else "capture_new")
                / (key + ".npz")
            )
            old = read(NATIVE / "private_packets" / (key + ".json"))
            score_error.append(abs(float(cap["score"]) - old["contrast"]))
            grad = arrays(NATIVE / "private_packets" / (key + ".npz"))[
                "body_gradient"
            ].astype(float)
            h16 = arrays(SOURCE / "private_inputs" / (key + ".npz"))
            if h16["state"].ndim != 2:
                raise ValueError("Cached H16 uses the token-by-hidden schema")
            h = h16["state"][h16["positions"]].astype(float)
            expected = []
            for f in range(4):
                d, _ = direction(h, h16["grain"], f)
                expected.append(float(np.sum(grad * d.astype(np.float32))))
            tau_error.append(float(abs(cap["tau"] - expected).max()))
            np.testing.assert_allclose(cap["tau"], expected, atol=1e-5, rtol=3e-4)
    assert (
        max(vector_error) < 1e-10
        and max(local_error) < 1e-10
        and max(score_error) < 2e-4
    )
    for r in new_readers:
        np.testing.assert_array_equal(r, new_readers[0])
    merrors = []
    for private, groupings in (
        (
            "private_actions.csv",
            [
                ("query_summary.csv", ["cohort", "arm", "query_name"]),
                ("factor_summary.csv", ["cohort", "arm", "query_name", "factor"]),
            ],
        ),
        (
            "private_contrasts.csv",
            [
                ("contrast_summary.csv", ["cohort", "arm"]),
                ("contrast_factors.csv", ["cohort", "arm", "factor"]),
            ],
        ),
    ):
        df = pd.read_csv(out / "analysis" / private)
        assert df.groupby("cohort").author_key.nunique().to_dict() == {
            "new8": 8,
            "source4": 4,
        }
        for filename, groups in groupings:
            tab = pd.read_csv(out / "analysis" / filename).set_index(
                groups + ["readout"]
            )
            for key, g in df.groupby(groups, sort=True):
                y = g.actual.to_numpy()
                for arm in ("compiled", "transported"):
                    a = g[arm].to_numpy()
                    row = tab.loc[key + (arm,)]
                    expected = {
                        "action_r": np.corrcoef(y, a)[0, 1],
                        "action_mse_over_zero": np.sum((a - y) ** 2) / np.sum(y * y),
                        "actual_action_rms": np.sqrt(np.mean(y * y)),
                        "predicted_action_rms": np.sqrt(np.mean(a * a)),
                    }
                    merrors.extend(abs(row[k] - v) for k, v in expected.items())
                    assert row.authors == g.author_key.nunique() and row.actions == len(
                        g
                    )
    assert max(merrors) < 1e-10
    ec = pd.read_csv(out / "analysis/error_accounting.csv")
    closure = float(
        abs(ec.total_mse - ec.selection_mse - ec.reader_mse - ec.cross_term).max()
    )
    assert closure < 1e-10
    dest = ROOT / "reports/grain_vector_mechanisms/selection_endpoint_tables"
    dest.mkdir(exist_ok=True)
    public = [n for n in p["files"] if not n.startswith("private_")]
    for n in public:
        (dest / n).write_bytes((out / "analysis" / n).read_bytes())
    source, new = frozen(out, "capture_source"), frozen(out, "capture_new")
    checks = source["checks"] + new["checks"]
    assert (
        len(checks) == 2
        and new["reverse_pairs"] == 24
        and new["jvps"] == 2
        and new["fd_forwards"] == 4
    )
    result = {
        "status": "PASS",
        "vector_replays": 24,
        "prediction_replays": 12,
        "csv_replays": 7,
        "key_vector_max_difference": max(vector_error),
        "local_projection_max_difference": max(local_error),
        "native_score_max_difference": max(score_error),
        "full_H16_chain_action_max_difference": max(tau_error),
        "jvp_reverse_max_difference": max(abs(c["jvp"] - c["reverse"]) for c in checks),
        "fd_dot_max_difference": max(abs(c["fd"] - c["dot"]) for c in checks),
        "independent_metric_max_difference": max(merrors),
        "error_energy_closure": closure,
        "new8_readers_identical": True,
        "native_seconds": new["native_seconds"],
        "loading_seconds": source["loading_seconds"] + new["loading_seconds"],
        "capture_wall_seconds": source["wall_seconds"] + new["wall_seconds"],
        "audit_code_sha256": sha(Path(__file__)),
        "public_tables": {n: sha(dest / n) for n in public},
    }
    write(out / "replay_audit.json", result)
    print(result)


if __name__ == "__main__":
    audit(BASE / "logic_selection_endpoint_v1")
