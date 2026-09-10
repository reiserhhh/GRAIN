#!/usr/bin/env python3
"""Independent cache-only checks and aggregate publication for NEXT56."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_source_offset_mechanism import (
    BASE,
    CANDIDATES,
    ENDPOINT,
    OLD,
    array_hash,
    arrays,
    body_directions,
    family,
    frozen,
    read,
    ref,
    sha,
    source_set,
    verify,
    verify_hashes,
    write,
)


def audit(out):
    """Reconstruct same-function derivatives, pairing and factorial error accounts."""
    m, pf = verify(out), frozen(out, "predictions")
    verify_hashes(ROOT, m["reused_candidates"])
    verify_hashes(ROOT, m["evaluation"])
    summaries = [
        read(out / n / "summary.json") for n in ("analysis", "replay_analysis")
    ]
    for n, s in zip(("analysis", "replay_analysis"), summaries, strict=True):
        verify_hashes(out / n, s["files"])
    if summaries[0] != summaries[1] or len(summaries[0]["files"]) != 26:
        raise ValueError("Complete 26-table cached replay required")
    max_contraction, max_state, metric_error, account_error = 0.0, 0.0, 0.0, 0.0
    folder = out / "analysis"
    tables = {n: pd.read_csv(folder / n) for n in summaries[0]["files"]}
    jobs = read(out / "private_jobs.json")
    source_scores = {}
    for j in jobs:
        if j["cohort"] == "source4":
            for q, k in zip(
                ("neuroticism", "confirmation_seeking"), j["query_keys"], strict=True
            ):
                source_scores[j["index"], q] = float(
                    arrays(ENDPOINT / "capture_source" / (k + ".npz"))["score"]
                )
    for j in jobs:
        model = arrays(out / "private_models" / f"joint_{ref(j)}_pointwise.npz")
        for q, k in zip(
            ("neuroticism", "confirmation_seeking"), j["query_keys"], strict=True
        ):
            data = arrays(out / "private_inputs" / (k + ".npz"))
            x, d = data["state"].astype(float), body_directions(data)
            f = family(x, d, model)
            p, old = (
                arrays(out / "predictions" / (k + ".npz")),
                arrays(OLD / "predictions" / (k + ".npz")),
            )
            target = arrays(
                ENDPOINT
                / ("capture_source" if j["cohort"] == "source4" else "capture_new")
                / (k + ".npz")
            )
            constant = np.mean(
                [source_scores[i, q] for i in source_set(j["cohort"], j["index"])]
            )
            if p["gradient"].shape != (2, *x.shape):
                raise ValueError("Two complete suffix gradients required")
            for a, (packet, i) in zip(
                CANDIDATES, ((old, 0), (p, 0), (p, 1), (old, 2)), strict=True
            ):
                mx, md = f[a]
                if array_hash(mx.astype(np.float32)) != str(
                    packet["x32_hash"][i]
                ) or array_hash(md.astype(np.float32)) != str(packet["d32_hash"][i]):
                    raise ValueError("Candidate state/tangent hashes differ")
                tau = np.sum(
                    packet["gradient"][i].astype(float)[None]
                    * md.astype(np.float32).astype(float),
                    axis=(1, 2),
                )
                max_contraction = max(
                    max_contraction, float(abs(tau - packet["tau"][i]).max())
                )
                rows = tables["private_actions.csv"]
                rows = rows[
                    (rows.author_key == j["key"])
                    & (rows.query_name == q)
                    & (rows.candidate == a)
                ].sort_values("factor")
                np.testing.assert_allclose(rows.predicted, packet["tau"][i], atol=1e-13)
                np.testing.assert_allclose(rows.actual, target["tau"], atol=1e-13)
                s = tables["private_scores.csv"]
                s = s[
                    (s.author_key == j["key"])
                    & (s.query_name == q)
                    & (s.candidate == a)
                ]
                if len(s) != 1:
                    raise ValueError("Unique complete score row required")
                np.testing.assert_allclose(s.predicted, packet["score"][i], atol=1e-13)
                np.testing.assert_allclose(s.actual, float(target["score"]), atol=1e-13)
                np.testing.assert_allclose(s.constant, constant, atol=1e-13)
            for i in (0, 1):
                max_state = max(
                    max_state,
                    float(
                        abs(
                            f["joint"][i]
                            - f["offset_only"][i]
                            - f["nonintercept_only"][i]
                            + f["identity"][i]
                        ).max()
                    ),
                )
    for private, group, n, score in (
        ("scores", ["cohort", "query_name", "candidate"], "scores", True),
        ("score_contrasts", ["cohort", "candidate"], "score_contrasts", True),
        ("actions", ["cohort", "query_name", "candidate"], "actions", False),
        (
            "actions",
            ["cohort", "query_name", "candidate", "factor"],
            "factor_actions",
            False,
        ),
        ("action_contrasts", ["cohort", "candidate"], "action_contrasts", False),
        (
            "action_contrasts",
            ["cohort", "candidate", "factor"],
            "factor_contrasts",
            False,
        ),
    ):
        raw, result = (
            tables["private_" + private + ".csv"],
            tables[n + ".csv"].set_index(group),
        )
        for k, g in raw.groupby(group, sort=True):
            y, p = g.actual.to_numpy(), g.predicted.to_numpy()
            prefix = "score" if score else "action"
            denom = (
                np.mean((y - g.constant.to_numpy()) ** 2) if score else np.mean(y**2)
            )
            check = {
                prefix + "_r": np.corrcoef(y, p)[0, 1],
                prefix + "_rmse": np.sqrt(np.mean((p - y) ** 2)),
                prefix
                + ("_mse_over_source_constant" if score else "_mse_over_zero"): np.mean(
                    (p - y) ** 2
                )
                / denom,
            }
            metric_error = max(
                metric_error,
                *(abs(float(result.loc[k, f]) - v) for f, v in check.items()),
            )
        e = tables["private_" + n + "_effects.csv"]
        checks = (
            e.offset + e.remainder - e.change,
            e.full - e.identity - e.change,
            e.full_squared_error - e.base_squared_error - e.error_change,
            e.offset_linear_error
            + e.remainder_linear_error
            + e.offset_squared
            + e.remainder_squared
            + e.cross_error
            - e.error_change,
        )
        account_error = max(account_error, *(float(abs(v).max()) for v in checks))
    for stem, keys, values in (
        (
            "score",
            ["author_key", "cohort", "candidate"],
            ["actual", "predicted", "constant"],
        ),
        (
            "action",
            ["author_key", "cohort", "candidate", "factor"],
            ["actual", "predicted"],
        ),
    ):
        raw = tables["private_" + stem + "s.csv"]
        a, b = [
            raw[raw.query_name == q].set_index(keys)[values].sort_index()
            for q in ("neuroticism", "confirmation_seeking")
        ]
        cf = (
            tables["private_" + stem + "_contrasts.csv"]
            .set_index(keys)[values]
            .sort_index()
        )
        np.testing.assert_allclose(cf, a - b, atol=1e-13)
    energy = tables["private_offset_geometry.csv"]
    energy_error = float(
        abs(
            energy.correction_energy
            - energy.offset_energy
            - energy.remainder_energy
            - energy.signed_cross_energy
        ).max()
    )
    if (
        max(max_state, max_contraction, metric_error, account_error, energy_error)
        > 1e-10
    ):
        raise ValueError("Independent exact accounting failed")
    if (pf["reverse_pairs"], pf["jvps"], pf["fd_forwards"], pf["completed_units"]) != (
        48,
        2,
        4,
        24,
    ) or len(pf["checks"]) != 2:
        raise ValueError("Native inventory differs")
    for c in pf["checks"]:
        for field in ("jvp", "fd"):
            np.testing.assert_allclose(
                c[field], c["reverse"], atol=m[field + "_atol"], rtol=m[field + "_rtol"]
            )
    if (
        pf["native_seconds"] > 120
        or pf["numeric_seconds"] > 60
        or pf["extra_denials"]
        or pf["denial_probes"] != 4
        or pf["current_target_reads"]
        or pf["reused_output_reads"]
        or pf["fits"]
        or pf["human_labels_read"]
    ):
        raise ValueError("Bound or input restriction differs")
    r = {
        "status": "PASS",
        "primary_commit": m["commit"],
        "native_seconds": pf["native_seconds"],
        "numeric_seconds": pf["numeric_seconds"],
        "loading_seconds": pf["loading_seconds"],
        "reverse_pairs": 48,
        "jvps": 2,
        "fd_forwards": 4,
        "cached_replayed_tables": 26,
        "candidate_packets": 24,
        "contraction_max_error": max_contraction,
        "state_identity_max_error": max_state,
        "metric_max_error": float(metric_error),
        "factorial_accounting_max_error": float(account_error),
        "energy_identity_max_error": energy_error,
        "decisions": summaries[0]["decisions"],
        "audit_native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "audit_code_sha256": sha(Path(__file__)),
    }
    write(out / "replay_audit.json", r)
    dst = ROOT / "reports/grain_vector_mechanisms/source_offset_tables"
    dst.mkdir(exist_ok=True)
    names = [n for n in summaries[0]["files"] if not n.startswith("private_")]
    if len(names) != 14:
        raise ValueError("Fourteen public aggregate tables required")
    for n in names:
        if "cohort" not in tables[n] or any(
            c in tables[n] for c in ("author_key", "text", "user_id")
        ):
            raise ValueError("Private table cannot be published")
        (dst / n).write_bytes((folder / n).read_bytes())
    write(dst / "audit.json", r)
    write(dst / "inventory.json", {n: sha(dst / n) for n in names + ["audit.json"]})
    print(r, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        audit(BASE / "logic_source_offset_mechanism_v1")
