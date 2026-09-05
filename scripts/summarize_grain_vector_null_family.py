#!/usr/bin/env python3
"""Replay complete author-label null draws and show finite Monte Carlo resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import task_score
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_vector_external_links import load_oof


def adjust_family(p):
    """Return Holm familywise and BH exploratory corrections without dropping tests."""
    p = np.asarray(p, dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid complete family p-values")
    order = np.argsort(p, kind="stable")
    holm, bh = np.empty_like(p), np.empty_like(p)
    holm[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p))))
    )
    bh[order] = np.minimum(
        1,
        np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[
            ::-1
        ],
    )
    return holm, bh


def exceedance_interval(k, n):
    """Clopper-Pearson interval for the underlying randomization exceedance probability."""
    return [
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    ]


def audit_null(folder, parent, source, budget):
    """Recompute each stored null statistic from its private prediction/permutation."""
    m = json.loads((folder / "manifest.json").read_text())
    s = json.loads((folder / "summary.json").read_text())
    if (
        material_hash({k: v for k, v in m.items() if k != "identity"}) != m["identity"]
        or s["parent_identity"] != m["identity"]
        or s["status"] != "COMPLETE"
        or m["draws"] != 199
        or s["draws"] != 199
        or m["bridge_labels_read"]
        or s["bridge_labels_read"]
    ):
        raise ValueError("Invalid or incomplete null seal")
    observed = load_oof(parent)
    pm = observed["manifest"]
    if (
        pm["identity"] != m["parent_identity"]
        or hashlib.sha256(
            (parent / "private_oof_predictions.csv").read_bytes()
        ).hexdigest()
        != m["observed_oof_sha256"]
    ):
        raise ValueError("Different observed source parent")
    bundles = [b for b in source if b["source_group"] == pm["source_group"]]
    if m["exclude_reference"]:
        bundles = exclude_reference(bundles, budget)
    if material_hash(bundles) != m["source_material_sha256"]:
        raise ValueError("Different null author ordering")
    lookup = {u: i for i, u in enumerate(observed["ids"])}
    order = [lookup[b["author_id"]] for b in bundles]
    y = observed["y"][order]
    statistic = task_score(y, observed["p"][order], m["task"])
    records = json.loads((folder / "permutation_records.json").read_text())
    if len(records) != 199:
        raise ValueError("Expected complete 199 indexed draws")
    values = []
    for j, r in enumerate(records):
        path = folder / "private_cache" / f"draw_{j:03d}.npz"
        if (
            r["draw"] != j
            or r["seed"] != 731 + j
            or hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]
        ):
            raise ValueError("Invalid indexed draw identity")
        with np.load(path, allow_pickle=False) as z:
            ix = np.random.default_rng(731 + j).permutation(len(y))
            if not np.array_equal(ix, z["permutation"]):
                raise ValueError("Different author label shuffle")
            score = task_score(y[ix], z["prediction"], m["task"])
        if not np.isclose(score, r["score"], atol=1e-12, rtol=0):
            raise ValueError("Null statistic cannot be replayed")
        values.append(score)
    values = np.asarray(values)
    k = int(np.sum(values >= statistic))
    p = (1 + k) / 200
    checks = {
        "observed": statistic,
        "null_mean": float(values.mean()),
        "null_sd": float(values.std(ddof=1)),
        "p_upper": p,
    }
    if any(
        not np.isclose(s[key], val, atol=1e-12, rtol=0) for key, val in checks.items()
    ):
        raise ValueError(
            "Stored null summary differs from independently replayed draws"
        )
    interval = exceedance_interval(k, 199)
    return {
        "case": folder.name,
        "task": m["task"],
        "arm": m["arm"],
        "feature": m["feature"],
        "authors": len(y),
        "draws": 199,
        **checks,
        "exceedances": k,
        "null_lower": float(np.quantile(values, 0.025)),
        "null_upper": float(np.quantile(values, 0.975)),
        "exceedance_probability_lower": interval[0],
        "exceedance_probability_upper": interval[1],
        "monte_carlo_resolution": 0.005,
        "observed_replay_error": s["observed_replay_error"],
        "parent_identity": m["parent_identity"],
    }


def main():
    """Keep missing tests in the family with conservative placeholders in partial output."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    if (
        material_hash({k: v for k, v in batch.items() if k != "identity"})
        != batch["identity"]
        or hashlib.sha256(a.views.read_bytes()).hexdigest() != batch["views_sha256"]
    ):
        raise ValueError("Changed family material")
    source = json.loads(a.views.read_text())
    budget = pd.read_csv(
        a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
    )
    rows, indices, pending = [], [], []
    for i, case in enumerate(batch["cases"]):
        folder = a.batch_dir / case["id"]
        if not (folder / "summary.json").exists():
            pending.append(case["id"])
            continue
        rows.append(audit_null(folder, Path(case["parent"]), source, budget))
        indices.append(i)
    if pending and not a.allow_partial:
        raise ValueError("Null family is not complete")
    if not rows:
        raise ValueError("No completed null")
    probabilities = np.ones(len(batch["cases"]))
    probabilities[indices] = [r["p_upper"] for r in rows]
    holm, bh = adjust_family(probabilities)
    for i, r in zip(indices, rows):
        r["holm_p_15"] = holm[i]
        r["bh_q_15_exploratory"] = bh[i]
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use new null audit directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    pd.DataFrame(rows).to_csv(out / "null_family.csv", index=False)
    summary = {
        "status": "PARTIAL" if pending else "COMPLETE",
        "verified_conditions": len(rows),
        "planned_conditions": len(batch["cases"]),
        "verified_draws": sum(r["draws"] for r in rows),
        "pending": pending,
        "bridge_labels_read": False,
        "new_label_values_read": False,
        "minimum_raw_p": 0.005,
        "minimum_holm_p_with_15_equal_floor_values": 0.075,
        "warning": "The .005 Monte Carlo floor is not an exact p or a ranking of strong signals.",
        "scope": "Unconditional source associations. Holm controls this 15-test family; BH is exploratory, not a global mechanism claim.",
        "batch_identity": batch["identity"],
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
