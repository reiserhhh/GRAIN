#!/usr/bin/env python3
"""Independent address-based readback of saved positive laws and summary ranks."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_same_evidence_hm137 as H


def main():
    """Rebuild event order from metadata and validate every saved witness."""
    parent = ROOT / H.read(H.SPEC)["out"]
    out = parent / "03_mean_fiber_annex"
    manifest = H.read(out / "manifest.json")
    for name, expected in manifest["code"].items():
        if H.sha(ROOT / name) != expected:
            raise ValueError("Changed annex source")
    for name, expected in H.read(out / "seal.json").items():
        if H.sha(name) != expected:
            raise ValueError("Changed annex artifact")
    spec = H.check(parent)["spec"]
    records = H.read(parent / "00/records.json")
    e = np.empty((96, 1024))
    for batch in H.read(parent / "01/batches.json"):
        if H.sha(batch["path"]) != batch["sha256"]:
            raise ValueError("Changed encoder unit")
        with np.load(batch["path"], allow_pickle=False) as z:
            for index, record_id in enumerate(z["ids"]):
                row = records[int(record_id)]
                address = tuple(row[k] for k in ("source", "bank", "family")) + (
                    (row["b"] + 1) // 2,
                    (row["c"] + 1) // 2,
                )
                position = np.ravel_multi_index(address, (2, 2, 6, 2, 2))
                raw = z["raw"][index]
                e[position] = raw / np.sqrt(sum(float(v * v) for v in raw))
    bundle = H.read(ROOT / spec["grain_bundle"])
    g = e @ np.array(bundle["directions"])
    i, j = np.triu_indices(15)
    second = g[:, i] * g[:, j]
    views = {
        "E5_full_mean": e,
        "GRAIN15_mean": g,
        "GRAIN15_mean_and_second_moments": np.column_stack([g, second]),
    }
    with np.load(
        ROOT / spec["previous"] / "02/observations.npz", allow_pickle=False
    ) as z:
        y = z["cube"][..., :2].reshape(96, 56)
    d = np.zeros((96, 24))
    for stratum in range(24):
        d[stratum * 4 : (stratum + 1) * 4, stratum] = [1, -1, -1, 1]
    summary = H.read(out / "summary.json")
    ranks = {}
    for view, x in views.items():
        a = d.T @ x
        eigen = np.linalg.eigvalsh(a @ a.T)
        numerical_floor = max(eigen.max(), 0) * 100 * np.finfo(float).eps * 24
        rank = int(np.count_nonzero(eigen > numerical_floor))
        ranks[view] = rank
        expected = next(r["rank"] for r in summary["ranks"] if r["view"] == view)
        if rank != expected:
            raise ValueError("Independent Gram-rank mismatch")
    table = pd.read_csv(out / "all_targets.csv")
    errors = {
        "summary_preservation": 0.0,
        "stratum_margins": 0.0,
        "saved_difference": 0.0,
        "second_moment_difference": 0.0,
    }
    counts = {view: 0 for view in views}
    targets = H.read(parent / "00/manifest.json")["previous_targets"] + ["action_fact"]
    with np.load(out / "witness_laws.npz", allow_pickle=False) as laws:
        for row in table.itertuples():
            r, p, q, u = (
                ("H", "full").index(row.program),
                ("A", "B").index(row.actor),
                targets.index(row.target),
                ("score", "logodds").index(row.unit),
            )
            index = np.ravel_multi_index((r, p, q, u), (2, 2, 7, 2))
            key = f"{row.view}__q{index}"
            if row.status != "POSITIVE_CONDITIONAL_MEAN_FIBER":
                if key + "__plus" in laws:
                    raise ValueError("Unreported saved positive law")
                continue
            plus, minus = laws[key + "__plus"], laws[key + "__minus"]
            if (
                min(plus.min(), minus.min()) <= 0
                or max(abs(plus.sum() - 1), abs(minus.sum() - 1)) > 1e-10
            ):
                raise ValueError("Invalid positive law")
            change = plus - minus
            err = max(
                abs(sum(float(change[t] * views[row.view][t, k]) for t in range(96)))
                for k in range(views[row.view].shape[1])
            )
            errors["summary_preservation"] = max(errors["summary_preservation"], err)
            for s, b, c in itertools.product(range(24), range(2), range(2)):
                cells = change[s * 4 : (s + 1) * 4].reshape(2, 2)
                errors["stratum_margins"] = max(
                    errors["stratum_margins"],
                    abs(float(cells[b, :].sum())),
                    abs(float(cells[:, c].sum())),
                )
            actual = sum(float(change[t] * y[t, index]) for t in range(96))
            errors["saved_difference"] = max(
                errors["saved_difference"], abs(actual - row.account_difference)
            )
            extra = float(np.max(abs(change @ second)))
            errors["second_moment_difference"] = max(
                errors["second_moment_difference"],
                abs(extra - row.second_moment_difference_max),
            )
            counts[row.view] += 1
    if counts != summary["witness_counts"] or max(errors.values()) > 1e-10:
        raise ValueError("Saved witness readback failed")
    result = {
        "status": "VERIFIED_ADDRESS_BASED_MEAN_FIBER_READBACK",
        "ranks": ranks,
        "positive_law_pairs": counts,
        "errors": errors,
        "rows": len(table),
        "verifier_sha256": H.sha(__file__),
        "same_agent_not_blind_replication": True,
    }
    target = out / "numeric_readback.json"
    if target.exists():
        if H.read(target) != result:
            raise ValueError("Different prior readback")
    else:
        H.write(target, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
