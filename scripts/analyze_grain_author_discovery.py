#!/usr/bin/env python3
"""Discovery-first real-author native-query comparison using frozen cached scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.external_heads import TRAITS
from scripts.analyze_grain_direct_source_scores import direct_prediction
from scripts.summarize_grain_vector_external_links import load_oof

TASKS = ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")
ARMS = ("raw", "cue_mask", "placebo_mask")
OBSERVERS = ("bge_large", "e5_large", "qwen_block_10")


def read(path):
    """Read one local JSON artifact without logging its private contents."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    """Hash an input file for a reproducible availability snapshot."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    """Write strict JSON, refusing nonfinite metrics."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def scalar_metrics(y, p, binary=False):
    """Report associations and fixed-threshold metrics without orienting to labels."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite([y, p]).all():
        raise ValueError("Require paired finite vectors")
    row = {"authors": len(y), "prediction_sd": float(np.std(p))}
    varied = len(y) >= 3 and np.ptp(y) > 0 and np.ptp(p) > 0
    r, pv = pearsonr(y, p) if varied else (None, None)
    row.update(
        pearson_r=float(r) if r is not None else None,
        pearson_p=float(pv) if pv is not None else None,
        spearman_rho=float(spearmanr(y, p).statistic) if varied else None,
    )
    bounds = (None, None)
    if varied and len(y) > 3:
        z = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15))
        bounds = np.tanh(z + np.array([-1, 1]) * norm.ppf(0.975) / np.sqrt(len(y) - 3))
    row.update(
        r_lower=float(bounds[0]) if bounds[0] is not None else None,
        r_upper=float(bounds[1]) if bounds[1] is not None else None,
        mae=float(np.abs(y - p).mean()),
        rmse=float(np.sqrt(((y - p) ** 2).mean())),
    )
    if binary:
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Binary targets required")
        both = len(np.unique(y)) == 2
        row.update(
            auc=float(roc_auc_score(y, p)) if both else None,
            balanced_accuracy=float(balanced_accuracy_score(y, p >= 0.5))
            if both
            else None,
            accuracy=float(np.mean(y == (p >= 0.5))),
            positive_rate_true=float(y.mean()),
            positive_rate_pred=float(np.mean(p >= 0.5)),
        )
    return row


def read_native(queue):
    """Load every completed shard and check saved hashes, not outcome performance."""
    manifest, status = read(queue / "manifest.json"), read(queue / "queue_status.json")
    completed = status["completed"]
    if (
        status.get("child_pid")
        or not completed
        or manifest.get("preflight_only") is not False
    ):
        raise ValueError("Require an idle real-cache snapshot")
    if len({r["shard"] for r in completed}) != len(completed):
        raise ValueError("Duplicate shard")
    records, inputs = {}, {}
    for item in completed:
        shard = queue / f"shard_{item['shard']:03d}"
        summary, sm = read(shard / "summary.json"), read(shard / "manifest.json")
        if (
            summary["status"] != "COMPLETE"
            or not summary["real_neural_collection"]
            or summary["parent_identity"] != sm["identity"]
            or sm["material_identity"] != manifest["material_identity"]
            or summary["model_signature"] != manifest["model_signature"]
        ):
            raise ValueError("Incomplete or mixed-model shard")
        inputs[str((shard / "summary.json").resolve())] = digest(shard / "summary.json")
        paths = sorted(
            p
            for p in (shard / "private_cache").glob("*.json")
            if not p.name.startswith("._")
        )
        if len(paths) != summary["authors"]:
            raise ValueError("Missing author cache")
        for path in paths:
            sha = digest(path)
            if summary["outputs_sha256"][path.relative_to(shard).as_posix()] != sha:
                raise ValueError("Cache hash mismatch")
            packet = read(path)
            uid = packet["author_id"]
            if uid in records:
                raise ValueError("Duplicate author")
            records[uid] = packet
            inputs[str(path.resolve())] = sha
    if len(records) != status["verified_authors"]:
        raise ValueError("Snapshot author count mismatch")
    return records, inputs, manifest, status


def freeze(args):
    """Fix all available authors and input digests before joining source outcomes."""
    if args.output_dir.exists():
        raise ValueError("Use a new discovery output directory")
    records, inputs, queue, status = read_native(args.queue_dir)
    material = read(args.material_dir / "manifest.json")
    if material["identity"] != queue["material_identity"]:
        raise ValueError("Native material mismatch")
    sources = {}
    for task in TASKS:
        for arm in ARMS:
            for observer in OBSERVERS:
                folder = args.primary_dir / f"{task}_{arm}_{observer}"
                m = read(folder / "manifest.json")
                if (
                    m["common_material_sha256"] != material["common_material_sha256"]
                    or m["bridge_labels_used"]
                    or not m["exclude_reference"]
                ):
                    raise ValueError("Source-head material mismatch")
                for name in (
                    "manifest.json",
                    "summary.json",
                    "private_oof_predictions.csv",
                ):
                    path = folder / name
                    sources[str(path.resolve())] = digest(path)
    payload = {
        "protocol": "GRAIN_D1_DISCOVERY_SNAPSHOT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "queue_identity": queue["identity"],
        "material_identity": material["identity"],
        "common_material_sha256": material["common_material_sha256"],
        "completed_shards": len(status["completed"]),
        "authors": len(records),
        "source_counts": dict(Counter(r["source_group"] for r in records.values())),
        "private_author_ids": sorted(records),
        "native_inputs": inputs,
        "source_inputs": sources,
        "script_sha256": digest(__file__),
        "source_outcomes_joined": False,
        "analysis_scope": "availability-fixed exploratory subset; full128 contract unchanged",
    }
    args.output_dir.mkdir(parents=True)
    write(args.output_dir / "membership_snapshot.json", payload)
    print(
        json.dumps(
            {k: payload[k] for k in ("completed_shards", "authors", "source_counts")}
        )
    )


def analyze(args):
    """Compare every source trait and observer on the same frozen available authors."""
    frozen = read(args.output_dir / "membership_snapshot.json")
    if (args.output_dir / "summary.json").exists():
        raise ValueError("Preserve completed discovery analysis")
    for path, sha in {**frozen["native_inputs"], **frozen["source_inputs"]}.items():
        if digest(path) != sha:
            raise ValueError("Frozen input changed")
    if digest(__file__) != frozen["script_sha256"]:
        raise ValueError("Analysis code changed after membership freeze")
    records, _, _, _ = read_native(args.queue_dir)
    if sorted(records) != frozen["private_author_ids"]:
        raise ValueError("Frozen cohort changed")
    metrics, links, predictions, populations, quality = [], [], [], [], []
    for task in TASKS:
        group = "big5_only" if task == "big5" else "mbti_only"
        names = TRAITS if task == "big5" else (task,)
        for arm in ARMS:
            packets = [
                load_oof(args.primary_dir / f"{task}_{arm}_{o}") for o in OBSERVERS
            ]
            anchor = packets[0]
            for packet in packets[1:]:
                if any(
                    not np.array_equal(packet[k], anchor[k])
                    for k in ("ids", "y", "folds")
                ):
                    raise ValueError("Unpaired statistical observers")
            ix = np.array(
                [
                    i
                    for i, uid in enumerate(anchor["ids"])
                    if uid in records and records[uid]["source_group"] == group
                ],
                dtype=int,
            )
            ids, y = anchor["ids"][ix], anchor["y"][ix]
            if len(ix) < 3:
                raise ValueError(f"Too few available source authors: {task}")
            p, diagnostic = direct_prediction(records, ids, task, arm)
            values = {
                "native_query": p,
                **{o: v["p"][ix] for o, v in zip(OBSERVERS, packets)},
            }
            populations.append(
                {"task": task, "arm": arm, "source_group": group, "authors": len(ids)}
            )
            q = pd.DataFrame(diagnostic)
            for trait, rows in q.groupby("trait"):
                quality.append(
                    {
                        "task": task,
                        "arm": arm,
                        "trait": trait,
                        "candidate_mass_median": float(rows.candidate_mass.median()),
                        "first_token_compliance": float(
                            rows.greedy_first_token_is_candidate.mean()
                        ),
                    }
                )
            for j, trait in enumerate(names):
                truth = y[:, j] if task == "big5" else y
                native = p[:, j] if task == "big5" else p
                for observer, value in values.items():
                    pred = value[:, j] if task == "big5" else value
                    metrics.append(
                        dict(
                            task=task,
                            trait=trait,
                            arm=arm,
                            observer=observer,
                            **scalar_metrics(truth, pred, task != "big5"),
                        )
                    )
                    if observer != "native_query":
                        corr = scalar_metrics(native, pred)
                        links.append(
                            {
                                "task": task,
                                "trait": trait,
                                "arm": arm,
                                "observer": observer,
                                "authors": len(ids),
                                "query_head_r": corr["pearson_r"],
                                "query_head_rho": corr["spearman_rho"],
                            }
                        )
                    predictions.extend(
                        {
                            "author_id": uid,
                            "task": task,
                            "trait": trait,
                            "arm": arm,
                            "observer": observer,
                            "y_true": float(t),
                            "prediction": float(v),
                        }
                        for uid, t, v in zip(ids, truth, pred)
                    )
    for name, rows in (
        ("metrics", metrics),
        ("query_head_links", links),
        ("private_predictions", predictions),
        ("populations", populations),
        ("native_quality", quality),
    ):
        pd.DataFrame(rows).to_csv(args.output_dir / f"{name}.csv", index=False)
    table = pd.DataFrame(metrics)
    headline = []
    for (task, arm, observer), rows in table.groupby(
        ["task", "arm", "observer"], sort=False
    ):
        key = "pearson_r" if task == "big5" else "auc"
        headline.append(
            {
                "task": task,
                "arm": arm,
                "observer": observer,
                "metric": key,
                "value": float(rows[key].mean()) if rows[key].notna().all() else None,
                "authors": int(rows.authors.iloc[0]),
            }
        )
    pd.DataFrame(headline).to_csv(args.output_dir / "headline.csv", index=False)
    summary = {
        "status": "COMPLETE_DISCOVERY",
        "scope": frozen["analysis_scope"],
        "completed_shards": frozen["completed_shards"],
        "authors_available": frozen["authors"],
        "metric_rows": len(metrics),
        "source_outcomes_joined": True,
        "bridge_outcomes_read": False,
        "new_forwards": 0,
        "new_fits": 0,
        "axes_flipped": 0,
        "inference": "descriptive discovery; future tuned use needs separate authors",
        "snapshot_sha256": digest(args.output_dir / "membership_snapshot.json"),
        "outputs_sha256": {p.name: digest(p) for p in args.output_dir.glob("*.csv")},
    }
    write(args.output_dir / "summary.json", summary)
    print(json.dumps(summary))


def main():
    """Run the separate membership-freeze and outcome-analysis steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("snapshot", "analyze"), required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--material-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(ROOT / "results"):
        raise ValueError("Private discovery outputs belong in ignored results/")
    (freeze if args.phase == "snapshot" else analyze)(args)


if __name__ == "__main__":
    main()
