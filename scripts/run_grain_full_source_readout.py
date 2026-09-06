#!/usr/bin/env python3
"""One bounded, source-only readout of verified full-corpus Qwen caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import (
    AXES,
    TRAITS,
    load_source_labels,
    metrics,
    predict_head,
    task_score,
)
from grain_research.vector_mechanisms.external_readout_cache import (
    cached_readout_matrix,
)
from grain_research.vector_mechanisms.external_readouts import ARMS, KINDS, SITES
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import (
    digest,
    nested_fit,
    require_observer_completion,
    support_table,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_links import bootstrap_score
from scripts.run_grain_vector_mechanism_external_queue import verify_completed
from scripts.run_grain_vector_mechanism_sparse_baselines import bounded_child


def source_inputs(a):
    """Validate complete original material, then select exactly one source group."""
    if a.readout == "logit_margins" and a.site != "block_10":
        raise ValueError("Margins are site-independent; use one block10 identifier")
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads(
        (a.preflight_dir / "common_material_manifest.json").read_text(encoding="utf-8")
    )
    parent = json.loads((a.parent_dir / "manifest.json").read_text(encoding="utf-8"))
    ids = [b["author_id"] for b in source]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(common["author_ids"])
        or material_hash(source) != common["material_sha256"]
        or material_hash({k: v for k, v in parent.items() if k != "identity"})
        != parent["identity"]
    ):
        raise ValueError("Changed common material, author union or source parent")
    group = "big5_only" if a.task == "big5" else "mbti_only"
    bundles = [b for b in source if b["source_group"] == group]
    if a.exclude_reference:
        budget = pd.read_csv(
            a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}
        )
        bundles = exclude_reference(bundles, budget)
    if not bundles:
        raise ValueError("No source-exclusive authors")
    folds = np.asarray([b["official_folds"][a.task] for b in bundles])
    if set(folds) != set(range(5)):
        raise ValueError("All five original source-specific folds required")
    queue = require_observer_completion(a.vector_dir, common["material_sha256"])
    evidence = {"queue_manifest_sha256": digest(a.vector_dir / "manifest.json")}
    # An apparently complete status cannot substitute for the cached commitments.
    for stage in ("bge_large", "e5_large", "qwen"):
        for shard in range(24):
            folder = a.vector_dir / stage / f"shard_{shard:02d}"
            verify_completed(
                folder, source, common, stage, 24, shard, queue["code_sha256"]
            )
            evidence[f"{stage}/shard_{shard:02d}"] = {
                "manifest_sha256": digest(folder / "manifest.json"),
                "status_sha256": digest(folder / "run_status.json"),
            }
    x = cached_readout_matrix(
        bundles, source, common, a.vector_dir, a.arm, a.site, a.readout
    )
    if x.ndim != 2 or len(x) != len(bundles) or not np.isfinite(x).all():
        raise ValueError("Malformed cached readout matrix")
    return source, common, parent, bundles, folds, x, evidence


def execute(a):
    """Seal features before target conversion; replay all saved source-only heads."""
    out = a.output_dir / "fit"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    _, common, parent, bundles, folds, x, evidence = source_inputs(a)
    ids = np.asarray([b["author_id"] for b in bundles])
    fields = {
        "protocol": "VM05_FULL_SOURCE_READOUT_V1",
        "task": a.task,
        "feature": "qwen_" + a.site,
        "readout": a.readout,
        "arm": a.arm,
        "source_group": bundles[0]["source_group"],
        "authors": len(ids),
        "feature_columns": x.shape[1],
        "source_material_sha256": material_hash(bundles),
        "source_author_commitment": material_hash(ids.tolist()),
        "fold_commitment": material_hash(folds.tolist()),
        "common_material_sha256": common["material_sha256"],
        "source_table_parent_identity": parent["identity"],
        "budget_sha256": digest(a.preflight_dir / "budget_by_author.csv")
        if a.exclude_reference
        else None,
        "feature_input_sha256": hashlib.sha256(
            np.ascontiguousarray(x).tobytes()
        ).hexdigest(),
        "observer_evidence": evidence,
        "seed": 731,
        "pca": not a.no_pca,
        "pca_applied": not a.no_pca,
        "exclude_reference": a.exclude_reference,
        "thread_control": "retain",
        "drop_seed": 731,
        "permutation_seed": None,
        "bridge_labels_used": False,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "preflight_only": a.preflight_only,
        "probability_type": "native_logistic_no_extra_calibrator"
        if a.task != "big5"
        else "not_probability",
        "calibration_validated": False,
        "scoring_code": {
            name: digest(ROOT / name)
            for name in (
                "scripts/run_grain_full_source_readout.py",
                "grain_research/vector_mechanisms/external_readout_cache.py",
                "grain_research/vector_mechanisms/external_readouts.py",
                "grain_research/vector_mechanisms/external_heads.py",
                "grain_research/vector_mechanisms/external_controls.py",
                "grain_research/vector_mechanisms/ready_comparison.py",
                "scripts/run_grain_vector_mechanism_external_queue.py",
            )
        },
    }
    fields["identity"] = material_hash(fields)
    write_json(out / "manifest.json", fields)
    archive_source(ROOT, out / "source_snapshot.zip")
    if a.preflight_only:
        write_json(
            out / "run_status.json",
            {
                "status": "PREFLIGHT_COMPLETE",
                "parent_identity": fields["identity"],
                "authors": len(ids),
                "source_label_values_read": False,
                "bridge_labels_read": False,
                "neural_forwards": 0,
                "model_refits": 0,
            },
        )
        return
    y, audit = load_source_labels(
        a.prepared_root, bundles, a.task, parent["source_tables"]
    )
    write_json(out / "source_label_audit.json", audit)
    support = support_table(y, ids, folds, a.task)
    pd.DataFrame(support).to_csv(out / "private_target_support.csv", index=False)
    if not all(r["supported"] for r in support):
        raise ValueError("Unsupported frozen source fold; do not resample")

    def checkpoint(fold, model, selection):
        joblib.dump(model, out / f"private_fold_{fold}.joblib", compress=3)
        write_json(out / "selection.json", selection)
        write_json(
            out / "run_status.json",
            {
                "status": "RUNNING",
                "completed_folds": fold + 1,
                "seconds": time.monotonic() - start,
                "bridge_labels_used": False,
            },
        )

    with threadpool_limits(limits=4):
        pred, _, rows, full, search, parameter = nested_fit(
            x, y, ids, folds, a.task, "dense", not a.no_pca, checkpoint
        )
        joblib.dump(full, out / "private_full_source_head.joblib", compress=3)
        replay = []
        for fold in range(5):
            te = np.flatnonzero(folds == fold)
            model = joblib.load(out / f"private_fold_{fold}.joblib")
            err = float(np.max(abs(predict_head(model, x[te], a.task) - pred[te])))
            if not np.isfinite(err) or err > 1e-10:
                raise ValueError("Saved full-source readout does not replay")
            replay.append({"fold": fold, "max_absolute_error": err, "authors": len(te)})
    names = TRAITS if a.task == "big5" else [a.task]
    yy, pp = (y, pred) if a.task == "big5" else (y[:, None], pred[:, None])
    pd.DataFrame(
        [
            {
                "author_id": uid,
                "fold": int(folds[i]),
                "trait": trait,
                "y_true": yy[i, j],
                "prediction": pp[i, j],
            }
            for i, uid in enumerate(ids)
            for j, trait in enumerate(names)
        ]
    ).to_csv(out / "private_oof_predictions.csv", index=False)
    pd.DataFrame(metrics(y, pred, a.task)).to_csv(
        out / "metrics_by_trait.csv", index=False
    )
    pd.DataFrame(rows).to_csv(out / "metrics_by_fold.csv", index=False)
    pd.DataFrame(replay).to_csv(out / "saved_head_replay.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": fields["identity"],
        "authors": len(ids),
        "task": a.task,
        "feature": fields["feature"],
        "arm": a.arm,
        "readout": a.readout,
        "score": task_score(y, pred, a.task),
        "score_name": "mean_pearson" if a.task == "big5" else "auc",
        "bootstrap": bootstrap_score(y, pred, a.task),
        "full_parameter": parameter,
        "full_search": search,
        "full_head_sha256": digest(out / "private_full_source_head.joblib"),
        "saved_fold_replay_max": max(r["max_absolute_error"] for r in replay),
        "bridge_labels_used": False,
        "neural_forwards": 0,
        "exclude_reference": a.exclude_reference,
        "thread_control": "retain",
        "permutation_seed": None,
        "full_train_authors": len(ids),
        "seconds": time.monotonic() - start,
        "scope": "Full source cached readout; neither direct LLM scoring nor context counterfactual",
        "open_controls": [
            "matched readout comparisons",
            "discussion controls",
            "permutation",
            "independent bridge",
        ],
    }
    summary["outputs_sha256"] = {
        f.name: digest(f)
        for f in out.iterdir()
        if f.is_file()
        and f.name not in ("manifest.json", "run_status.json", "source_snapshot.zip")
        and not f.name.startswith("._")
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_status.json", summary)


def main():
    """Run one owned child with an explicit time/RSS limit and fresh output path."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "views",
        "preflight-dir",
        "parent-dir",
        "prepared-root",
        "vector-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--task", choices=("big5", *AXES), required=True)
    p.add_argument("--site", choices=SITES, required=True)
    p.add_argument("--readout", choices=(*KINDS, "logit_margins"), required=True)
    p.add_argument("--arm", choices=ARMS, default="raw")
    p.add_argument("--exclude-reference", action="store_true")
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    a.output_dir = a.output_dir.resolve()
    if (
        not a.output_dir.is_relative_to(ROOT / "results")
        or not 60 <= a.max_seconds <= 7200
    ):
        raise ValueError("Require private results path and bounded runtime")
    if a.worker:
        execute(a)
        return
    a.output_dir.mkdir(parents=True, exist_ok=False)

    def update(pid, elapsed, rss, peak):
        write_json(
            a.output_dir / "supervisor_status.json",
            {
                "status": "RUNNING",
                "child_pid": pid,
                "seconds": elapsed,
                "rss_bytes": rss,
                "peak_rss_bytes": peak,
                "bridge_labels_read": False,
            },
        )

    try:
        with (a.output_dir / "execution.log").open("w") as log:
            resource = bounded_child(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker",
                ],
                log,
                a.max_seconds,
                8 * 1024**3,
                update,
            )
        status = json.loads(
            (a.output_dir / "fit/run_status.json").read_text(encoding="utf-8")
        )
        if status["status"] not in ("COMPLETE", "PREFLIGHT_COMPLETE"):
            raise ValueError("Child did not finish the declared operation")
        result = {
            "status": status["status"],
            **resource,
            "authors": status["authors"],
            "bridge_labels_read": False,
        }
        write_json(a.output_dir / "supervisor_status.json", result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(
            a.output_dir / "supervisor_status.json",
            {
                "status": "FAILED_STOP",
                "exception_type": type(error).__name__,
            },
        )
        raise


if __name__ == "__main__":
    main()
