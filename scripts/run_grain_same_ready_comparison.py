#!/usr/bin/env python3
"""One bounded source-only same-ready comparator/increment, not a new model search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import (
    TRAITS,
    load_source_labels,
    metrics,
    predict_head,
    task_score,
)
from grain_research.vector_mechanisms.external_readout_cache import (
    cached_readout_matrix,
)
from grain_research.vector_mechanisms.external_readouts import KINDS as READOUT_KINDS
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import (
    FEATURES,
    digest,
    feature_matrix,
    load_registry_inputs,
    nested_fit,
    require_observer_completion,
    support_table,
)
from grain_research.vector_mechanisms.ready_registry import TASKS
from scripts.prepare_grain_same_ready_registry import COHORTS
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_links import (
    bootstrap_score,
    dense_features,
)
from scripts.run_grain_vector_mechanism_external_queue import verify_completed
from scripts.run_grain_vector_mechanism_sparse_baselines import bounded_child


def execute(a):
    """Prepare immutable features first, then fit only permitted source labels."""
    out = a.output_dir / "fit"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    readout = getattr(a, "readout", "body_token")
    if readout != "body_token" and not a.feature.removeprefix("grain_").startswith(
        "qwen_"
    ):
        raise ValueError("Nondefault readout applies only to a Qwen feature block")
    source, common, reg, profiles, registry_manifest = load_registry_inputs(
        a.views, a.registry_dir, a.registry_audit_dir, a.profile_dir, a.preflight_dir
    )

    observer_evidence = {}

    def observer(bundles, feature, arm):
        if a.vector_dir is None:
            raise ValueError("Neural comparison waits for the full24-shard queue")
        queue = require_observer_completion(a.vector_dir, common["material_sha256"])
        # Status alone is not acceptance: replay every cache commitment before fitting.
        for stage in ("bge_large", "e5_large", "qwen"):
            for shard in range(24):
                folder = a.vector_dir / stage / f"shard_{shard:02d}"
                verify_completed(
                    folder, source, common, stage, 24, shard, queue["code_sha256"]
                )
                observer_evidence[f"{stage}/shard_{shard:02d}"] = {
                    "manifest_sha256": digest(folder / "manifest.json"),
                    "status_sha256": digest(folder / "run_status.json"),
                }
        observer_evidence["queue_manifest_sha256"] = digest(
            a.vector_dir / "manifest.json"
        )
        if readout != "body_token":
            return cached_readout_matrix(
                bundles,
                source,
                common,
                a.vector_dir,
                arm,
                feature.removeprefix("qwen_"),
                readout,
            )
        opts = SimpleNamespace(
            feature=feature,
            arm=arm,
            vector_dir=a.vector_dir,
            preflight_dir=a.preflight_dir,
        )
        return dense_features(bundles, source, common, opts)

    x, bundles, ix, blocks = feature_matrix(
        source, reg, profiles, a.cohort, a.task, a.arm, a.feature, observer
    )
    ids, folds = reg["author_ids"][ix], reg[a.task + "_fold"][ix]
    kind = "word_char" if a.feature == "word_char" else "dense"
    parent = json.loads((a.parent_dir / "manifest.json").read_text())
    if (
        material_hash({k: v for k, v in parent.items() if k != "identity"})
        != parent["identity"]
    ):
        raise ValueError("Source-table parent identity mismatch")
    fields = {
        "protocol": "VM05_SAME_READY_COMPARISON_V1",
        "task": a.task,
        "feature": a.feature,
        "readout": readout if "qwen_" in a.feature else "not_qwen",
        "arm": a.arm,
        "cohort": a.cohort,
        "authors": len(ids),
        "source_group": bundles[0]["source_group"],
        "source_material_sha256": material_hash(bundles),
        "registry_identity": registry_manifest["identity"],
        "registry_audit_sha256": digest(a.registry_audit_dir / "summary.json"),
        "source_table_parent_identity": parent["identity"],
        "common_material_sha256": common["material_sha256"],
        "feature_input_sha256": material_hash(x.tolist())
        if kind == "word_char"
        else hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        "feature_blocks": blocks,
        "observer_evidence": observer_evidence,
        "scaler_fit": "inner_source_only",
        "pca": not a.no_pca,
        "pca_applied": kind == "dense" and not a.no_pca,
        "thread_control": "retain",
        "seed": 731,
        "permutation_seed": None,
        "bridge_labels_used": False,
        "preflight_only": a.preflight_only,
        "probability_type": "native_logistic_no_extra_calibrator"
        if a.task != "big5"
        else "not_probability",
        "separate_calibration_fitted": False,
        "calibration_validated": False,
        "scoring_code": {
            name: digest(ROOT / name)
            for name in (
                "scripts/run_grain_same_ready_comparison.py",
                "grain_research/vector_mechanisms/ready_comparison.py",
                "grain_research/vector_mechanisms/ready_registry.py",
                "grain_research/vector_mechanisms/external_heads.py",
                "scripts/run_grain_vector_mechanism_external_links.py",
                "scripts/run_grain_vector_mechanism_external_features.py",
                "scripts/run_grain_vector_mechanism_external_queue.py",
                "grain_research/vector_mechanisms/external_readouts.py",
                "grain_research/vector_mechanisms/external_readout_cache.py",
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
                "model_refits": 0,
                "neural_forwards": 0,
                "feature_blocks": blocks,
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
        raise ValueError(
            "Frozen cohort lacks target support; keep failure, do not alter folds"
        )

    def checkpoint(fold, model, selection):
        joblib.dump(model, out / f"private_fold_{fold}.joblib", compress=3)
        write_json(out / "selection.json", selection)
        write_json(
            out / "run_status.json",
            {
                "status": "RUNNING",
                "completed_folds": fold + 1,
                "seconds": time.monotonic() - start,
            },
        )

    with threadpool_limits(limits=4):
        pred, _, rows, full, search, parameter = nested_fit(
            x, y, ids, folds, a.task, kind, not a.no_pca, checkpoint
        )
        joblib.dump(full, out / "private_full_source_head.joblib", compress=3)
        replay = []
        for fold in range(5):
            test = np.flatnonzero(folds == fold)
            model = joblib.load(out / f"private_fold_{fold}.joblib")
            err = float(np.max(abs(predict_head(model, x[test], a.task) - pred[test])))
            if not np.isfinite(err) or err > 1e-10:
                raise ValueError("Saved ready head does not replay")
            replay.append(
                {"fold": fold, "max_absolute_error": err, "authors": len(test)}
            )
    names = TRAITS if a.task == "big5" else [a.task]
    yy, pp = (y, pred) if a.task == "big5" else (y[:, None], pred[:, None])
    pd.DataFrame(
        [
            {
                "author_id": uid,
                "fold": int(folds[i]),
                "trait": t,
                "y_true": yy[i, j],
                "prediction": pp[i, j],
            }
            for i, uid in enumerate(ids)
            for j, t in enumerate(names)
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
        "feature": a.feature,
        "arm": a.arm,
        "score": task_score(y, pred, a.task),
        "readout": fields["readout"],
        "bootstrap": bootstrap_score(y, pred, a.task),
        "full_parameter": parameter,
        "full_search": search,
        "full_head_sha256": digest(out / "private_full_source_head.joblib"),
        "bridge_labels_used": False,
        "saved_fold_replay_max": max(r["max_absolute_error"] for r in replay),
        "seconds": time.monotonic() - start,
        "scope": "Ready-only retain-thread source refit, not whole-source OOF restriction or causal mechanism",
        "open_controls": [
            "paired same-ready comparisons",
            "discussion controls",
            "permutation",
            "independent bridge",
        ],
    }
    summary["outputs_sha256"] = {
        p.name: digest(p)
        for p in out.iterdir()
        if p.is_file()
        and p.name not in ("manifest.json", "run_status.json", "source_snapshot.zip")
        and not p.name.startswith("._")
    }
    write_json(out / "summary.json", summary)
    write_json(out / "run_status.json", summary)


def main():
    """Supervise one owned CPU child; never duplicate or resume partial fits."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "views",
        "registry-dir",
        "registry-audit-dir",
        "profile-dir",
        "preflight-dir",
        "parent-dir",
        "prepared-root",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--vector-dir", type=Path)
    p.add_argument("--cohort", choices=COHORTS, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--feature", choices=FEATURES, required=True)
    p.add_argument("--arm", choices=("raw", "cue_mask", "placebo_mask"), default="raw")
    p.add_argument("--no-pca", action="store_true")
    p.add_argument(
        "--readout", choices=(*READOUT_KINDS, "logit_margins"), default="body_token"
    )
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
    a.output_dir.mkdir(exist_ok=False, parents=True)

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
        status = json.loads((a.output_dir / "fit/run_status.json").read_text())
        if status["status"] not in ("COMPLETE", "PREFLIGHT_COMPLETE"):
            raise ValueError("Child did not complete declared operation")
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
            {"status": "FAILED_STOP", "exception_type": type(error).__name__},
        )
        raise


if __name__ == "__main__":
    main()
