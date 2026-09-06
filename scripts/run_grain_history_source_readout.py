#!/usr/bin/env python3
"""One source-exclusive history/baseline readout after all128 history caches complete."""

import argparse
import hashlib
import json
import sys
import time
import zipfile
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
from grain_research.vector_mechanisms.history_collection import (
    KINDS,
    SITES,
    qualification_packet,
)
from grain_research.vector_mechanisms.history_instrument import registered_cases
from grain_research.vector_mechanisms.history_readout_cache import (
    COHORTS,
    complete_history_index,
    history_matrix,
    source_indices,
)
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import (
    digest,
    load_registry_inputs,
    nested_fit,
    require_observer_completion,
    support_table,
)
from grain_research.vector_mechanisms.ready_registry import TASKS
from scripts.collect_grain_source_history import CODE as COLLECTOR_CODE
from scripts.collect_grain_source_history import require_technical
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_features import bundle_arms
from scripts.run_grain_vector_mechanism_external_links import (
    bootstrap_score,
    dense_features,
)
from scripts.run_grain_vector_mechanism_external_queue import verify_completed
from scripts.run_grain_vector_mechanism_sparse_baselines import bounded_child

FEATURES = (
    "word_char",
    "bge_large",
    "e5_large",
    "qwen_body",
    "sorted_distances",
    "history",
    "grain_history",
)
CODE = (
    "scripts/run_grain_history_source_readout.py",
    "grain_research/vector_mechanisms/history_readout_cache.py",
    "grain_research/vector_mechanisms/history_cache.py",
    "grain_research/vector_mechanisms/history_collection.py",
    "grain_research/vector_mechanisms/history_instrument.py",
    "grain_research/vector_mechanisms/ready_comparison.py",
    "grain_research/vector_mechanisms/ready_registry.py",
    "grain_research/vector_mechanisms/external_heads.py",
    "grain_research/vector_mechanisms/external_controls.py",
    "grain_research/vector_mechanisms/external_material.py",
    "scripts/run_grain_vector_mechanism_external_links.py",
    "scripts/run_grain_vector_mechanism_external_features.py",
    "scripts/run_grain_vector_mechanism_external_queue.py",
)


def source_inputs(a):
    """Reconcile strict/ready cohorts and validate complete caches before target conversion."""
    if a.feature not in FEATURES or a.site not in SITES[1:] or a.kind not in KINDS:
        raise ValueError(
            "Require a declared history feature and non-input observation site"
        )
    if (
        a.feature in ("sorted_distances", "grain_history")
        and a.cohort != "history_ready"
    ):
        raise ValueError(
            "Formal GRAIN features require the identical all-arm-ready history cohort"
        )
    by_author, whole, material, audit = registered_cases(
        a.views, a.material_dir, a.audit_dir
    )
    # Do not retain two full copies of the36-comment source during registry loading.
    del by_author
    source, common, reg, profiles, rm = load_registry_inputs(
        a.views, a.registry_dir, a.registry_audit_dir, a.profile_dir, a.preflight_dir
    )
    ix = source_indices(source, whole, reg, a.cohort, a.task)
    bundles = [source[i] for i in ix]
    folds = reg[a.task + "_fold"][ix]
    qm, _, model = qualification_packet(a.qualification_dir, material, audit, ROOT)
    collector_code = {n: digest(ROOT / n) for n in COLLECTOR_CODE}
    technical = require_technical(a.technical_parent, qm, material, collector_code)
    index, evidence = complete_history_index(
        a.history_dir, whole, material, audit, collector_code, qm, technical, model
    )
    parent = json.loads((a.parent_dir / "manifest.json").read_text(encoding="utf-8"))
    if (
        material_hash({k: v for k, v in parent.items() if k != "identity"})
        != parent["identity"]
    ):
        raise ValueError("Source-table parent changed")
    blocks = []
    if a.feature == "word_char":
        x = np.asarray(["\n\n".join(bundle_arms(b)[a.arm]) for b in bundles])
    elif a.feature in ("history", "grain_history"):
        h = history_matrix(bundles, whole, index, a.arm, a.site, a.kind)
        if a.feature == "grain_history":
            g = profiles[a.arm + "_sorted_distances"][ix]
            if g.shape != (len(ix), 16) or not np.isfinite(g).all():
                raise ValueError(
                    "History-ready profile is not supported finite16-dimensional g"
                )
            x = np.concatenate([g, h], axis=1)
            blocks.append({"name": "sorted_distances", "start": 0, "stop": 16})
        else:
            x = h
        blocks.append(
            {
                "name": "history_" + a.kind,
                "start": x.shape[1] - 3072,
                "stop": x.shape[1],
            }
        )
    elif a.feature == "sorted_distances":
        x = profiles[a.arm + "_sorted_distances"][ix]
        if x.shape != (len(ix), 16) or not np.isfinite(x).all():
            raise ValueError("Missing accepted history-ready GRAIN profile")
        blocks.append({"name": "sorted_distances", "start": 0, "stop": 16})
    else:
        queue = require_observer_completion(a.vector_dir, common["material_sha256"])
        for stage in ("bge_large", "e5_large", "qwen"):
            for i in range(24):
                folder = a.vector_dir / stage / f"shard_{i:02d}"
                verify_completed(
                    folder, source, common, stage, 24, i, queue["code_sha256"]
                )
                evidence[f"original/{stage}/{i}"] = {
                    "manifest_sha256": digest(folder / "manifest.json"),
                    "status_sha256": digest(folder / "run_status.json"),
                }
        opts = SimpleNamespace(
            feature="qwen_" + a.site if a.feature == "qwen_body" else a.feature,
            arm=a.arm,
            vector_dir=a.vector_dir,
            preflight_dir=a.preflight_dir,
        )
        x = dense_features(bundles, source, common, opts)
        if x.ndim != 2 or len(x) != len(ix) or not np.isfinite(x).all():
            raise ValueError("Invalid matched historical baseline matrix")
        blocks.append({"name": opts.feature, "start": 0, "stop": x.shape[1]})
    evidence.update(
        material_identity=material["identity"],
        audit_identity=audit["identity"],
        registry_identity=rm["identity"],
        qualification_identity=qm["identity"],
        technical_identity=technical["identity"],
        common_material_sha256=common["material_sha256"],
        budget_sha256=digest(a.preflight_dir / "budget_by_author.csv"),
    )
    return bundles, folds, x, blocks, evidence, parent


def execute(a):
    """Use the unchanged source-inner estimator and save/replay all official-fold heads."""
    start = time.monotonic()
    out = a.output_dir / "fit"
    out.mkdir(exist_ok=False)
    bundles, folds, x, blocks, evidence, parent = source_inputs(a)
    ids = np.array([b["author_id"] for b in bundles])
    kind = "word_char" if a.feature == "word_char" else "dense"
    if any(
        b["source_group"] != ("big5_only" if a.task == "big5" else "mbti_only")
        for b in bundles
    ):
        raise ValueError("Only source-exclusive target rows can reach fitting")
    m = {
        "protocol": "VM05_HISTORY_SOURCE_READOUT_V1",
        "task": a.task,
        "cohort": a.cohort,
        "feature": a.feature,
        "site": a.site
        if a.feature in ("history", "grain_history", "qwen_body")
        else None,
        "history_kind": a.kind if a.feature in ("history", "grain_history") else None,
        "arm": a.arm,
        "authors": len(ids),
        "source_group": bundles[0]["source_group"],
        "source_material_sha256": material_hash(bundles),
        "author_commitment": material_hash(ids.tolist()),
        "fold_commitment": material_hash(folds.tolist()),
        "source_table_parent_identity": parent["identity"],
        "feature_input_sha256": material_hash(x.tolist())
        if kind == "word_char"
        else hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        "feature_blocks": blocks,
        "evidence": evidence,
        "pca": not a.no_pca,
        "pca_applied": kind == "dense" and not a.no_pca,
        "seed": 731,
        "inner_cv": "3 author folds, seed731",
        "outer_cv": "5 source-specific official folds",
        "thread_control": "retain",
        "separate_calibration_fitted": False,
        "bridge_labels_used": False,
        "reference_authors_excluded": True,
        "fixed_reference_history_used": a.feature in ("history", "grain_history"),
        "reference_budget_comparison": "shared permitted dictionary; sparse/pooled baselines do not consume substituted donor texts",
        "neural_forwards": 0,
        "preflight_only": a.preflight_only,
        "scoring_code": {n: digest(ROOT / n) for n in CODE},
    }
    m["identity"] = material_hash(m)
    write_json(out / "manifest.json", m)
    archive_source(ROOT, out / "source_snapshot.zip")
    with zipfile.ZipFile(out / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED) as z:
        for n in CODE:
            if n not in z.namelist():
                z.write(ROOT / n, n)
    if a.preflight_only:
        write_json(
            out / "run_status.json",
            {
                "status": "PREFLIGHT_COMPLETE",
                "parent_identity": m["identity"],
                "authors": len(ids),
                "source_label_values_read": False,
                "model_refits": 0,
                "bridge_labels_read": False,
                "neural_forwards": 0,
            },
        )
        return
    y, label_audit = load_source_labels(
        a.prepared_root, bundles, a.task, parent["source_tables"]
    )
    write_json(out / "source_label_audit.json", label_audit)
    support = support_table(y, ids, folds, a.task)
    pd.DataFrame(support).to_csv(out / "private_target_support.csv", index=False)
    if not all(r["supported"] for r in support):
        raise ValueError("Unsupported source folds; do not resample or retune support")

    def checkpoint(fold, model, selection):
        joblib.dump(model, out / f"private_fold_{fold}.joblib", compress=3)
        write_json(out / "selection.json", selection)
        write_json(
            out / "run_status.json",
            {
                "status": "RUNNING",
                "completed_folds": fold + 1,
                "bridge_labels_used": False,
            },
        )

    with threadpool_limits(limits=4):
        pred, _, fold_rows, full, search, parameter = nested_fit(
            x, y, ids, folds, a.task, kind, not a.no_pca, checkpoint
        )
        joblib.dump(full, out / "private_full_source_head.joblib", compress=3)
        replay = []
        for fold in range(5):
            selected = folds == fold
            actual = predict_head(
                joblib.load(out / f"private_fold_{fold}.joblib"), x[selected], a.task
            )
            err = float(np.max(abs(actual - pred[selected])))
            if not np.isfinite(err) or err > 1e-10:
                raise ValueError("Saved history source head fails replay")
            replay.append(
                {
                    "fold": fold,
                    "authors": int(selected.sum()),
                    "max_absolute_error": err,
                }
            )
    yy, pp = (y, pred) if a.task == "big5" else (y[:, None], pred[:, None])
    names = TRAITS if a.task == "big5" else [a.task]
    pd.DataFrame(
        [
            {
                "author_id": u,
                "fold": int(folds[i]),
                "trait": t,
                "y_true": yy[i, j],
                "prediction": pp[i, j],
            }
            for i, u in enumerate(ids)
            for j, t in enumerate(names)
        ]
    ).to_csv(out / "private_oof_predictions.csv", index=False)
    for name, rows in (
        ("metrics_by_trait", metrics(y, pred, a.task)),
        ("metrics_by_fold", fold_rows),
        ("saved_head_replay", replay),
    ):
        pd.DataFrame(rows).to_csv(out / (name + ".csv"), index=False)
    s = {
        "status": "COMPLETE",
        "parent_identity": m["identity"],
        "task": a.task,
        "feature": a.feature,
        "cohort": a.cohort,
        "arm": a.arm,
        "authors": len(ids),
        "full_train_authors": len(ids),
        "score": task_score(y, pred, a.task),
        "score_name": "mean_pearson" if a.task == "big5" else "auc",
        "bootstrap": bootstrap_score(y, pred, a.task),
        "full_parameter": parameter,
        "full_search": search,
        "full_head_sha256": digest(out / "private_full_source_head.joblib"),
        "saved_fold_replay_max": max(r["max_absolute_error"] for r in replay),
        "bridge_labels_used": False,
        "neural_forwards": 0,
        "separate_calibration_fitted": False,
        "scope": "source-trained history readability; no native trait or human causal claim",
        "seconds": time.monotonic() - start,
    }
    s["outputs_sha256"] = {
        f.name: digest(f)
        for f in out.iterdir()
        if f.is_file()
        and f.name not in ("manifest.json", "source_snapshot.zip", "run_status.json")
        and not f.name.startswith("._")
    }
    write_json(out / "summary.json", s)
    write_json(out / "run_status.json", s)


def main():
    """Run one owned time/RSS-bounded child; never launch a grid or new model weights."""
    p = argparse.ArgumentParser(description=__doc__)
    for n in (
        "views",
        "preflight-dir",
        "parent-dir",
        "prepared-root",
        "registry-dir",
        "registry-audit-dir",
        "profile-dir",
        "material-dir",
        "audit-dir",
        "qualification-dir",
        "technical-parent",
        "history-dir",
        "vector-dir",
        "output-dir",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--cohort", choices=COHORTS, required=True)
    p.add_argument("--feature", choices=FEATURES, required=True)
    p.add_argument("--site", choices=SITES[1:], default="block_10")
    p.add_argument("--kind", choices=KINDS, default="current_context")
    p.add_argument("--arm", choices=("raw", "cue_mask", "placebo_mask"), default="raw")
    p.add_argument("--no-pca", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--max-seconds", type=int, default=3600)
    a = p.parse_args()
    a.output_dir = a.output_dir.resolve()
    if (
        not a.output_dir.is_relative_to(ROOT / "results")
        or not 60 <= a.max_seconds <= 7200
    ):
        raise ValueError("Private results and bounded runtime required")
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
        with (a.output_dir / "execution.log").open("x", encoding="utf-8") as log:
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
        s = json.loads(
            (a.output_dir / "fit/run_status.json").read_text(encoding="utf-8")
        )
        if s["status"] not in ("COMPLETE", "PREFLIGHT_COMPLETE"):
            raise ValueError("History child incomplete")
        result = {
            "status": s["status"],
            "authors": s["authors"],
            "bridge_labels_read": False,
            **resource,
        }
        write_json(a.output_dir / "supervisor_status.json", result)
        print(json.dumps(result), flush=True)
    except Exception as e:
        write_json(
            a.output_dir / "supervisor_status.json",
            {"status": "FAILED_STOP", "exception_type": type(e).__name__},
        )
        raise


if __name__ == "__main__":
    main()
