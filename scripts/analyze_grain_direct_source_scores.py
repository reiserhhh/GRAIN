#!/usr/bin/env python3
"""Complete-only native direct versus audited source-head comparison, without fitting."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.direct_material import (
    load_registration,
    verify_tokenizer,
)
from grain_research.vector_mechanisms.direct_queue import semantic_parent, verify_shard
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.external_heads import TRAITS, metrics
from grain_research.vector_mechanisms.external_material import shard_material
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.collect_grain_direct_traits import CODE, qualified_parent
from scripts.run_grain_direct_traits_queue import QUEUE_CODE, SEMANTIC_CODE, fingerprint
from scripts.run_grain_neural_sensitivity_batch import validate_primary
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_neural_source_queue import validate_result
from scripts.summarize_grain_neural_source_primary import OBSERVERS, score
from scripts.summarize_grain_vector_external_links import load_oof

ARMS = ("raw", "cue_mask", "placebo_mask")
TASKS = ("big5", "EI_cont", "SN_cont", "TF_cont", "JP_cont")


def require_complete(manifest, status):
    """Reject partial/preflight queues before opening any saved source outcomes."""
    completed = status.get("completed", [])
    if (
        manifest.get("protocol") != "VM05_DIRECT_QUEUE_V1"
        or manifest.get("preflight_only") is not False
        or material_hash({k: v for k, v in manifest.items() if k != "identity"})
        != manifest.get("identity")
        or manifest.get("shards") != 128
        or status.get("status") != "COMPLETE"
        or status.get("parent_identity") != manifest["identity"]
        or status.get("child_pid")
        or status.get("real_neural_collection") is not True
        or any(
            p.get(k) is not False
            for p in (manifest, status)
            for k in ("labels_read", "bridge_outcomes_read")
        )
        or len(completed) != 128
        or {r.get("shard") for r in completed} != set(range(128))
        or sum(r.get("authors", 0) for r in completed) != manifest.get("authors")
        or status.get("verified_authors") != manifest.get("authors")
    ):
        raise ValueError(
            "Complete128-shard real direct queue required before source OOF access"
        )


def direct_prediction(records, ids, task, arm):
    """Use every registered view; never drop an inconvenient native answer."""
    expected_group = "big5_only" if task == "big5" else "mbti_only"
    names = [t.lower() for t in TRAITS] if task == "big5" else [task]
    if arm not in ARMS or len(set(ids)) != len(ids):
        raise ValueError("Invalid arm or repeated source author")
    predictions, diagnostic = [], []
    for uid in ids:
        data = records[uid]
        if data["author_id"] != uid or data["source_group"] != expected_group:
            raise ValueError(
                "Native evaluation cannot use another source role or bridge"
            )
        values = []
        for trait in names:
            rows = [
                r for r in data["readouts"] if r["task"] == trait and r["arm"] == arm
            ]
            if len(rows) != 3 or {r["view"] for r in rows} != {0, 1, 2}:
                raise ValueError("Native score requires all three original views")
            scores = np.asarray([r["conditional_score"] for r in rows])
            if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError("Invalid native conditional score")
            mean = float(scores.mean())
            if mean != data["view_means"][f"{arm}__{trait}"]:
                raise ValueError("Native equal-view aggregation changed")
            values.append(mean)
            diagnostic.extend(
                dict(
                    author_id=uid,
                    trait=trait,
                    **{
                        k: r[k]
                        for k in (
                            "view",
                            "candidate_mass",
                            "conditional_entropy",
                            "greedy_first_token_is_candidate",
                        )
                    },
                )
                for r in rows
            )
        predictions.append(values)
    p = np.asarray(predictions)
    return (p if task == "big5" else p[:, 0]), diagnostic


def paired_scores(y, predictions, folds, task, names, pairs, draws=1000):
    """Shared author-within-fold bootstrap; retain undefined contrasts explicitly."""
    y, predictions, folds = np.asarray(y), np.asarray(predictions), np.asarray(folds)
    if (
        task not in TASKS
        or draws < 20
        or len(set(names)) != len(names)
        or predictions.shape != (len(names), *y.shape)
        or folds.shape != (len(y),)
        or set(folds) != set(range(5))
        or not np.isfinite(y).all()
        or not np.isfinite(predictions).all()
        or (task == "big5" and (y.ndim != 2 or y.shape[1] != 5))
        or (task != "big5" and (y.ndim != 1 or not np.isin(y, [0, 1]).all()))
        or any(
            a == b or not 0 <= a < len(names) or not 0 <= b < len(names)
            for a, b in pairs
        )
    ):
        raise ValueError(
            "Require paired source outcomes, predictions and five official folds"
        )
    parts = [np.flatnonzero(folds == f) for f in range(5)]

    def estimate(indices):
        joined = np.concatenate(indices)
        return np.asarray(
            [
                [score(y[joined], p[joined], task) for p in predictions],
                np.mean(
                    [
                        [score(y[ix], p[ix], task) for p in predictions]
                        for ix in indices
                    ],
                    axis=0,
                ),
            ]
        )

    observed = estimate(parts)
    rng = np.random.default_rng(731)
    samples = np.asarray(
        [
            estimate([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(draws)
        ]
    )
    rows = []
    for a, b in pairs:
        for k, estimand in enumerate(("pooled", "mean_official_fold")):
            delta = observed[k, a] - observed[k, b]
            differences = samples[:, k, a] - samples[:, k, b]
            finite = differences[np.isfinite(differences)]
            defined = bool(np.isfinite(delta) and len(finite) >= 0.95 * draws)
            bounds = np.quantile(finite, [0.025, 0.975]) if defined else [None, None]
            rows.append(
                {
                    "candidate": names[a],
                    "baseline": names[b],
                    "estimand": estimand,
                    "candidate_score": float(observed[k, a])
                    if np.isfinite(observed[k, a])
                    else None,
                    "baseline_score": float(observed[k, b])
                    if np.isfinite(observed[k, b])
                    else None,
                    "delta": float(delta) if np.isfinite(delta) else None,
                    "lower": bounds[0],
                    "upper": bounds[1],
                    "draws": draws,
                    "undefined_draws": draws - len(finite),
                    "status": "DEFINED_POINTWISE"
                    if defined
                    else "UNDEFINED_OR_LOW_SUPPORT",
                    "familywise_confirmed": False,
                }
            )
    return rows


def analyze(a):
    """Verify all native caches and audited OOF parents, then save all fixed contrasts."""
    queue = json.loads((a.queue_dir / "manifest.json").read_text(encoding="utf-8"))
    status_path = a.queue_dir / "queue_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    require_complete(queue, status)
    if (
        (a.queue_dir / "queue.lock").exists()
        or a.output_dir.exists()
        or not a.output_dir.resolve().is_relative_to(ROOT / "results")
    ):
        raise ValueError(
            "Require idle completed queue and fresh private analysis output"
        )
    by_author, whole, material = load_registration(a.views, a.material_dir)
    code = fingerprint(CODE)
    qm, qs = qualified_parent(a.qualification_dir, material, code, whole)
    verify_tokenizer(a.tokenizer_dir, material)
    tokenizer = AutoTokenizer.from_pretrained(
        str(a.tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    semantic = semantic_parent(
        a.semantic_dir,
        material,
        qm["identity"],
        qs["model_signature"],
        fingerprint(SEMANTIC_CODE),
        tokenizer,
    )
    if (
        queue["material_identity"] != material["identity"]
        or queue["author_commitment"] != material_hash([c["author_id"] for c in whole])
        or queue["authors"] != len(whole)
        or queue["code_sha256"] != fingerprint(QUEUE_CODE)
        or queue["collector_code_sha256"] != code
        or queue["semantic_parent"] != semantic
        or queue["qualification_identity"] != qm["identity"]
        or queue["model_signature"] != qs["model_signature"]
    ):
        raise ValueError(
            "Direct queue material, code or qualified native observer changed"
        )
    common = json.loads(
        (a.preflight_dir / "common_material_manifest.json").read_text(encoding="utf-8")
    )
    if material["common_material_sha256"] != common["material_sha256"]:
        raise ValueError(
            "Direct and external source heads do not share full text material"
        )
    source = exclude_reference(
        list(by_author.values()),
        pd.read_csv(a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}),
    )
    wanted = {b["author_id"] for b in source if b["source_group"] != "bridge"}
    committed = {r["shard"]: r for r in status["completed"]}
    records, inputs, verified = {}, {}, []
    for i in range(128):
        folder = a.queue_dir / f"shard_{i:03d}"
        result = verify_shard(
            folder,
            whole,
            material,
            code,
            128,
            i,
            preflight=False,
            qualification=qm["identity"],
            model_signature=qs["model_signature"],
        )
        if committed[i] != {"shard": i, **result}:
            raise ValueError("Native shard differs from terminal queue acceptance")
        verified.append(result)
        inputs[f"shard_{i:03d}"] = {
            n: digest(folder / n) for n in ("manifest.json", "summary.json")
        }
        for c in shard_material(whole, 128, i):
            uid = c["author_id"]
            if uid in wanted:
                if uid in records:
                    raise ValueError("Repeated native author across shards")
                records[uid] = json.loads(
                    (
                        folder / "private_cache" / (material_hash(uid) + ".json")
                    ).read_text(encoding="utf-8")
                )
    if (
        set(records) != wanted
        or sum(r["cache_bytes"] for r in verified) != status["cache_bytes"]
    ):
        raise ValueError("Native source author union or total cache size changed")
    # Labels enter only here, via already audited source-only OOF packets, never a bridge table.
    primary, audit = validate_primary(
        a.primary_dir, a.primary_audit_dir, a.resource_dir
    )
    if (
        digest(a.views) != primary["views_sha256"]
        or digest(a.preflight_dir / "manifest.json")
        != primary["preflight_manifest_sha256"]
    ):
        raise ValueError("Audited primary input differs")
    scores, traits, folds_out, paired, quality, prediction_rows = [], [], [], [], [], []
    native = {}
    for task in TASKS:
        for arm in ARMS:
            packets = []
            for observer in OBSERVERS:
                packet = load_oof(a.primary_dir / f"{task}_{arm}_{observer}")
                validate_result(
                    packet,
                    {"task": task, "arm": arm, "feature": observer},
                    source,
                    common["material_sha256"],
                )
                if packets and any(
                    not np.array_equal(packet[k], packets[0][k])
                    for k in ("ids", "y", "folds")
                ):
                    raise ValueError("External source observers are not paired")
                packets.append(packet)
            anchor = packets[0]
            p, diagnostic = direct_prediction(records, anchor["ids"], task, arm)
            y, folds = anchor["y"], anchor["folds"]
            native[task, arm] = {"ids": anchor["ids"], "y": y, "folds": folds, "p": p}
            names = ("native_direct", *OBSERVERS)
            predictions = np.asarray([p, *(r["p"] for r in packets)])
            for observer, pred in zip(names, predictions, strict=True):
                by_fold = [
                    score(y[folds == f], pred[folds == f], task) for f in range(5)
                ]
                scores.append(
                    {
                        "task": task,
                        "arm": arm,
                        "observer": observer,
                        "authors": len(y),
                        "pooled_score": score(y, pred, task),
                        "mean_fold_score": float(np.mean(by_fold)),
                    }
                )
                for fold in range(5):
                    for row in metrics(y[folds == fold], pred[folds == fold], task):
                        folds_out.append(
                            dict(
                                task=task, arm=arm, observer=observer, fold=fold, **row
                            )
                        )
                trait_metrics = metrics(y, pred, task)
                if task != "big5":
                    trait_metrics[0]["log_loss"] = float(
                        log_loss(y, np.clip(pred, 1e-15, 1 - 1e-15), labels=[0, 1])
                    )
                traits.extend(
                    dict(task=task, arm=arm, observer=observer, **row)
                    for row in trait_metrics
                )
            paired.extend(
                dict(task=task, arm=arm, comparison="native_vs_external", **r)
                for r in paired_scores(
                    y, predictions, folds, task, names, ((0, 1), (0, 2), (0, 3))
                )
            )
            df = pd.DataFrame(diagnostic)
            for trait, group in df.groupby("trait"):
                quality.append(
                    {
                        "task": task,
                        "arm": arm,
                        "trait": trait,
                        "authors": group.author_id.nunique(),
                        "logical_rows": len(group),
                        "greedy_first_token_compliance": float(
                            group.greedy_first_token_is_candidate.mean()
                        ),
                        "candidate_mass_min": float(group.candidate_mass.min()),
                        "candidate_mass_median": float(group.candidate_mass.median()),
                        "conditional_entropy_mean": float(
                            group.conditional_entropy.mean()
                        ),
                        "complete_response_parse_rate_measured": False,
                        "rejected_authors": 0,
                    }
                )
            for j, trait in enumerate(TRAITS if task == "big5" else [task]):
                for i, uid in enumerate(anchor["ids"]):
                    prediction_rows.append(
                        {
                            "author_id": uid,
                            "fold": int(folds[i]),
                            "task": task,
                            "arm": arm,
                            "trait": trait,
                            "y_true": float(y[i, j] if task == "big5" else y[i]),
                            "prediction": float(p[i, j] if task == "big5" else p[i]),
                        }
                    )
            print(
                json.dumps({"evaluated_task": task, "arm": arm, "authors": len(y)}),
                flush=True,
            )
        group = [native[task, arm] for arm in ARMS]
        if any(
            not np.array_equal(r[k], group[0][k])
            for r in group[1:]
            for k in ("ids", "y", "folds")
        ):
            raise ValueError("Native mask arms differ in authors, labels or folds")
        paired.extend(
            dict(task=task, arm="paired_masks", comparison="native_mask_ablation", **r)
            for r in paired_scores(
                group[0]["y"],
                [r["p"] for r in group],
                group[0]["folds"],
                task,
                ARMS,
                ((1, 0), (2, 0), (1, 2)),
            )
        )
    if (len(scores), len(traits), len(folds_out), len(paired), len(quality)) != (
        60,
        108,
        540,
        120,
        27,
    ):
        raise ValueError("Incomplete declared direct-source comparison family")
    analysis_code = (
        "scripts/analyze_grain_direct_source_scores.py",
        "tests/test_grain_direct_source_scores.py",
        "scripts/run_grain_neural_sensitivity_batch.py",
        "scripts/summarize_grain_vector_external_links.py",
        "scripts/summarize_grain_neural_source_primary.py",
        "scripts/analyze_grain_ei_source_gap.py",
        "grain_research/vector_mechanisms/external_controls.py",
        "grain_research/vector_mechanisms/external_heads.py",
    )
    manifest = {
        "protocol": "VM05_NATIVE_DIRECT_SOURCE_COMPARISON_V1",
        "queue_identity": queue["identity"],
        "queue_status_sha256": digest(status_path),
        "material_identity": material["identity"],
        "common_material_sha256": common["material_sha256"],
        "primary_identity": primary["identity"],
        "primary_audit_identity": audit["identity"],
        "source_budget_sha256": digest(a.preflight_dir / "budget_by_author.csv"),
        "native_shard_inputs": inputs,
        "audited_source_inputs": audit["inputs"],
        "native_aggregation": "Unchanged arithmetic mean of three complete12-comment view scores",
        "source_cohort": "Full-source reference-free; task-specific official folds",
        "new_model_fits": 0,
        "calibration_fits": 0,
        "bridge_outcomes_read": False,
        "bootstrap": "1000 author-within-fold shared draws, seed731; pointwise conditional on existing predictions, not refit/familywise",
        "implemented_after_primary_results_before_full_native_source_results": True,
        "code_sha256": {n: digest(ROOT / n) for n in analysis_code},
    }
    manifest["identity"] = material_hash(manifest)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(a.output_dir / "manifest.json", manifest)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        for n in analysis_code:
            if n not in z.namelist():
                z.write(ROOT / n, n)
    tables = {
        "observer_scores.csv": scores,
        "metrics_by_trait.csv": traits,
        "metrics_by_fold.csv": folds_out,
        "paired_comparison.csv": paired,
        "native_readout_quality.csv": quality,
        "private_native_predictions.csv": prediction_rows,
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(a.output_dir / name, index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "scored_conditions": len(scores),
        "paired_rows": len(paired),
        "quality_rows": len(quality),
        "undefined_contrasts": sum(r["status"] != "DEFINED_POINTWISE" for r in paired),
        "new_neural_forwards": 0,
        "model_refits": 0,
        "source_label_tables_read": False,
        "existing_source_oof_outcomes_reused": True,
        "bridge_labels_read": False,
        "program_complete": False,
        "scope": "Native task response versus external decoding; differing instructions/pooling and prior training, not isolated reasoning causality or calibrated personality measurement",
        "outputs_sha256": {n: digest(a.output_dir / n) for n in tables},
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    """Run only after all native source shards are complete and copied; no partial switch."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "queue-dir",
        "views",
        "material-dir",
        "qualification-dir",
        "semantic-dir",
        "tokenizer-dir",
        "preflight-dir",
        "primary-dir",
        "primary-audit-dir",
        "resource-dir",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
