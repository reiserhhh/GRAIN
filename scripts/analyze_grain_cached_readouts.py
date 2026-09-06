#!/usr/bin/env python3
"""Compare the complete315 cached readouts to matched body heads, without refitting."""

import argparse
import fcntl
import json
import sys
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import metrics
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from scripts.run_grain_cached_readout_batch import (
    ARMS,
    LEASE_FLAGS,
    READOUTS,
    SITES,
    TASKS,
    cases,
    process_absent,
    source_members,
    validate_result,
)
from scripts.run_grain_neural_sensitivity_batch import (
    validate_primary,
    validate_replay,
)
from scripts.run_grain_neural_sensitivity_batch import (
    validate_result as validate_body,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_neural_source_primary import score
from scripts.summarize_grain_vector_external_links import load_oof


def require_complete(manifest, status):
    """Fail before OOF access unless every frozen case has a terminal acceptance."""
    expected = {c["id"] for c in cases()}
    completed = status.get("completed", [])
    if (
        manifest.get("protocol") != "VM05_CACHED_SECONDARY_READOUT_BATCH_V1"
        or material_hash({k: v for k, v in manifest.items() if k != "identity"})
        != manifest.get("identity")
        or manifest.get("cases") != cases()
        or manifest.get("cohort") != "full_source_reference_free"
        or manifest.get("pca") is not True
        or manifest.get("seed") != 731
        or manifest.get("thread_control") != "retain"
        or manifest.get("readout_selection") != "none"
        or manifest.get("bridge_labels_read") is not False
        or manifest.get("neural_forwards") != 0
        or status.get("status") != "COMPLETE"
        or status.get("planned") != 315
        or status.get("child_pid") is not None
        or status.get("bridge_labels_read") is not False
        or status.get("program_complete") is not False
        or len(completed) != 315
        or {r.get("case") for r in completed} != expected
    ):
        raise ValueError("Complete315-case source family required before OOF access")


def readouts(site):
    """Margin diagnostics are site-independent and appear once, at the block10 ID."""
    if site not in SITES:
        raise ValueError("Unknown frozen site")
    return (
        "body_token",
        *READOUTS,
        *(("logit_margins",) if site == "block_10" else ()),
    )


def require_pairing(packets):
    """Reject unequal authors, target values or official folds rather than intersecting."""
    if len(packets) < 2 or any(
        not np.array_equal(packets[0][key], p[key])
        for p in packets[1:]
        for key in ("ids", "y", "folds")
    ):
        raise ValueError(
            "Readout contrasts require identical authors, targets and folds"
        )


def paired_readouts(y, predictions, folds, task, site, draws=1000):
    """Share within-fold author resamples; retain all paired conditional OOF intervals."""
    names = readouts(site)
    y, predictions, folds = map(np.asarray, (y, predictions, folds))
    if (
        task not in TASKS
        or predictions.shape != (len(names), *y.shape)
        or folds.shape != (len(y),)
        or set(folds) != set(range(5))
        or not np.isfinite(y).all()
        or not np.isfinite(predictions).all()
        or (task == "big5" and (y.ndim != 2 or y.shape[1] != 5))
        or (task != "big5" and (y.ndim != 1 or not np.isin(y, [0, 1]).all()))
        or not isinstance(draws, int)
        or isinstance(draws, bool)
        or draws < 20
    ):
        raise ValueError(
            "Require finite fixed-family predictions on five official folds"
        )
    parts = [np.flatnonzero(folds == f) for f in range(5)]

    def estimates(indices):
        joined = np.concatenate(indices)
        by_fold = np.array(
            [[score(y[ix], p[ix], task) for p in predictions] for ix in indices]
        )
        return np.array(
            [
                [score(y[joined], p[joined], task) for p in predictions],
                by_fold.mean(0),
            ]
        )

    observed = estimates(parts)
    if not np.isfinite(observed).all():
        raise ValueError("Undefined source metric: not enough per-fold target support")
    rng = np.random.default_rng(731)
    sampled = np.array(
        [
            estimates([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(draws)
        ]
    )
    finite = sampled[np.isfinite(sampled).all(axis=(1, 2))]
    if len(finite) < 0.95 * draws:
        raise ValueError("Insufficient bootstrap class or trait support")
    fold_scores = np.array(
        [[score(y[ix], p[ix], task) for p in predictions] for ix in parts]
    )
    pairs = [(i, 0, "matched_body_reference") for i in range(1, len(names))]
    pairs += [
        (
            names.index("decision_disposition"),
            names.index("decision_behavior"),
            "exploratory_task_contrast",
        ),
        (
            names.index("decision_task_delta"),
            names.index("decision_behavior"),
            "exploratory_task_contrast",
        ),
    ]
    rows = []
    for candidate, baseline, kind in pairs:
        for k, estimand in enumerate(("pooled", "mean_official_fold")):
            low, high = np.quantile(
                finite[:, k, candidate] - finite[:, k, baseline], [0.025, 0.975]
            )
            rows.append(
                {
                    "candidate": names[candidate],
                    "baseline": names[baseline],
                    "contrast_type": kind,
                    "estimand": estimand,
                    "candidate_score": observed[k, candidate],
                    "baseline_score": observed[k, baseline],
                    "delta": observed[k, candidate] - observed[k, baseline],
                    "lower": low,
                    "upper": high,
                    "positive_fold_count": int(
                        np.sum(fold_scores[:, candidate] > fold_scores[:, baseline])
                    ),
                    "draws": draws,
                    "undefined_draws": draws - len(finite),
                    "familywise_confirmed": False,
                }
            )
    return rows


def baseline_packet(a, task, arm, site, members, common, audits):
    """Verify the exact existing same-site body-token head, not a selected new fit."""
    case = {"task": task, "arm": arm, "feature": "qwen_" + site, "pca": True}
    base = f"{task}_{arm}_qwen_{site}"
    case["id"] = base + "_pca16"
    folder = (
        a.primary_dir / base if site == "block_10" else a.sensitivity_dir / case["id"]
    )
    root = a.primary_dir if site == "block_10" else a.sensitivity_dir
    key = base if site == "block_10" else case["id"]
    audit = audits[0] if site == "block_10" else audits[1]
    expected_names = {
        f"{key}/{name}"
        for name in (
            "manifest.json",
            "summary.json",
            "private_oof_predictions.csv",
            "private_full_source_head.joblib",
            *(f"private_fold_{f}.joblib" for f in range(5)),
        )
    } | {f"{key}_replay/{name}" for name in ("summary.json", "replay_by_fold.csv")}
    hashes = audit["inputs"][key]
    if set(hashes) != expected_names or any(
        digest(root / n) != sha for n, sha in hashes.items()
    ):
        raise ValueError("Previously audited body head/replay artifact changed")
    result = load_oof(folder)
    m, s = result["manifest"], result["summary"]
    validate_body(result, case, members[task], common["material_sha256"])
    if (
        m.get("readout", "body_token") != "body_token"
        or digest(folder / "private_full_source_head.joblib") != s["full_head_sha256"]
    ):
        raise ValueError("Changed primary body readout/head")
    if site != "block_10":
        validate_replay(a.sensitivity_dir / (case["id"] + "_replay"), case)
    else:
        validate_replay(a.primary_dir / (base + "_replay"), {**case, "id": base})
    return result, folder


def analyze(a):
    """Verify the full family before reading OOF, then compute a non-selective report."""
    m = json.loads((a.batch_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((a.batch_dir / "batch_status.json").read_text(encoding="utf-8"))
    require_complete(m, status)
    if not process_absent(status["pid"], "run_grain_cached_readout_batch.py"):
        raise ValueError("Wait for the completed batch owner to exit")
    if a.output_dir.exists() or not a.output_dir.resolve().is_relative_to(
        ROOT / "results"
    ):
        raise ValueError("Fresh private result directory required")
    paths = {
        "views": a.views,
        "common": a.preflight_dir / "common_material_manifest.json",
        "budget": a.preflight_dir / "budget_by_author.csv",
        "source_parent": a.parent_dir / "manifest.json",
        "vector_parent": a.vector_dir / "manifest.json",
        "ready_paired": a.paired_dir / "summary.json",
        **{n: getattr(a, n) / "batch_manifest.json" for n in LEASE_FLAGS},
    }
    if set(paths) != set(m["input_sha256"]):
        raise ValueError("Changed full-family input scope")

    def unchanged():
        if any(digest(p) != m["input_sha256"][n] for n, p in paths.items()) or any(
            digest(ROOT / n) != sha for n, sha in m["scoring_code"].items()
        ):
            raise ValueError("Frozen secondary source/code parent changed")

    unchanged()
    _, primary_audit = validate_primary(
        a.primary_dir, a.primary_audit_dir, a.resource_dir
    )
    layer_audit, layer_summary = verified_packet(a.layer_audit_dir)
    layer_batch = json.loads(
        (a.sensitivity_dir / "batch_manifest.json").read_text(encoding="utf-8")
    )
    layer_cases = {
        f"{task}_{arm}_qwen_{site}_pca16"
        for task in TASKS
        for arm in ARMS
        for site in SITES
        if site != "block_10"
    }
    if (
        layer_audit["sensitivity_identity"] != layer_batch["identity"]
        or layer_audit["primary_audit_identity"] != primary_audit["identity"]
        or set(layer_audit["inputs"]) != layer_cases
        or set(layer_audit["accepted_arms"]) != set(ARMS)
        or layer_summary["bridge_labels_read"]
        or layer_summary["model_refits"]
    ):
        raise ValueError("Complete previously audited body-layer family required")
    members, common, parent = source_members(a)
    if m["authors_by_task"] != {t: len(v) for t, v in members.items()}:
        raise ValueError("Changed reference-free source membership")
    accepted = {r["case"]: r for r in status["completed"]}
    packets, inputs = {}, {}
    names = (
        "manifest.json",
        "summary.json",
        "private_oof_predictions.csv",
        "private_full_source_head.joblib",
    )
    for case in cases():
        folder = a.batch_dir / case["id"] / "fit"
        if (
            validate_result(
                folder, case, members, common, parent, m["input_sha256"]["budget"]
            )
            != accepted[case["id"]]
        ):
            raise ValueError("Result differs from its terminal acceptance")
        packets[case["task"], case["arm"], case["site"], case["readout"]] = load_oof(
            folder
        )
        inputs[case["id"]] = {
            n: digest(folder / n) for n in (*names, "saved_head_replay.csv")
        }
    rows, scores, traits = [], [], []
    for task in TASKS:
        for arm in ARMS:
            for site in SITES:
                baseline, folder = baseline_packet(
                    a, task, arm, site, members, common, (primary_audit, layer_audit)
                )
                baseline_key = f"{task}_{arm}_{site}_body_token"
                inputs[baseline_key] = {n: digest(folder / n) for n in names}
                names0 = readouts(site)
                group = [
                    baseline,
                    *(packets[task, arm, site, kind] for kind in names0[1:]),
                ]
                require_pairing(group)
                rows.extend(
                    dict(
                        task=task, arm=arm, site=site, authors=len(baseline["ids"]), **r
                    )
                    for r in paired_readouts(
                        baseline["y"],
                        [p["p"] for p in group],
                        baseline["folds"],
                        task,
                        site,
                    )
                )
                for kind, packet in zip(names0, group, strict=True):
                    tags = {"task": task, "arm": arm, "site": site, "readout": kind}
                    scores.append(
                        {
                            **tags,
                            "authors": len(packet["ids"]),
                            "pooled_score": packet["summary"]["score"],
                            "mean_fold_score": packet["mean_fold_score"],
                        }
                    )
                    traits.extend(
                        {**tags, **r} for r in metrics(packet["y"], packet["p"], task)
                    )
                print(
                    json.dumps(
                        {"task": task, "arm": arm, "site": site, "heads": len(group)}
                    ),
                    flush=True,
                )
    unchanged()
    if len(scores) != 375 or len(rows) != 870:
        raise ValueError("Incomplete matched-body and exploratory comparison family")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    code = m["scoring_code"] | {
        n: digest(ROOT / n)
        for n in (
            "scripts/analyze_grain_cached_readouts.py",
            "scripts/run_grain_neural_sensitivity_batch.py",
            "scripts/summarize_grain_neural_source_primary.py",
            "scripts/analyze_grain_ei_source_gap.py",
            "scripts/summarize_grain_vector_external_links.py",
        )
    }
    manifest = {
        "protocol": "VM05_COMPLETE_SECONDARY_READOUT_COMPARISON_V1",
        "batch_identity": m["identity"],
        "body_audit_identities": [primary_audit["identity"], layer_audit["identity"]],
        "input_sha256": inputs,
        "scoring_code": code,
        "bootstrap_seed": 731,
        "draws": 1000,
        "analysis_implemented_during_execution": True,
        "inference": "Pointwise author-within-fold paired bootstrap conditional on fitted OOF; not familywise or refit inference",
        "primary_contrasts": 315,
        "exploratory_contrasts": 120,
        "estimands": ["pooled", "mean_official_fold"],
        "selection": "none",
        "bridge_labels_read": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(a.output_dir / "manifest.json", manifest)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        for n in code:
            if n not in z.namelist():
                z.write(ROOT / n, n)
    tables = {
        "observer_scores.csv": scores,
        "metrics_by_trait.csv": traits,
        "paired_readouts.csv": rows,
    }
    for name, values in tables.items():
        pd.DataFrame(values).to_csv(a.output_dir / name, index=False)
    summary = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "secondary_heads": 315,
        "existing_body_heads": 60,
        "paired_rows": len(rows),
        "model_refits": 0,
        "neural_forwards": 0,
        "source_label_tables_read": False,
        "existing_source_oof_outcomes_reused": True,
        "bridge_labels_read": False,
        "program_complete": False,
        "familywise_confirmation": False,
        "outputs_sha256": {n: digest(a.output_dir / n) for n in tables},
    }
    write_json(a.output_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    """Do not compete for CPU with any fitter; never summarize a partial winner set."""
    parser = argparse.ArgumentParser(description=__doc__)
    for n in (
        "batch_dir",
        "views",
        "preflight_dir",
        "parent_dir",
        "vector_dir",
        "paired_dir",
        "primary_audit_dir",
        "layer_audit_dir",
        *LEASE_FLAGS,
        "output_dir",
    ):
        parser.add_argument("--" + n.replace("_", "-"), type=Path, required=True)
    a = parser.parse_args()
    # Check terminal completeness before taking leases or touching saved outcomes.
    require_complete(
        json.loads((a.batch_dir / "batch_manifest.json").read_text(encoding="utf-8")),
        json.loads((a.batch_dir / "batch_status.json").read_text(encoding="utf-8")),
    )
    with ExitStack() as stack:
        for folder in (a.batch_dir, *(getattr(a, n) for n in LEASE_FLAGS)):
            handle = stack.enter_context((folder / ".execution.lock").open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        analyze(a)


if __name__ == "__main__":
    main()
