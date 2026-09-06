#!/usr/bin/env python3
"""Paired original-dimension versus PCA16 audit of all nine declared Big5 heads."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_controls import exclude_reference
from grain_research.vector_mechanisms.inventory import archive_source
from grain_research.vector_mechanisms.ready_comparison import digest
from scripts.run_grain_neural_sensitivity_batch import (
    cases,
    validate_primary,
    validate_replay,
    validate_result,
)
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.summarize_grain_neural_source_primary import score
from scripts.summarize_grain_vector_external_links import load_oof


def paired_readouts(candidate, baseline, draws=1000):
    """Use shared within-fold author samples; preserve pooled and mean-fold estimands."""
    if (
        candidate["manifest"]["task"] != "big5"
        or baseline["manifest"]["task"] != "big5"
        or any(
            not np.array_equal(candidate[k], baseline[k]) for k in ("ids", "y", "folds")
        )
        or set(candidate["folds"]) != set(range(5))
        or draws < 20
    ):
        raise ValueError(
            "Require paired Big5 authors, outcomes and five official folds"
        )
    y, predictions = candidate["y"], [candidate["p"], baseline["p"]]
    parts = [np.flatnonzero(candidate["folds"] == f) for f in range(5)]

    def estimates(indices):
        joined = np.concatenate(indices)
        pooled = [score(y[joined], p[joined], "big5") for p in predictions]
        within = [[score(y[ix], p[ix], "big5") for p in predictions] for ix in indices]
        return np.asarray([pooled, np.mean(within, axis=0)])

    observed = estimates(parts)
    if not np.isfinite(observed).all():
        raise ValueError("Undefined observed prediction correlation")
    rng = np.random.default_rng(731)
    samples = np.asarray(
        [
            estimates([rng.choice(ix, len(ix), replace=True) for ix in parts])
            for _ in range(draws)
        ]
    )
    finite = samples[np.isfinite(samples).all(axis=(1, 2))]
    if len(finite) < 0.95 * draws:
        raise ValueError("Insufficient bootstrap trait support")
    rows = []
    for i, estimand in enumerate(("pooled", "mean_official_fold")):
        low, high = np.quantile(finite[:, i, 0] - finite[:, i, 1], [0.025, 0.975])
        rows.append(
            {
                "estimand": estimand,
                "no_pca_score": observed[i, 0],
                "pca16_score": observed[i, 1],
                "delta": observed[i, 0] - observed[i, 1],
                "lower": low,
                "upper": high,
                "draws": draws,
                "undefined_draws": draws - len(finite),
            }
        )
    return rows


def main():
    """Read all nine completed/replayed pairs without opening any new outcome table."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "batch-dir",
        "primary-dir",
        "audit-dir",
        "resource-dir",
        "views",
        "preflight-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    primary, audit = validate_primary(a.primary_dir, a.audit_dir, a.resource_dir)
    batch = json.loads((a.batch_dir / "batch_manifest.json").read_text())
    if (
        batch["protocol"] != "VM05_FROZEN_NEURAL_SENSITIVITIES_V1"
        or material_hash({k: v for k, v in batch.items() if k != "identity"})
        != batch["identity"]
        or batch["cases"] != cases()
        or batch["primary_identity"] != primary["identity"]
        or batch["primary_audit_identity"] != audit["identity"]
        or digest(a.views) != primary["views_sha256"]
        or digest(a.preflight_dir / "manifest.json")
        != primary["preflight_manifest_sha256"]
        or any(digest(ROOT / n) != sha for n, sha in batch["code_sha256"].items())
    ):
        raise ValueError("Changed declared sensitivity parent or input")
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Common material changed")
    source = exclude_reference(
        source,
        pd.read_csv(a.preflight_dir / "budget_by_author.csv", dtype={"author_id": str}),
    )
    pairs, inputs, rows = [], {}, []
    for c in cases()[:9]:
        folder = a.batch_dir / c["id"]
        candidate = load_oof(folder)
        validate_result(candidate, c, source, common["material_sha256"])
        replay = folder.with_name(folder.name + "_replay")
        validate_replay(replay, c)
        if (
            digest(folder / "private_full_source_head.joblib")
            != candidate["summary"]["full_head_sha256"]
        ):
            raise ValueError("Original-dimension source head changed")
        base_name = f"{c['task']}_{c['arm']}_{c['feature']}"
        baseline = load_oof(a.primary_dir / base_name)
        validate_result(baseline, c | {"pca": True}, source, common["material_sha256"])
        for key in (
            "common_material_sha256",
            "source_material_sha256",
            "feature_input_sha256",
        ):
            if candidate["manifest"][key] != baseline["manifest"][key]:
                raise ValueError(
                    "PCA contrast changed more than the projection/readout fitting"
                )
        files = [
            folder / n
            for n in (
                "manifest.json",
                "summary.json",
                "private_oof_predictions.csv",
                "private_full_source_head.joblib",
            )
        ]
        files += [folder / f"private_fold_{f}.joblib" for f in range(5)]
        files += [replay / "summary.json", replay / "replay_by_fold.csv"]
        inputs[c["id"]] = {
            str(path.relative_to(a.batch_dir)): digest(path) for path in files
        }
        pairs.append((c, candidate, baseline))
    for c, candidate, baseline in pairs:
        rows.extend(
            {
                "observer": c["feature"],
                "arm": c["arm"],
                "authors": len(candidate["ids"]),
                **r,
            }
            for r in paired_readouts(candidate, baseline)
        )
    a.output_dir.mkdir(parents=True, exist_ok=False)
    code = (
        "scripts/analyze_grain_projection_sensitivity.py",
        "scripts/summarize_grain_neural_source_primary.py",
        "scripts/run_grain_neural_sensitivity_batch.py",
    )
    m = {
        "protocol": "VM05_BIG5_PROJECTION_SENSITIVITY_V1",
        "primary_audit_identity": audit["identity"],
        "sensitivity_identity": batch["identity"],
        "inputs": inputs,
        "code_sha256": {n: digest(ROOT / n) for n in code},
        "bootstrap": "1000 author-within-fold draws, seed731, pointwise conditional on fitted OOF",
        "analysis_implementation_after_point_estimates": True,
    }
    m["identity"] = material_hash(m)
    write_json(a.output_dir / "manifest.json", m)
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    with zipfile.ZipFile(
        a.output_dir / "source_snapshot.zip", "a", zipfile.ZIP_DEFLATED
    ) as z:
        for name in code:
            if name not in z.namelist():
                z.write(ROOT / name, name)
    path = a.output_dir / "paired_projection.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    s = {
        "status": "COMPLETE",
        "parent_identity": m["identity"],
        "pairs": len(pairs),
        "rows": len(rows),
        "authors_per_pair": len(pairs[0][1]["ids"]),
        "source_label_tables_read": False,
        "existing_oof_outcomes_reused": True,
        "bridge_labels_read": False,
        "model_refits": 0,
        "program_complete": False,
        "outputs_sha256": {path.name: digest(path)},
        "scope": "All9 predeclared Big5 compression contrasts, not simultaneous/refit inference or a causal signal-erasure result",
    }
    write_json(a.output_dir / "summary.json", s)
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
