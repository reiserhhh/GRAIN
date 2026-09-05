#!/usr/bin/env python3
"""Replay stored source-fold models on exactly their held-out author evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.external_heads import TRAITS, predict_head
from grain_research.vector_mechanisms.inventory import archive_source
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_external_features import bundle_arms
from scripts.run_grain_vector_mechanism_external_links import dense_features
from scripts.summarize_grain_vector_external_links import load_oof


def main():
    """Verify model/output correspondence without refitting or new label access."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--vector-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new replay output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    source = json.loads(a.views.read_text(encoding="utf-8"))
    common = json.loads((a.preflight_dir / "common_material_manifest.json").read_text())
    if material_hash(source) != common["material_sha256"]:
        raise ValueError("Changed common material")
    cache, records = {}, []
    with threadpool_limits(limits=4):
        for folder in a.runs:
            result = load_oof(folder)
            m = result["manifest"]
            bundles = [b for b in source if b["source_group"] == m["source_group"]]
            if material_hash(bundles) != m["source_material_sha256"]:
                raise ValueError("Changed source author evidence")
            key = (m["source_group"], m["feature"], m["arm"])
            if key not in cache:
                if m["feature"] == "word_char":
                    x = np.asarray(
                        ["\n\n".join(bundle_arms(b)[m["arm"]]) for b in bundles]
                    )
                else:
                    opts = SimpleNamespace(
                        feature=m["feature"],
                        arm=m["arm"],
                        preflight_dir=a.preflight_dir,
                        vector_dir=a.vector_dir,
                    )
                    x = dense_features(bundles, source, common, opts)
                ix = {b["author_id"]: i for i, b in enumerate(bundles)}
                cache[key] = (x, ix)
            x, ix = cache[key]
            for fold in range(5):
                test = np.flatnonzero(result["folds"] == fold)
                model = joblib.load(folder / f"private_fold_{fold}.joblib")
                prediction = predict_head(
                    model, x[[ix[u] for u in result["ids"][test]]], m["task"]
                )
                error = float(np.max(abs(prediction - result["p"][test])))
                if not np.isfinite(error) or error > 1e-10:
                    raise ValueError("Stored fold prediction does not replay")
                # Actual sparse pipeline never applies PCA despite old CLI request metadata.
                records.append(
                    {
                        "run": folder.name,
                        "fold": fold,
                        "authors": len(test),
                        "max_absolute_error": error,
                        "pca_applied": "pca" in model.named_steps,
                        "target_columns": len(TRAITS) if m["task"] == "big5" else 1,
                    }
                )
            print(json.dumps({"run": folder.name, "verified_folds": 5}), flush=True)
    pd.DataFrame(records).to_csv(out / "replay_by_fold.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "runs": len(a.runs),
        "fold_models": len(records),
        "max_absolute_error": max(r["max_absolute_error"] for r in records),
        "new_label_values_read": False,
        "model_refits": 0,
        "bridge_labels_read": False,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
