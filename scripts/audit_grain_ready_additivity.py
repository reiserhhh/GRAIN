#!/usr/bin/env python3
"""Check the algebraic scope of saved hybrid heads without fitting or real targets."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.ready_comparison import digest, verified_packet
from grain_research.vector_mechanisms.ready_registry import TASKS
from scripts.run_grain_vector_mechanism_controls import write_json


def audit_head(model, split, task):
    """Verify zero mixed finite differences on the identity/logit scale, not probability."""
    if list(model.named_steps) != ["scale", "pca", "head"]:
        raise ValueError("Expected fixed scaler/PCA/linear head pipeline")
    scale, pca, head = (model.named_steps[n] for n in ("scale", "pca", "head"))
    if (
        type(head).__name__ != ("Ridge" if task == "big5" else "LogisticRegression")
        or pca.whiten
        or not 0 < split < len(scale.mean_)
    ):
        raise ValueError("Different head family or coordinate contract")
    p = len(scale.mean_)
    rng = np.random.default_rng(731)
    base = scale.mean_ + rng.normal(size=(8, p)) * scale.scale_ * 0.1
    dg, dh = np.zeros_like(base), np.zeros_like(base)
    dg[:, :split] = rng.normal(size=(8, split)) * scale.scale_[:split] * 0.1
    dh[:, split:] = rng.normal(size=(8, p - split)) * scale.scale_[split:] * 0.1
    points = [base, base + dg, base + dh, base + dg + dh]
    score = model.predict if task == "big5" else model.decision_function
    values = [np.asarray(score(x)) for x in points]
    mixed = values[3] - values[1] - values[2] + values[0]
    # PCA and scaling are affine: coefficients pull back exactly to input space.
    beta = np.atleast_2d(head.coef_) @ pca.components_ / scale.scale_[None, :]
    if task == "big5":
        expected = dg @ beta.T
        observed = values[1] - values[0]
    else:
        expected = (dg @ beta.T).ravel()
        observed = values[1] - values[0]
    errors = [float(np.max(abs(mixed))), float(np.max(abs(expected - observed)))]
    if not np.isfinite(errors).all() or max(errors) > 1e-10:
        raise ValueError("Saved pipeline does not reproduce additive pullback")
    return {
        "score_scale": "identity" if task == "big5" else "log_odds",
        "synthetic_rectangles": len(base),
        "mixed_difference_max": errors[0],
        "coefficient_pullback_max": errors[1],
        "psychological_interaction_tested": False,
    }


def main():
    """Audit all60 hybrid saved heads after complete paired acceptance, no real-data refit."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("batch-dir", "paired-dir", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    paired, summary = verified_packet(a.paired_dir)
    batch = json.loads(
        (a.batch_dir / "batch_manifest.json").read_text(encoding="utf-8")
    )
    if (
        paired["protocol"] != "VM05_RAW_READY_PAIRED_COMPARISON_V1"
        or paired["batch_identity"] != batch["identity"]
        or summary["new_source_heads_accepted"] != 34
        or summary["paired_rows"] != 90
    ):
        raise ValueError("Complete raw ready paired parent required")
    rows, inputs = [], {}
    for task in TASKS:
        for feature in ("grain_qwen_block_10", "grain_bge_large"):
            case = f"{task}_raw_{feature}"
            folder = a.batch_dir / case / "fit"
            m, _ = verified_packet(folder)
            if (
                m["task"] != task
                or m["feature"] != feature
                or m["feature_blocks"][0]
                != {"name": "sorted_distances", "column_start": 0, "column_stop": 16}
            ):
                raise ValueError("Changed task or hybrid block ordering")
            for label in (*range(5), "full"):
                name = (
                    "private_full_source_head.joblib"
                    if label == "full"
                    else f"private_fold_{label}.joblib"
                )
                sha = digest(folder / name)
                if sha != paired["inputs"][case][name]:
                    raise ValueError("Saved head differs from paired parent")
                inputs[f"{case}/{name}"] = sha
                rows.append(
                    {
                        "task": task,
                        "feature": feature,
                        "fold": label,
                        **audit_head(joblib.load(folder / name), 16, task),
                    }
                )
    out = a.output_dir.resolve()
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private results directory required")
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "saved_head_additivity.csv", index=False)
    m = {
        "protocol": "VM05_READY_SAVED_HEAD_ADDITIVITY_V1",
        "paired_identity": paired["identity"],
        "inputs": inputs,
        "scoring_code": {
            "scripts/audit_grain_ready_additivity.py": digest(Path(__file__))
        },
        "scope": "Post-hoc mathematical instrument audit, not source prediction selection",
    }
    m["identity"] = material_hash(m)
    write_json(out / "manifest.json", m)
    s = {
        "status": "COMPLETE",
        "parent_identity": m["identity"],
        "saved_heads": len(rows),
        "synthetic_rectangles": sum(r["synthetic_rectangles"] for r in rows),
        "mixed_difference_max": max(r["mixed_difference_max"] for r in rows),
        "coefficient_pullback_max": max(r["coefficient_pullback_max"] for r in rows),
        "model_refits": 0,
        "real_feature_rows_read": False,
        "source_label_tables_read": False,
        "bridge_labels_read": False,
        "personality_interaction_identified": False,
        "outputs_sha256": {
            "saved_head_additivity.csv": digest(out / "saved_head_additivity.csv")
        },
    }
    write_json(out / "summary.json", s)
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
