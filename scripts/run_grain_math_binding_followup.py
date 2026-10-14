#!/usr/bin/env python3
"""Result-guided joint-calibration packet with new, cold confirmation families."""

import argparse
import json
from pathlib import Path

import numpy as np
from run_grain_math_response import (
    DEFAULT_OUT,
    MODELS,
    ROOT,
    M,
    checkpoint,
    sha,
    write_csv,
    write_json,
)


def prepare(out):
    """Freeze independent content cores, reusing the original operation wrappers."""
    dest = out / "06/00"
    if (dest / "material_bank.jsonl").exists():
        return
    families = json.loads(
        (ROOT / "configs/grain_math_confirmation_families_v1.json").read_text()
    )
    old = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    prior = json.loads(
        (ROOT / "configs/grain_math_material_families_v1.json").read_text()
    )[0]
    materials = []
    for f, (name, setup, positive, negative) in enumerate(families, 32):
        for k, bits in enumerate(M.product((-1, 1), repeat=4)):
            row = old[k].copy()
            p = bits[0]
            row["text"] = (
                row["text"]
                .replace(prior[1], setup)
                .replace(
                    prior[2] if p == 1 else prior[3], positive if p == 1 else negative
                )
            )
            row.update(
                material_id=f"f{f:02d}_" + "".join(str((b + 1) // 2) for b in bits),
                family=f,
                content_name=name,
                role="confirmation",
                event_id=f"synthetic_f{f:02d}",
            )
            materials.append(row)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "material_bank.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in materials)
    )
    write_json(
        dest / "reader_registry.json",
        json.loads((out / "00/reader_registry.json").read_text()),
    )
    write_json(
        dest / "protocol_before_run.json",
        {
            "source": "original 16 source families",
            "validation": "original 8 validation families",
            "old_test": "diagnostic only; no parameter selection",
            "confirmation": "16 fresh cores; same frozen operations, single score pass",
            "hypothesis": "joint calibration of strong first-order readouts avoids weak interaction-direction inversion",
            "model": "3 first-order ridge readouts, source LOFO second-stage multilinear vs additive calibration",
            "budget": {
                "new_texts": 256,
                "encodings": 512,
                "neural_training": 0,
                "api": 0,
            },
            "bank_sha256": sha(dest / "material_bank.jsonl"),
            "result_guided": True,
        },
    )
    checkpoint(out, "06", "materials_frozen", [str(dest / "protocol_before_run.json")])


def run(out):
    """Calibrate only on old source/validation, then score new families once."""
    dest = out / "06"
    if (dest / "confirmation_metrics.json").exists():
        return
    old = [
        json.loads(s) for s in (out / "00/material_bank.jsonl").read_text().splitlines()
    ]
    fresh = [
        json.loads(s)
        for s in (dest / "00/material_bank.jsonl").read_text().splitlines()
    ]
    bank = old + fresh
    bits, groups, roles = (
        np.array([r["bits"] for r in bank]),
        np.array([r["family"] for r in bank]),
        np.array([r["role"] for r in bank]),
    )
    candidates, metadata = {}, {}
    vectors = {}
    for reader in MODELS:
        x = np.concatenate(
            [
                np.load(out / f"01B/embeddings/{reader}.npy"),
                np.load(dest / f"01B/embeddings/{reader}.npy"),
            ]
        ).astype(float)
        vectors[reader] = x
        candidates[reader], metadata[reader] = {}, {}
        for arm, joint in (("joint_binding", True), ("additive_binding", False)):
            model, metadata[reader][arm] = M.fit_binding_model(
                x, bits, groups, roles, joint
            )
            candidates[reader][arm] = model
            path = dest / f"reader_bundles/{reader}_{arm}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **model)
        for arm in ("ridge", "paired_dual"):
            with np.load(out / f"01B/reader_bundles/{reader}_{arm}.npz") as bundle:
                candidates[reader][arm + "_r1"] = {k: bundle[k] for k in bundle.files}
    write_json(dest / "source_fits.json", metadata)
    write_json(
        dest / "model_freeze.json",
        {
            "phase": "before_confirmation",
            "source_families": np.unique(groups[roles == "source"]).tolist(),
            "validation_families": np.unique(groups[roles == "validation"]).tolist(),
            "bundles": {
                p.name: sha(p)
                for p in (dest / "reader_bundles").glob("*.npz")
                if not p.name.startswith("._")
            },
        },
    )
    metrics, family_rows, decomposition, incremental = {}, [], [], []
    confirm = roles == "confirmation"
    for reader, models in candidates.items():
        x = vectors[reader]
        metrics[reader] = {}
        for arm, model in models.items():
            scores = M.score_bundle(model, x)
            np.save(dest / f"scores_{reader}_{arm}.npy", scores)
            metrics[reader][arm] = []
            for target in range(2):
                effects = M.family_effects(
                    scores[confirm], bits[confirm], groups[confirm], target
                )
                gain = np.array([r["target_gain"] for r in effects])
                mse = np.array(
                    [
                        np.mean(
                            (
                                scores[confirm & (groups == f), target]
                                - M.text_targets(bits[confirm & (groups == f)])[
                                    :, target
                                ]
                            )
                            ** 2
                        )
                        for f in np.unique(groups[confirm])
                    ]
                )
                metrics[reader][arm].append(
                    {
                        "target": target,
                        "gain": float(gain.mean()),
                        "gain_ci": M.bootstrap_interval(gain),
                        "style_rms": float(np.mean([r["style_rms"] for r in effects])),
                        "other_gain": float(
                            np.mean([r["other_target_gain"] for r in effects])
                        ),
                        "unit_rmse": float(np.sqrt(mse.mean())),
                        "mse_ci": M.bootstrap_interval(mse),
                    }
                )
                family_rows.extend(
                    {
                        "reader": reader,
                        "arm": arm,
                        "target": target,
                        "mse": float(mse[i]),
                        **r,
                    }
                    for i, r in enumerate(effects)
                )
                errors = (scores - M.text_targets(bits))[:, target]
                for role in ("test", "confirmation"):
                    mask = roles == role
                    offset = np.array(
                        [
                            np.mean(errors[mask & (groups == f)]) ** 2
                            for f in np.unique(groups[mask])
                        ]
                    ).mean()
                    total = float(np.mean(errors[mask] ** 2))
                    decomposition.append(
                        {
                            "reader": reader,
                            "arm": arm,
                            "target": target,
                            "role": role,
                            "total_mse": total,
                            "between_family_offset_mse": float(offset),
                            "within_family_error_mse": total - float(offset),
                            "offset_fraction": float(offset / total),
                        }
                    )
        for target in range(2):
            rr = [
                r
                for r in family_rows
                if r["reader"] == reader and r["target"] == target
            ]
            joint = {r["family"]: r["mse"] for r in rr if r["arm"] == "joint_binding"}
            for arm in ("additive_binding", "ridge_r1", "paired_dual_r1"):
                delta = np.array(
                    [r["mse"] - joint[r["family"]] for r in rr if r["arm"] == arm]
                )
                incremental.append(
                    {
                        "reader": reader,
                        "target": target,
                        "comparator": arm,
                        "joint_mse_improvement": float(delta.mean()),
                        "ci": M.bootstrap_interval(delta),
                    }
                )
    write_json(dest / "confirmation_metrics.json", metrics)
    write_csv(dest / "confirmation_by_family.csv", family_rows)
    write_csv(dest / "incremental_comparison.csv", incremental)
    write_csv(dest / "error_decomposition.csv", decomposition)
    checkpoint(
        out,
        "06",
        "complete",
        [
            str(dest / "confirmation_metrics.json"),
            str(dest / "incremental_comparison.csv"),
        ],
    )


def main():
    """Run preparation or inference separately from the two offline encoder jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else run)(args.out)
    print(f"binding stage={args.stage} complete", flush=True)


if __name__ == "__main__":
    main()
