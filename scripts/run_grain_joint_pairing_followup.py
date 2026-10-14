#!/usr/bin/env python3
"""Post-r1 mechanism test: an event-paired product, not another inverse penalty."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.joint_conditional_observation import (
    build_meter,
    calibrate,
    conditional_function,
    factorial,
    predict_function,
    read_events,
)
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_joint_conditional_observation import DEFAULT, dump, read, sha


def setup(parent):
    """Freeze the separately stated post-result question before new source fitting."""
    out = parent / "paired_followup"
    out.mkdir(exist_ok=False)
    inputs = [
        Path(__file__),
        ROOT / "grain_research/joint_conditional_observation.py",
        ROOT / "docs/GRAIN_JOINT_PAIRING_FOLLOWUP_V1.md",
        parent / "cold_source/observations.csv",
        parent / "cold_application/observations.csv",
        parent / "public_bank.csv",
        parent / "generate/evidence.npz",
    ]
    dump(out / "freeze.json", {str(p): sha(p) for p in inputs})
    dump(
        out / "status.json",
        {
            "post_result_mechanism_followup": True,
            "extra_neural_calls": 0,
            "extra_events": 0,
        },
    )
    print("followup setup frozen", flush=True)


def stage(parent, name):
    """Check separate frozen inputs without rewriting r1 artifacts."""
    out = parent / "paired_followup"
    for p, digest in read(out / "freeze.json").items():
        assert sha(p) == digest, p
    folder = out / name
    folder.mkdir(exist_ok=False)
    return read(parent / "config.json"), folder


def fit(parent):
    """Fit only the original source corners, including one declared product."""
    cfg, folder = stage(parent, "fit")
    inputs = [parent / "cold_source/observations.csv", parent / "public_bank.csv"]
    outputs = [
        folder / n
        for n in ["calibration.json", "meters.json", "guard.json", "seal.json"]
    ]
    denied = install(
        inputs,
        outputs,
        [parent / "cold_application/observations.csv", parent / "generate/truth.npz"],
        [ROOT],
    )
    f = pd.read_csv(inputs[1]).merge(
        pd.read_csv(inputs[0]), on="case_id", validate="one_to_one"
    )
    assert f.source.all() and len(f) == 32
    y = f[cfg["channels"]].to_numpy(float)
    y = np.c_[y, y[:, 8] * y[:, 9]]
    model = calibrate(
        y, f.condition, f[["worry", "reassurance"]], cfg["covariance_shrinkage"]
    )
    meters = {}
    for arm, index in [("nli_product", [8, 9, 10]), ("joint_product", list(range(11)))]:
        for z in cfg["conditions"]:
            meters[f"{arm}__{z}"] = build_meter(
                model, z, index, tolerance=cfg["svd_tolerance"]
            )
    dump(folder / "calibration.json", model)
    dump(folder / "meters.json", meters)
    dump(folder / "guard.json", {"denied": denied, "source_texts": 32})
    dump(folder / "seal.json", {"meters_sha256": sha(folder / "meters.json")})
    print("followup source frozen", flush=True)


def apply(parent):
    """Use paired or deliberately unpaired same-author event readouts, without truth."""
    cfg, folder = stage(parent, "apply")
    mp = parent / "paired_followup/fit/meters.json"
    assert sha(mp) == read(mp.parent / "seal.json")["meters_sha256"]
    inputs = [
        mp,
        parent / "cold_source/observations.csv",
        parent / "cold_application/observations.csv",
        parent / "generate/evidence.npz",
    ]
    outputs = [folder / n for n in ["functions.npz", "guard.json", "seal.json"]]
    denied = install(inputs, outputs, [parent / "generate/truth.npz"], [ROOT])
    meters = read(mp)
    frame = pd.concat([pd.read_csv(inputs[1]), pd.read_csv(inputs[2])]).sort_values(
        "case_id"
    )
    raw = frame[cfg["channels"]].to_numpy(float)
    results = {}
    with np.load(inputs[3], allow_pickle=False) as data:
        for key in data.files:
            indices = data[key]
            name = key.removesuffix("_case_indices")
            base = raw[indices]
            for arm in ["nli_product", "joint_product", "joint_product_unpaired"]:
                product = base[..., 8] * (
                    np.roll(base[..., 9], 1, axis=2)
                    if arm.endswith("unpaired")
                    else base[..., 9]
                )
                y = np.concatenate([base, product[..., None]], axis=3)
                fitted_arm = arm.removesuffix("_unpaired")
                moments = np.stack(
                    [
                        read_events(y[:, i], meters[f"{fitted_arm}__{z}"])
                        for i, z in enumerate(cfg["conditions"])
                    ],
                    axis=1,
                )
                f = conditional_function(moments[:, [0, 2]])
                prefix = f"{name}__{arm}__"
                results[prefix + "level"] = f["level"]
                results[prefix + "finite_change"] = 2 * f["slope"]
                results[prefix + "new_condition"] = predict_function(f, 0.5)
    np.savez_compressed(folder / "functions.npz", **results)
    dump(
        folder / "guard.json",
        {"denied": denied, "extra_events": 0, "middle_used_for_fit": False},
    )
    dump(
        folder / "seal.json",
        {"functions_sha256": sha(folder / "functions.npz"), "meters_sha256": sha(mp)},
    )
    print("followup functions frozen", flush=True)


def evaluate(parent):
    """Compare all moment targets plus a perfect event-count sampling reference."""
    cfg, folder = stage(parent, "evaluate")
    ap = parent / "paired_followup/apply/functions.npz"
    assert sha(ap) == read(ap.parent / "seal.json")["functions_sha256"]
    bank = pd.read_csv(parent / "public_bank.csv").sort_values("case_id")
    event_ref = factorial(bank[["worry", "reassurance"]])[:, 1:]
    rows = []
    with (
        np.load(ap, allow_pickle=False) as pred,
        np.load(parent / "generate/evidence.npz", allow_pickle=False) as evidence,
        np.load(parent / "generate/truth.npz", allow_pickle=False) as truth,
    ):
        for seed in cfg["seeds"]:
            for style in ["source_style", "held_style"]:
                name = f"s{seed}_{style}"
                oracle = conditional_function(
                    event_ref[evidence[name + "_case_indices"]][:, [0, 2]]
                )
                perfect = {
                    "level": oracle["level"],
                    "finite_change": 2 * oracle["slope"],
                    "new_condition": predict_function(oracle, 0.5),
                }
                for arm in [
                    "nli_product",
                    "joint_product",
                    "joint_product_unpaired",
                    "perfect_event_mean",
                ]:
                    for target in perfect:
                        actual = (
                            truth[f"s{seed}_level"]
                            if target == "level"
                            else 2 * truth[f"s{seed}_slope"]
                            if target == "finite_change"
                            else truth[f"s{seed}_condition_moments"][:, 1]
                        )
                        estimate = (
                            perfect[target]
                            if arm == "perfect_event_mean"
                            else pred[f"{name}__{arm}__{target}"]
                        )
                        for j, moment in enumerate(cfg["targets"]):
                            rows.append(
                                {
                                    "seed": seed,
                                    "style": style,
                                    "arm": arm,
                                    "target": target,
                                    "moment": moment,
                                    "mse": float(
                                        np.mean((estimate[:, j] - actual[:, j]) ** 2)
                                    ),
                                    "bias": float(
                                        np.mean(estimate[:, j] - actual[:, j])
                                    ),
                                    "correlation": float(
                                        np.corrcoef(estimate[:, j], actual[:, j])[0, 1]
                                    ),
                                }
                            )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "metrics_by_seed.csv", index=False)
    summary = (
        frame.groupby(["style", "arm", "target", "moment"], sort=False)[
            ["mse", "bias", "correlation"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(folder / "comparison.csv", index=False)
    print(summary[summary.target == "new_condition"].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["setup", "fit", "apply", "evaluate"])
    parser.add_argument("--parent", type=Path, default=DEFAULT)
    args = parser.parse_args()
    globals()[args.stage](args.parent.resolve())
