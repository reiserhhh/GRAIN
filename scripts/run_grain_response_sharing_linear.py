#!/usr/bin/env python3
"""Post-result signed random-slope reference, not a replacement of frozen r1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.response_sharing import apply_function, compile_function
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_joint_conditional_observation import dump, read, sha
from scripts.run_grain_response_sharing import DEFAULT, TARGETS


def fit(parent, out):
    """Freeze a three-component reference and use original source moments only."""
    out.mkdir(exist_ok=False)
    cfg = read(parent / "config.json")
    paths = [
        Path(__file__).resolve(),
        ROOT / "docs/GRAIN_RESPONSE_SHARING_LINEAR_REFERENCE.md",
        ROOT / "grain_research/response_sharing.py",
        parent / "fit/models.json",
        parent / "config.json",
    ]
    dump(out / "freeze.json", {str(p): sha(p) for p in paths})
    source = parent / "fit/models.json"
    denied = install(
        [source],
        [out / "models.json", out / "fit_guard.json"],
        [parent / "generate"],
        [ROOT],
    )
    models = read(source)
    z = np.array(cfg["conditions"])
    basis = np.array([np.ones((5, 5)), np.outer(z, z), np.eye(5)])
    design = basis.reshape(3, -1).T
    result = {}
    for key, model in models.items():
        if key.endswith("_unpaired"):
            continue
        coef, residual = nnls(design, np.array(model["raw_signal"]).ravel())
        model["kernel"] = np.einsum("k,kij->ij", coef, basis)
        model["linear_components"], model["fit_residual"] = coef, residual
        gradient = design.T @ (design @ coef - np.array(model["raw_signal"]).ravel())
        assert gradient.min() > -1e-10
        np.testing.assert_allclose(gradient[coef > 1e-10], 0, atol=1e-10)
        result[key] = model
    dump(out / "models.json", result)
    dump(
        out / "fit_guard.json",
        {"denied": denied, "models_sha256": sha(out / "models.json")},
    )
    print("9 source-only random-slope references fit", flush=True)


def apply(parent, out, stage):
    """Apply one observed support; independently check mean/covariance algebra."""
    cfg = read(parent / "config.json")
    for path, digest in read(out / "freeze.json").items():
        assert sha(path) == digest
    assert sha(out / "models.json") == read(out / "fit_guard.json")["models_sha256"]
    folder = out / stage
    folder.mkdir(exist_ok=False)
    model_file, data_file = (
        out / "models.json",
        parent / f"generate/application_{stage}.npz",
    )
    denied = install(
        [model_file, data_file],
        [
            folder / n
            for n in ["programs.json", "predictions.npz", "audit.json", "seal.json"]
        ],
        [parent / "generate/truth.npz"]
        + [
            parent / f"generate/application_{s}.npz"
            for s in cfg["stages"]
            if s != stage
        ],
        [ROOT],
    )
    models, predictions, programs, checks, worst = read(model_file), {}, {}, 0, 0.0
    obs = cfg["stages"][stage]
    with np.load(data_file, allow_pickle=False) as data:
        for key, model in models.items():
            p = compile_function(model, obs)
            programs[key] = p
            k, r, m = (
                np.array(model["kernel"]),
                np.array(model["noise"]),
                np.array(model["mean"]),
            )
            s = (k + r)[np.ix_(obs, obs)]
            w = cho_solve(cho_factor(s), k[obs]).T
            np.testing.assert_allclose(w, p["weights"], atol=1e-12)
            np.testing.assert_allclose(k - w @ k[obs], p["covariance"], atol=1e-12)
            for style in ["source_style", "held_style"]:
                y = data[key + "_" + style]
                f = apply_function(p, y)["value"]
                independent = m + (y - m[obs]) @ w.T
                error = float(np.max(abs(f - independent)))
                assert error < 1e-10
                worst = max(worst, error)
                predictions[key + "_" + style] = f
                checks += f.size
    dump(folder / "programs.json", programs)
    np.savez_compressed(folder / "predictions.npz", **predictions)
    dump(
        folder / "audit.json",
        {
            "status": "PASS",
            "values_checked": checks,
            "max_error": worst,
            "denied": denied,
        },
    )
    dump(
        folder / "seal.json",
        {n: sha(folder / n) for n in ["predictions.npz", "programs.json"]},
    )
    print(stage, checks, "values independently checked", flush=True)


def evaluate(parent, out):
    """Report all reference results after its fixed scoring stages are sealed."""
    cfg = read(parent / "config.json")
    rows = []
    for stage in cfg["stages"]:
        for name, digest in read(out / stage / "seal.json").items():
            assert sha(out / stage / name) == digest
    with np.load(parent / "generate/truth.npz", allow_pickle=False) as truth:
        for stage in cfg["stages"]:
            programs = read(out / stage / "programs.json")
            with np.load(out / stage / "predictions.npz", allow_pickle=False) as data:
                for seed in cfg["seeds"]:
                    for world in cfg["worlds"]:
                        key = f"s{seed}_{world}"
                        for style in ["source_style", "held_style"]:
                            for target, c in TARGETS.items():
                                c = np.array(c)
                                y, p = (
                                    truth[key + "_application"] @ c,
                                    data[key + "_" + style] @ c,
                                )
                                variance = float(c @ programs[key]["covariance"] @ c)
                                rows.append(
                                    {
                                        "seed": seed,
                                        "world": world,
                                        "stage": stage,
                                        "style": style,
                                        "target": target,
                                        "mse": np.mean((p - y) ** 2),
                                        "correlation": np.corrcoef(p, y)[0, 1]
                                        if np.std(p) > 1e-12
                                        else np.nan,
                                        "coverage": np.mean(
                                            abs(p - y)
                                            <= 1.959964 * np.sqrt(max(variance, 0))
                                        ),
                                    }
                                )
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print("162 result rows;reference complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["fit", "center", "response", "added", "evaluate"]
    )
    parser.add_argument("--parent", type=Path, default=DEFAULT)
    args = parser.parse_args()
    parent = args.parent.resolve()
    out = parent / "linear_reference"
    if args.stage in ["fit", "evaluate"]:
        globals()[args.stage](parent, out)
    else:
        apply(parent, out, args.stage)
