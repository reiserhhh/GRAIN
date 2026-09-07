#!/usr/bin/env python3
"""P1L: zero-forward setting-held-out readouts of common entity-state differences."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    """Compute source/cache provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ridge_predict(x_train, y_train, x_test):
    """Fit only training centering/scale and fixed unit-kernel Ridge alpha1."""
    center, mean = x_train.mean(axis=0), y_train.mean(axis=0)
    x = x_train - center
    scale = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    if scale < 1e-12:
        return np.broadcast_to(mean, (len(x_test), len(mean))).copy()
    x, test = x / scale, (x_test - center) / scale
    weights = np.linalg.solve(x @ x.T + np.eye(len(x)), y_train - mean)
    return mean + test @ x.T @ weights


def cross_predict(x, y, groups, names, model):
    """Leave both names from each setting out together; never tune on outcomes."""
    result = np.zeros_like(y, dtype=float)
    for group in sorted(set(groups)):
        test = groups == group
        train = ~test
        if model == "mean":
            result[test] = y[train].mean(axis=0)
        elif model == "name_mean":
            for i in np.flatnonzero(test):
                result[i] = y[train & (names == names[i])].mean(axis=0)
        else:
            features = (
                np.linalg.norm(x, axis=1, keepdims=True) if model == "norm_ridge" else x
            )
            if model not in ("norm_ridge", "vector_ridge"):
                raise ValueError("Undeclared model")
            result[test] = ridge_predict(features[train], y[train], features[test])
    return result


def load_arrays(parent):
    """Read audited canonical entity states and exact finite response coefficients."""
    material = json.loads((parent / "materials.json").read_text())["rows"]
    layouts = json.loads((parent / "token_layouts.json").read_text())
    coefficient = pd.read_csv(parent / "coefficients.csv")
    cells = sorted({(r["group_id"], r["entity"]) for r in material})
    target_keys = list(
        itertools.product(
            ("native", "N"),
            ("event", "trait", "venue"),
            ("constant", "polarity", "code", "semantic"),
        )
    )
    x, y = [], []
    for setting, name in cells:
        subset = [
            r
            for r in material
            if r["group_id"] == setting
            and r["entity"] == name
            and r["criterion"] == "event"
            and r["polarity"] == r["code"] == 1
        ]
        states = {}
        for r in subset:
            n = layouts[r["case_id"]]["N"]
            if len(n) != 1:
                raise ValueError("Single name token required")
            states[r["role_sign"]] = np.load(
                parent / "activations" / (r["case_id"] + ".npz")
            )["block_16"][n[0]].astype(float)
        if set(states) != {-1, 1} or len(subset) != 2:
            raise ValueError("Incomplete role pair")
        x.append(states[1] - states[-1])
        values = []
        for channel, criterion, term in target_keys:
            sub = coefficient[
                (coefficient.group_id == setting)
                & (coefficient.entity == name)
                & (coefficient.channel == channel)
                & (coefficient.criterion == criterion)
            ]
            if len(sub) != 1:
                raise ValueError("Incomplete target grid")
            values.append(float(sub.iloc[0][term]))
        y.append(values)
    return np.array(x), np.array(y), cells, target_keys


def run(args):
    """Run fixed CPU readouts and the complete24-setting permutation diagnostic."""
    start = time.monotonic()
    parent = args.parent_dir
    for name, digest in json.loads((parent / "output_hashes.json").read_text()).items():
        if sha(parent / name) != digest:
            raise ValueError("Parent artifact changed")
    manifest = json.loads((parent / "manifest.json").read_text())
    if json.loads((parent / "run_status.json").read_text())["status"] != "COMPLETE":
        raise ValueError("Parent incomplete")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "docs/GRAIN_ENTITY_OPERATOR_P1L_PLAN.md",
        ROOT / "tests/test_grain_entity_operator.py",
    ]
    identity = {
        "parent_identity": manifest["identity"],
        "new_neural_calls": 0,
        "alpha": 1,
        "split": "leave_one_setting_out",
        "code": {str(p.relative_to(ROOT)): sha(p) for p in sources},
    }
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT))
    (out / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    try:
        x, y, cells, target_keys = load_arrays(parent)
        groups = np.array([k[0] for k in cells])
        names = np.array([k[1] for k in cells])
        if (
            x.shape[0] != 8
            or y.shape != (8, 24)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("Bad input matrix")
        np.savez_compressed(out / "arrays.npz", x=x, y=y, groups=groups, names=names)
        predictions = {
            m: cross_predict(x, y, groups, names, m)
            for m in ("mean", "name_mean", "norm_ridge", "vector_ridge")
        }
        rows, metrics = [], []
        for model, pred in predictions.items():
            for i, (group, name) in enumerate(cells):
                for j, (channel, criterion, term) in enumerate(target_keys):
                    rows.append(
                        {
                            "model": model,
                            "group_id": group,
                            "entity": name,
                            "channel": channel,
                            "criterion": criterion,
                            "term": term,
                            "observed": y[i, j],
                            "predicted": pred[i, j],
                        }
                    )
            for channel, criterion, term in target_keys:
                j = target_keys.index((channel, criterion, term))
                error = float(np.sum((pred[:, j] - y[:, j]) ** 2))
                ref = float(np.sum((predictions["mean"][:, j] - y[:, j]) ** 2))
                zero = float(np.sum(y[:, j] ** 2))
                metrics.append(
                    {
                        "model": model,
                        "channel": channel,
                        "criterion": criterion,
                        "term": term,
                        "sse": error,
                        "mean_sse": ref,
                        "zero_sse": zero,
                        "relative_to_mean": error / ref if ref > 1e-12 else None,
                        "relative_to_zero": error / zero if zero > 1e-12 else None,
                    }
                )
        pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
        pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
        totals = [
            {
                "model": m,
                "sse": float(np.sum((pred - y) ** 2)),
                "relative_to_mean": float(
                    np.sum((pred - y) ** 2) / np.sum((predictions["mean"] - y) ** 2)
                ),
            }
            for m, pred in predictions.items()
        ]
        pd.DataFrame(totals).to_csv(out / "summary.csv", index=False)
        permutations = []
        unique = sorted(set(groups))
        index = {cell: i for i, cell in enumerate(cells)}
        for permutation in itertools.permutations(unique):
            if time.monotonic() - start > 60:
                raise RuntimeError("P1L CPU time budget exceeded")
            mapping = dict(zip(unique, permutation))
            perm_y = np.array([y[index[mapping[g], name]] for g, name in cells])
            pred = cross_predict(x, perm_y, groups, names, "vector_ridge")
            permutations.append(
                {
                    "mapping": "|".join(permutation),
                    "identity": list(permutation) == unique,
                    "sse": float(np.sum((pred - perm_y) ** 2)),
                }
            )
        pd.DataFrame(permutations).to_csv(out / "permutations.csv", index=False)
        observed = float(np.sum((predictions["vector_ridge"] - y) ** 2))
        singular = np.linalg.svd(x - x.mean(axis=0), compute_uv=False)
        result = {
            "status": "COMPLETE",
            "new_neural_calls": 0,
            "rows": 8,
            "settings": 4,
            "targets": 24,
            "seconds": time.monotonic() - start,
            "permutations": 24,
            "permutation_tail_fraction": float(
                np.mean([p["sse"] <= observed + 1e-10 for p in permutations])
            ),
            "x_centered_singular_values": singular.tolist(),
            "x_centered_rank": int(np.linalg.matrix_rank(x - x.mean(axis=0))),
            "interpretation": "Four authored setting folds; functional response prediction, not personality generalization.",
        }
        (out / "run_status.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "output_hashes.json").write_text(
            json.dumps(
                {
                    str(p.relative_to(out)): sha(p)
                    for p in out.rglob("*")
                    if p.is_file() and not p.name.startswith("._")
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(result))
    except Exception as exc:
        (out / "run_status.json").write_text(
            json.dumps({"status": "FAILED", "error": str(exc), "new_neural_calls": 0})
            + "\n"
        )
        raise


def main():
    """Run one existing-cache packet, without fitting the LLM or reading people."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
