#!/usr/bin/env python3
"""Independent sklearn and Gram-spectrum replay of P1L cached vector readouts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def sha(path):
    """Return a local artifact digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_oof(x, y, groups, names, model):
    """Use sklearn's estimator, not the collector's dual-system implementation."""
    prediction = np.empty_like(y)
    for g in np.unique(groups):
        held = groups == g
        train = ~held
        if model == "mean":
            prediction[held] = np.mean(y[train], axis=0)
        elif model == "name_mean":
            for i in np.flatnonzero(held):
                prediction[i] = np.mean(y[train & (names == names[i])], axis=0)
        else:
            features = (
                np.sqrt(np.sum(x * x, axis=1))[:, None] if model == "norm_ridge" else x
            )
            centered = features[train] - features[train].mean(axis=0)
            scale = np.sqrt(np.einsum("ij,ij->", centered, centered) / len(centered))
            if scale < 1e-12:
                prediction[held] = y[train].mean(axis=0)
            else:
                estimator = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky")
                estimator.fit(features[train] / scale, y[train])
                prediction[held] = estimator.predict(features[held] / scale)
    return prediction


def audit(result, parent):
    """Verify arrays originate in the declared source and replay all held-out results."""
    status = json.loads((result / "run_status.json").read_text())
    manifest = json.loads((result / "manifest.json").read_text())
    if status["status"] != "COMPLETE" or status["new_neural_calls"] != 0:
        raise ValueError("Incomplete/noncached packet")
    for directory in (result, parent):
        for name, digest in json.loads(
            (directory / "output_hashes.json").read_text()
        ).items():
            if sha(directory / name) != digest:
                raise ValueError("Source output changed")
    if (
        manifest["parent_identity"]
        != json.loads((parent / "manifest.json").read_text())["identity"]
    ):
        raise ValueError("Wrong parent")
    with zipfile.ZipFile(result / "source_snapshot.zip") as z:
        for name, digest in manifest["code"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("Source snapshot changed")
    arrays = np.load(result / "arrays.npz")
    x, y, groups, names = [arrays[k] for k in ("x", "y", "groups", "names")]
    if (
        y.shape != (8, 24)
        or len(set(groups)) != 4
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Bad cached shape")
    material = json.loads((parent / "materials.json").read_text())["rows"]
    layouts = json.loads((parent / "token_layouts.json").read_text())
    targets = pd.read_csv(parent / "coefficients.csv")
    columns = list(
        itertools.product(
            ("native", "N"),
            ("event", "trait", "venue"),
            ("constant", "polarity", "code", "semantic"),
        )
    )
    source_error = 0.0
    for i, (group, name) in enumerate(zip(groups, names)):
        subset = [
            r
            for r in material
            if r["group_id"] == group
            and r["entity"] == name
            and r["criterion"] == "event"
            and r["polarity"] == r["code"] == 1
        ]
        signs = {}
        for r in subset:
            signs[r["role_sign"]] = np.load(
                parent / "activations" / (r["case_id"] + ".npz")
            )["block_16"][layouts[r["case_id"]]["N"][0]].astype(float)
        source_error = max(source_error, float(abs(signs[1] - signs[-1] - x[i]).max()))
        for j, (channel, criterion, term) in enumerate(columns):
            sub = targets[
                (targets.group_id == group)
                & (targets.entity == name)
                & (targets.channel == channel)
                & (targets.criterion == criterion)
            ]
            if len(sub) != 1:
                raise ValueError("Missing source coefficient")
            source_error = max(source_error, abs(y[i, j] - sub.iloc[0][term]))
    predictions = pd.read_csv(result / "predictions.csv")
    metrics = pd.read_csv(result / "metrics.csv")
    summary = pd.read_csv(result / "summary.csv")
    if len(predictions) != 768 or len(metrics) != 96 or len(summary) != 4:
        raise ValueError("Incomplete readout grid")
    models = ("mean", "name_mean", "norm_ridge", "vector_ridge")
    fresh = {m: independent_oof(x, y, groups, names, m) for m in models}
    index = {(g, n): i for i, (g, n) in enumerate(zip(groups, names))}
    prediction_error, metric_error = 0.0, 0.0
    for r in predictions.itertuples():
        i, j = (
            index[r.group_id, r.entity],
            columns.index((r.channel, r.criterion, r.term)),
        )
        prediction_error = max(
            prediction_error,
            abs(fresh[r.model][i, j] - r.predicted),
            abs(y[i, j] - r.observed),
        )
    for r in metrics.itertuples():
        j = columns.index((r.channel, r.criterion, r.term))
        error = float(np.sum((fresh[r.model][:, j] - y[:, j]) ** 2))
        mean = float(np.sum((fresh["mean"][:, j] - y[:, j]) ** 2))
        zero = float(np.dot(y[:, j], y[:, j]))
        values = {"sse": error, "mean_sse": mean, "zero_sse": zero}
        if mean > 1e-12:
            values["relative_to_mean"] = error / mean
        if zero > 1e-12:
            values["relative_to_zero"] = error / zero
        metric_error = max(
            metric_error, *(abs(v - getattr(r, k)) for k, v in values.items())
        )
    for r in summary.itertuples():
        value = float(np.sum((fresh[r.model] - y) ** 2))
        metric_error = max(
            metric_error,
            abs(value - r.sse),
            abs(value / np.sum((fresh["mean"] - y) ** 2) - r.relative_to_mean),
        )
    saved = pd.read_csv(result / "permutations.csv").set_index("mapping")
    unique = sorted(set(groups))
    permutation_error, tail = 0.0, 0
    observed = float(np.sum((fresh["vector_ridge"] - y) ** 2))
    if len(saved) != 24 or saved.index.duplicated().any():
        raise ValueError("Permutation count/uniqueness failed")
    for p in itertools.permutations(range(4)):
        lookup = dict(zip(unique, [unique[i] for i in p]))
        permuted = np.array([y[index[lookup[g], n]] for g, n in zip(groups, names)])
        pred = independent_oof(x, permuted, groups, names, "vector_ridge")
        error = float(np.sum((pred - permuted) ** 2))
        permutation_error = max(
            permutation_error,
            abs(error - saved.loc["|".join(unique[i] for i in p), "sse"]),
        )
        tail += int(error <= observed + 1e-10)
    if tail / 24 != status["permutation_tail_fraction"]:
        raise ValueError("Permutation tail differs")
    centered = x - x.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered @ centered.T)[::-1]
    spectrum_error = float(
        abs(eigenvalues - np.array(status["x_centered_singular_values"]) ** 2).max()
    )
    if (
        max(
            source_error,
            prediction_error,
            metric_error,
            permutation_error,
            spectrum_error,
        )
        > 1e-8
    ):
        raise ValueError("Independent numerical replay failed")
    return {
        "status": "VERIFIED",
        "new_neural_calls": 0,
        "source_array_max": source_error,
        "sklearn_oof_max": prediction_error,
        "metric_replay_max": metric_error,
        "permutation_replay_max": permutation_error,
        "gram_spectrum_max": spectrum_error,
        "permutation_tail_fraction": tail / 24,
        "interpretation": "Four authored settings; code/criterion response prediction, not human accuracy or unique binding subspace.",
    }


def main():
    """Write a separate independent audit receipt and source copy."""
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("result-dir", "parent-dir", "output-dir"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    result = audit(a.result_dir, a.parent_dir)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    result["source_sha256"] = sha(Path(__file__))
    (a.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
