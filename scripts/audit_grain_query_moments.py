#!/usr/bin/env python3
"""Independent arithmetic, raw replay and frozen external joins for M05."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.data_views import stable_key
from scripts.run_grain_query_moment_operator import (
    ARMS,
    NATIVE,
    QUERIES,
    arrays,
    current,
    directions,
    load_models,
    read,
    sha,
    verify,
    write,
)


def label_namespace(labels):
    """Use the existing cohort pseudonyms, rejecting ambiguous author joins."""
    labels = labels.copy()
    labels.index = [stable_key(str(uid), ID_SEED)[:24] for uid in labels.index]
    if not labels.index.is_unique:
        raise ValueError("Duplicate or colliding label pseudonyms")
    return labels


def direct(g, h, model):
    """Explicit scalar/cell loops independent of the application's implementation."""
    n = len(h)
    z = (g - model["g_mean"]) / model["g_scale"]
    x1 = np.c_[np.ones(n), z]
    x2 = np.c_[x1, *[z[:, i] * z[:, j] for i in range(4) for j in range(i, 4)]]
    a1, a2 = x1 @ model["beta1"] / n, x2 @ model["beta2"] / n
    residual = h.copy()
    _, index = np.unique(g, axis=0, return_inverse=True)
    for cell in np.unique(index):
        mask = index == cell
        residual[mask] -= h[mask].mean(axis=0)
    r = residual @ model["query_basis"]
    value = np.trace((r.T @ r / n) @ model["within_matrix"])
    field = 2 / n * r @ model["within_matrix"] @ model["query_basis"].T
    scores = (
        np.array([np.sum(a1 * h), np.sum(a2 * h), np.sum(a2 * h) + value])
        + model["offsets"]
    )
    return scores, np.array([a1, a2, a2 + field])


def arithmetic(out):
    """Reconstruct every scalar/directional prediction from frozen current inputs."""
    contract, models = verify(out), load_models(out)
    pred = pd.read_csv(out / "application/private_predictions.csv")
    error = 0.0
    for job in contract["jobs"]:
        data = current(job)
        ds = directions(data, models)
        for query, model in models.items():
            value, field = direct(data["g"], data["h"], model)
            sub = pred[(pred.author_key == job["key"]) & (pred["query"] == query)]
            for row in sub.itertuples():
                ai = ARMS.index(row.arm)
                error = max(
                    error,
                    abs(value[ai] - row.score),
                    abs(np.sum(field[ai] * ds[row.direction]) - row.effect),
                )
    if error > 1e-9:
        raise ArithmeticError("Independent scalar/field replay failed")
    return error


def comparisons(out):
    """Paired author bootstrap, keeping all directions of an author together."""
    frame = pd.read_csv(out / "evaluation/private_comparison.csv")
    rng, rows = np.random.default_rng(91359), []
    for (cohort, query), data in frame.groupby(["cohort", "query"]):
        authors = sorted(data.author_key.unique())
        for scope in ("old4", "within2", "random"):
            subset = (
                data[data.direction.str.startswith("F")]
                if scope == "old4"
                else data[data.direction.isin(["within_" + q for q in QUERIES])]
                if scope == "within2"
                else data[data.direction == "within_random"]
            )
            old, new = [], []
            for author in authors:
                f = subset[subset.author_key == author]
                old.append(
                    np.mean(
                        (
                            f[f.arm == "quadratic"].effect
                            - f[f.arm == "quadratic"].native_effect
                        )
                        ** 2
                    )
                )
                new.append(
                    np.mean(
                        (
                            f[f.arm == "within_augmented"].effect
                            - f[f.arm == "within_augmented"].native_effect
                        )
                        ** 2
                    )
                )
            old, new = np.array(old), np.array(new)
            draws = rng.integers(0, len(authors), size=(999, len(authors)))
            delta = np.sqrt(old[draws].mean(1)) - np.sqrt(new[draws].mean(1))
            rows.append(
                {
                    "cohort": cohort,
                    "query": query,
                    "scope": scope,
                    "authors": len(authors),
                    "rmse_reduction": np.sqrt(old.mean()) - np.sqrt(new.mean()),
                    "lower": np.quantile(delta, 0.025),
                    "upper": np.quantile(delta, 0.975),
                    "positive_authors": int((old > new).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(out / "evaluation/paired_comparisons.csv", index=False)


def cold_replay(out):
    """Compare recomputed raw material coordinates after cold predictions are frozen."""
    frozen = read(out / "cold/summary.json")
    for name, expected in frozen["files"].items():
        if sha(out / "cold" / name) != expected:
            raise ValueError("Cold coordinates changed")
    original = current(
        next(
            j
            for j in read(out / "contract.json")["jobs"]
            if j["cohort"] == "new8" and j["index"] == 0
        )
    )
    replay = arrays(out / "cold/reused_author0.npz")
    errors = {
        name: float(np.max(abs(original[name] - replay[name]))) for name in ("g", "h")
    }
    if errors["g"] > 1e-10 or errors["h"] > 1e-4:
        raise ArithmeticError("Raw replay changed frozen material coordinates")
    if frozen["prefix_calls"] != 2 or frozen["denied_cache_probes"] != 2:
        raise ValueError("Cold call/cache contract differs")
    for trace in frozen["traces"]:
        if trace["completed_blocks"] != list(range(17)) or trace["forbidden_calls"]:
            raise ValueError("Forbidden cold suffix/logit computation")
    return errors


def external_join(out):
    """Inspect available Big5 links only AFTER immutable scores, without fitting.

    This small historical panel is an interface diagnostic. Every trait is retained;
    no name, sign, rank or model is selected from these exploratory correlations.
    """
    label_path = Path(
        "/Volumes/mobile3/projects/project persona/data_sets/prepared/pandora_official/pandora_official_big5_prepared.csv"
    )
    traits = [
        "Extraversion",
        "Neuroticism",
        "Agreeableness",
        "Conscientiousness",
        "Openness",
    ]
    labels = pd.read_csv(
        label_path, usecols=["user_id", *traits], dtype={"user_id": str}
    ).set_index("user_id")
    labels = label_namespace(labels)
    jobs = read(NATIVE / "private_jobs.json")
    lookup = {}
    for job in jobs:
        key = f"{job['cohort']}_a{job['index']:02d}"
        directory = (ROOT / job["old_gradient"]).parent.parent
        meta = next(
            r
            for r in read(directory / "private_design.json")
            if r["index"] == job["index"]
        )
        lookup[key] = str(meta["author_id"])
    predictions = pd.read_csv(
        out / "application/private_predictions.csv"
    ).drop_duplicates(["author_key", "query", "arm"])
    rows, linkage = [], []
    for (cohort, query, arm), group in predictions.groupby(["cohort", "query", "arm"]):
        for trait in traits:
            pairs = [
                (r.score, labels.loc[lookup[r.author_key], trait])
                for r in group.itertuples()
                if lookup[r.author_key] in labels.index
            ]
            pairs = np.asarray(pairs).reshape(-1, 2)
            pairs = pairs[np.isfinite(pairs).all(1)]
            r = (
                float(np.corrcoef(pairs.T)[0, 1])
                if len(pairs) >= 3 and min(pairs.std(0)) > 0
                else np.nan
            )
            rows.append(
                {
                    "cohort": cohort,
                    "query": query,
                    "arm": arm,
                    "criterion": trait,
                    "n": len(pairs),
                    "pearson": r,
                    "status": "small_reused_panel_no_selection",
                }
            )
    for key, uid in lookup.items():
        linkage.append({"author_key": key, "has_big5": uid in labels.index})
    pd.DataFrame(rows).to_csv(out / "evaluation/external_links.csv", index=False)
    write(
        out / "evaluation/external_join.json",
        {
            "label_sha256": sha(label_path),
            "id_namespace": "stable_key(raw_user_id, cohort.ID_SEED)[:24]",
            "rows": linkage,
            "fit": False,
            "cohort_separate": True,
            "confirmation_seeking_specific_scale_available": False,
        },
    )
    return {
        "matched_authors": sum(r["has_big5"] for r in linkage),
        "total_authors": len(linkage),
        "correlation_rows": len(rows),
    }


def main():
    """Run independent checks without modifying the compiled candidate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    for directory in (out / "application", out / "models"):
        for name, expected in read(directory / "freeze.json")["files"].items():
            if sha(directory / name) != expected:
                raise ValueError("Frozen artifact changed")
    comparisons(out)
    result = {
        "status": "pass",
        "max_arithmetic_error": arithmetic(out),
        "cold_replay": cold_replay(out),
        "external_linkage": external_join(out),
    }
    write(out / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
