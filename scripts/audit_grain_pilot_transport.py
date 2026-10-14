#!/usr/bin/env python3
"""Independent covariance scoring, text address and partition checks for M11/M12."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_grain_query_moment_operator import read, sha, write


def independent_scores(y, z, u, t, fit, targets):
    """Direct joint-normal solve, independent of production posterior/kernels."""
    x = np.c_[np.ones(len(t)), z, u]
    v = np.asarray(fit["covariance"])
    state = fit["stationary"] * np.exp(-fit["rate"] * abs(t[:, None] - t))
    c = block_diag(v, state)
    h = np.c_[x, np.eye(len(t))]
    m = np.r_[fit["mean"], np.zeros(len(t))]
    inv = np.linalg.inv(h @ c @ h.T + fit["noise"] * np.eye(len(t)))
    k = c @ h.T @ inv
    estimated = m + (y - h @ m) @ k.T
    covariance = c - k @ h @ c
    gap = np.diff(t)
    w = np.r_[gap[0] / 2, (gap[:-1] + gap[1:]) / 2, gap[-1] / 2] / (t[-1] - t[0])
    rows, variance = [], []
    for target in targets:
        l = np.zeros(len(m))
        if target in ("level", "condition", "partner"):
            l[("level", "condition", "partner").index(target)] = 1
        elif target in ("current_state", "current_response"):
            l[-1] = 1
            if target == "current_response":
                l[:3] = [1, z[-1], u[-1]]
        elif target == "window_state":
            l[3:] = w
        elif target == "conditional_profile":
            l[:3] = [1, 0.5, 0]
        else:
            raise ValueError(target)
        rows.append(estimated @ l)
        variance.append(l @ covariance @ l)
    return np.stack(rows, axis=1), np.broadcast_to(variance, (len(y), len(targets)))


def inverse_meter(y, z, fitted):
    """Rebuild a saved same-unit map without fitting or accessing test truth."""
    a = np.asarray(fitted["coef"])
    offset, gain = a[:2]
    if fitted["conditional"]:
        offset, gain = offset + a[2] * z, gain + a[3] * z
    return (y - offset) / gain


def audit_gaussian(out, cfg):
    """Recompute all learned non-oracle scores and bootstrap variance accounting."""
    folder = out / "gaussian"
    fits = read(folder / "fits.json")
    predictions = pd.read_csv(folder / "predictions.csv")
    error, checked = 0.0, 0
    for seed in cfg["seeds"]:
        source = np.load(folder / f"source_{seed}.npz")
        z, u, t = (source[k] for k in ("condition", "partner", "times"))
        for world in cfg["worlds"]:
            key = f"{seed}_{world}"
            data = np.load(folder / f"evaluation_{key}.npz")
            y, fit = data["observations"].mean(axis=2), fits[key]
            arms = {
                "source_raw": (y, fits[f"source_{seed}"]),
                "source_anchor": (
                    inverse_meter(y, z, fit["global_map"]),
                    fits[f"source_{seed}"],
                ),
                "pilot_global": (
                    inverse_meter(y, z, fit["global_map"]),
                    fit["global_fit"],
                ),
                "pilot_conditional": (
                    inverse_meter(y, z, fit["conditional_map"]),
                    fit["local_fit"],
                ),
            }
            for arm, (values, fitted) in arms.items():
                p, v = independent_scores(values, z, u, t, fitted, cfg["targets"])
                part = predictions[
                    (predictions.seed == seed)
                    & (predictions.world == world)
                    & (predictions.arm == arm)
                ]
                for j, target in enumerate(cfg["targets"]):
                    item = part[part.target == target].sort_values("author")
                    error = max(
                        error,
                        float(np.max(abs(item.score.to_numpy() - p[:, j]))),
                        float(np.max(abs(item.variance.to_numpy() - v[:, j]))),
                    )
                    checked += 2 * len(item)
            boot = np.load(folder / f"bootstrap_scores_{key}.npz")
            pm, vw = boot["scores"].mean(axis=0), boot["variances"].mean(axis=0)
            vb = boot["scores"].var(axis=0, ddof=1)
            for j, target in enumerate(cfg["targets"]):
                item = predictions[
                    (predictions.seed == seed)
                    & (predictions.world == world)
                    & (predictions.arm == "pilot_mixture")
                    & (predictions.target == target)
                ].sort_values("author")
                for column, value in (
                    ("score", pm[:, j]),
                    ("variance", (vw + vb)[:, j]),
                    ("within_variance", vw[:, j]),
                    ("construction_variance", vb[:, j]),
                ):
                    error = max(
                        error, float(np.max(abs(item[column].to_numpy() - value)))
                    )
                    checked += len(item)
    if error > 1e-8:
        raise AssertionError(f"Score replay error {error}")
    return {"checked_values": checked, "maximum_error": error}


def audit_text(out):
    """Check each public-case pointer and independently rebuild two-cell scores."""
    folder = out / "text"
    bank = pd.read_csv(
        ROOT
        / "results/grain_vector_mechanisms/mechanism_todo_v1/condition_accumulation_r1/public_scored_bank.csv"
    )
    lookup = bank.set_index("case_id").evidence_hat.to_dict()
    count, error = 0, 0.0
    for path in folder.glob("*.npz"):
        if path.name.startswith("._"):
            continue
        data = np.load(path)
        values = np.array([lookup[a] for a in data["addresses"].ravel()]).reshape(
            data["observations"].shape
        )
        error = max(error, float(np.max(abs(values - data["observations"]))))
        count += values.size
    predictions = pd.read_csv(folder / "predictions.csv")
    fits = read(folder / "fits.json")
    nscore = 0
    for (query, seed, world), part in predictions.groupby(["query", "seed", "world"]):
        key = f"{query}_{seed}"
        data = np.load(folder / f"evaluation_{key}_{world}.npz")
        mean = data["observations"][:, :, :4].mean(axis=2)
        for arm, fitted in (
            ("source", fits[f"source_{key}"]),
            ("local_pilot", fits[f"{key}_{world}"]),
        ):
            v, noise, prior = (
                np.asarray(fitted[k]) for k in ("covariance", "noise", "mean")
            )
            k = v @ np.linalg.inv(v + np.diag(noise / 4))
            score = prior + (mean - prior) @ k.T
            cov = v - k @ v
            for target, weights in (
                ("level", [0.5, 0.5]),
                ("response", [-0.5, 0.5]),
                ("positive_condition", [0, 1]),
            ):
                weights = np.asarray(weights)
                item = part[(part.arm == arm) & (part.target == target)].sort_values(
                    "author"
                )
                error = max(
                    error,
                    float(np.max(abs(item.score.to_numpy() - score @ weights))),
                    float(
                        np.max(abs(item.variance.to_numpy() - weights @ cov @ weights))
                    ),
                )
                nscore += len(item) * 2
    if error > 1e-8:
        raise AssertionError(f"Text replay error {error}")
    return {"case_addresses": count, "checked_scores": nscore, "maximum_error": error}


def partitions_and_metrics(out):
    """Verify source/pilot/evaluation identity separation and every metric row."""
    rows_checked, discrepancy = 0, 0.0
    for stage in ("gaussian", "text"):
        folder = out / stage
        ids = pd.read_csv(folder / "partitions.csv")
        if ids.id.duplicated().any():
            raise AssertionError("Synthetic source/pilot/test identities overlap")
        for left, right in (
            ("source", "pilot"),
            ("source", "evaluation"),
            ("pilot", "evaluation"),
        ):
            if set(ids[ids.partition == left].id) & set(ids[ids.partition == right].id):
                raise AssertionError("Partition overlap")
        data = pd.read_csv(folder / "predictions.csv")
        for filename in ("metrics.csv", "metrics_by_seed.csv"):
            table = pd.read_csv(folder / filename)
            keys = [
                k for k in ("world", "query", "seed", "target", "arm") if k in table
            ]
            for row in table.to_dict("records"):
                mask = np.ones(len(data), bool)
                for key in keys:
                    mask &= data[key].to_numpy() == row[key]
                item = data[mask]
                if len(item) != row["n"]:
                    raise AssertionError("Metric sample count mismatch")
                e = item.score.to_numpy() - item.truth.to_numpy()
                expected = {
                    "rmse": np.sqrt(np.mean(e**2)),
                    "bias": np.mean(e),
                    "coverage95": np.mean(
                        abs(e) <= 1.959963984540054 * np.sqrt(item.variance)
                    ),
                    "mean_variance": item.variance.mean(),
                    "within_variance": item.within_variance.mean(),
                    "construction_variance": item.construction_variance.mean(),
                }
                discrepancy = max(
                    discrepancy, max(abs(row[k] - v) for k, v in expected.items())
                )
                rows_checked += 1
    if discrepancy > 1e-8:
        raise AssertionError("Metric mismatch")
    natural = pd.read_csv(out / "natural/private_partitions.csv")
    if natural.author_id.duplicated().any():
        raise AssertionError("Natural source/test author overlap")
    scores = pd.read_csv(out / "natural/private_relative_positions.csv")
    if set(scores.author_id) & set(natural[natural.partition == "source"].author_id):
        raise AssertionError("Reference authors scored as held out")
    return {
        "metric_rows": rows_checked,
        "maximum_error": discrepancy,
        "natural_source": int((natural.partition == "source").sum()),
        "natural_evaluation": int((natural.partition == "evaluation").sum()),
    }


def comparisons(out):
    """Paired error changes conditional on the three realized pilots, not power gates."""
    rows = []
    rng = np.random.default_rng(91390)
    arms = {
        "gaussian": [
            ("source_raw", "source_anchor"),
            ("source_anchor", "pilot_global"),
            ("pilot_global", "pilot_conditional"),
            ("source_raw", "pilot_mixture"),
        ],
        "text": [("source", "local_pilot"), ("local_pilot", "pilot_mixture")],
    }
    for stage, pairs in arms.items():
        data = pd.read_csv(out / stage / "predictions.csv")
        keys = [k for k in ("world", "query", "target") if k in data]
        for values, part in data.groupby(keys):
            for old, new in pairs:
                left = part[part.arm == old].set_index(["seed", "author"])
                right = (
                    part[part.arm == new].set_index(["seed", "author"]).loc[left.index]
                )
                e0, e1 = (
                    (left.score - left.truth).to_numpy() ** 2,
                    (right.score - right.truth).to_numpy() ** 2,
                )
                blocks = [
                    np.flatnonzero(left.index.get_level_values("seed") == seed)
                    for seed in sorted(data.seed.unique())
                ]
                bootstrap = []
                for _ in range(999):
                    index = np.concatenate(
                        [rng.choice(b, size=len(b), replace=True) for b in blocks]
                    )
                    bootstrap.append(
                        np.sqrt(e0[index].mean()) - np.sqrt(e1[index].mean())
                    )
                rows.append(
                    {
                        "stage": stage,
                        **dict(zip(keys, values)),
                        "old": old,
                        "new": new,
                        "n": len(left),
                        "delta_rmse": np.sqrt(e0.mean()) - np.sqrt(e1.mean()),
                        "lower95": np.quantile(bootstrap, 0.025),
                        "upper95": np.quantile(bootstrap, 0.975),
                        "positive_seed_count": sum(
                            np.sqrt(e0[b].mean()) > np.sqrt(e1[b].mean())
                            for b in blocks
                        ),
                        "scope": "Author bootstrap conditional on realized independent pilot fits",
                    }
                )
    pd.DataFrame(rows).to_csv(out / "paired_comparisons.csv", index=False)


def main():
    """Audit current artifacts, record numerical limitations without hiding them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    cfg, contract = read(out / "config.json"), read(out / "contract.json")
    for path, digest in contract["files"].items():
        if sha(ROOT / path) != digest:
            raise AssertionError(f"Input changed: {path}")
    with threadpool_limits(limits=1):
        gaussian = audit_gaussian(out, cfg)
        text = audit_text(out)
        numerical = partitions_and_metrics(out)
        comparisons(out)
    summary = read(out / "gaussian/summary.json")
    result = {
        "status": "PASS_WITH_RECORDED_FIT_LIMITS",
        "gaussian": gaussian,
        "text": text,
        "partitions_metrics": numerical,
        "bootstrap_nonconverged": summary["optimizer_nonconverged"],
        "models_at_bounds": summary["models_at_bounds"],
        "nonconverged_handling": "Retained as frozen finite-iteration bootstrap fits, not declared exact optima; no test-selected retries",
        "source_parameters_learned": True,
        "source_parameters_supplied_truth": False,
        "human_labels": False,
        "meps_access": False,
    }
    write(out / "audit.json", result)
    paths = sorted(
        p
        for p in out.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.name != "delivery_manifest.json"
    )
    write(
        out / "delivery_manifest.json", {str(p.relative_to(out)): sha(p) for p in paths}
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
