#!/usr/bin/env python3
"""Replay HM95 with edge-null projection and off-diagonal covariance formulas."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_grain_conditional_current_hm95 as R


def alt_lift(counts, refs):
    """Project oriented edges with an incidence nullspace, not node potentials."""
    ns = counts.shape[-1]
    edges = list(itertools.combinations(range(ns), 2))
    incidence = np.zeros((ns, len(edges)))
    for k, (i, j) in enumerate(edges):
        incidence[i, k], incidence[j, k] = 1, -1
    projection = np.eye(len(edges)) - np.linalg.pinv(incidence) @ incidence
    total = counts.sum((2, 3, 4, 5))
    raw = counts / total[..., None, None, None, None]
    q = raw.sum((-2, -1))
    raw = np.broadcast_to(
        raw[:, :, None], (*counts.shape[:2], len(refs), *counts.shape[2:])
    )
    opportunity = q[:, :, None, :, :, None, None] * refs[None, None]
    laws = np.stack((raw, opportunity, raw - opportunity), 3)
    signed = np.stack([(laws[..., i, j] - laws[..., j, i]) / 2 for i, j in edges], -1)
    projected = signed @ projection.T
    cell = np.zeros(laws.shape)
    for k, (i, j) in enumerate(edges):
        cell[..., i, j], cell[..., j, i] = projected[..., k], -projected[..., k]
    cell = cell.reshape(*cell.shape[:4], -1, ns, ns)
    total_cell = cell.sum(-3)
    hidden = cell - total_cell[..., None, :, :] / cell.shape[-3]
    return {
        "resolved": cell.reshape(*cell.shape[:4], -1),
        "visible": (total_cell / np.sqrt(cell.shape[-3])).reshape(*cell.shape[:4], -1),
        "hidden": hidden.reshape(*hidden.shape[:4], -1),
    }


def alt_covariance(x, strata):
    """Subtract all distinct-author cross-pairs from same-author cross-pairs."""
    result = np.zeros(x.shape[2:-1])
    for group in np.unique(strata):
        block = x[strata == group]
        n = len(block)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            products = (block[:, a] * block[:, b]).sum(-1).sum(0)
            all_pairs = (block[:, a].sum(0) * block[:, b].sum(0)).sum(-1)
            off_diagonal = (all_pairs - products) / (n * (n - 1))
            result += n / len(x) * (products / n - off_diagonal) / 3
    return result


def alt_counts(states, authors, views, conditions, n_authors):
    """Rebuild each edge-position histogram by encoded-address bincounts."""
    parts = []
    for r in range(3):
        address = (
            ((authors * 3 + views) * 6 + conditions) * 4 + states[:, r]
        ) * 4 + states[:, r + 1]
        parts.append(
            np.bincount(address, minlength=n_authors * 3 * 6 * 16).reshape(
                n_authors, 3, 6, 4, 4
            )
        )
    return np.stack(parts, axis=3)


def verify(out):
    """Check frozen inputs, all null draws, all target actions and Fisher accounts."""
    cfg = R.verify(out)
    for stage in ("01", "02", "03"):
        R.R.previous(out, stage)
    if (out / "verification.json").exists():
        raise ValueError("Keep prior verification")
    source = ROOT / cfg["natural_source"]
    split = R.R.read(source / "01/partition.json")
    if set(split["source"]) & set(split["application"]):
        raise AssertionError("Author overlap")
    metadata = R.R.read(source / "02/private_metadata.json")
    with np.load(source / "02/private_author_accounts.npz", allow_pickle=False) as data:
        states = np.concatenate([data[row["key"] + "_states"] for row in metadata])
    with np.load(out / "01/source_bundle.npz", allow_pickle=False) as data:
        bundle = {key: data[key] for key in data.files}
    with np.load(out / "02/private_features.npz", allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    with np.load(out / "03/null_and_covariance.npz", allow_pickle=False) as data:
        evaluation = {key: data[key] for key in data.files}
    error, checks = 0.0, 0

    def compare(a, b):
        """Track all checked scalars; stop on a genuine numerical discrepancy."""
        nonlocal error, checks
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError("Shape/nonfinite verification failure")
        error = max(error, float(np.max(abs(a - b))))
        checks += a.size
        if error > 1e-10:
            raise AssertionError(f"Alternate arithmetic failed: {error}")

    refs = bundle["references"]
    with np.load(source / "01/source_model.npz", allow_pickle=False) as data:
        totals = data["source_counts"].sum(0)
        compare(refs, np.stack([data["reference_" + arm] for arm in cfg["arms"]]))
    alpha = cfg["source_markov_pseudocount"]
    initial = totals[:, 0].sum(-1) + alpha
    initial /= initial.sum(1, keepdims=True)
    transitions = totals + alpha
    transitions /= transitions.sum(-1, keepdims=True)
    patterns = np.array(list(itertools.product(range(4), repeat=4)))
    probability = np.zeros((6, 256))
    pairs = np.zeros((6, 3, 4, 4))
    for z, k in itertools.product(range(6), range(256)):
        seq = patterns[k]
        probability[z, k] = initial[z, seq[0]] * np.prod(
            [transitions[z, r, seq[r], seq[r + 1]] for r in range(3)]
        )
        for r in range(3):
            pairs[z, r, seq[r], seq[r + 1]] += probability[z, k]
    for name, value in (
        ("initial", initial),
        ("transitions", transitions),
        ("patterns", patterns),
        ("probabilities", probability),
        ("pair_laws", pairs),
    ):
        compare(value, bundle[name])
    authors, views, conditions, strata = (
        saved[k] for k in ("authors", "views", "conditions", "strata")
    )
    counts = alt_counts(states, authors, views, conditions, len(strata))
    compare(counts, saved["counts"])
    feature = alt_lift(counts, refs)
    expected = alt_lift(counts.sum((-2, -1))[..., None, None] * pairs[None, None], refs)
    for key in R.SPACES:
        compare(feature[key], saved[key])
        compare(expected[key], saved["expected_" + key])
    observed = np.stack([alt_covariance(feature[k], strata) for k in R.SPACES], -1)
    common_mean = np.stack([alt_covariance(expected[k], strata) for k in R.SPACES], -1)
    compare(observed, evaluation["observed"])
    compare(common_mean, evaluation["expected"])
    compare(observed[..., 0], observed[..., 1] + observed[..., 2])
    compare(
        (feature["resolved"] ** 2).sum(-1),
        (feature["visible"] ** 2).sum(-1) + (feature["hidden"] ** 2).sum(-1),
    )
    cdf = probability.cumsum(1)
    cdf[:, -1] = 1
    rng = np.random.default_rng(cfg["seed"])
    for draw in range(cfg["ghost_draws"]):
        random = rng.random(len(conditions))
        indices = np.zeros(len(conditions), int)
        for z in range(6):
            selected = conditions == z
            indices[selected] = np.searchsorted(cdf[z], random[selected], side="right")
        ghost_counts = alt_counts(
            patterns[indices], authors, views, conditions, len(strata)
        )
        compare(ghost_counts.sum((-2, -1)), counts.sum((-2, -1)))
        ghost_feature = alt_lift(ghost_counts, refs)
        ghost_value = np.stack(
            [alt_covariance(ghost_feature[k], strata) for k in R.SPACES], -1
        )
        compare(ghost_value, evaluation["ghost"][draw])
        compare(ghost_value[:, 1], observed[:, 1])
        if draw % 100 == 0:
            print("HM95 alternate null replay", draw + 1, flush=True)
    table = pd.read_csv(out / "03/covariance_accounts.csv").set_index(
        ["arm", "channel", "space"]
    )
    for a, c, s in itertools.product(range(4), range(3), range(3)):
        row = table.loc[cfg["arms"][a], R.CHANNELS[c], R.SPACES[s]]
        null, value = evaluation["ghost"][:, a, c, s], observed[a, c, s]
        compare(
            row[
                [
                    "covariance",
                    "source_common_expectation",
                    "delta_over_source_common",
                    "ghost_mean",
                    "ghost_lower",
                    "ghost_upper",
                    "ghost_p",
                ]
            ].to_numpy(float),
            [
                value,
                common_mean[a, c, s],
                value - common_mean[a, c, s],
                null.mean(),
                np.quantile(null, 0.025),
                np.quantile(null, 0.975),
                (1 + np.count_nonzero(null >= value - R.TOL)) / (len(null) + 1),
            ],
        )
    ai, ci = (
        cfg["arms"].index("joint"),
        R.CHANNELS.index("response"),
    )
    rng = np.random.default_rng(cfg["seed"] + 1)
    for draw in range(cfg["bootstrap_draws"]):
        indices = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(strata == g), np.sum(strata == g), replace=True
                )
                for g in np.unique(strata)
            ]
        )
        score = alt_covariance(
            feature["hidden"][indices, :, ai, ci], strata[indices]
        ) - alt_covariance(expected["hidden"][indices, :, ai, ci], strata[indices])
        compare(score, evaluation["primary_bootstrap"][draw])
    compare(
        np.quantile(evaluation["primary_bootstrap"], (0.025, 0.975)),
        R.R.read(out / "03/summary.json")["primary_delta_author_CI"],
    )
    paths = R.R.read(out / "00/application_paths.json")
    with np.load(out / "02/toy_laws.npz", allow_pickle=False) as data:
        laws = data["laws"]
    scores = pd.read_csv(out / "02/toy_scores.csv").set_index(
        ["author", "family", "field", "target"]
    )
    names = R.R.read(ROOT / cfg["vector_source"] / "00/manifest.json")["spec"][
        "targets"
    ]
    pooled_first = None
    for p, record in enumerate(paths):
        local = np.zeros((4, 3, 3))
        for c, seq in enumerate(record["paths"]):
            for i, j in zip(seq, seq[1:] + seq[:1], strict=True):
                local[c, i, j] += 1 / (4 * len(seq))
        compare(local, laws[p])
        pooled = local.sum(0)
        if pooled_first is None:
            pooled_first = pooled
        compare(pooled, pooled_first)
        for f, h, t in itertools.product(range(6), range(3), range(6)):
            value = sum(
                local[c, i, j] * bundle["kernel"][c, f, i, j, h, t]
                for c, i, j in itertools.product(range(4), range(3), range(3))
            )
            flat = sum(
                pooled[i, j]
                * sum(bundle["kernel"][c, f, i, j, h, t] for c in range(4))
                / 4
                for i, j in itertools.product(range(3), range(3))
            )
            reset = sum(
                local[c, i, j] * bundle["reset"][c, f, j, h, t]
                for c, i, j in itertools.product(range(4), range(3), range(3))
            )
            row = scores.loc[
                record["author"],
                names[f],
                ("uniform_value", "local_query", "local_bound_query")[h],
                names[t],
            ]
            compare(
                row[["conditional", "pooled", "reset"]].to_numpy(float),
                [value, flat, reset],
            )
    truth = pd.DataFrame(R.R.read(out / "00/evaluation_truth.json"))
    joined = scores.reset_index().merge(truth, on="author", validate="many_to_one")
    contrasts = []
    for group, rows in joined.groupby(["pattern", "family", "field", "target"]):
        plus = (
            rows[rows.sign == 1][["conditional", "pooled", "reset"]].mean().to_numpy()
        )
        minus = (
            rows[rows.sign == -1][["conditional", "pooled", "reset"]].mean().to_numpy()
        )
        contrasts.append((*group, *(plus - minus)))
    alternate = pd.DataFrame(
        contrasts,
        columns=[
            "pattern",
            "family",
            "field",
            "target",
            "conditional",
            "pooled",
            "reset",
        ],
    ).set_index(["pattern", "family", "field", "target"])
    actual = pd.read_csv(out / "03/conditional_target_actions.csv").set_index(
        alternate.index.names
    )
    compare(alternate.to_numpy(), actual.loc[alternate.index].to_numpy())
    with np.load(out / "03/fisher_witness.npz", allow_pickle=False) as data:
        base, tangents = data["base"], data["tangents"]
        full = np.array(
            [
                [
                    sum(
                        tangents[a, c, i, j] * tangents[b, c, i, j] / base[c, i, j]
                        for c, i, j in itertools.product(range(4), range(3), range(3))
                    )
                    for b in range(3)
                ]
                for a in range(3)
            ]
        )
        compare(full, data["full"])
        compare(full, 0.24 * np.eye(3))
        compare(tangents.sum(1), np.zeros((3, 3, 3)))
        compare(data["pooled"], np.zeros((3, 3)))
        if (base + tangents[0]).min() <= 0:
            raise AssertionError("Invalid finite probability support")
    result = {
        "status": "VERIFIED_FINITE_HM95_NOT_GLOBAL",
        "maximum_alternate_error": error,
        "alternate_scalar_checks": checks,
        "null_draws_replayed": cfg["ghost_draws"],
        "bootstrap_draws_replayed": cfg["bootstrap_draws"],
        "actual_toy_scores_checked": len(scores),
        "same_agent_not_third_party": True,
        "new_neural_calls": 0,
        "original_scope_complete": False,
    }
    R.R.write(out / "verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    verify(args.out or ROOT / R.R.read(R.CFG)["output"])
