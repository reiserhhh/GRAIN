#!/usr/bin/env python3
"""S1: quantify local function, common-query effects and two-edit nonadditivity."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_grain_functional_basis import author_folds
from scripts.analyze_grain_real_author_path import prediction_quality
from scripts.run_grain_multiquery_sensitivity import QUERIES
from scripts.run_grain_real_author_path import BASE, read, sha, write


def corr(x, y):
    """Finite Pearson correlation, returning None for a constant observable."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(x @ y / denom, -1, 1)) if denom > 1e-14 else None


def cluster_r_interval(x, y, groups, draws=1000):
    """Resample entire authors, retaining all paired edit/view observations."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    unique = sorted(set(groups))
    indices = [np.flatnonzero(groups == name) for name in unique]
    rng = np.random.default_rng(20260908)
    values = []
    for _ in range(draws):
        selected = np.concatenate(
            [indices[j] for j in rng.integers(len(unique), size=len(unique))]
        )
        r = corr(x[selected], y[selected])
        if r is not None:
            values.append(r)
    return {
        "r": corr(x, y),
        "r_lower": float(np.quantile(values, 0.025)) if values else None,
        "r_upper": float(np.quantile(values, 0.975)) if values else None,
    }


def load_panel(folder):
    """Read only hash-verified complete two-view author units, without labels."""
    manifest, material = (
        read(folder / "manifest.json"),
        read(folder / "private_material.json"),
    )
    if sha(folder / "private_material.json") != manifest["material_sha256"]:
        raise ValueError("Frozen material changed")
    records = []
    for path in sorted((folder / "private_packets").glob("panel_*.json")):
        packet = read(path)
        row = material[packet["material_row"]]
        if any(packet[k] != row[k] for k in ("author_id", "panel_index", "view")):
            raise ValueError("Incorrect author/view join")
        if packet["state_sha256"] != sha(path.with_suffix(".npz")) or set(
            packet["queries"]
        ) != set(QUERIES):
            raise ValueError("Incomplete/corrupted query packet")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            vectors = dict(z)
        records.append((row, packet, vectors))
    counts = pd.Series([row["author_id"] for row, _, _ in records]).value_counts()
    complete = set(counts[counts == 2].index)
    records = [item for item in records if item[0]["author_id"] in complete]
    if len(complete) < 3:
        raise ValueError("At least three complete authors needed")
    return manifest, records


def response_rows(records):
    """Keep natural Delta, single-position M and local g*d as separate observables."""
    rows, joint, attribution = [], [], []
    for material, packet, vectors in records:
        prefix = {
            k: material[k] for k in ("author_id", "view", "source_group", "panel_index")
        }
        for query in QUERIES:
            q = packet["queries"][query]
            scores = {
                name: item["conditional_score"] for name, item in q["readouts"].items()
            }
            h, g = (
                vectors[f"{query}__full"].astype(float),
                vectors[f"{query}__gradient"].astype(float),
            )
            for edit in ("e1", "e2"):
                d = vectors[f"{query}__{edit}"].astype(float) - h
                rows.append(
                    {
                        **prefix,
                        "query": query,
                        "edit": edit,
                        "full_score": scores["full"],
                        "natural": scores[edit] - scores["full"],
                        "patch": scores["patch_" + edit] - scores["full"],
                        "linear": float(g @ d),
                        "removed_state_linear": -float(
                            np.sum(np.asarray(q["comment_attribution"])[material[edit]])
                        ),
                        "query_plus_removed": float(g @ d)
                        - float(
                            np.sum(np.asarray(q["comment_attribution"])[material[edit]])
                        ),
                        "gradient_norm": float(np.linalg.norm(g)),
                        "displacement_norm": float(np.linalg.norm(d)),
                        "match_within_tolerance": material["matching"][
                            "within_tolerance"
                        ],
                    }
                )
            if "joint" in scores:
                d1, d2, d12 = [
                    vectors[f"{query}__{name}"].astype(float) - h
                    for name in ("e1", "e2", "joint")
                ]
                mixed = d12 - d1 - d2
                norm = float(np.linalg.norm(d12))
                joint.append(
                    {
                        **prefix,
                        "query": query,
                        "interaction": scores["joint"]
                        - scores["e1"]
                        - scores["e2"]
                        + scores["full"],
                        "joint_natural": scores["joint"] - scores["full"],
                        "mixed_state_norm": float(np.linalg.norm(mixed)),
                        "joint_state_norm": norm,
                        "mixed_state_relative": float(np.linalg.norm(mixed) / norm)
                        if norm > 1e-10
                        else None,
                        "mixed_state_gradient": float(g @ mixed),
                    }
                )
            for j, value in enumerate(q["comment_attribution"]):
                attribution.append(
                    {
                        **prefix,
                        "query": query,
                        "comment": j,
                        "attribution": value,
                        "tokens": q["body_token_counts"][j],
                        "in_e1": j in material["e1"],
                        "in_e2": j in material["e2"],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(joint), pd.DataFrame(attribution)


def paired_shared(rows):
    """Compare matched queries and remove common edit/view means descriptively."""
    results = []
    for signal in ("natural", "patch", "linear"):
        pivot = rows.pivot(
            index=["author_id", "view", "edit"], columns="query", values=signal
        ).sort_index()
        if pivot.isna().any().any():
            raise ValueError("Unpaired query responses")
        centered = pivot - pivot.groupby(level=["view", "edit"]).transform("mean")
        groups = pivot.index.get_level_values("author_id").to_numpy()
        for left, right in itertools.combinations(QUERIES, 2):
            for mode, data in (("raw", pivot), ("edit_view_centered", centered)):
                results.append(
                    {
                        "signal": signal,
                        "query_left": left,
                        "query_right": right,
                        "mode": mode,
                        **cluster_r_interval(data[left], data[right], groups),
                    }
                )
    return pd.DataFrame(results)


def frozen_transfer(records, frozen):
    """Evaluate previously frozen S0 coefficients on new authors and new edit rules."""
    if not frozen.exists():
        return []
    manifest = read(frozen / "manifest.json")
    results = []
    for spec in manifest["models"]:
        path = frozen / spec["file"]
        if sha(path) != spec["sha256"]:
            raise ValueError("Frozen S0 operator changed")
        with np.load(path, allow_pickle=False) as z:
            model = dict(z)
        x, y, groups = [], [], []
        query = spec["task"]
        for row, packet, vectors in records:
            q = packet["queries"][query]["readouts"]
            for edit in ("e1", "e2"):
                x.append(vectors[f"{query}__{edit}"] - vectors[f"{query}__full"])
                y.append(
                    q["patch_" + edit]["conditional_score"]
                    - q["full"]["conditional_score"]
                )
                groups.append(row["author_id"])
        features = np.asarray(x, float)
        if not bool(model["full"]):
            features = features @ model["basis"]
        p = (features / model["scale"]) @ model["coef"]
        results.append(
            {
                "query": query,
                "method": spec["method"],
                "authors": len(set(groups)),
                **prediction_quality(y, p),
                **cluster_r_interval(y, p, groups),
            }
        )
    return results


def balanced_projection(d, g, method, rank):
    """Return V,W with P=VW^T, using training-only uncentered state/gradient moments."""
    d, g = np.asarray(d, float), np.asarray(g, float)
    if d.shape != g.shape or d.ndim != 2:
        raise ValueError("Aligned state and gradient rows required")
    if method in ("variance", "gradient"):
        _, s, vt = np.linalg.svd(d if method == "variance" else g, full_matrices=False)
        k = min(rank, int(np.sum(s > max(s[0], 1.0) * 1e-10)))
        basis = vt[:k].T
        return basis, basis
    if method != "balanced":
        raise ValueError("Unknown paired functional projection")
    # R_d=D^T/sqrt(n), R_g=G^T/sqrt(n); W^T V=I on retained nonzero singular values.
    u, s, vt = np.linalg.svd(g @ d.T / len(d), full_matrices=False)
    k = min(rank, int(np.sum(s > max(s[0], 1e-10) * 1e-8)))
    factor = np.sqrt(len(d) * s[:k])
    return (d.T @ vt[:k].T) / factor, (g.T @ u[:, :k]) / factor


def functional_projections(records):
    """Measure paired g^T(I-P)d loss on held authors, not new projected-patch outcomes."""
    result = []
    for query in QUERIES:
        displacement, gradients, observed, groups = [], [], [], []
        for row, packet, vectors in records:
            scores = packet["queries"][query]["readouts"]
            for edit in ("e1", "e2"):
                displacement.append(
                    vectors[f"{query}__{edit}"] - vectors[f"{query}__full"]
                )
                gradients.append(vectors[f"{query}__gradient"])
                observed.append(
                    scores["patch_" + edit]["conditional_score"]
                    - scores["full"]["conditional_score"]
                )
                groups.append(row["author_id"])
        d, g, y = (
            np.asarray(displacement, float),
            np.asarray(gradients, float),
            np.asarray(observed),
        )
        full = np.sum(d * g, axis=1)
        folds = author_folds(groups, min(5, len(set(groups))), "s1_projection_cv")
        for method in ("variance", "gradient", "balanced"):
            for rank in (1, 2, 4, 8):
                p = np.zeros(len(d))
                for fold in sorted(set(folds)):
                    train = folds != fold
                    v, w = balanced_projection(d[train], g[train], method, rank)
                    p[~train] = np.sum((d[~train] @ w) * (g[~train] @ v), axis=1)
                result.append(
                    {
                        "query": query,
                        "method": method,
                        "rank": rank,
                        "authors": len(set(groups)),
                        "paired_loss": float(np.mean((full - p) ** 2)),
                        "paired_loss_to_zero": float(
                            np.mean((full - p) ** 2) / np.mean(full**2)
                        ),
                        **prediction_quality(y, p),
                    }
                )
    return pd.DataFrame(result)


def shared_author_spaces(rows):
    """Describe AJIVE-style joint author spaces at fixed block ranks, with permutation references."""
    summary, scores = [], []
    rng = np.random.default_rng(20260908)
    for signal in ("natural", "patch"):
        blocks = [
            rows[rows["query"] == q]
            .pivot(index="author_id", columns=["view", "edit"], values=signal)
            .sort_index()
            for q in QUERIES
        ]
        if any(not b.index.equals(blocks[0].index) for b in blocks):
            raise ValueError("Unpaired authors in multi-block geometry")
        centered = [b.to_numpy() - b.to_numpy().mean(0) for b in blocks]
        for block_rank in (1, 2):
            bases = [
                np.linalg.svd(b, full_matrices=False)[0][:, :block_rank]
                for b in centered
            ]
            u, singular, _ = np.linalg.svd(np.column_stack(bases), full_matrices=False)
            null = np.asarray(
                [
                    np.linalg.svd(
                        np.column_stack([b[rng.permutation(len(b))] for b in bases]),
                        compute_uv=False,
                    )[:2]
                    for _ in range(500)
                ]
            )
            for joint_rank in (1, 2):
                joint = u[:, :joint_rank]
                fractions = [
                    float(np.linalg.norm(joint.T @ b) ** 2 / np.linalg.norm(b) ** 2)
                    for b in centered
                ]
                summary.append(
                    {
                        "signal": signal,
                        "block_rank": block_rank,
                        "joint_rank": joint_rank,
                        "singular_value": float(singular[joint_rank - 1]),
                        "permuted_median": float(np.median(null[:, joint_rank - 1])),
                        "permuted_p": float(
                            (
                                1
                                + np.sum(
                                    null[:, joint_rank - 1] >= singular[joint_rank - 1]
                                )
                            )
                            / 501
                        ),
                        **{
                            q + "_projection_fraction": f
                            for q, f in zip(QUERIES, fractions, strict=True)
                        },
                    }
                )
            for j, author in enumerate(blocks[0].index):
                scores.append(
                    {
                        "author_id": author,
                        "signal": signal,
                        "block_rank": block_rank,
                        "J1": float(u[j, 0]),
                        "J2": float(u[j, 1]),
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(scores)


def evidence_shared(attribution):
    """Compare matched-comment attribution profiles with and without token-count normalization."""
    rows = []
    for (_, _), frame in attribution.groupby(["author_id", "view"]):
        for left, right in itertools.combinations(QUERIES, 2):
            a, b = [
                frame[frame["query"] == q].sort_values("comment") for q in (left, right)
            ]
            for mode in ("raw", "per_token"):
                x = a.attribution.to_numpy() / (
                    a.tokens.to_numpy() if mode == "per_token" else 1
                )
                y = b.attribution.to_numpy() / (
                    b.tokens.to_numpy() if mode == "per_token" else 1
                )
                rows.append(
                    {"left": left, "right": right, "mode": mode, "r": corr(x, y)}
                )
    return (
        pd.DataFrame(rows)
        .groupby(["left", "right", "mode"])
        .agg(mean_r=("r", "mean"), median_r=("r", "median"), views=("r", "count"))
        .reset_index()
    )


def main():
    """Write a bounded pilot/full analysis without touching acquisition files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frozen-s0",
        type=Path,
        default=BASE / "functional_basis_s0_r1/frozen_operators",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private analysis must remain in ignored results")
    out.mkdir(parents=True, exist_ok=False)
    _, records = load_panel(args.panel)
    rows, joint, attribution = response_rows(records)
    authors = rows.author_id.nunique()
    metrics, interactions = [], []
    for query, group in rows.groupby("query"):
        for target, predictor in (
            ("patch", "linear"),
            ("natural", "patch"),
            ("natural", "linear"),
            ("natural", "removed_state_linear"),
            ("natural", "query_plus_removed"),
        ):
            metrics.append(
                {
                    "query": query,
                    "target": target,
                    "predictor": predictor,
                    "authors": authors,
                    **prediction_quality(group[target], group[predictor]),
                    **cluster_r_interval(
                        group[target], group[predictor], group.author_id
                    ),
                }
            )
        if not joint.empty:
            j = joint[joint["query"] == query]
            rms = float(np.sqrt(np.mean(j.interaction**2)))
            natural_rms = float(np.sqrt(np.mean(j.joint_natural**2)))
            interactions.append(
                {
                    "query": query,
                    "authors": j.author_id.nunique(),
                    "interaction_mean": float(j.interaction.mean()),
                    "interaction_rms": rms,
                    "joint_response_rms": natural_rms,
                    "interaction_to_joint_rms": rms / natural_rms
                    if natural_rms > 1e-10
                    else None,
                    "authors_abs_interaction_above_1e_minus_5": int(
                        (j.interaction.abs() > 1e-5).sum()
                    ),
                    "median_state_nonadditivity": float(
                        j.mixed_state_relative.median()
                    ),
                    "gradient_mixed_vs_score_interaction_r": corr(
                        j.mixed_state_gradient, j.interaction
                    ),
                }
            )
    shared = paired_shared(rows)
    transfer = frozen_transfer(records, args.frozen_s0)
    with threadpool_limits(limits=2):
        projection = functional_projections(records)
        joint_space, joint_scores = shared_author_spaces(rows)
    evidence = evidence_shared(attribution)
    rows.to_csv(out / "private_responses.csv", index=False)
    joint.to_csv(out / "private_joint_responses.csv", index=False)
    attribution.to_csv(out / "private_comment_attribution.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "paired_response_metrics.csv", index=False)
    pd.DataFrame(interactions).to_csv(out / "interaction_metrics.csv", index=False)
    shared.to_csv(out / "shared_response_metrics.csv", index=False)
    pd.DataFrame(transfer).to_csv(out / "frozen_s0_transfer.csv", index=False)
    projection.to_csv(out / "functional_projections.csv", index=False)
    joint_space.to_csv(out / "shared_author_spaces.csv", index=False)
    joint_scores.to_csv(out / "private_joint_author_scores.csv", index=False)
    evidence.to_csv(out / "shared_evidence_profiles.csv", index=False)
    summary = {
        "status": "COMPLETE_PANEL" if authors == 24 else "BOUNDED_PILOT",
        "authors": authors,
        "views": len(records),
        "labels_read": False,
        "panel_manifest_sha256": sha(args.panel / "manifest.json"),
        "script_sha256": sha(__file__),
        "metrics": metrics,
        "interactions": interactions,
        "transfer": transfer,
        "shared": shared.astype(object)
        .where(pd.notna(shared), None)
        .to_dict("records"),
        "projections": projection.astype(object)
        .where(pd.notna(projection), None)
        .to_dict("records"),
        "author_spaces": joint_space.astype(object)
        .where(pd.notna(joint_space), None)
        .to_dict("records"),
        "evidence": evidence.astype(object)
        .where(pd.notna(evidence), None)
        .to_dict("records"),
    }
    write(out / "summary.json", summary)
    report = f"# S1 query sensitivity: {authors} complete authors\n\n"
    report += "Computational effect targets, no psychological outcomes.\n\n## Local and natural effects\n\n"
    report += pd.DataFrame(metrics).to_markdown(index=False, floatfmt=".4f")
    report += "\n\n## Shared author effects after edit/view centering\n\n"
    report += shared[shared["mode"] == "edit_view_centered"].to_markdown(
        index=False, floatfmt=".4f"
    )
    report += "\n\n## Joint edit interaction\n\n" + pd.DataFrame(
        interactions
    ).to_markdown(index=False, floatfmt=".4f")
    report += "\n\n## Frozen S0 transfer\n\n" + pd.DataFrame(transfer).to_markdown(
        index=False, floatfmt=".4f"
    )
    report += "\n\nIntervals resample authors; shared centered intervals condition on fitted cell means.\n"
    report += "Attributions are block16 gradient-times-state, not input gradients.\n"
    report += "Removed-state sums are zero-hidden-state surrogates, not exact text deletion derivatives.\n"
    report += "\n## Paired functional projection (measured test gradients, not LLM-free inference)\n\n"
    report += projection[projection["rank"].isin([1, 8])].to_markdown(
        index=False, floatfmt=".4f"
    )
    report += "\n\n## Descriptive joint author spaces\n\n" + joint_space.to_markdown(
        index=False, floatfmt=".4f"
    )
    report += "\n\n## Matched-comment evidence profiles\n\n" + evidence.to_markdown(
        index=False, floatfmt=".4f"
    )
    (out / "report.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(pd.DataFrame(interactions).to_string(index=False), flush=True)
    print(pd.DataFrame(transfer).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
