#!/usr/bin/env python3
"""M25/M26: frozen multi-query author geometry and compact functional readout."""

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

from grain_research.vector_mechanisms.functional_profiles import (
    FixedAuthorCoordinates,
    FunctionalQuery,
    MatchedQueryProjection,
    profile_pair_auc,
    relative_rms_error,
)
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    corr,
    read,
    sha,
    write,
)

LEVEL = BASE / "complete_level_response_v1"
QUERIES = ("TF_cont", "neuroticism", "confirmation_seeking")


def paired_metrics(x, y, authors):
    """Author-cluster uncertainty for matched scalar functions, never text-IID CI."""
    x, y, authors = np.asarray(x, float), np.asarray(y, float), np.asarray(authors)
    unique = sorted(set(authors))
    groups = [np.flatnonzero(authors == a) for a in unique]
    rng, samples = np.random.default_rng(20260909), []
    for _ in range(1000):
        ix = np.concatenate(
            [groups[j] for j in rng.integers(len(groups), size=len(groups))]
        )
        r = corr(x[ix], y[ix])
        if r is not None:
            samples.append(r)
    error, zero = np.mean((x - y) ** 2), np.mean(x * x)
    return {
        "authors": len(unique),
        "events": len(x),
        "r": corr(x, y),
        "r_lower": float(np.quantile(samples, 0.025)) if samples else None,
        "r_upper": float(np.quantile(samples, 0.975)) if samples else None,
        "mse": float(error),
        "mse_over_zero": float(error / zero) if zero > 1e-14 else None,
    }


def load_states(folder):
    """Load only complete hash-bound states and their already available teacher."""
    m = read(folder / "manifest.json")
    if read(folder / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("State collection incomplete")
    if (
        sha(folder / "private_design.json") != m["input_sha256"]
        or sha(LEVEL / "reference.npz") != m["reference_sha256"]
    ):
        raise ValueError("Design/reference changed")
    design = read(folder / "private_design.json")
    records, hashes = [], {}
    for d in design:
        path = folder / "private_packets" / f"cell_{d['index']:03d}.json"
        meta = read(path)
        if any(
            meta[k] != d[k]
            for k in (
                "index",
                "author_id",
                "view",
                "query",
                "condition",
                "role",
                "native_score",
            )
        ):
            raise ValueError("State join changed")
        if sha(path.with_suffix(".npz")) != meta["array_sha256"]:
            raise ValueError("State array changed")
        hashes[str(path.relative_to(ROOT))] = sha(path)
        hashes[str(path.with_suffix(".npz").relative_to(ROOT))] = meta["array_sha256"]
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            records.append((d, z["P"].astype(float), z["Q"].astype(float)))
    with np.load(LEVEL / "reference.npz", allow_pickle=False) as z:
        arrays = dict(z)
    return m, records, hashes, arrays


def readout(arrays, query, rank=None):
    """Construct a fixed query function from the pre-existing two-author matrix."""
    get = lambda k: arrays[f"{query}__{k}"]
    return FunctionalQuery(get("Z"), get("h"), get("G"), get("g"), get("b"), rank)


def response_table(parts):
    """Subtract each matching full state; queries share the same edits and views."""
    by_key = {
        (r["author_id"], r["view"], r["query"]): r
        for r in parts
        if r["condition"] == "full"
    }
    out = []
    for r in parts:
        if r["condition"] == "full":
            continue
        full = by_key[(r["author_id"], r["view"], r["query"])]
        base = {k: r[k] for k in ("author_id", "view", "query", "condition", "role")}
        for field in ("mean", "configuration", "final", "score", "native_score"):
            base[field] = r[field] - full[field]
        out.append(base)
    return pd.DataFrame(out)


def compact_readers(records, arrays, out):
    """Choose ranks by reference-only functional error, never evaluation labels."""
    compact, checks, frozen = {}, [], {}
    for q in QUERIES:
        rows = [
            (d, p, h)
            for d, p, h in records
            if d["query"] == q and d["role"] == "reference"
        ]
        full = readout(arrays, q)
        exact = {
            (d["author_id"], d["view"], d["condition"]): full.parts(p, h)[
                "configuration"
            ]
            for d, p, h in rows
        }
        ranks = sorted({1, 2, 4, 8, 16, min(full.p_ref.shape)})
        selected = None
        for rank in ranks:
            f = readout(arrays, q, rank)
            pred = {
                (d["author_id"], d["view"], d["condition"]): f.parts(p, h)[
                    "configuration"
                ]
                for d, p, h in rows
            }
            errors = {}
            for c in ("full", "e1", "e2"):
                keys = [k for k in exact if k[2] == c]
                a = [
                    exact[k] - (0 if c == "full" else exact[(*k[:2], "full")])
                    for k in keys
                ]
                b = [
                    pred[k] - (0 if c == "full" else pred[(*k[:2], "full")])
                    for k in keys
                ]
                errors[c] = relative_rms_error(a, b)
            checks.append({"query": q, "rank": rank, **errors})
            if selected is None and max(errors.values()) <= 0.10:
                selected, compact[q] = rank, f
        if selected is None:
            raise RuntimeError("Full-rank exact fallback failed")
        f = compact[q]
        u, s, v = np.linalg.svd(f.config_coef, full_matrices=False)
        left, right = u[:, :selected] * s[:selected], v[:selected]
        frozen.update(
            {
                f"{q}__{name}": value
                for name, value in {
                    "left": left,
                    "right": right,
                    "mean_coef": f.mean_coef,
                    "q_coef": f.g_final,
                    "p_ref": f.p_ref,
                    "q_ref": f.q_ref,
                    "anchor": f.anchor,
                    "rank": selected,
                }.items()
            }
        )
    np.savez_compressed(out / "compact_functions.npz", **frozen)
    pd.DataFrame(checks).to_csv(out / "reference_rank_checks.csv", index=False)
    return compact, {q: compact[q].rank for q in QUERIES}


def shared_functions(events, out):
    """Exact shared/private accounting on independent matched author responses."""
    metric, links, saved, frozen = [], [], [], {}
    for signal in ("configuration", "score", "native_score"):
        pivot = (
            events.pivot(
                index=["author_id", "view", "condition", "role"],
                columns="query",
                values=signal,
            )
            .loc[:, list(QUERIES)]
            .sort_index()
        )
        if pivot.isna().any().any():
            raise ValueError("Unmatched multi-query responses")
        index = pivot.index.to_frame(index=False)
        conditions = (index.view.astype(str) + "/" + index.condition).to_numpy()
        train = index.role.to_numpy() == "reference"
        model = MatchedQueryProjection().fit(pivot.to_numpy()[train], conditions[train])
        z, shared, private = model.transform(pivot.to_numpy(), conditions)
        frozen[signal] = {
            "queries": list(QUERIES),
            "conditions": model.conditions,
            "centers": {k: v.tolist() for k, v in model.centers.items()},
            "scale": model.scale.tolist(),
            "active": model.active.tolist(),
            "coef": model.coef.tolist(),
            "reference_gram": model.gram.tolist(),
        }
        for q, query in enumerate(QUERIES):
            eval_ix = ~train
            a, b, r = z[eval_ix, q], shared[eval_ix, q], private[eval_ix, q]
            metric.append(
                {
                    "signal": signal,
                    "query": query,
                    **paired_metrics(a, b, index.author_id.to_numpy()[eval_ix]),
                    "total_energy": float(np.mean(a * a)),
                    "shared_energy": float(np.mean(b * b)),
                    "private_energy": float(np.mean(r * r)),
                    "twice_shared_private": float(2 * np.mean(b * r)),
                    "accounting_error": float(np.max(np.abs(a - b - r))),
                }
            )
            for i, row in index.iterrows():
                saved.append(
                    {
                        **row.to_dict(),
                        "signal": signal,
                        "query": query,
                        "standardized": z[i, q],
                        "shared": shared[i, q],
                        "private": private[i, q],
                    }
                )
        for left, right in itertools.combinations(range(3), 2):
            links.append(
                {
                    "signal": signal,
                    "left": QUERIES[left],
                    "right": QUERIES[right],
                    **paired_metrics(
                        z[~train, left],
                        z[~train, right],
                        index.author_id.to_numpy()[~train],
                    ),
                }
            )
    write(out / "matched_query_projection.json", frozen)
    pd.DataFrame(saved).to_csv(out / "private_shared_coordinates.csv", index=False)
    pd.DataFrame(metric).to_csv(out / "shared_private_metrics.csv", index=False)
    pd.DataFrame(links).to_csv(out / "functional_links.csv", index=False)
    return metric, links


def author_profiles(parts, events, out):
    """Freeze twelve-author norms and compare paired held-author text profiles."""
    level = pd.DataFrame(parts)
    level = (
        level[level.condition == "full"]
        .pivot(index=["author_id", "view", "role"], columns="query", values="score")
        .loc[:, list(QUERIES)]
    )
    level.columns = [f"level/{q}" for q in QUERIES]
    response = events.pivot(
        index=["author_id", "view", "role"],
        columns=["condition", "query"],
        values="configuration",
    )
    response = response.loc[:, [(e, q) for e in ("e1", "e2") for q in QUERIES]]
    response.columns = [f"{e}/configuration/{q}" for e in ("e1", "e2") for q in QUERIES]
    profile = level.join(response, validate="one_to_one").sort_index()
    if profile.isna().any().any():
        raise ValueError("Profile incomplete")
    ref = profile[profile.index.get_level_values("role") == "reference"]
    norms = FixedAuthorCoordinates().fit(
        ref.groupby(level="author_id").mean().to_numpy()
    )
    scored = pd.DataFrame(
        norms.transform(profile.to_numpy()),
        index=profile.index,
        columns=profile.columns,
    )
    write(
        out / "author_reference.json",
        {
            "coordinates": list(profile.columns),
            "center": norms.center.tolist(),
            "scale": norms.scale.tolist(),
            "active": norms.active.tolist(),
            "reference_authors": 12,
            "construction": "one row per reference author, average two views first",
        },
    )
    profile.to_csv(out / "private_author_raw.csv")
    scored.to_csv(out / "private_author_positions.csv")
    evaluation = scored[scored.index.get_level_values("role") == "evaluation"]
    a, b = [evaluation.xs(v, level="view").sort_index() for v in (0, 1)]
    if list(a.index) != list(b.index):
        raise ValueError("Paired author views mismatch")
    rng, metrics, coordinate = np.random.default_rng(20260909), [], []
    for name, columns in (
        ("level", list(a.columns[:3])),
        ("configuration_response", list(a.columns[3:])),
        ("combined", list(a.columns)),
    ):
        x, y = a[columns].to_numpy(), b[columns].to_numpy()
        auc, distances = profile_pair_auc(x, y)
        null = [profile_pair_auc(x, y[rng.permutation(len(y))])[0] for _ in range(1000)]
        metrics.append(
            {
                "profile": name,
                "authors": len(x),
                "dimensions": len(columns),
                "same_author_auc": auc,
                "permutation_p": float((1 + sum(v >= auc for v in null)) / 1001),
                "null_95_upper": float(np.quantile(null, 0.95)),
                "matched_distance": float(np.diag(distances).mean()),
                "unmatched_distance": float(
                    distances[~np.eye(len(x), dtype=bool)].mean()
                ),
            }
        )
    for c in a.columns:
        coordinate.append(
            {
                "coordinate": c,
                **paired_metrics(
                    a[c].to_numpy(),
                    b[c].to_numpy(),
                    a.index.get_level_values("author_id").to_numpy(),
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "author_profile_metrics.csv", index=False)
    pd.DataFrame(coordinate).to_csv(out / "author_coordinate_metrics.csv", index=False)
    return metrics, coordinate


def main():
    """Develop a single frozen construction, then report all held-author arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.state_dir.resolve(), args.output_dir.resolve()
    if not source.is_relative_to(BASE) or not out.is_relative_to(BASE):
        raise ValueError("Use ignored private result paths")
    manifest, records, hashes, arrays = load_states(source)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "run_status.json", {"status": "RUNNING"})
    with threadpool_limits(2):
        compact, ranks = compact_readers(records, arrays, out)
        functions = {q: readout(arrays, q) for q in QUERIES}
        parts, compact_parts = [], []
        for d, p, q in records:
            meta = {
                k: d[k]
                for k in (
                    "author_id",
                    "view",
                    "query",
                    "condition",
                    "role",
                    "native_score",
                )
            }
            parts.append({**meta, **functions[d["query"]].parts(p, q)})
            compact_parts.append({**meta, **compact[d["query"]].parts(p, q)})
        events, compact_events = response_table(parts), response_table(compact_parts)
        pd.DataFrame(parts).to_csv(out / "private_functional_parts.csv", index=False)
        events.to_csv(out / "private_responses.csv", index=False)
        pd.DataFrame(compact_parts).to_csv(
            out / "private_compact_parts.csv", index=False
        )
        shared, links = shared_functions(events, out)
        profiles, coordinates = author_profiles(parts, events, out)
        metrics, compression = [], []
        for query in QUERIES:
            for kind, frame, small in (
                ("level", pd.DataFrame(parts), pd.DataFrame(compact_parts)),
                ("response", events, compact_events),
            ):
                ix = (frame["query"] == query) & (frame.role == "evaluation")
                if kind == "level":
                    ix &= frame.condition == "full"
                g, c = frame[ix], small[ix]
                metrics.append(
                    {
                        "query": query,
                        "kind": kind,
                        **paired_metrics(g.native_score, g.score, g.author_id),
                    }
                )
                for field in ("score", "configuration"):
                    compression.append(
                        {
                            "query": query,
                            "kind": kind,
                            "field": field,
                            "rank": ranks[query],
                            **paired_metrics(g[field], c[field], g.author_id),
                            "relative_rms_error": relative_rms_error(
                                g[field], c[field]
                            ),
                        }
                    )
        pd.DataFrame(metrics).to_csv(out / "native_readout_metrics.csv", index=False)
        pd.DataFrame(compression).to_csv(out / "compact_metrics.csv", index=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SHARED_AUTHOR_PROFILES_M25_M26_V1",
            "state_manifest_sha256": sha(source / "manifest.json"),
            "private_state_hashes": hashes,
            "reference_sha256": manifest["reference_sha256"],
            "script_sha256": sha(__file__),
            "module_sha256": sha(
                ROOT / "grain_research/vector_mechanisms/functional_profiles.py"
            ),
            "human_labels_read": False,
            "new_neural_calls": 0,
            "reference_authors": 12,
            "evaluation_authors": 12,
            "compact_ranks": ranks,
            "compact_sha256": sha(out / "compact_functions.npz"),
            "shared_projection_sha256": sha(out / "matched_query_projection.json"),
            "author_reference_sha256": sha(out / "author_reference.json"),
        },
    )
    write(
        out / "summary.json",
        {
            "native_readout": metrics,
            "compact": compression,
            "shared_private": shared,
            "functional_links": links,
            "author_profiles": profiles,
            "author_coordinates": coordinates,
            "compact_ranks": ranks,
            "human_labels_read": False,
        },
    )
    write(
        out / "run_status.json",
        {"status": "COMPLETE", "new_neural_calls": 0, "human_labels_read": False},
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(profiles).to_string(index=False))
    print("Frozen compact ranks:", ranks)


if __name__ == "__main__":
    main()
