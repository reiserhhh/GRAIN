#!/usr/bin/env python3
"""HM94: source-freeze, cold-apply and evaluate directed natural evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_opportunity_current as M
from scripts import run_grain_math_natural_summary as OLD

CFG = ROOT / "configs/grain_opportunity_current_hm94.json"
OUT = ROOT / "results/grain_math_response_v1/opportunity_current_r1"
PLAN = ROOT / "docs/GRAIN_OPPORTUNITY_CURRENT_PLAN_HM94.md"
CODE = (
    PLAN,
    CFG,
    Path(__file__).resolve(),
    ROOT / "grain_research/construction/math_opportunity_current.py",
    ROOT / "tests/test_grain_math_opportunity_current.py",
    ROOT / "scripts/verify_grain_opportunity_current_hm94.py",
    ROOT / "scripts/run_grain_math_natural_summary.py",
    ROOT / "grain_research/construction/math_author_current.py",
)


def read(path):
    """Read an explicit safe manifest or generated aggregate metadata."""
    return json.loads(path.read_text())


def write(path, value):
    """Write strict generated JSON, not scientific data supplied by a model."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    """Hash an explicit allowed file without opening any raw-material source."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(out):
    """Seal protocol and inherited numeric caches before new outcome analysis."""
    hashes, raw_metadata_hash = OLD.binding()
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in CODE})
    out.mkdir(parents=True, exist_ok=False)
    (out / "00").mkdir()
    write(
        out / "00/manifest.json",
        {
            "protocol": "HM94_OPPORTUNITY_AND_RESPONSE_CURRENT",
            "hashes": hashes,
            "raw_hash_metadata_only": raw_metadata_hash,
            "config": read(CFG),
            "new_calls": 0,
            "new_neural_training": 0,
            "external_labels_read": False,
            "retrospective_numeric_cache": True,
            "original_scope_complete": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print("PREPARED_HM94; no new outcomes read", flush=True)


def sealed(out):
    """Check frozen code and the numeric-only input whitelist."""
    m = read(out / "00/manifest.json")
    for key, expected in m["hashes"].items():
        if sha(ROOT / key) != expected:
            raise ValueError("Sealed input/code changed: " + key)
    OLD.binding()
    return m["config"]


def stage(out, name):
    """Create a non-overwriting recoverable stage directory."""
    cfg = sealed(out)
    p = out / name
    p.mkdir(exist_ok=False)
    write(p / "checkpoint.json", {"status": "RUNNING", "completed": 0})
    return cfg, p, time.monotonic()


def finish(path, summary):
    """Seal completed finite outputs without altering earlier stages."""
    write(path / "summary.json", summary)
    write(path / "checkpoint.json", {"status": summary["status"]})
    write(
        path / "hashes.json",
        {
            p.name: sha(p)
            for p in path.iterdir()
            if p.is_file() and not p.name.startswith("._") and p.name != "hashes.json"
        },
    )
    print(summary, flush=True)


def load_data():
    """Join existing slice scores and embeddings at exact cached comment addresses."""
    supports, replay = OLD.load_support()
    frame = pd.read_csv(
        OLD.SLICES / "private_factor_slices.csv",
        usecols=["author_id", "view", "comment", "slice", *OLD.FC],
        dtype={"author_id": str},
    )
    frame["opaque"] = frame.author_id.map(OLD.identifier)
    indexed = frame.set_index(["opaque", "view", "comment", "slice"])[
        OLD.FC
    ].sort_index()
    data = {}
    for uid, s in supports.items():
        x = np.stack(
            [
                indexed.loc[(uid, int(v), int(c))].loc[[0, 1, 2, 3]].to_numpy(float)
                for v, c in zip(s["panel"], s["comment"], strict=True)
            ]
        )
        if not np.allclose(x.mean(1), s["g"], atol=1e-12, rtol=0):
            raise ValueError("Exact comment-to-slice binding failed")
        data[uid] = {
            "x": x,
            "h": s["h"],
            "view": s["panel"],
            "comment": s["comment"],
            "y": s["y"],
            "queries": s["queries"],
        }
    if len(data) != 276 or sum(len(d["x"]) for d in data.values()) != 9165:
        raise ValueError("Unexpected complete support")
    return data, replay


def partition(data, cfg):
    """Keep all views of an author together; labels never select source authors."""
    source, application = [], []
    for battery in (4, 5):
        ids = [uid for uid, d in data.items() if len(d["queries"]) == battery]
        ids.sort(
            key=lambda u: hashlib.sha256(f"{cfg['seed']}::{u}".encode()).hexdigest()
        )
        cut = int(np.ceil(len(ids) * cfg["source_fraction"]))
        source.extend(ids[:cut])
        application.extend(ids[cut:])
    if set(source) & set(application) or set(source + application) != set(data):
        raise ValueError("Author partition leaked or lost observations")
    return sorted(source), sorted(application)


def transform(d, model):
    """Cold-assign frozen states/conditions, without fitting an application object."""
    x = (d["x"].reshape(-1, 4) - model["g_mean"]) / model["g_scale"]
    states = M.nearest(x, model["g_centers"]).reshape(-1, 4)
    h = (d["h"] - model["h_mean"]) @ model["h_components"].T
    conditions = M.nearest(h, model["h_centers"])
    return states, conditions


def source(out):
    """Fit unsupervised reference geometry ONLY on the predetermined source authors."""
    cfg, p, started = stage(out, "01")
    data, replay = load_data()
    ids, app = partition(data, cfg)
    g = np.concatenate([data[u]["x"].reshape(-1, 4) for u in ids])
    h = np.concatenate([data[u]["h"] for u in ids])
    mu, scale = g.mean(0), g.std(0)
    if (scale < 1e-12).any():
        raise ValueError("Source coordinate is degenerate")
    km = KMeans(n_clusters=cfg["states"], random_state=cfg["seed"], n_init=10).fit(
        (g - mu) / scale
    )
    pca = PCA(
        n_components=cfg["condition_pca_dimensions"],
        svd_solver="randomized",
        random_state=cfg["seed"],
    ).fit(h)
    hc = KMeans(n_clusters=cfg["conditions"], random_state=cfg["seed"], n_init=10).fit(
        pca.transform(h)
    )
    model = {
        "g_mean": mu,
        "g_scale": scale,
        "g_centers": km.cluster_centers_,
        "h_mean": pca.mean_,
        "h_components": pca.components_,
        "h_centers": hc.cluster_centers_,
    }
    counts = []
    for uid in ids:
        x, z = transform(data[uid], model)
        counts.append(M.pair_counts(x, z, cfg["states"], cfg["conditions"]))
    references, support, sparse = M.build_references(
        counts, cfg["pseudocount"], cfg["minimum_reference_authors"]
    )
    np.savez_compressed(
        p / "source_model.npz",
        **model,
        **{f"reference_{k}": v for k, v in references.items()},
        source_counts=np.stack(counts),
        source_support=support,
        fallback=sparse,
    )
    write(p / "partition.json", {"source": ids, "application": app})
    finish(
        p,
        {
            "status": "FROZEN_SOURCE_HM94",
            "source_authors": len(ids),
            "application_authors": len(app),
            "source_comments": len(h),
            "context_pca_explained": float(pca.explained_variance_ratio_.sum()),
            "source_distinct_author_support": support.tolist(),
            "fallback_cells": int(sparse.sum()),
            "previous_numeric_replay_error": replay,
            "wall_seconds": time.monotonic() - started,
            "new_calls": 0,
            "new_predictors": 0,
        },
    )


def previous(out, name):
    """Check a preceding stage's seal before applying its frozen products."""
    p = out / name
    for filename, expected in read(p / "hashes.json").items():
        if sha(p / filename) != expected:
            raise ValueError("Stage changed: " + filename)


def apply(out):
    """Cold-apply source geometry/references and save every signed author account."""
    cfg, p, started = stage(out, "02")
    previous(out, "01")
    data, replay = load_data()
    split = read(out / "01/partition.json")
    source_ids, app = partition(data, cfg)
    if source_ids != split["source"] or app != split["application"]:
        raise ValueError("Partition changed")
    with np.load(out / "01/source_model.npz", allow_pickle=False) as z:
        model = {k: z[k] for k in z.files}
    arrays, metadata = {}, []
    rng = np.random.default_rng(cfg["seed"])
    for index, uid in enumerate(app):
        d = data[uid]
        x, z = transform(d, model)
        key = f"a{index:03d}"
        arrays[key + "_states"], arrays[key + "_conditions"] = x, z
        arrays[key + "_views"], arrays[key + "_proxy"] = d["view"], d["y"]
        counts, qs, laws, parts, nulls = [], [], [], [], []
        for v in range(3):
            mask = d["view"] == v
            c = M.pair_counts(x[mask], z[mask], cfg["states"], cfg["conditions"])
            counts.append(c)
            account = [
                M.reference_account(c, model["reference_" + arm]) for arm in cfg["arms"]
            ]
            j = np.stack([value[0] for value in account])
            qs.append(account[0][1])
            laws.append(j)
            parts.append(np.array([[M.components(a) for a in row] for row in j]))
            shuffled = []
            for _ in range(cfg["within_comment_shuffle_draws"]):
                sx = np.stack([row[rng.permutation(4)] for row in x[mask]])
                cc = M.pair_counts(sx, z[mask], cfg["states"], cfg["conditions"])
                jj = cc.sum((0, 1)) / cc.sum()
                shuffled.append(M.components(jj)[2])
            nulls.append(shuffled)
        arrays.update(
            {
                key + "_" + name: np.array(value)
                for name, value in (
                    ("counts", counts),
                    ("q", qs),
                    ("laws", laws),
                    ("parts", parts),
                    ("shuffle_cycles", nulls),
                )
            }
        )
        metadata.append(
            {
                "key": key,
                "opaque_author": uid,
                "queries": d["queries"],
                "comments": len(x),
            }
        )
        if index % 30 == 0:
            write(p / "checkpoint.json", {"status": "RUNNING", "completed": index + 1})
            print(f"Cold application {index + 1}/{len(app)}", flush=True)
        if time.monotonic() - started > cfg["maximum_cpu_seconds_per_stage"]:
            raise TimeoutError("Finite stage budget exhausted; preserve checkpoint")
    np.savez_compressed(p / "private_author_accounts.npz", **arrays)
    write(p / "private_metadata.json", metadata)
    finish(
        p,
        {
            "status": "COLD_APPLIED_HM94",
            "application_authors": len(app),
            "application_comments": sum(r["comments"] for r in metadata),
            "new_calls": 0,
            "new_fits": 0,
            "previous_numeric_replay_error": replay,
            "wall_seconds": time.monotonic() - started,
        },
    )


def correlation(x, y):
    """Compute Pearson on finite authors with explicit constant-series handling."""
    a, b = x - np.mean(x), y - np.mean(y)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return None if denominator <= 1e-14 else float(a @ b / denominator)


def evaluate(out):
    """Measure author/reference structure; never select a predictor or rewrite scores."""
    cfg, p, started = stage(out, "03")
    previous(out, "01")
    previous(out, "02")
    metadata = read(out / "02/private_metadata.json")
    with np.load(out / "02/private_author_accounts.npz", allow_pickle=False) as z:
        accounts = {k: z[k] for k in z.files}
    parts = np.stack([accounts[r["key"] + "_parts"] for r in metadata])
    q = np.stack([accounts[r["key"] + "_q"] for r in metadata]).sum(3)
    # parts axes: author,view,reference,raw/opportunity/response,S/G/C,state,state.
    cycle = parts[..., 2, :, :]
    edges = np.triu_indices(cfg["states"], 1)
    reproducibility, variance = [], []
    for arm_index, arm in enumerate(cfg["arms"]):
        centered = cycle[:, :, arm_index] - cycle[:, :, arm_index].mean(0)
        energy = np.mean(np.sum(centered**2, (-2, -1)), (0, 1))
        cross = float(
            2 * np.mean(np.sum(centered[:, :, 1] * centered[:, :, 2], (-2, -1)))
        )
        variance.append(
            {
                "arm": arm,
                "raw_variance": energy[0],
                "opportunity_variance": energy[1],
                "response_variance": energy[2],
                "cross_term": cross,
                "identity_error": float(abs(energy[0] - energy[1] - energy[2] - cross)),
            }
        )
        for channel_index, channel in enumerate(("raw", "opportunity", "response")):
            feature = cycle[:, :, arm_index, channel_index, edges[0], edges[1]]
            score, distance = M.matched_author_closeness(
                feature, q, cfg["matched_strangers"]
            )
            estimate = M.bootstrap_mean(
                score.mean(1), cfg["bootstrap_draws"], cfg["seed"]
            )
            reproducibility.append(
                {
                    "arm": arm,
                    "channel": channel,
                    "auc": estimate[0],
                    "ci_lower": estimate[1],
                    "ci_upper": estimate[2],
                    "pair_01": float(score[:, 0].mean()),
                    "pair_02": float(score[:, 1].mean()),
                    "pair_12": float(score[:, 2].mean()),
                    "matched_context_squared_distance": float(distance.mean()),
                }
            )
    pd.DataFrame(variance).to_csv(p / "cycle_variance_accounts.csv", index=False)
    pd.DataFrame(reproducibility).to_csv(p / "author_closeness.csv", index=False)
    raw = cycle[:, :, 0, 0]
    null = np.stack([accounts[r["key"] + "_shuffle_cycles"] for r in metadata])
    order_excess = (np.sum(raw**2, (-2, -1)) - np.sum(null**2, (-2, -1)).mean(2)).mean(
        1
    )
    order = M.bootstrap_mean(order_excess, cfg["bootstrap_draws"], cfg["seed"])
    rng, correlations = np.random.default_rng(cfg["seed"]), []
    joint = cfg["arms"].index("joint")
    for query in OLD.QUERIES:
        ix = [i for i, r in enumerate(metadata) if query in r["queries"]]
        y = np.array(
            [
                accounts[metadata[i]["key"] + "_proxy"][
                    :, metadata[i]["queries"].index(query)
                ].mean()
                for i in ix
            ]
        )
        for channel_index, channel in enumerate(("raw", "opportunity", "response")):
            for a, b in ((1, 2), (1, 3), (2, 3)):
                x = cycle[ix, :, joint, channel_index, a, b].mean(1)
                value = correlation(x, y)
                if value is None:
                    pv, lower, upper = None, None, None
                else:
                    permutations = [
                        correlation(x, y[rng.permutation(len(y))])
                        for _ in range(cfg["permutation_draws"])
                    ]
                    pv = (1 + sum(abs(v) >= abs(value) for v in permutations)) / (
                        cfg["permutation_draws"] + 1
                    )
                    boot = []
                    for _ in range(cfg["bootstrap_draws"]):
                        sample = rng.integers(0, len(y), len(y))
                        rr = correlation(x[sample], y[sample])
                        if rr is not None:
                            boot.append(rr)
                    lower, upper = map(float, np.quantile(boot, (0.025, 0.975)))
                correlations.append(
                    {
                        "query_proxy": query,
                        "channel": channel,
                        "cycle_coordinate": f"C{a}{b}",
                        "n": len(ix),
                        "r": value,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "permutation_p": pv,
                    }
                )
    table = pd.DataFrame(correlations)
    finite = table.permutation_p.notna()
    values = table.loc[finite, "permutation_p"].to_numpy()
    ordering = np.argsort(values)
    corrected = np.minimum.accumulate(
        (values[ordering] * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty(len(values))
    adjusted[ordering] = np.minimum(corrected, 1)
    table["fdr_q"] = np.nan
    table.loc[finite, "fdr_q"] = adjusted
    table.to_csv(p / "query_proxy_links.csv", index=False)
    toy_rows = []
    toy = M.toy_worlds()
    np.savez_compressed(p / "toy_accounts.npz", **toy)
    for name, law in toy.items():
        c = np.array([M.components(j)[2] for j in law])
        toy_rows.append(
            {
                "world": name,
                "raw_cycle_norm": float(np.linalg.norm(c[0])),
                "opportunity_cycle_norm": float(np.linalg.norm(c[1])),
                "response_cycle_norm": float(np.linalg.norm(c[2])),
                "decomposition_error": float(np.max(abs(c[0] - c[1] - c[2]))),
            }
        )
    pd.DataFrame(toy_rows).to_csv(p / "toy_mechanisms.csv", index=False)
    coarsened = (
        toy["forward_opportunity_only"][0] - toy["unordered_opportunity_only"][0]
    )
    finish(
        p,
        {
            "status": "EVALUATED_FINITE_HM94_NOT_GLOBAL",
            "source_authors": 92,
            "application_authors": len(metadata),
            "inherited_authors": 276,
            "inherited_comments": 9165,
            "within_comment_only": True,
            "fixed_reference_bootstrap_not_source_uncertainty": True,
            "order_excess_mean_CI": order,
            "toy_coarse_false_response_cycle_norm": float(
                np.linalg.norm(M.components(coarsened)[2])
            ),
            "maximum_variance_identity_error": max(
                row["identity_error"] for row in variance
            ),
            "proxy_tests": len(table),
            "proxy_fdr_05": int((table.fdr_q < 0.05).sum()),
            "new_calls": 0,
            "new_predictors": 0,
            "human_personality_accuracy_measured": False,
            "wall_seconds": time.monotonic() - started,
        },
    )


def main():
    """Run exactly one finite stage; old runs/stages are never silently replaced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "source", "apply", "evaluate"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
