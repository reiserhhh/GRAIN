#!/usr/bin/env python3
"""Reconcile GRAIN/query/criterion connections at matched statistical levels.

Uses existing private numerical caches only. No models, texts, APIs, new factor
fits or human-label model selection. Public output is an aggregate report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.factor_catalog import bh_adjust
from grain_research.vector_mechanisms.logic_levels import (
    coordinate_accounting,
    covariance_accounting,
    infer_levels,
)

BASE = ROOT / "results/grain_vector_mechanisms"
PLAN = ROOT / "docs/GRAIN_LOGIC_RESEARCH_PLAN_V1.md"


def sha(path):
    """Hash evidence without printing private values."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    """Write finite, inspectable local JSON."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load():
    """Join exact, unique author/view keys; reject missingness or join expansion."""
    dynamic = BASE / "dynamic_path_channels_v1"
    files = {
        "factors": BASE / "factor_path_routes_v1/source_r2/private_factor_views.csv",
        "descriptors": dynamic / "within_comment/private_dynamic_views.csv",
        "paths": dynamic / "private_path_views.csv",
        "criteria": BASE / "all_factor_source_m29_v1/private_external_links.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in files.items()}
    g, d, p, c = (frames[name] for name in files)
    keys = ["author_id", "view"]
    for frame, pk in (
        (g, keys),
        (d, keys),
        (p, keys + ["query"]),
        (c, ["author_id", "query"]),
    ):
        if frame.duplicated(pk).any() or frame[pk].isna().any().any():
            raise ValueError("Duplicate or missing input key")
    original = [n for n in g if n.startswith(("whole:", "native:", "fixed_128_cross:"))]
    desc = [
        n
        for n in d
        if n.startswith(("level::", "disp::", "drift::", "roughness::", "area::"))
    ]
    if len(original) != 12 or len(desc) != 22:
        raise ValueError("Unexpected frozen feature catalog")
    merged = g[keys + original].merge(
        d[keys + desc], on=keys, how="outer", validate="one_to_one"
    )
    ids = sorted(g.author_id.unique())
    if len(ids) != 276 or len(merged) != 828 or merged.isna().any().any():
        raise ValueError("Corrected 276-author reference-disjoint panel required")
    index = pd.MultiIndex.from_product([ids, range(3)], names=keys)
    merged = merged.set_index(keys).loc[index].reset_index()
    p = p[p.author_id.isin(ids)].copy()
    c = c[c.author_id.isin(ids)].copy()
    expected = {
        "EI_cont",
        "SN_cont",
        "TF_cont",
        "JP_cont",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    }
    if set(p["query"]) != expected or set(c["query"]) != expected:
        raise ValueError("Nine exact query endpoints required")
    if len(p) != 3951 or len(c) != 1317:
        raise ValueError("Unexpected endpoint coverage")
    audit = {
        "authors": len(ids),
        "views": 3,
        "features": len(original + desc),
        "unique_keys": True,
        "complete_views": True,
        "path_rows": len(p),
        "criterion_rows": len(c),
        "source_hashes": {
            str(path.relative_to(ROOT)): sha(path) for path in files.values()
        },
    }
    return merged, p, c, original, desc, audit


def execute(out, *, bootstrap, permutations):
    """Run the frozen finite comparison; never overwrite an earlier packet."""
    if out.exists():
        raise ValueError("Use a new output directory; prior packets are immutable")
    features, paths, criteria, original, desc, audit = load()
    out.mkdir(parents=True)
    manifest = {
        "protocol": "GRAIN_LOGIC_LEVELS_V1",
        "seed": 20260910,
        "bootstrap": bootstrap,
        "permutations": permutations,
        "new_model_calls": 0,
        "new_prediction_fits": 0,
        "coordinate_fits": "in-sample algebra check only",
        "source": audit,
        "plan_sha256": sha(PLAN),
        "script_sha256": sha(__file__),
        "module_sha256": sha(ROOT / "grain_research/vector_mechanisms/logic_levels.py"),
    }
    write(out / "manifest.json", manifest)
    links, accounts, charts, endpoint = [], [], [], []
    columns = original + desc
    for query, p in paths.groupby("query", sort=True):
        ids = sorted(p.author_id.unique())
        expected = 63 if query.endswith("_cont") else 213
        if len(ids) != expected or len(p) != 3 * expected:
            raise ValueError("Wrong query cohort")
        index = pd.MultiIndex.from_product([ids, range(3)], names=["author_id", "view"])
        x = (
            features.set_index(["author_id", "view"])
            .loc[index, columns]
            .to_numpy(float)
            .reshape(len(ids), 3, len(columns))
        )
        y = (
            p.set_index(["author_id", "view"])
            .loc[index, "score"]
            .to_numpy(float)
            .reshape(len(ids), 3)
        )
        criterion = (
            criteria[criteria["query"].eq(query)]
            .set_index("author_id")
            .loc[ids, "y_true"]
            .to_numpy(float)
        )
        if query.endswith("_cont") and not set(criterion).issubset({0.0, 1.0}):
            raise ValueError("Binary axis coding changed")
        estimate = infer_levels(
            x, y, criterion, bootstrap=bootstrap, permutations=permutations
        )
        for level, values in estimate.items():
            for j, feature in enumerate(columns):
                links.append(
                    {
                        "query": query,
                        "feature": feature,
                        "level": level,
                        "authors": len(ids),
                        **{k: v[j] for k, v in values.items()},
                    }
                )
        account = covariance_accounting(x, y)
        for j, feature in enumerate(columns):
            accounts.append(
                {
                    "query": query,
                    "feature": feature,
                    **{k: v[j] for k, v in account.items()},
                }
            )
        endpoint.append(
            {
                "query": query,
                "authors": len(ids),
                "function_criterion_r": float(np.corrcoef(y.mean(1), criterion)[0, 1]),
            }
        )
        rng = np.random.default_rng(20260910)
        transformations = [
            (f"rotation_{i}", np.linalg.qr(rng.normal(size=(12, 12)))[0])
            for i in range(8)
        ]
        transformations.append(("fixed_scaling", np.diag(np.geomspace(0.1, 10, 12))))
        for level in ("between", "within"):
            if level == "between":
                xx, yy = x[:, :, :12].mean(1), y.mean(1)
            else:
                xx = (x[:, :, :12] - x[:, :, :12].mean(1)[:, None]).reshape(-1, 12)
                yy = (y - y.mean(1)[:, None]).ravel()
            for name, a in transformations:
                charts.append(
                    {
                        "query": query,
                        "level": level,
                        "transform": name,
                        **coordinate_accounting(xx, yy, a),
                    }
                )
        write(
            out / "status.json",
            {
                "status": "RUNNING",
                "completed_query": query,
                "completed_queries": len(endpoint),
            },
        )
        print(
            f"Completed {query}: {len(ids)} authors, {len(columns)} features",
            flush=True,
        )
    table = pd.DataFrame(links)
    table["q"] = np.nan
    for _, rows in table.groupby("level"):
        table.loc[rows.index, "q"] = bh_adjust(rows.p.to_numpy())
    table.to_csv(out / "level_links.csv", index=False)
    pd.DataFrame(accounts).to_csv(out / "covariance_accounting.csv", index=False)
    pd.DataFrame(charts).to_csv(out / "coordinate_accounting.csv", index=False)
    pd.DataFrame(endpoint).to_csv(out / "function_criterion.csv", index=False)
    write(
        out / "status.json",
        {
            "status": "COMPLETE",
            "queries": 9,
            "associations": len(table),
            "coordinate_cases": len(charts),
            "max_covariance_closure_error": float(
                pd.DataFrame(accounts).closure_error.abs().max()
            ),
            "public_contains_individual_rows": False,
        },
    )
    print(json.dumps(json.loads((out / "status.json").read_text())), flush=True)


def main():
    """CLI for a complete cache-only run or a lower-resample smoke packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "logic_closure_v1/levels_r1"
    )
    parser.add_argument("--bootstrap", type=int, default=1999)
    parser.add_argument("--permutations", type=int, default=3999)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        execute(args.output, bootstrap=args.bootstrap, permutations=args.permutations)


if __name__ == "__main__":
    main()
