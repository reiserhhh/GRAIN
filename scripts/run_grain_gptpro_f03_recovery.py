#!/usr/bin/env python3
"""Restore paired F03 designs and source-learned IPW with isolated scoring."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent / "project persona"
sys.path.insert(0, str(ROOT))

from grain_research.gptpro_foundation_v3.f00 import sha256
from grain_research.gptpro_foundation_v3.f02_recovery import paired_bootstrap
from grain_research.gptpro_foundation_v3.f03 import (
    WORLD_VALUES,
    generate_records,
    grouped,
)
from grain_research.gptpro_foundation_v3.f03_identification import (
    bootstrap_endpoints,
    population_view,
)
from grain_research.gptpro_foundation_v3.f03_recovery import (
    fit_source_selection,
    score_views,
)


def dump(path, value):
    """Write generated JSON with finite Python values."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    """Write generated records with their explicit union schema."""
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def records(path):
    """Read only a caller-declared role-specific compressed observation file."""
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_phase(root, design):
    """Learn shared selection parameters only from source observations."""
    path = root / "source.jsonl.gz"
    if design == "fixed":
        model = None  # Selection is an assigned intervention, not a learned classifier.
    else:
        model = fit_source_selection(records(path))
    dump(root / "model.json", model)
    dump(
        root / "fit_manifest.json",
        {
            "source_hash": sha256(path),
            "model_hash": sha256(root / "model.json"),
            "input_files": ["source.jsonl.gz"],
            "model_status": "ASSIGNED_SELECTION"
            if model is None
            else "CONVERGED"
            if model["converged"]
            else "NOT_CONVERGED",
        },
    )


def prediction_phase(root, design, prefixes):
    """Apply fixed source slopes to each person's nested prefix, without truth."""
    fit = json.loads((root / "fit_manifest.json").read_text())
    if sha256(root / "model.json") != fit["model_hash"]:
        raise ValueError("source model changed")
    model = json.loads((root / "model.json").read_text())
    output = []
    endpoint_rows = []
    for role in ("validation", "application"):
        for author, rows in grouped(records(root / f"{role}.jsonl.gz")).items():
            for prefix in prefixes:
                for result in score_views(rows[:prefix], model, design):
                    output.append(
                        {"author_id": author, "role": role, "prefix": prefix, **result}
                    )
                for z in (-1, 1):
                    seed = int(
                        hashlib.sha256(
                            f"20260915/{author}/{prefix}/{z}".encode()
                        ).hexdigest()[:16],
                        16,
                    )
                    intervals = bootstrap_endpoints(rows[:prefix], z, seed=seed)
                    endpoint_rows.append(
                        {
                            "author_id": author,
                            "role": role,
                            "prefix": prefix,
                            "z": z,
                            **{
                                k: json.dumps(v) if isinstance(v, list) else v
                                for k, v in intervals.items()
                            },
                        }
                    )
    table(root / "predictions.csv", output)
    table(root / "endpoint_bootstrap.csv", endpoint_rows)


def score_phase(root, world, design, seed):
    """Read latent response targets only after predictions are frozen."""
    truth = {r["author_id"]: r for r in json.loads((root / "truth.json").read_text())}
    output = []
    with (root / "predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            target = truth[row["author_id"]]["r"]["-" if int(row["z"]) == -1 else "+"]
            base = {
                "world": world,
                "design": design,
                "seed": seed,
                "author_id": row["author_id"],
                "role": row["role"],
                "prefix": row["prefix"],
                "view": row["view"],
                "z": row["z"],
                "target": target,
            }
            for method in ("naive", "fixed_beta", "ipw"):
                value = float(row[method]) if row[method] != "" else None
                output.append(
                    {
                        **base,
                        "method": method,
                        "estimate": value,
                        "bias": value - target if value is not None else None,
                        "squared_error": (value - target) ** 2
                        if value is not None
                        else None,
                    }
                )
            for method, lowkey, highkey in [
                ("general_bound", "bound_low", "bound_high"),
                (
                    "independent_opportunity_bound",
                    "independent_O_bound_low",
                    "independent_O_bound_high",
                ),
            ]:
                low = float(row[lowkey]) if row[lowkey] != "" else None
                high = float(row[highkey]) if row[highkey] != "" else None
                output.append(
                    {
                        **base,
                        "method": method,
                        "bound_low": low,
                        "bound_high": high,
                        "contains_target": int(low <= target <= high)
                        if low is not None
                        else None,
                        "width": high - low if low is not None else None,
                        "scope": "sample_plugin_bounds_not_confidence_intervals",
                    }
                )
    table(root / "scored.csv", output)


def summarize(out, roots):
    """Pair author-level losses across methods/designs and preserve missingness."""
    all_rows = []
    for root in roots:
        with (root / "scored.csv").open(newline="") as handle:
            all_rows.extend(csv.DictReader(handle))
    table(out / "scored.csv", all_rows)
    cells = defaultdict(list)
    for row in all_rows:
        cells[
            tuple(
                row[k]
                for k in ("world", "design", "seed", "role", "prefix", "view", "method")
            )
        ].append(row)
    summaries = []
    for key, rows in cells.items():
        result = dict(
            zip(("world", "design", "seed", "role", "prefix", "view", "method"), key)
        )
        result.update(
            authors=len({row["author_id"] for row in rows}), condition_rows=len(rows)
        )
        for field in ("squared_error", "bias", "contains_target", "width"):
            values = [float(row[field]) for row in rows if row.get(field, "") != ""]
            result[field] = float(np.mean(values)) if values else None
            result[field + "_n"] = len(values)
        summaries.append(result)
    table(out / "metrics.csv", summaries)
    authors = defaultdict(list)
    for row in all_rows:
        if (
            row["role"] == "application"
            and row["view"] == "all_opportunities"
            and row.get("squared_error", "") != ""
        ):
            authors[
                (
                    row["world"],
                    row["design"],
                    int(row["seed"]),
                    int(row["prefix"]),
                    row["method"],
                    row["author_id"],
                )
            ].append(float(row["squared_error"]))
    comparisons = []
    seeds = sorted({key[2] for key in authors})
    prefixes = sorted({key[3] for key in authors})
    for world in WORLD_VALUES:
        for prefix in prefixes:
            for name, left, right in [
                ("ipw_vs_naive_free", ("free", "ipw"), ("free", "naive")),
                ("ipw_vs_naive_common", ("common", "ipw"), ("common", "naive")),
                ("fixed_vs_free", ("fixed", "fixed_beta"), ("free", "naive")),
                ("common_vs_free", ("common", "naive"), ("free", "naive")),
            ]:
                diffs = {}
                complete = 0
                for seed in seeds:
                    a = {
                        key[-1]: np.mean(v)
                        for key, v in authors.items()
                        if key[:5] == (world, left[0], seed, prefix, left[1])
                        and len(v) == 2
                    }
                    b = {
                        key[-1]: np.mean(v)
                        for key, v in authors.items()
                        if key[:5] == (world, right[0], seed, prefix, right[1])
                        and len(v) == 2
                    }
                    ids = sorted(a.keys() & b.keys())
                    complete += len(ids)
                    if ids:
                        diffs[seed] = np.array([a[u] - b[u] for u in ids])
                if len(diffs) == len(seeds):
                    comparisons.append(
                        {
                            "world": world,
                            "prefix": prefix,
                            "comparison": name,
                            "complete_paired_authors": complete,
                            "complete_case_scope": "both conditions observed for both estimators",
                            **paired_bootstrap(diffs),
                        }
                    )
    table(out / "paired_comparisons.csv", comparisons)


def full_run(args):
    """Run original four worlds, three paired designs, three seeds and prefixes."""
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("preserve old revisions")
    out.mkdir(parents=True)
    seeds = [20260915] if args.smoke else [20260915, 20260916, 20260917]
    counts = (
        {"source": 12, "validation": 6, "application": 6}
        if args.smoke
        else {"source": 300, "validation": 100, "application": 200}
    )
    prefixes = [4, 8] if args.smoke else [32, 128, 512]
    started = datetime.now(timezone.utc).isoformat()
    sources = [
        Path(__file__),
        ROOT / "grain_research/gptpro_foundation_v3/f03.py",
        ROOT / "grain_research/gptpro_foundation_v3/f03_recovery.py",
        ROOT / "grain_research/gptpro_foundation_v3/f03_identification.py",
        ROOT / "grain_research/gptpro_foundation_v3/f02_recovery.py",
        PARENT / "docs/GRAIN_GPTPRO_ATOMIC_PROTOCOLS_V3.md",
    ]
    (out / "source_archive").mkdir()
    for path in sources:
        (out / "source_archive" / (sha256(path) + "_" + path.name)).write_bytes(
            path.read_bytes()
        )
    dump(
        out / "input_manifest.json",
        {"files": [{"path": str(p), "sha256": sha256(p)} for p in sources]},
    )
    dump(
        out / "protocol.json",
        {
            "seeds": seeds,
            "counts": counts,
            "prefixes": prefixes,
            "worlds": list(WORLD_VALUES),
            "designs": ["free", "fixed", "common"],
            "source_only_selection": True,
            "truth_process": "score only",
            "smoke": args.smoke,
        },
    )
    exact = []
    for r, pi, kappa, lam, gamma, z, design in itertools.product(
        [0.2, 0.5, 0.8],
        [0.4, 0.8],
        [-0.5, 0.5],
        [0, 0.8],
        [0, 1],
        [-1, 1],
        ["free", "common", "fixed"],
    ):
        law = population_view(r, pi, kappa, lam, gamma, z, design)
        for key in ("general_bound", "independent_opportunity_bound"):
            assert law[key][0] - 1e-12 <= r <= law[key][1] + 1e-12
        exact.append(
            {
                "r": r,
                "pi": pi,
                "kappa": kappa,
                "lambda": lam,
                "gamma": gamma,
                "z": z,
                "design": design,
                **law,
            }
        )
    dump(out / "exact_population_cells.json", exact)
    roots = []
    for seed in seeds:
        for world in WORLD_VALUES:
            paired_truth = None
            paired_material = None
            for design in ("free", "fixed", "common"):
                root = out / f"{world}_{seed}_{design}"
                root.mkdir()
                rows, truth = generate_records(
                    world, design, seed, events=max(prefixes), counts=counts
                )
                truth_signature = hashlib.sha256(
                    json.dumps(
                        [{k: v for k, v in r.items() if k != "design"} for r in truth],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                material_signature = hashlib.sha256(
                    json.dumps(
                        [(r["author_id"], r["t"], r["condition"]) for r in rows],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                if paired_truth is not None and (
                    truth_signature != paired_truth
                    or material_signature != paired_material
                ):
                    raise ValueError("design pairing failed")
                paired_truth, paired_material = truth_signature, material_signature
                for role in counts:
                    with gzip.open(
                        root / f"{role}.jsonl.gz", "wt", compresslevel=1
                    ) as handle:
                        for row in rows:
                            if row["role"] == role:
                                handle.write(json.dumps(row) + "\n")
                dump(root / "truth.json", truth)
                del rows, truth
                for phase in ("source", "predict", "score"):
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--phase",
                        phase,
                        "--output",
                        str(root),
                        "--design",
                        design,
                        "--world",
                        world,
                        "--seed",
                        str(seed),
                        "--prefixes",
                        *[str(p) for p in prefixes],
                    ]
                    result = subprocess.run(
                        command, cwd=ROOT, text=True, capture_output=True, check=False
                    )
                    (root / f"{phase}.log").write_text(result.stdout + result.stderr)
                    if result.returncode:
                        raise RuntimeError(f"{root.name}/{phase}: see saved log")
                roots.append(root)
                dump(
                    out / "checkpoint.json",
                    {
                        "completed_design_runs": [r.name for r in roots],
                        "complete": False,
                    },
                )
                print(
                    json.dumps({"completed": root.name, "design_runs": len(roots)}),
                    flush=True,
                )
    summarize(out, roots)
    dump(
        out / "summary.json",
        {
            "status": "NUMERICAL_RECOVERY_REVIEW_PENDING",
            "smoke": args.smoke,
            "design_runs": len(roots),
            "paired_author_instances": len(seeds)
            * len(WORLD_VALUES)
            * sum(counts.values()),
            "records": len(roots) * sum(counts.values()) * max(prefixes),
            "parent_cards_promoted": [],
            "pending": [
                "old choice-design target comparison",
                "original card-level acceptance",
            ],
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    dump(
        out / "output_manifest.json",
        {
            "files": [
                {"path": str(p.relative_to(out)), "sha256": sha256(p)}
                for p in sorted(out.rglob("*"))
                if p.is_file()
            ]
        },
    )


def main():
    """Dispatch isolated computation phases or the full recovery battery."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", choices=["source", "predict", "score"])
    parser.add_argument("--world")
    parser.add_argument("--design")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[32, 128, 512])
    args = parser.parse_args()
    if args.phase == "source":
        source_phase(args.output, args.design)
    elif args.phase == "predict":
        prediction_phase(args.output, args.design, args.prefixes)
    elif args.phase == "score":
        score_phase(args.output, args.world, args.design, args.seed)
    else:
        full_run(args)


if __name__ == "__main__":
    main()
