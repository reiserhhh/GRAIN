#!/usr/bin/env python3
"""HM63: freeze, measure and verify target-active natural summary fibers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_natural_summary as M

BASE = ROOT / "results/grain_vector_mechanisms"
SOURCE = BASE / "logic_configuration_extraction_v1"
SLICES = BASE / "dynamic_path_channels_v1/within_comment"
PRIOR = BASE / "logic_material_coupling_v1_r2"
OUT = ROOT / "results/grain_math_response_v1/natural_summary_observability_r1"
PLAN = ROOT / "docs/GRAIN_NATURAL_SUMMARY_OBSERVABILITY_PLAN_V1.md"
FC = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
FACTORS = [f"whole:F{i:02d}" for i in range(1, 5)]
QUERIES = (
    "EI_cont",
    "SN_cont",
    "TF_cont",
    "JP_cont",
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
)
NUMERIC_INPUTS = (
    SOURCE / "feature_manifest.json",
    SOURCE / "private_feature_rows.csv",
    SOURCE / "private_features.npz",
    SLICES / "manifest.json",
    SLICES / "private_factor_slices.csv",
    PRIOR / "manifest.json",
    PRIOR / "prediction_freeze.json",
    PRIOR / "private_coupling_views.csv",
    PRIOR / "private_local_maps.npz",
)
CODE = (
    PLAN,
    Path(__file__).resolve(),
    ROOT / "grain_research/construction/math_natural_summary.py",
    ROOT / "tests/test_grain_math_natural_summary.py",
)


def read(path):
    """Read a small numerical manifest, never a raw-material document."""
    return json.loads(path.read_text())


def write(path, data):
    """Save generated aggregate metadata with strict finite JSON numbers."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    """Hash only an explicit numeric/code whitelist."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def binding():
    """Check prior numerical seals; compare raw hashes without opening raw text."""
    pm = read(PRIOR / "manifest.json")
    pf = read(PRIOR / "prediction_freeze.json")
    fm, sm = read(SOURCE / "feature_manifest.json"), read(SLICES / "manifest.json")
    raw_key = (
        "results/grain_vector_mechanisms/source_remainder_s9_r1/private_material.json"
    )
    if not (pm["hashes"][raw_key] == fm["hashes"][raw_key] == sm["sources"][raw_key]):
        raise ValueError("Prior source hashes do not identify the same comments")
    hashes = {}
    for path in NUMERIC_INPUTS:
        key = str(path.relative_to(ROOT))
        actual = sha(path)
        if path.parent == PRIOR and path.name in pf["files"]:
            expected = pf["files"][path.name]
        elif key in pm["hashes"]:
            expected = pm["hashes"][key]
        else:
            expected = actual  # Bind the prior manifests themselves below.
        if actual != expected:
            raise ValueError("Frozen numeric input changed: " + key)
        hashes[key] = actual
    if pm["queries"] != list(QUERIES) or pf["human_labels_read"]:
        raise ValueError("Unexpected frozen query protocol")
    return hashes, pm["hashes"][raw_key]


def prepare(out):
    """Freeze this bounded protocol before inspecting its new outcomes."""
    hashes, material_hash = binding()
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in CODE})
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_NATURAL_SUMMARY_OBSERVABILITY_V1",
            "hashes": hashes,
            "source_material_hash_metadata_only": material_hash,
            "arms": list(M.ARMS),
            "queries": list(QUERIES),
            "expected_authors": 276,
            "expected_comments": 9165,
            "expected_author_query_arms": 9219,
            "radius": 0.5,
            "max_cpu_wall_seconds": 300,
            "new_fits": 0,
            "new_model_calls": 0,
            "raw_text_or_external_labels_read": False,
            "unit": "four frozen lexical coordinates per comment; proxy query h*w",
            "not_native_history_or_personality_truth": True,
        },
    )
    print({"status": "PREPARED_HM63", "authors": 276, "new_model_calls": 0}, flush=True)


def sealed(out):
    """Verify only this packet's explicit safe input/code list."""
    manifest = read(out / "manifest.json")
    for key, expected in manifest["hashes"].items():
        if sha(ROOT / key) != expected:
            raise ValueError("HM63 sealed input changed: " + key)
    binding()
    return manifest


def identifier(author):
    """Use a local opaque cache key rather than exporting an author identifier."""
    return hashlib.sha256(str(author).encode()).hexdigest()[:24]


def load_support():
    """Join exact cached comment addresses and replay prior numerical observations."""
    meta = pd.read_csv(
        SOURCE / "private_feature_rows.csv",
        usecols=["author_id", "view", "edit", "group"],
        dtype={"author_id": str},
    )
    meta["feature_index"] = np.arange(len(meta))
    with np.load(SOURCE / "private_features.npz", allow_pickle=False) as z:
        flat, offsets = z["bags"], z["offsets"]
    if len(offsets) != len(meta) + 1 or offsets[0] != 0 or offsets[-1] != len(flat):
        raise ValueError("Invalid numerical cache offsets")
    meta = meta[meta.group.isin(["big5_only", "mbti_only"]) & (meta.edit == "source")]
    if meta.duplicated(["author_id", "view"]).any() or len(meta) != 828:
        raise ValueError("Exact complete original source panels required")
    slices = pd.read_csv(
        SLICES / "private_factor_slices.csv",
        usecols=["author_id", "view", "comment", "slice", *FC],
        dtype={"author_id": str},
    )
    keys = ["author_id", "view", "comment"]
    if slices.duplicated([*keys, "slice"]).any() or len(slices) != 36660:
        raise ValueError("Duplicate or incomplete slice input")
    groups = slices.groupby(keys, sort=True)
    if not groups["slice"].agg(lambda x: tuple(sorted(x)) == (0, 1, 2, 3)).all():
        raise ValueError("Each comment must contain the same four distinct slices")
    comments = groups[FC].mean()
    if len(comments) != 9165 or not np.isfinite(comments.to_numpy()).all():
        raise ValueError("Unexpected comment support")
    prior = pd.read_csv(
        PRIOR / "private_coupling_views.csv",
        dtype={"author_id": str},
        usecols=[
            "author_id",
            "view",
            "query",
            "fold",
            "factor",
            "U",
            "S",
            "g_variance",
            "p_variance",
            "eligible_comments",
        ],
    )
    if (
        len(prior) != 15804
        or prior.duplicated(["author_id", "view", "query", "factor"]).any()
    ):
        raise ValueError("Invalid previous observation keys")
    prior = prior.set_index(["author_id", "view", "query", "factor"])
    with np.load(PRIOR / "private_local_maps.npz", allow_pickle=False) as z:
        maps = {k: z[k] for k in z.files if k.endswith("__coef")}
    if len(maps) != 45:
        raise ValueError("Expected exactly45 previously frozen OOF maps")
    supports, replay = {}, 0.0
    for author, panels in meta.groupby("author_id", sort=True):
        if (
            len(panels) != 3
            or set(panels.view) != {0, 1, 2}
            or panels.group.nunique() != 1
        ):
            raise ValueError("Incomplete or mixed source author")
        qs = QUERIES[:4] if panels.group.iloc[0] == "mbti_only" else QUERIES[4:]
        gg, hh, pp, cc, yy, folds = [], [], [], [], [], None
        for row in panels.sort_values("view").itertuples():
            data = comments.loc[(author, row.view)].sort_index()
            indexes = data.index.to_numpy(int)
            g = data.to_numpy(float)
            bag = flat[offsets[row.feature_index] : offsets[row.feature_index + 1]]
            if (
                bag.shape != (12, 1024)
                or not 7 <= len(g) <= 12
                or indexes.min() < 0
                or indexes.max() >= 12
            ):
                raise ValueError("Invalid exact comment-to-embedding address")
            h = bag[indexes].astype(float)
            values, panel_folds = [], []
            for query in qs:
                old = prior.loc[(author, row.view, query)].loc[FACTORS]
                if old.fold.nunique() != 1 or old.eligible_comments.ne(len(g)).any():
                    raise ValueError("Frozen map fold/comment count mismatch")
                fold = int(old.fold.iloc[0])
                panel_folds.append(fold)
                psi = h @ maps[f"{query}__{fold}__coef"]
                gc, pc = g - g.mean(0), psi - psi.mean()
                actual = np.column_stack(
                    (
                        g.mean(0),
                        np.mean(gc * pc[:, None], 0),
                        np.mean(gc**2, 0),
                        np.full(4, np.mean(pc**2)),
                    )
                )
                expected = old[["U", "S", "g_variance", "p_variance"]].to_numpy()
                replay = max(replay, float(np.max(abs(actual - expected))))
                values.append(psi)
            if folds is not None and folds != panel_folds:
                raise ValueError("An author changed OOF folds across panels")
            folds = panel_folds
            gg.append(g)
            hh.append(h)
            yy.append(np.column_stack(values))
            pp.append(np.full(len(g), row.view))
            cc.append(indexes)
        key = identifier(author)
        if key in supports:
            raise ValueError("Opaque identifier collision")
        supports[key] = {
            "g": np.vstack(gg),
            "h": np.vstack(hh),
            "y": np.vstack(yy),
            "panel": np.concatenate(pp),
            "comment": np.concatenate(cc),
            "queries": list(qs),
            "folds": folds,
        }
    if (
        len(supports) != 276
        or sum(len(s["g"]) for s in supports.values()) != 9165
        or replay > 1e-10
    ):
        raise ValueError("Prior numerical observations do not replay")
    return supports, replay


def raw_moments(g, panel, weights, arm, h):
    """Compute original-coordinate expectations without the projection implementation."""
    indicators = (panel[:, None] == np.unique(panel)[None, :]).astype(float)
    if arm.startswith("rotated"):
        g = g @ M.ROTATION
    parts = [weights @ indicators]
    if arm == "embedding_mean":
        parts.append(weights @ h)
    elif arm == "panel_diag":
        for value in np.unique(panel):
            mask = panel == value
            parts.extend((weights[mask] @ g[mask], weights[mask] @ (g[mask] ** 2)))
    else:
        parts.append(weights @ g)
        if arm != "pooled_mean":
            parts.append(weights @ (g**2))
        if arm.endswith("full"):
            parts.append(
                weights
                @ np.column_stack(
                    [g[:, i] * g[:, j] for i in range(4) for j in range(i + 1, 4)]
                )
            )
    return np.concatenate([p.ravel() for p in parts])


def run(out):
    """Execute all authors/queries/arms once; never fit or select a predictor."""
    manifest = sealed(out)
    if (out / "checkpoint.json").exists():
        raise ValueError("Preserve previous attempt; no automatic rerun")
    write(out / "checkpoint.json", {"status": "RUNNING", "completed_authors": 0})
    started = time.monotonic()
    supports, replay = load_support()
    records, arrays, full_rotation_error = [], {}, 0.0
    for count, (key, s) in enumerate(supports.items(), 1):
        g, h, panel, y = s["g"], s["h"], s["panel"], s["y"]
        for name in ("g", "panel", "comment", "y"):
            arrays[f"{key}__{name}"] = s[name]
        full_residual = None
        for arm in M.ARMS:
            result = M.witness(M.design(g, panel, arm, h), y, manifest["radius"])
            for name in ("plus", "minus", "residual"):
                arrays[f"{key}__{arm}__{name}"] = result[name]
            if arm == "pooled_full":
                full_residual = result["residual"]
            if arm == "rotated_full":
                full_rotation_error = max(
                    full_rotation_error,
                    float(np.max(abs(result["residual"] - full_residual))),
                )
            for j, query in enumerate(s["queries"]):
                records.append(
                    {
                        "author_key": key,
                        "query": query,
                        "fold": s["folds"][j],
                        "arm": arm,
                        "comments": len(g),
                        "rank": result["rank"],
                        "null_dimension": result["null_dimension"],
                        "active": bool(result["active"][j]),
                        "residual_fraction": float(
                            result["residual_energy_fraction"][j]
                        ),
                        "feasible_gap": float(result["gap"][j]),
                        "query_sd": float(y[:, j].std()),
                        "qr_error": result["qr_error"],
                        "summary_error": result["summary_error"],
                    }
                )
        if count % 50 == 0 or count == len(supports):
            write(
                out / "checkpoint.json",
                {
                    "status": "RUNNING",
                    "completed_authors": count,
                    "rows": len(records),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            print(
                {"authors": count, "total": len(supports), "rows": len(records)},
                flush=True,
            )
        if time.monotonic() - started > manifest["max_cpu_wall_seconds"]:
            raise TimeoutError("HM63 bounded CPU budget exceeded; preserve checkpoint")
    if len(records) != 9219 or full_rotation_error > 1e-8:
        raise ValueError("Incomplete battery or full covariance rotation failure")
    table = pd.DataFrame(records)
    if table.duplicated(["author_key", "query", "arm"]).any():
        raise ValueError("Duplicate outcome key")
    table.to_csv(out / "private_author_query_arms.csv", index=False)
    np.savez_compressed(out / "private_witnesses.npz", **arrays)
    aggregate = (
        table.groupby(["query", "arm"], sort=True)
        .agg(
            authors=("author_key", "nunique"),
            active_pairs=("active", "sum"),
            mean_residual_fraction=("residual_fraction", "mean"),
            median_residual_fraction=("residual_fraction", "median"),
            mean_feasible_gap=("feasible_gap", "mean"),
            max_feasible_gap=("feasible_gap", "max"),
            mean_query_sd=("query_sd", "mean"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            min_null_dimension=("null_dimension", "min"),
            max_null_dimension=("null_dimension", "max"),
        )
        .reset_index()
    )
    aggregate.to_csv(out / "aggregate_summary.csv", index=False)
    files = [
        "private_author_query_arms.csv",
        "private_witnesses.npz",
        "aggregate_summary.csv",
    ]
    summary = {
        "status": "COMPLETE_FINITE_HM63_NOT_GLOBAL",
        "authors": len(supports),
        "comments": 9165,
        "author_query_arms": len(records),
        "seconds": time.monotonic() - started,
        "input_replay_max_error": replay,
        "full_covariance_rotation_max_error": full_rotation_error,
        "new_calls_fits_labels": [0, 0, 0],
        "mean_embedding_proxy_control_is_algebraic": True,
        "not_pair_of_actual_people_not_significance_or_personality_effect": True,
        "original_scope_complete": False,
        "files": {name: sha(out / name) for name in files},
    }
    write(out / "summary.json", summary)
    write(
        out / "checkpoint.json", {"status": summary["status"], "completed_authors": 276}
    )
    print(summary, flush=True)


def verify(out):
    """Recalculate all witnesses with raw weighted moments and direct query sums."""
    sealed(out)
    summary = read(out / "summary.json")
    for name, expected in summary["files"].items():
        if sha(out / name) != expected:
            raise ValueError("Outcome changed: " + name)
    supports, replay = load_support()
    table = pd.read_csv(out / "private_author_query_arms.csv")
    table = table.set_index(["author_key", "query", "arm"])
    errors = {
        "raw_moments": 0.0,
        "direct_gap": 0.0,
        "norm_gap": 0.0,
        "probability_mass": 0.0,
        "stored_support": 0.0,
        "residual_fraction": 0.0,
    }
    verified, weight_min, weight_max = 0, 1.0, 0.0
    with np.load(out / "private_witnesses.npz", allow_pickle=False) as z:
        for key, s in supports.items():
            g, h, panel, y = s["g"], s["h"], s["panel"], s["y"]
            for name in ("g", "panel", "comment", "y"):
                errors["stored_support"] = max(
                    errors["stored_support"],
                    float(np.max(abs(z[f"{key}__{name}"] - s[name]))),
                )
            for arm in M.ARMS:
                plus, minus = z[f"{key}__{arm}__plus"], z[f"{key}__{arm}__minus"]
                residual = z[f"{key}__{arm}__residual"]
                for j, query in enumerate(s["queries"]):
                    row = table.loc[(key, query, arm)]
                    p, m = plus[:, j], minus[:, j]
                    weight_min = min(weight_min, float(min(p.min(), m.min()) * len(g)))
                    weight_max = max(weight_max, float(max(p.max(), m.max()) * len(g)))
                    errors["probability_mass"] = max(
                        errors["probability_mass"],
                        float(max(abs(p.sum() - 1), abs(m.sum() - 1))),
                    )
                    raw = raw_moments(g, panel, p, arm, h) - raw_moments(
                        g, panel, m, arm, h
                    )
                    errors["raw_moments"] = max(
                        errors["raw_moments"], float(np.max(abs(raw)))
                    )
                    gap = float(p @ y[:, j] - m @ y[:, j])
                    errors["direct_gap"] = max(
                        errors["direct_gap"], abs(gap - row.feasible_gap)
                    )
                    norm_gap = (
                        np.sum(residual[:, j] ** 2)
                        / (len(g) * np.max(abs(residual[:, j])))
                        if row.active
                        else 0.0
                    )
                    errors["norm_gap"] = max(errors["norm_gap"], abs(gap - norm_gap))
                    power = np.sum((y[:, j] - y[:, j].mean()) ** 2)
                    fraction = (
                        np.sum(residual[:, j] ** 2) / power if power > 1e-20 else 0.0
                    )
                    errors["residual_fraction"] = max(
                        errors["residual_fraction"],
                        abs(fraction - row.residual_fraction),
                    )
                    verified += 1
    if (
        verified != 9219
        or max(errors.values()) > 1e-8
        or weight_min < 0.5 - 1e-9
        or weight_max > 1.5 + 1e-9
    ):
        raise ArithmeticError("Independent weighted-moment audit failed")
    # Aggregate outcome readback, not a second projection execution.
    aggregate = pd.read_csv(out / "aggregate_summary.csv")
    for row in aggregate.itertuples():
        selected = table.xs((row.query, row.arm), level=("query", "arm"))
        if (
            len(selected) != row.authors
            or int(selected.active.sum()) != row.active_pairs
        ):
            raise ValueError("Aggregate count mismatch")
        for field, column, op in (
            ("mean_residual_fraction", "residual_fraction", "mean"),
            ("median_residual_fraction", "residual_fraction", "median"),
            ("mean_feasible_gap", "feasible_gap", "mean"),
            ("max_feasible_gap", "feasible_gap", "max"),
        ):
            if (
                abs(float(getattr(selected[column], op)()) - getattr(row, field))
                > 1e-10
            ):
                raise ValueError("Aggregate value mismatch")
    report = {
        "status": "VERIFIED_FINITE_HM63_NOT_GLOBAL",
        "verified_author_query_arms": verified,
        "errors": errors,
        "input_replay_max_error": replay,
        "weights_times_n_min_max": [weight_min, weight_max],
        "same_agent_alternate_arithmetic_not_independent_replication": True,
    }
    write(out / "verification.json", report)
    print(report, flush=True)


def main():
    """Expose explicit prepare/run/verify stages with no automatic restart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "verify"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        try:
            {"prepare": prepare, "run": run, "verify": verify}[args.stage](args.out)
        except Exception as error:
            if args.out.is_dir() and args.stage != "prepare":
                write(
                    args.out / ("failure_" + args.stage + ".json"),
                    {"status": "FAILED_PRESERVE_ATTEMPT", "error": repr(error)},
                )
            raise


if __name__ == "__main__":
    main()
