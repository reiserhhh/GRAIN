#!/usr/bin/env python3
"""HM57: replay frozen logits with fixed pools; no neural or fitting imports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_event_probability_pooling as M
from grain_research.evidence_history import active_records

SPEC = ROOT / "configs/grain_math_event_probability_pooling_v1.json"
OUT = ROOT / "results/grain_math_response_v1/event_probability_pooling_r1"
CODE = (
    "configs/grain_math_event_probability_pooling_v1.json",
    "docs/GRAIN_EVENT_PROBABILITY_POOLING_PLAN_V1.md",
    "grain_research/construction/math_event_probability_pooling.py",
    "grain_research/evidence_history.py",
    "scripts/run_grain_math_event_probability_pooling.py",
    "scripts/verify_grain_math_event_probability_pooling.py",
    "tests/test_grain_math_event_probability_pooling.py",
)
FIELDS = (
    "unit",
    "parent_unit",
    "role",
    "content",
    "query",
    "family",
    "polarity",
    "order",
    "regime",
)


def read(path):
    """Read local JSON without executing or loading a model."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash local evidence in bounded chunks, including preserved vector arrays."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    """Save a generated research artifact with strict JSON number handling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    """Save the full fixed roster as CSV, preserving null rather than fake zero."""
    if not rows:
        raise ValueError("Empty research tables are not complete")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_snapshot(spec):
    """Bind every candidate logit to original frozen inputs and raw metadata."""
    source = ROOT / spec["source"]
    manifest = read(source / "00/manifest.json")
    if (
        read(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM56_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM56 required")
    if (
        read(source / "binding_verification.json")["status"]
        != "VERIFIED_HM56_RAW_COMPONENT_BINDING"
    ):
        raise ValueError("HM56 raw binding companion required")
    if read(source / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("No partial or live acquisition may be used")
    lineage = {}

    def bind(path, expected=None):
        path = Path(path)
        name = str(path.relative_to(ROOT))
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError("Frozen source drift: " + name)
        lineage[name] = actual

    for name in (
        "00/manifest.json",
        "verification.json",
        "binding_verification.json",
        "run_status.json",
    ):
        bind(source / name)
    for name, expected in manifest["code"].items():
        bind(ROOT / name, expected)
        bind(source / "00/code" / name, expected)
    for name, expected in manifest["assets"].items():
        bind(source / name, expected)
    parent = ROOT / manifest["spec"]["parent"]
    for name, expected in manifest["parent_files"].items():
        bind(parent / name, expected)
    original_hashes = read(source / "03/artifact_hashes.json")
    bind(source / "03/artifact_hashes.json")
    for name, expected in original_hashes.items():
        bind(source / name, expected)
    rows = read(source / "00/rows.json")
    if [r["unit"] for r in rows] != list(range(spec["expected_readouts"])):
        raise ValueError("Full canonical readout roster required")
    values = np.asarray(spec["candidate_values"], float) / 9
    if manifest["alphabet"]["values"] != values.tolist():
        raise ValueError("Candidate alphabet changed")
    logits, paths, maximum = [], [], 0.0
    for row in rows:
        if row["reused"]:
            item = manifest["reused"][str(row["parent_unit"])]
            path = parent / item["meta"]
            bind(path, item["meta_sha256"])
            expected_array = item["array_sha256"]
            raw = read(path)
            if (
                row["query"] != row["content"]
                or raw["unit"] != row["parent_unit"]
                or raw["target"] != row["content"]
            ):
                raise ValueError("Reused raw observation identity mismatch")
        else:
            path = source / f"01/units/unit_{row['unit']:04d}.json"
            bind(path)
            raw = read(path)
            if any(raw[k] != row[k] for k in FIELDS):
                raise ValueError("Fresh raw observation identity mismatch")
            expected_array = raw["array_sha256"]
        if raw["array_sha256"] != expected_array:
            raise ValueError("Original array metadata mismatch")
        bind(path.with_suffix(".npz"), expected_array)
        a = np.asarray(raw["native"]["candidate_logits"], float)
        p = M.probabilities(a)
        if a.shape != (10,):
            raise ValueError("Exactly ten raw candidate logits required")
        maximum = max(
            maximum,
            abs(float(p @ values) - raw["native"]["conditional_score"]),
            float(np.max(np.abs(p - raw["native"]["conditional_probabilities"]))),
        )
        logits.append(a)
        paths.append(str(path.relative_to(ROOT)))
    if maximum > spec["arithmetic_tolerance"]:
        raise ValueError("Raw candidate readout replay failed")
    return rows, np.asarray(logits), paths, lineage, manifest["targets"], maximum


def qualified_weights(row, basis):
    """Count own current events or own mentions using explicit history identities."""
    history = row["history"]
    if basis == "event":
        records = [r for r in active_records(history)["records"] if r["actor"] == "A"]
        if len({r["event_id"] for r in records}) != len(records):
            raise ValueError("Finite design needs one current claim per occasion")
    elif basis == "mention":
        records = [
            r
            for r in history
            if r["actor"] == "A" and r["kind"] in {"new", "alias", "revise"}
        ]
    else:
        raise ValueError("Unknown declared basis")
    if not records:
        raise ValueError("Missing own evidence is not a zero score")
    counts = Counter((r["family"], r["polarity"]) for r in records)
    keys = sorted(counts)
    return keys, [counts[k] / len(records) for k in keys], len(records)


def prepare(out):
    """Freeze CPU-only formulas and complete source links before new summaries."""
    if out.exists():
        raise FileExistsError("Preserve prior runs")
    spec = read(SPEC)
    rows, logits, paths, lineage, targets, maximum = source_snapshot(spec)
    if spec["max_new_neural_calls"] != 0 or spec["fit_search_API_training_private"]:
        raise ValueError("Zero-call no-fit packet required")
    if (
        len(targets) != spec["expected_queries"]
        or sum(r["role"] == "atom" for r in rows) != spec["expected_atom_readouts"]
    ):
        raise ValueError("Incomplete atom/query roster")
    for folder in ("00/code", "03"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    write(out / "00/rows.json", rows)
    write(out / "00/raw_paths.json", paths)
    np.save(out / "00/raw_logits.npy", logits, allow_pickle=False)
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = sha(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM57",
            "spec": spec,
            "targets": targets,
            "values": (np.asarray(spec["candidate_values"], float) / 9).tolist(),
            "code": code,
            "source_hashes": lineage,
            "assets": {
                n: sha(out / n)
                for n in ("00/rows.json", "00/raw_paths.json", "00/raw_logits.npy")
            },
            "raw_replay_max_error": maximum,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_is_result_after_reference": True,
            "new_neural_calls": 0,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "new_neural_calls": 0,
            "fit_calls": 0,
            "API_calls": 0,
            "source_readouts": len(rows),
        },
    )
    print("PREPARED HM57: all 1296 raw readouts, no new model calls")


def check(out):
    """Check frozen new code and every linked source without inference."""
    m = read(out / "00/manifest.json")
    for name, expected in m["code"].items():
        if sha(ROOT / name) != expected or sha(out / "00/code" / name) != expected:
            raise ValueError("Frozen HM57 code drift: " + name)
    for name, expected in m["assets"].items():
        if sha(out / name) != expected:
            raise ValueError("Frozen HM57 input drift: " + name)
    for name, expected in m["source_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Linked HM56 source drift: " + name)
    return m


def pairs(vectors, m):
    """Enumerate all regime and order pairs with exact shared query coordinates."""
    for c, f, p, order in itertools.product(m["targets"], (0, 1), (0, 1), (0, 1)):
        for name, hi, lo in m["spec"]["contrasts"]:
            a, b = vectors[c, f, p, order, hi], vectors[c, f, p, order, lo]
            yield (
                {
                    "content": c,
                    "family": f,
                    "polarity": p,
                    "order": order,
                    "contrast": name,
                    "hi": a["parent_unit"],
                    "lo": b["parent_unit"],
                },
                a,
                b,
            )
    for c, f, p, regime in itertools.product(
        m["targets"], (0, 1), (0, 1), m["spec"]["regimes"]
    ):
        a, b = vectors[c, f, p, 1, regime], vectors[c, f, p, 0, regime]
        yield (
            {
                "content": c,
                "family": f,
                "polarity": p,
                "order": -1,
                "contrast": "order_reversed_" + regime,
                "hi": a["parent_unit"],
                "lo": b["parent_unit"],
            },
            a,
            b,
        )


def rms(values):
    """Return descriptive fixed-material RMS, never a population effect size."""
    a = np.asarray(values, float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Nonempty finite rows required")
    return float(np.sqrt(np.mean(a**2)))


def summaries(scalars, collisions, tolerance):
    """Describe all fixed references and collision lower bounds without selection."""
    aggregation = []
    for basis, regime in itertools.product(
        ("event", "mention"),
        ("balanced", "alias_focus", "distinct_focus", "other_focus"),
    ):
        chosen = [r for r in scalars if r["basis"] == basis and r["regime"] == regime]
        for query in ("ALL", *dict.fromkeys(r["query"] for r in chosen)):
            rows = (
                chosen if query == "ALL" else [r for r in chosen if r["query"] == query]
            )
            aggregation.append(
                {
                    "basis": basis,
                    "regime": regime,
                    "query": query,
                    "scalar_rows": len(rows),
                    "native_minus_mixture_rms": rms([r["total_change"] for r in rows]),
                    "geometric_minus_mixture_rms": rms(
                        [r["pooling_change"] for r in rows]
                    ),
                    "native_minus_geometric_rms": rms(
                        [r["fixed_reference_residual"] for r in rows]
                    ),
                    "signed_cross_mean": float(
                        np.mean([r["score_cross"] for r in rows])
                    ),
                    "native_mixture_tv_mean": float(
                        np.mean([r["native_mixture_tv"] for r in rows])
                    ),
                    "native_geometric_tv_mean": float(
                        np.mean([r["native_geometric_tv"] for r in rows])
                    ),
                    "geometric_outside_scalar_endpoints": sum(
                        r["geometric_outside_endpoints"] for r in rows
                    ),
                    "native_outside_scalar_endpoints": sum(
                        r["native_outside_endpoints"] for r in rows
                    ),
                }
            )
    collision_summary = []
    for contrast, basis, pool in itertools.product(
        dict.fromkeys(r["contrast"] for r in collisions),
        ("event", "mention"),
        ("mixture", "geometric"),
    ):
        rows = [
            r
            for r in collisions
            if (r["contrast"], r["basis"], r["pool"]) == (contrast, basis, pool)
        ]
        same = [r for r in rows if r["same_summary"]]
        collision_summary.append(
            {
                "contrast": contrast,
                "basis": basis,
                "pool": pool,
                "paired_cases": len(rows),
                "same_summary_count": len(same),
                "same_summary_native_changes": sum(
                    r["native_profile_change_norm"] > tolerance for r in same
                ),
                "maximum_summary_difference": max(
                    r["summary_max_difference"] for r in rows
                ),
                "native_profile_change_rms": rms(
                    [r["native_profile_change_norm"] for r in rows]
                ),
                "minimum_common_profile_rms": rms(
                    [r["minimum_common_profile_rms"] for r in same]
                )
                if same
                else None,
                "native_probability_tv_mean": float(
                    np.mean([r["native_probability_tv_mean"] for r in rows])
                ),
            }
        )
    return aggregation, collision_summary


def analyze(out):
    """Replay every raw distribution, reference geometry and same-summary pair."""
    m = check(out)
    if read(out / "run_status.json")["status"] != "PREPARED":
        raise ValueError("Analyze once; preserve prior terminal artifacts")
    start = time.monotonic()
    rows, logits = (
        read(out / "00/rows.json"),
        np.load(out / "00/raw_logits.npy", allow_pickle=False),
    )
    targets, v = m["targets"], np.asarray(m["values"])
    atoms, accounts = {}, {}
    for r in rows:
        if r["role"] == "atom":
            key = r["content"], r["family"], r["polarity"], r["query"]
            if key in atoms:
                raise ValueError("Duplicate isolated evidence/query")
            atoms[key] = logits[r["unit"]]
        else:
            group = accounts.setdefault(r["parent_unit"], {})
            if r["query"] in group:
                raise ValueError("Duplicate account/query")
            group[r["query"]] = r
    if len(accounts) != m["spec"]["expected_accounts"]:
        raise ValueError("Missing whole-account profiles")
    scalars, probability_rows, vectors = [], [], {}
    for unit, group in accounts.items():
        if set(group) != set(targets):
            raise ValueError("Incomplete six-query profile")
        r = group[targets[0]]
        if any(
            any(
                doc[k] != r[k]
                for k in (
                    "content",
                    "family",
                    "polarity",
                    "order",
                    "regime",
                    "body",
                    "history",
                )
            )
            for doc in group.values()
        ):
            raise ValueError("Query crossing changed the underlying account")
        key = r["content"], r["family"], r["polarity"], r["order"], r["regime"]
        result = {
            "parent_unit": unit,
            "native": np.array(
                [M.probabilities(logits[group[q]["unit"]]) for q in targets]
            ),
        }
        base = {
            k: r[k]
            for k in ("parent_unit", "content", "family", "polarity", "order", "regime")
        }
        for basis in m["spec"]["bases"]:
            keys, w, count = qualified_weights(r, basis)
            per_query = []
            for j, q in enumerate(targets):
                a = np.array([atoms[r["content"], f, p, q] for f, p in keys])
                d = M.decomposition(result["native"][j], a, w, v)
                ends = M.probabilities(a) @ v
                scalar = {
                    k: value
                    for k, value in d.items()
                    if not isinstance(value, np.ndarray)
                }
                scalar.update(
                    count=count,
                    focal_weight=w[keys.index((r["family"], r["polarity"]))],
                    geometric_outside_endpoints=bool(
                        d["geometric"] < min(ends) - m["spec"]["arithmetic_tolerance"]
                        or d["geometric"]
                        > max(ends) + m["spec"]["arithmetic_tolerance"]
                    ),
                    native_outside_endpoints=bool(
                        d["native"] < min(ends) - m["spec"]["arithmetic_tolerance"]
                        or d["native"] > max(ends) + m["spec"]["arithmetic_tolerance"]
                    ),
                )
                path = M.curve_derivatives(
                    atoms[r["content"], r["family"], r["polarity"], q],
                    atoms[r["content"], 1 - r["family"], 1 - r["polarity"], q],
                    scalar["focal_weight"],
                    v,
                )
                scalar.update(
                    e_path_first=path["first"],
                    e_path_second=path["second"],
                    e_path_identity_error=abs(path["score"] - d["geometric"]),
                )
                scalars.append(base | {"query": q, "basis": basis} | scalar)
                for digit in range(10):
                    n, b, g = (
                        d[k][digit]
                        for k in (
                            "native_probability",
                            "mixture_probability",
                            "geometric_probability",
                        )
                    )
                    probability_rows.append(
                        base
                        | {
                            "query": q,
                            "basis": basis,
                            "digit": digit,
                            "native": n,
                            "mixture": b,
                            "geometric": g,
                            "pooling_change": g - b,
                            "fixed_reference_residual": n - g,
                            "total_change": n - b,
                            "cross": 2 * (g - b) * (n - g),
                        }
                    )
                per_query.append(d)
            result[basis] = {
                pool: np.array([d[pool + "_probability"] for d in per_query])
                for pool in ("mixture", "geometric")
            }
        vectors[key] = result
    collisions, changes = [], []
    for base, hi, lo in pairs(vectors, m):
        for basis, pool in itertools.product(
            m["spec"]["bases"], ("mixture", "geometric")
        ):
            c = M.collision(
                hi["native"],
                lo["native"],
                hi[basis][pool],
                lo[basis][pool],
                v,
                m["spec"]["summary_equality_tolerance"],
            )
            collisions.append(base | {"basis": basis, "pool": pool} | c)
        for basis in m["spec"]["bases"]:
            n, b, g = (
                (hi[k] - lo[k]) @ v
                if k == "native"
                else (hi[basis][k] - lo[basis][k]) @ v
                for k in ("native", "mixture", "geometric")
            )
            for j, q in enumerate(targets):
                changes.append(
                    base
                    | {
                        "basis": basis,
                        "query": q,
                        "native_change": n[j],
                        "mixture_change": b[j],
                        "geometric_change": g[j],
                        "pooling_change": g[j] - b[j],
                        "fixed_reference_residual": n[j] - g[j],
                        "total_change": n[j] - b[j],
                        "cross": 2 * (g[j] - b[j]) * (n[j] - g[j]),
                    }
                )
    aggregation, collision_summary = summaries(
        scalars, collisions, m["spec"]["arithmetic_tolerance"]
    )
    outputs = {
        "scalar_decomposition": scalars,
        "probability_decomposition": probability_rows,
        "paired_changes": changes,
        "collisions": collisions,
        "pooling_summary": aggregation,
        "collision_summary": collision_summary,
    }
    for name, data in outputs.items():
        table(out / f"03/{name}.csv", data)
    summary = {
        "packet": "HM57",
        "status": "ANALYZED_FINITE_NOT_GLOBAL",
        "source_readouts": len(rows),
        "accounts": len(accounts),
        "scalar_decompositions": len(scalars),
        "probability_coordinates": len(probability_rows),
        "paired_change_coordinates": len(changes),
        "collision_certificates": len(collisions),
        "new_neural_calls": 0,
        "fit_calls": 0,
        "API_calls": 0,
        "cpu_analysis_seconds": time.monotonic() - start,
        "maximum_identity_error": max(
            r[k]
            for r in scalars
            for k in (
                "score_identity_error",
                "score_energy_error",
                "probability_energy_error",
                "mixture_kl_identity_error",
                "geometric_kl_identity_error",
                "e_path_identity_error",
            )
        ),
        "pooling_results": aggregation,
        "collision_results": collision_summary,
        "references_not_causal_fractions": True,
        "is_personality_accuracy": False,
        "original_scope_complete": False,
    }
    write(out / "03/summary.json", summary)
    write(
        out / "03/artifact_hashes.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in sorted((out / "03").iterdir())
            if p.name != "artifact_hashes.json"
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "ANALYZED",
            "new_neural_calls": 0,
            "fit_calls": 0,
            "API_calls": 0,
            "source_readouts": len(rows),
            "cpu_analysis_seconds": summary["cpu_analysis_seconds"],
        },
    )
    print(
        {
            k: v
            for k, v in summary.items()
            if k not in ("pooling_results", "collision_results")
        }
    )


def main():
    """Expose prepare/analyze only: this runner has no collect or fitting stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze}[args.phase](args.output)


if __name__ == "__main__":
    main()
