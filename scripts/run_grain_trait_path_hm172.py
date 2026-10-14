#!/usr/bin/env python3
"""Join natural directed event observations and source-oriented trait meaning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_trait_path as F
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_trait_anchor_hm171 as P

CONFIG = "configs/grain_trait_path_hm172.json"
DEFAULT = ROOT / "results/grain_math_response_v1/trait_path_observation_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_TRAIT_PATH_PLAN_HM172.md",
    "grain_research/construction/math_trait_path.py",
    "scripts/run_grain_trait_path_hm172.py",
    "scripts/score_grain_trait_path_cold.py",
    "scripts/verify_grain_trait_path_hm172.py",
    "tests/test_grain_trait_path.py",
)
STAGES = ("compiled_seal.json", "cold_seal.json", "analysis_seal.json")
ROLES = ("source_train", "source_validation", "held")


def parent(spec):
    """Authenticate the old target packet and its same-evidence parent."""
    p = ROOT / spec["parent"]
    old = P.check(p, P.STAGES)
    return p, old


def check(out, stages=()):
    """Verify all scientific bytes without mutating or recovering old runs."""
    manifest = R.check(out, stages)
    parent(manifest["config"])
    return manifest


def prepare(out):
    """Freeze the exact old event family before any new trait association."""
    spec = R.read(ROOT / CONFIG)
    p, old = parent(spec)
    data = R.prepared(ROOT / old["config"]["parent"])
    node = R.read(p / "source_law.json")
    P.labels(old["config"])  # Only the already authorized exact alignment.
    states = R.F.states(data["coordinates"].reshape(-1, 15), node["coordinate_cutoffs"])
    states = states.reshape(624, 36) + 4 * data["conditions"]
    paths = np.stack([states[:, a:b] for a, b in spec["windows"]], axis=1)
    if paths.shape != (624, 2, 18) or len(data["pairs"]) != 21216:
        raise ValueError("Changed same-event family")
    if any(
        np.any(np.diff(data["times"][:, a:b], axis=1) <= 0) for a, b in spec["windows"]
    ):
        raise ValueError("Timestamp ties or reversed publication order")
    j = np.asarray([[F.edge_mass(row) for row in u] for u in paths])
    b = np.asarray([[F.fixed_end_mean(row) for row in u] for u in paths])
    source = data["source_train"]
    support = (j[source] > 0).sum((0, 1))
    if np.any(support == 0):
        raise ValueError("Unsupported source edge;no partition/smoothing rescue")
    out.mkdir(parents=True, exist_ok=False)
    parent_files = [
        str((p / name).relative_to(ROOT))
        for name in (
            "manifest.json",
            "source_law.json",
            "source_diagnostics.json",
            "cold_scores.jsonl",
            "compiled_seal.json",
            "cold_seal.json",
            "analysis_seal.json",
        )
    ]
    obs_path = str(
        (ROOT / old["config"]["parent"] / "observations.npz").relative_to(ROOT)
    )
    inputs = parent_files + [
        obs_path,
        old["config"]["labels"],
        old["config"]["material"],
    ]
    R.write(
        out / "manifest.json",
        {
            "config": spec,
            "code": {k: R.digest(ROOT / k) for k in SCIENCE},
            "inputs": {k: R.digest(ROOT / k) for k in inputs},
            "new_application_association_examined_before_seal": False,
            "original_scope_complete": False,
        },
    )
    R.archive(
        out / "observations.npz",
        paths=paths,
        joint=j,
        bag=b,
        **{k: data[k] for k in ROLES},
    )
    with (out / "cold_inputs.jsonl").open("x") as f:
        for u in range(624):
            for w, (start, stop) in enumerate(spec["windows"]):
                record = {
                    "author_key": f"window_{u}_{w}",
                    "events": [
                        {
                            "event_id": f"event_{u}_{int(data['original_units'][u, t])}",
                            "condition": int(data["conditions"][u, t]),
                            "coordinates": data["coordinates"][u, t].tolist(),
                        }
                        for t in range(start, stop)
                    ],
                }
                f.write(json.dumps(record, allow_nan=False) + "\n")
    quality = {
        "authors": 624,
        "windows": 1248,
        "records": 22464,
        "edges": 21216,
        "role_counts": {k: len(data[k]) for k in ROLES},
        "source_edge_window_support": support.tolist(),
        "source64_edges_supported": True,
        "timestamp_ties": 0,
        "same_coordinate_cuts_and_conditions": True,
        "new_neural_forwards": 0,
        "raw_questionnaire_or_text_reads": 0,
        "history_reused_not_lockbox": True,
        "reorder_is_numeric_not_new_text_or_causal_action": True,
    }
    R.write(out / "input_quality.json", quality)
    R.seal(
        out,
        "prepared_seal.json",
        ("observations.npz", "cold_inputs.jsonl", "input_quality.json"),
    )
    print(json.dumps(quality))


def observations(out):
    """Read locally retained directed ledgers without external scale outcomes."""
    return R.prepared(out)


def compile_source(out):
    """Build one ordered observer, exact bag/end observer and fixed label null."""
    manifest = check(out)
    p, old = parent(manifest["config"])
    z, node = observations(out), R.read(p / "source_law.json")
    src = z["source_train"]
    binary = (P.labels(old["config"])[src] > node["label_cutoffs"]).astype(int)
    cuts = node["coordinate_cutoffs"]
    perm = np.asarray(
        R.read(p / "source_diagnostics.json")["source_profile_permutation"]
    )
    matrices = {
        "ordered_source": z["joint"][src].mean(1),
        "bag_endpoint_source": z["bag"][src].mean(1),
        "permuted_source": z["joint"][src].mean(1),
    }
    files, laws = [], {}
    for name, mass in matrices.items():
        law = F.compile_law(
            mass,
            binary[perm] if name == "permuted_source" else binary,
            cuts,
            node,
            name,
        )
        F.verify_law(law)
        R.write(out / f"{name}.json", law)
        laws[name] = law
        files.append(f"{name}.json")
    halves = [np.arange(len(src))[k::2] for k in range(2)]
    half_laws = [
        F.compile_law(matrices["ordered_source"][ix], binary[ix], cuts, node, "half")
        for ix in halves
    ]
    for law in half_laws:
        F.verify_law(law)
    R.write(
        out / "source_half_laws.json",
        {"source_positions": [h.tolist() for h in halves], "laws": half_laws},
    )
    cycles = [F.parts(np.asarray(law["field"]))[2] for law in half_laws]
    cross = (cycles[0] * cycles[1]).sum((0, 1))
    norms = [np.sqrt((c**2).sum((0, 1))) for c in cycles]
    cycle_cos = [
        None
        if norms[0][q] * norms[1][q] == 0
        else float(cross[q] / (norms[0][q] * norms[1][q]))
        for q in range(5)
    ]
    full = F.parts(np.asarray(laws["ordered_source"]["field"]))
    node_kernel = np.repeat(np.asarray(node["probabilities"])[:, None, :], 8, axis=1)
    diag = {
        "source_half_cycle_cosine": cycle_cos,
        "source_half_cycle_norms": [a.tolist() for a in norms],
        "ordered_field_part_squared_norms": [
            np.sum(f**2, axis=(0, 1)).tolist() for f in full
        ],
        "old_node_reader_cycle_max": float(np.max(np.abs(F.parts(node_kernel)[2]))),
        "same_HM171_source_label_permutation": perm.tolist(),
        "fields_not_selected_or_refit": True,
        "source_binary_positivity_is_constructed_not_validation": True,
    }
    R.write(out / "source_diagnostics.json", diag)
    R.seal(
        out,
        "compiled_seal.json",
        (*files, "source_half_laws.json", "source_diagnostics.json"),
    )
    print(
        json.dumps(
            {
                "source_authors": len(src),
                "half_cycle_cosine": cycle_cos,
                "old_node_cycle_max": diag["old_node_reader_cycle_max"],
            }
        )
    )


def cold(out):
    """Use an independent outcome-free CLI for all three predetermined readers."""
    spec = check(out, STAGES[:1])["config"]
    for name in spec["readers"]:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_trait_path_cold.py"),
                "--bundle",
                str(out / f"{name}.json"),
                "--sha256",
                R.digest(out / f"{name}.json"),
                "--input",
                str(out / "cold_inputs.jsonl"),
                "--output",
                str(out / f"{name}_scores.jsonl"),
            ],
            check=True,
            timeout=spec["max_seconds"],
        )
    R.seal(out, "cold_seal.json", tuple(f"{k}_scores.jsonl" for k in spec["readers"]))


def score_arrays(out, name):
    """Read all outcomes, checking opaque author/window identity and no labels."""
    rows = [
        json.loads(line)
        for line in (out / f"{name}_scores.jsonl").read_text().splitlines()
    ]
    if len(rows) != 1248 or any(
        row["author_key"] != f"window_{i // 2}_{i % 2}"
        or row["labels_received"] is not False
        or row["reader"] != name
        for i, row in enumerate(rows)
    ):
        raise ValueError("Changed cold ledger addresses")
    return {
        k: np.asarray([r["accounts"][k] for r in rows]).reshape(624, 2, 5)
        for k in rows[0]["accounts"]
    }


def permutation_scores(paths, field, expected, seed, count):
    """Move entire internal events coherently; keep bags/endpoints and reader."""
    rng = np.random.default_rng(seed)
    cycle = F.parts(field)[2]
    fixed_cycle = np.einsum("uwij,ijq->uwq", expected, cycle)
    fixed_full = np.einsum("uwij,ijq->uwq", expected, field)
    scores = np.empty((count, 624, 2, 5, 2))
    for k in range(count):
        indices = np.argsort(rng.random(paths[:, :, 1:-1].shape), axis=-1)
        inner = np.take_along_axis(paths[:, :, 1:-1], indices, axis=-1)
        p = np.concatenate((paths[:, :, :1], inner, paths[:, :, -1:]), axis=-1)
        scores[k, ..., 0] = cycle[p[:, :, :-1], p[:, :, 1:]].mean(2) - fixed_cycle
        scores[k, ..., 1] = field[p[:, :, :-1], p[:, :, 1:]].mean(2) - fixed_full
    return scores


def analyze(out):
    """Keep every target/component and null; never choose a larger correlation."""
    manifest = check(out, STAGES[:2])
    started = time.perf_counter()
    p, old = parent(manifest["config"])
    y, z, spec = P.labels(old["config"]), observations(out), manifest["config"]
    scores = {k: score_arrays(out, k) for k in spec["readers"]}
    old_scores = P.score_arrays(p / "cold_scores.jsonl")["relative"]
    rows, continuity, errors = [], [], []
    for reader, a in scores.items():
        errors += [
            float(np.max(np.abs(a["relative"] - a["bag_expected"] - a["reorder"]))),
            float(
                np.max(
                    np.abs(a["relative"] - a["traffic"] - a["endpoint"] - a["cycle"])
                )
            ),
            float(
                np.max(np.abs(a["reorder"] - a["reorder_traffic"] - a["reorder_cycle"]))
            ),
            float(np.max(np.abs(a["reorder_endpoint"]))),
            float(np.max(np.abs(a["old_relative"] - old_scores))),
        ]
        for role in ROLES:
            ids = z[role]
            for q, trait in enumerate(F.TRAITS):
                for component, values in a.items():
                    zero = component == "reorder_endpoint"
                    continuity.append(
                        {
                            "reader": reader,
                            "role": role,
                            "trait": trait,
                            "component": component,
                            "authors": len(ids),
                            "early_late_r": None
                            if zero
                            else R.correlation(values[ids, 0, q], values[ids, 1, q]),
                        }
                    )
                    for w in range(2):
                        value = values[ids, w, q]
                        r = None if zero else R.correlation(value, y[ids, q])
                        interval = P.fisher_interval(r, len(ids))
                        rows.append(
                            {
                                "reader": reader,
                                "role": role,
                                "window": w,
                                "trait": trait,
                                "component": component,
                                "authors": len(ids),
                                "r": r,
                                "fisher95_low": interval[0],
                                "fisher95_high": interval[1],
                                "covariance": P.covariance(value, y[ids, q]),
                                "score_mean": float(value.mean()),
                                "score_sd": float(value.std()),
                                "constant_by_algebra": zero,
                            }
                        )
    R.table(out / "metrics.csv", rows)
    R.table(out / "continuity.csv", continuity)
    law = R.read(out / "ordered_source.json")
    null = permutation_scores(
        z["paths"],
        np.asarray(law["field"]),
        z["bag"],
        spec["seed"],
        spec["permutations"],
    )
    R.archive(out / "null_scores.npz", scores=null)
    null_rows, null_r = [], np.empty((3, 2, 5, 2, spec["permutations"]))
    for ri, role in enumerate(ROLES):
        ids = z[role]
        for w in range(2):
            for q, trait in enumerate(F.TRAITS):
                for k, component in enumerate(("reorder_cycle", "reorder")):
                    obs = R.correlation(
                        scores["ordered_source"][component][ids, w, q], y[ids, q]
                    )
                    vals = np.asarray(
                        [R.correlation(row[ids, w, q, k], y[ids, q]) for row in null],
                        float,
                    )
                    null_r[ri, w, q, k] = vals
                    tail = (
                        None
                        if obs is None
                        else float(
                            (1 + np.sum(np.abs(vals) >= abs(obs))) / (len(vals) + 1)
                        )
                    )
                    null_rows.append(
                        {
                            "role": role,
                            "window": w,
                            "trait": trait,
                            "component": component,
                            "observed_r": obs,
                            "null_r_mean": float(vals.mean()),
                            "two_sided_tail_rank": tail,
                            "permutations": len(vals),
                        }
                    )
    R.table(out / "order_null_metrics.csv", null_rows)
    R.archive(out / "null_correlations.npz", correlations=null_r)
    elapsed = time.perf_counter() - started
    summary = {
        "trait_order": list(F.TRAITS),
        "readers": spec["readers"],
        "roles": {k: len(z[k]) for k in ROLES},
        "windows": 1248,
        "edges": 21216,
        "metric_rows": len(rows),
        "continuity_rows": len(continuity),
        "null_rows": len(null_rows),
        "max_account_and_old_node_error": max(errors),
        "analysis_seconds": elapsed,
        "new_neural_forwards": 0,
        "cohort_reused_not_lockbox": True,
        "source_binary_positivity_is_constructed": True,
        "null_is_fixed_bag_and_end_order_not_all_opportunity_laws": True,
        "no_target_or_component_selection": True,
        "original_scope_complete": False,
    }
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analysis_seal.json",
        (
            "metrics.csv",
            "continuity.csv",
            "null_scores.npz",
            "null_correlations.npz",
            "order_null_metrics.csv",
            "summary.json",
        ),
    )
    print(json.dumps(summary))
    if elapsed > spec["max_seconds"]:
        raise RuntimeError("CPU cap exceeded;completed output retained,do not rerun")


def main():
    """Run exactly one explicit immutable stage; no implicit research queue."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "compile", "cold", "analyze", "check"))
    p.add_argument("--out", type=Path, default=DEFAULT)
    a = p.parse_args()
    if a.stage == "check":
        check(a.out, STAGES)
        print("ALL_SCIENTIFIC_SEALS_UNCHANGED")
    else:
        {
            "prepare": prepare,
            "compile": compile_source,
            "cold": cold,
            "analyze": analyze,
        }[a.stage](a.out)


if __name__ == "__main__":
    main()
