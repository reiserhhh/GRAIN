#!/usr/bin/env python3
"""Same matched action on three fixed caches, with signed source evidence ledgers."""

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
from grain_research.construction import math_action_representation as D
from grain_research.construction import math_response_function as M
from scripts import run_grain_matched_transition_hm168 as T

A, R = T.A, T.R
DEFAULT = ROOT / "results/grain_math_response_v1/action_representation_r1"
CONFIG = ROOT / "configs/grain_action_representation_hm169.json"
SCIENCE = (
    "configs/grain_action_representation_hm169.json",
    "docs/GRAIN_ACTION_REPRESENTATION_PLAN_HM169.md",
    "grain_research/construction/math_action_representation.py",
    "tests/test_grain_action_representation.py",
    "scripts/run_grain_action_representation_hm169.py",
    "scripts/verify_grain_action_representation_hm169.py",
    "scripts/score_grain_response_function.py",
    "grain_research/construction/math_response_function.py",
)


def require_seal(base, name):
    """Authenticate immutable artifact dictionaries without writing to predecessors."""
    for path, expected in A.read(base / name).items():
        if A.digest(base / path) != expected:
            raise ValueError("Sealed scientific artifact changed: " + str(base / path))


def check(out):
    """Check fixed methods, inputs, row identities and the full predecessor lineage."""
    manifest = A.read(out / "manifest.json")
    T.check(ROOT / manifest["config"]["predecessor"])
    for section, base in (
        ("science_hashes", ROOT),
        ("input_hashes", ROOT),
        ("prepared_hashes", out),
    ):
        for name, expected in manifest[section].items():
            if A.digest(base / name) != expected:
                raise ValueError("Frozen file changed: " + name)
    return manifest


def pair_rows(articles):
    """Require exactly one high and low observation per complete article pair."""
    grouped = {}
    for i, meta in enumerate(articles):
        ids = grouped.setdefault(meta["pair_id"], {})
        if meta["pole"] in ids or meta["pole"] not in (-1, 1):
            raise ValueError("Duplicate or unknown article pole")
        ids[meta["pole"]] = i
    result = []
    for ids in grouped.values():
        if set(ids) != {-1, 1}:
            raise ValueError("Incomplete pair cannot be a dependence observation")
        lo, hi = ids[-1], ids[1]
        keys = ("pair_id", "axis", "family", "role", "situation", "style", "seed")
        if any(articles[lo][k] != articles[hi][k] for k in keys):
            raise ValueError("High/low records do not share production addresses")
        result.append({k: articles[hi][k] for k in keys} | {"lo": lo, "hi": hi})
    return result


def paired(values, pairs):
    """Take high-minus-low differences in the registered whole-profile order."""
    return np.array([values[p["hi"]] - values[p["lo"]] for p in pairs])


def prepare(out):
    """Freeze post-outcome methods and exact same-evidence numeric representations."""
    config = A.read(CONFIG)
    parent, vector = ROOT / config["predecessor"], ROOT / config["vector_source"]
    old = T.check(parent)
    for seal in (
        "source_capture_seal.json",
        "application_capture_seal.json",
        "analysis_seal.json",
    ):
        require_seal(parent, seal)
    for seal in ("generation_seal.json", "observation_seal.json"):
        require_seal(vector, seal)
    for stage in ("source", "application"):
        if A.read(parent / (stage + "_collection_status.json"))["status"] != "COMPLETE":
            raise ValueError("Only complete terminal matched actions admitted")
    if any(
        A.read(vector / (stage + "_status.json"))["status"] != "COMPLETE"
        for stage in ("generation", "observation")
    ):
        raise ValueError("Historical vector cache not complete")
    material = A.read(vector / "materials.json")
    source = A.read(parent / "source_articles.json")
    application = A.read(parent / "application_articles.json")
    if (
        len(source) != config["source_rows"]
        or len(application) != config["application_rows"]
    ):
        raise ValueError("Current support cannot be reselected")
    ids = [m["old_index"] for m in source + application]
    if len(set(ids)) != len(ids) or any(
        material[i]["text_id"] != m["text_id"]
        for i, m in zip(ids, source + application, strict=True)
    ):
        raise ValueError(
            "Vector row/text identity collision or source/application overlap"
        )
    if {m["role"] for m in source} != {"discovery"} or {m["axis"] for m in source} != {
        "extraversion"
    }:
        raise ValueError("Historical E-only source support changed")
    pairs = {"source": pair_rows(source), "application": pair_rows(application)}
    if (
        len(pairs["source"]) != config["source_pairs"]
        or len(pairs["application"]) != config["application_pairs"]
    ):
        raise ValueError("Complete whole-pair support changed")
    starts = T.P.profiles(parent, "source")[0]
    app_starts = T.P.profiles(parent, "application")[0]
    context = ROOT / old["config"]["context_predecessor"]
    app_ends = T.P.profiles(context, "application")[0]
    with np.load(parent / "source.npz", allow_pickle=False) as saved:
        grain_source, actions, groups = (
            saved["features"].copy(),
            saved["responses"] - starts,
            saved["groups"].copy(),
        )
    with np.load(parent / "application_features.npz", allow_pickle=False) as saved:
        grain_application = saved["features"].copy()
    app_actions = app_ends - app_starts
    if not np.isfinite(actions).all() or not np.isfinite(app_actions).all():
        raise ValueError("Missing observations remain missing; no zero filling")
    expected_groups = np.array([m["situation"] + "|" + m["style"] for m in source])
    np.testing.assert_array_equal(groups, expected_groups)
    prior_pairs = A.read(parent / "complete_pairs.json")
    if [p["pair_id"] for p in pairs["application"]] != [
        p["pair_id"] for p in prior_pairs
    ]:
        raise ValueError("Native pair order changed")
    with np.load(parent / "paired_actions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(
            M.order_fields(paired(app_actions, pairs["application"])), saved["native"]
        )
    views = {"grain_tensor625": (grain_source, grain_application)}
    units = {}
    with np.load(vector / "numeric_observations.npz", allow_pickle=False) as saved:
        for name in config["views"][1:]:
            x = saved[name].astype(float)
            if x.shape != (702, 1536) or not np.isfinite(x).all():
                raise ValueError("Full historical contextual cache required")
            xs, xa = x[ids[: len(source)]], x[ids[len(source) :]]
            center = xs.mean(0)
            scale = float(np.sqrt(np.mean(np.sum((xs - center) ** 2, axis=1))))
            if scale <= 0:
                raise ValueError("Source view has no declared Euclidean unit")
            views[name] = ((xs - center) / scale, (xa - center) / scale)
            units[name] = {"origin": center, "scale": np.array(scale)}
    out.mkdir(parents=True, exist_ok=False)
    names = [
        "source_actions.npz",
        "application_actions.npz",
        "source_articles.json",
        "application_articles.json",
        "pairs.json",
    ]
    R.npz(out / names[0], responses=actions, groups=groups)
    R.npz(out / names[1], responses=app_actions)
    A.write(out / names[2], source)
    A.write(out / names[3], application)
    A.write(out / names[4], pairs)
    for name, (xs, xa) in views.items():
        folder = out / name
        folder.mkdir()
        for filename, values in (
            ("source_features.npz", {"features": xs}),
            ("application_features.npz", {"features": xa}),
        ):
            R.npz(folder / filename, **values)
            names.append(name + "/" + filename)
        if name in units:
            R.npz(folder / "units.npz", **units[name])
            names.append(name + "/units.npz")
    inputs = [
        parent / name
        for name in (
            "manifest.json",
            "source_capture_seal.json",
            "application_capture_seal.json",
            "analysis_seal.json",
            "source_program.npz",
            "cold_readings.npz",
            "complete_pairs.json",
        )
    ]
    inputs += [
        vector / name
        for name in (
            "manifest.json",
            "materials.json",
            "generation_seal.json",
            "observation_seal.json",
            "numeric_observations.npz",
        )
    ]
    quality = {
        "matched_rows": len(ids),
        "unique_old_rows": len(set(ids)),
        "source_pairs": len(pairs["source"]),
        "application_pairs": len(pairs["application"]),
        "missing_profile_values": 0,
        "source_generation_axes": ["extraversion"],
        "source_selection": "historical nonrandom prefix",
        "refused_rows_retained": sum(
            m["old_audit"]["refusal"] for m in source + application
        ),
        "truncated_rows_retained": sum(
            m["old_audit"]["truncated"] for m in source + application
        ),
    }
    A.write(out / "input_quality.json", quality)
    names.append("input_quality.json")
    A.write(
        out / "manifest.json",
        {
            "packet": "HM169",
            "config": config,
            "opportunities": old["opportunities"],
            "science_hashes": {name: A.digest(ROOT / name) for name in SCIENCE},
            "input_hashes": {
                str(path.relative_to(ROOT)): A.digest(path) for path in inputs
            },
            "prepared_hashes": {name: A.digest(out / name) for name in names},
            "post_outcome_analysis": True,
            "source_fit_uses_application_answers": False,
            "scope": "same synthetic articles and model-history actions;not human personality",
            "new_neural_forwards": 0,
            "original_scope_complete": False,
        },
    )
    print(json.dumps(quality), flush=True)


def compile_source(out):
    """Compile all fixed covariance components without reading application actions."""
    manifest = check(out)
    with np.load(out / "source_actions.npz", allow_pickle=False) as saved:
        y, groups = saved["responses"], saved["groups"]
    receipts = {}
    for view in manifest["config"]["views"]:
        with np.load(out / view / "source_features.npz", allow_pickle=False) as saved:
            x = saved["features"]
        origin, full, within, between = M.response_components(x, y, groups)
        coefficients = np.stack([full, within, between])
        if view == "grain_tensor625":
            with np.load(
                ROOT / manifest["config"]["predecessor"] / "source_program.npz",
                allow_pickle=False,
            ) as old:
                np.testing.assert_array_equal(origin, old["origin"])
                np.testing.assert_array_equal(coefficients, old["coefficients"][:3])
        path = out / view / "source_program.npz"
        R.npz(path, origin=origin, coefficients=coefficients)
        receipts[str(path.relative_to(out))] = A.digest(path)
    A.write(out / "source_program_seal.json", receipts)
    print("All three source-only programs sealed; no inverse/predictor fit", flush=True)


def cold(out):
    """Run three separate numeric-only processes, with no outcome/metadata inputs."""
    manifest = check(out)
    require_seal(out, "source_program_seal.json")
    receipts = {}
    for view in manifest["config"]["views"]:
        folder = out / view
        command = [
            sys.executable,
            str(ROOT / "scripts/score_grain_response_function.py"),
            "--features",
            str(folder / "application_features.npz"),
            "--program",
            str(folder / "source_program.npz"),
            "--output",
            str(folder / "cold_readings.npz"),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
        receipts[view + "/cold_readings.npz"] = A.digest(folder / "cold_readings.npz")
    A.write(out / "cold_seal.json", receipts)


def analyze(out):
    """Account for all representations, source groups and fixed pair-profile nulls."""
    manifest = check(out)
    config = manifest["config"]
    require_seal(out, "source_program_seal.json")
    require_seal(out, "cold_seal.json")
    started = time.monotonic()
    pairs = A.read(out / "pairs.json")
    with np.load(out / "source_actions.npz", allow_pickle=False) as saved:
        source_y, source_groups = saved["responses"], saved["groups"]
    with np.load(out / "application_actions.npz", allow_pickle=False) as saved:
        app_y = saved["responses"]
    native = M.order_fields(paired(app_y, pairs["application"]))
    paired_source = M.order_fields(paired(source_y, pairs["source"]))
    pair_data = pairs["source"] + pairs["application"]
    native_all = np.concatenate([paired_source, native])
    roles = np.array([p["role"] for p in pair_data])
    role_indices = {role: np.flatnonzero(roles == role) for role in config["roles"]}
    blocks = np.array([p["situation"] + "|" + p["style"] for p in pair_data])
    permutations = {
        role: D.profile_permutations(
            blocks[ids], config["permutations"], config["seed"] + k
        )
        for k, (role, ids) in enumerate(role_indices.items())
    }
    null_values = {
        role: np.empty((3, 2, 2, config["permutations"], 2)) for role in config["roles"]
    }
    cold_pairs, ledgers, source_weights = [], [], []
    alignments, group_rows, dependences = [], [], []
    for view_index, view in enumerate(config["views"]):
        with np.load(out / view / "source_features.npz", allow_pickle=False) as saved:
            xs = saved["features"]
        with np.load(
            out / view / "application_features.npz", allow_pickle=False
        ) as saved:
            xa = saved["features"]
        with np.load(out / view / "cold_readings.npz", allow_pickle=False) as saved:
            cold_pair = M.order_fields(paired(saved["readings"], pairs["application"]))
        if view == "grain_tensor625":
            with np.load(
                ROOT / config["predecessor"] / "paired_actions.npz", allow_pickle=False
            ) as saved:
                np.testing.assert_array_equal(cold_pair, saved["cold"][:, :3])
        cold_pairs.append(cold_pair)
        dx = paired(xa, pairs["application"])
        groups, weights, ledger = D.source_group_ledger(xs, source_y, source_groups, dx)
        ledger = M.order_fields(ledger)
        np.testing.assert_allclose(ledger.sum(2), cold_pair, atol=1e-12, rtol=1e-12)
        ledgers.append(ledger)
        source_weights.append(weights)
        all_dx = np.concatenate([paired(xs, pairs["source"]), dx])
        for role, ids in role_indices.items():
            for channel_index, channel in enumerate(R.CHANNELS):
                for field_index, field in enumerate(("even", "odd")):
                    accounts = D.dependence_accounts(
                        all_dx[ids],
                        native_all[ids, channel_index, :, field_index],
                        blocks[ids],
                    )
                    pooled = D.distance_account(accounts[0], accounts[1])
                    blocked = D.stratified_account(accounts[2], len(ids))
                    null = D.permuted_dependence(accounts, permutations[role])
                    null_values[role][view_index, channel_index, field_index] = null
                    record = {
                        "view": view,
                        "role": role,
                        "channel": channel,
                        "order_field": field,
                        "pairs": len(ids),
                        "blocks": len(np.unique(blocks[ids])),
                    }
                    for mode_index, (mode, account) in enumerate(
                        (("pooled", pooled), ("blocked", blocked))
                    ):
                        value = account["distance_covariance_squared"]
                        record.update({mode + "_" + k: v for k, v in account.items()})
                        tail = int((null[:, mode_index] >= value).sum())
                        record.update(
                            {
                                mode + "_null_mean": float(null[:, mode_index].mean()),
                                mode + "_observed_minus_null_mean": float(
                                    value - null[:, mode_index].mean()
                                ),
                                mode + "_null_ge_observed": tail,
                                mode + "_conditional_tail_rank": (1 + tail)
                                / (1 + len(null)),
                            }
                        )
                    dependences.append(record)
        for role in ("development", "evaluation"):
            for axis in ("extraversion", "neuroticism"):
                ids = np.array(
                    [
                        p["role"] == role and p["axis"] == axis
                        for p in pairs["application"]
                    ]
                )
                for arm_index, arm in enumerate(config["arms"]):
                    for channel_index, channel in enumerate(R.CHANNELS):
                        for opportunity_index, opportunity in enumerate(
                            manifest["opportunities"]
                        ):
                            for field_index, field in enumerate(("even", "odd")):
                                key = {
                                    "view": view,
                                    "role": role,
                                    "axis": axis,
                                    "arm": arm,
                                    "channel": channel,
                                    "opportunity": opportunity,
                                    "order_field": field,
                                }
                                observed = native[
                                    ids, channel_index, opportunity_index, field_index
                                ]
                                reading = cold_pair[
                                    ids,
                                    arm_index,
                                    channel_index,
                                    opportunity_index,
                                    field_index,
                                ]
                                alignments.append(
                                    key | M.signed_alignment(reading, observed)
                                )
                                for g, group in enumerate(groups):
                                    contribution = ledger[
                                        ids,
                                        arm_index,
                                        g,
                                        channel_index,
                                        opportunity_index,
                                        field_index,
                                    ]
                                    group_rows.append(
                                        key
                                        | {
                                            "source_group": str(group),
                                            "pairs": int(ids.sum()),
                                            "signed_product_mean": float(
                                                np.mean(contribution * observed)
                                            ),
                                            "contribution_rms": float(
                                                np.sqrt(np.mean(contribution**2))
                                            ),
                                        }
                                    )
        if time.monotonic() - started > config["cpu_seconds_cap"]:
            raise TimeoutError(
                "Finite CPU cap reached; preserve files and do not retry"
            )
    A.table(out / "action_alignment.csv", alignments)
    A.table(out / "distance_dependence.csv", dependences)
    A.table(out / "signed_source_group_contributions.csv", group_rows)
    R.npz(out / "paired_actions.npz", native=native, cold=np.stack(cold_pairs, axis=1))
    R.npz(
        out / "source_group_contributions.npz",
        contributions=np.stack(ledgers, axis=1),
        source_weights=np.stack(source_weights, axis=1),
        source_groups=groups,
    )
    R.npz(
        out / "permutation_accounts.npz",
        **{role + "_indices": p for role, p in permutations.items()},
        **{role + "_null": value for role, value in null_values.items()},
    )
    result = {}
    for view in config["views"]:
        result[view] = {}
        for channel in R.CHANNELS:
            for field in ("even", "odd"):
                result[view][channel + "_" + field] = {
                    arm: sum(
                        r["cosine"] is not None and r["cosine"] > 0
                        for r in alignments
                        if r["view"] == view
                        and r["role"] == "evaluation"
                        and r["arm"] == arm
                        and r["channel"] == channel
                        and r["order_field"] == field
                    )
                    for arm in config["arms"]
                }
    summary = {
        "status": "COMPLETE",
        "analysis_seconds": time.monotonic() - started,
        "alignment_rows": len(alignments),
        "dependence_rows": len(dependences),
        "group_account_rows": len(group_rows),
        "source_rows": len(source_y),
        "application_rows": len(app_y),
        "application_pairs": len(native),
        "evaluation_positive_cells_of8": result,
        "fixed_permutations": config["permutations"],
        "post_outcome_analysis": True,
        "new_neural_forwards": 0,
        "application_fit": False,
        "independence_claim": False,
        "human_personality_claim": False,
        "original_scope_complete": False,
    }
    if len(alignments) != 576 or len(dependences) != 36 or len(group_rows) != 4032:
        raise ValueError("Incomplete registered account family")
    A.write(out / "summary.json", summary)
    files = (
        "action_alignment.csv",
        "distance_dependence.csv",
        "signed_source_group_contributions.csv",
        "paired_actions.npz",
        "source_group_contributions.npz",
        "permutation_accounts.npz",
        "summary.json",
    )
    A.write(out / "analysis_seal.json", {name: A.digest(out / name) for name in files})
    print(json.dumps(summary), flush=True)


def main():
    """Run one immutable CPU-only stage without a model/request/retry queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "compile", "cold", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {"prepare": prepare, "compile": compile_source, "cold": cold, "analyze": analyze}[
        args.stage
    ](args.out.resolve())


if __name__ == "__main__":
    main()
