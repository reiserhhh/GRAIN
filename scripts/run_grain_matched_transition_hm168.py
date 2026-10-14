#!/usr/bin/env python3
"""Freeze one matched source action, cold-read vectors, then observe new starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_matched_transition as T
from scripts import run_grain_context_response_hm167 as P

A, R, H, M = P.A, P.R, P.H, P.M
DEFAULT = ROOT / "results/grain_math_response_v1/matched_context_transition_r1"
CONFIG = ROOT / "configs/grain_matched_transition_hm168.json"
SCIENCE = (
    "configs/grain_matched_transition_hm168.json",
    "docs/GRAIN_MATCHED_TRANSITION_PLAN_HM168.md",
    "grain_research/construction/math_matched_transition.py",
    "scripts/run_grain_matched_transition_hm168.py",
    "scripts/verify_grain_matched_transition_hm168.py",
    "tests/test_grain_matched_transition.py",
    "scripts/run_grain_context_response_hm167.py",
    "scripts/score_grain_response_function.py",
)
ARMS = (
    ["matched_full", "matched_within", "matched_between"]
    + [f"matched_relink{k}" for k in range(1, 6)]
    + ["mixed_start_full", "mixed_start_within", "mixed_start_between"]
)


def check(out):
    """Authenticate new science and the full sealed destination observation."""
    manifest = P.check(out)
    context = ROOT / manifest["config"]["context_predecessor"]
    P.check(context)
    for name, expected in manifest["context_parent_hashes"].items():
        if A.digest(context / name) != expected:
            raise ValueError("Sealed destination changed: " + name)
    for name in ("source_capture_seal.json", "application_capture_seal.json"):
        for path, expected in A.read(context / name).items():
            if A.digest(context / path) != expected:
                raise ValueError("Destination endpoint capture changed: " + path)
    return manifest


def prepare(out):
    """Inventory operation coverage and bind exactly the registered new starts."""
    from transformers import AutoTokenizer

    config = A.read(CONFIG)
    parent, context = (
        ROOT / config["predecessor"],
        ROOT / config["context_predecessor"],
    )
    old = P.check(context)
    for seal in ("source_program_seal.json", "cold_seal.json"):
        R.artifact(context, seal)
    for name, expected in A.read(context / "analysis_seal.json").items():
        if A.digest(context / name) != expected:
            raise ValueError("Destination analysis changed")
    if any(
        A.read(context / (s + "_collection_status.json"))["status"] != "COMPLETE"
        for s in ("source", "application")
    ):
        raise ValueError("Only terminal complete destination observations admitted")
    history = ROOT / config["history_source"]
    requests = A.read(history / "requests.json")
    tasks = {
        r["messages"][1]["content"]
        for r in requests
        if r["role"] == "discovery"
        and r["situation"] == config["start_task"]
        and r["style"] == config["start_task_style_for_address"]
        and r["policy"] == 0
    }
    if len(tasks) != 1 or old["config"]["common_task"] != config["destination_task"]:
        raise ValueError("Exactly the metadata-registered operation required")
    start_task, destination_task = tasks.pop(), old["common_task"]
    lookup = {
        (r["pair_id"], r["history"], r["opportunity"], r["order"]): r
        for r in requests
        if r["policy"] == 0 and r["history"] != 0
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(H.OLD.MODEL), local_files_only=True, trust_remote_code=False
    )
    history_captures = H.results(history)
    source_reuses = {r["unit_id"]: r for r in history_captures.values()}
    old_hashes = set(source_reuses)
    old_hashes.update(r["unit_id"] for r in R.captures(parent).values())
    for stage in ("source", "application"):
        old_hashes.update(r["unit_id"] for r in P.captures(context, stage).values())
    stages, addresses, original_keys, matched_keys, metadata = {}, {}, {}, {}, {}
    for stage in ("source", "application"):
        articles = A.read(context / (stage + "_articles.json"))
        metadata[stage] = articles
        units, specs, seen, original, matched = [], [], {}, [], []
        for i, meta in enumerate(articles):
            for c, opportunity in enumerate(old["opportunities"]):
                for order in (0, 1):
                    prior = lookup[meta["pair_id"], meta["pole"], opportunity, order]
                    messages = [r.copy() for r in prior["messages"]]
                    original.append(
                        T.transition_address(
                            messages, messages[1]["content"], destination_task
                        )
                    )
                    operation = T.transition_address(
                        messages, start_task, destination_task
                    )
                    matched.append(operation)
                    messages[1]["content"] = start_task
                    if any(messages[k] != prior["messages"][k] for k in (0, 2, 3)):
                        raise ValueError("Body/system/opportunity changed")
                    ids = H.M.rendered_token_ids(
                        tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True
                        )
                    )
                    if len(ids) > config["max_prompt_tokens"]:
                        raise ValueError("Full history outside support; no truncation")
                    key = hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()
                    ).hexdigest()
                    if stage == "application" and key in old_hashes:
                        raise ValueError("New application start already observed")
                    if key not in seen:
                        seen[key] = len(units)
                        reused = source_reuses.get(key) if stage == "source" else None
                        units.append(
                            {
                                "unit_id": key,
                                "prompt_ids": ids,
                                "reuse_old_unit": None
                                if reused is None
                                else reused["unit_index"],
                            }
                        )
                    specs.append(
                        {
                            "article": i,
                            "opportunity": c,
                            "order": order,
                            "unit_index": seen[key],
                            "original_unit_index": prior["unit_index"],
                            "text_id": meta["text_id"],
                            "body_sha256": hashlib.sha256(
                                messages[2]["content"].encode()
                            ).hexdigest(),
                            "operation_id": operation,
                            "production_condition": meta["situation"],
                            "production_style": meta["style"],
                        }
                    )
        stages[stage], addresses[stage] = units, specs
        original_keys[stage], matched_keys[stage] = original, matched
    inventory = {
        "mixed_start": T.support_account(
            original_keys["source"], original_keys["application"]
        ),
        "matched_start": T.support_account(
            matched_keys["source"], matched_keys["application"]
        ),
        "existing_interfaces": {
            "HM139": "source-equivalent vector functions need not agree on new actions",
            "HM148": "a fixed tensor response depends on both location and displacement",
            "HM163": "report-edit path consistency is not natural author dynamics",
            "HM167": "common destination does not define one source/application transition",
        },
        "destination_already_observed": True,
        "new_start_already_observed": False,
        "matching_implies_population_transport": False,
    }
    source_new = sum(u["reuse_old_unit"] is None for u in stages["source"])
    if source_new > config["max_new_source_inputs"] or (
        len(stages["application"]) != config["application_inputs"]
    ):
        raise ValueError("Fixed new-start support/cap changed")
    if inventory["matched_start"]["unsupported_application_units"]:
        raise ValueError("Application operation missing in source")
    source_destination, _ = P.profiles(context, "source")
    with np.load(context / "source.npz", allow_pickle=False) as saved:
        source_x, groups = saved["features"].copy(), saved["groups"].copy()
    with np.load(context / "application_features.npz", allow_pickle=False) as saved:
        app_x = saved["features"].copy()
    out.mkdir(parents=True, exist_ok=False)
    names = ["inventory.json", "source.npz", "application_features.npz"]
    A.write(out / "inventory.json", inventory)
    R.npz(
        out / "source.npz",
        features=source_x,
        responses=source_destination,
        groups=groups,
    )
    R.npz(out / "application_features.npz", features=app_x)
    for stage in ("source", "application"):
        (out / (stage + "_batches")).mkdir()
        for suffix, values in (
            ("units", stages[stage]),
            ("addresses", addresses[stage]),
            ("articles", metadata[stage]),
        ):
            name = stage + "_" + suffix + ".json"
            A.write(out / name, values)
            names.append(name)
    context_names = [
        "manifest.json",
        "source_capture_seal.json",
        "application_capture_seal.json",
        "analysis_seal.json",
        "source_program.npz",
        "source_program_seal.json",
        "cold_seal.json",
        "source_articles.json",
        "application_articles.json",
        "source_addresses.json",
        "application_addresses.json",
    ]
    parent_names = [
        "manifest.json",
        "source_program_seal.json",
        "cold_seal.json",
        "analysis_seal.json",
        "capture_seal.json",
    ]
    A.write(
        out / "manifest.json",
        {
            "packet": "HM168",
            "config": config,
            "common_task": start_task,
            "destination_task": destination_task,
            "opportunities": old["opportunities"],
            "arms": ARMS,
            "option_token_ids": old["option_token_ids"],
            "model_weights_sha256": old["model_weights_sha256"],
            "source_new": source_new,
            "source_reused": len(stages["source"]) - source_new,
            "application_new": len(stages["application"]),
            "science_hashes": {n: A.digest(ROOT / n) for n in SCIENCE},
            "prepared_hashes": {n: A.digest(out / n) for n in names},
            "parent_hashes": {n: A.digest(parent / n) for n in parent_names},
            "context_parent_hashes": {n: A.digest(context / n) for n in context_names},
            "application_start_outcomes_at_prepare": 0,
            "destination_already_observed": True,
            "original_scope_complete": False,
        },
    )
    print(
        json.dumps(
            {
                "source_new": source_new,
                "source_reused": len(stages["source"]) - source_new,
                "application_new": len(stages["application"]),
                "inventory": inventory,
            }
        ),
        flush=True,
    )


def compile_source(out):
    """Freeze the registered action accounts without inspecting application starts."""
    manifest = check(out)
    if (
        A.read(out / "source_collection_status.json")["status"] != "COMPLETE"
        or (out / "application_collection_status.json").exists()
    ):
        raise ValueError("Complete source/absent new application starts required")
    start, _ = P.profiles(out, "source")
    with np.load(out / "source.npz", allow_pickle=False) as saved:
        x, end, g = saved["features"], saved["responses"], saved["groups"]
    origin, full, within, between = T.action_components(x, start, end, g)
    profiles = end - start
    coefficients = [full, within, between]
    for shift in range(1, 6):
        coefficients.append(
            M.response_components(x, M.relink_profiles(profiles, g, shift), g)[1]
        )
    context = ROOT / manifest["config"]["context_predecessor"]
    with np.load(context / "source_program.npz", allow_pickle=False) as saved:
        coefficients.extend(saved["coefficients"][-3:])
        np.testing.assert_array_equal(origin, saved["origin"])
    direct_end, direct_start = (M.response_components(x, y, g) for y in (end, start))
    R.npz(
        out / "source_program.npz", origin=origin, coefficients=np.array(coefficients)
    )
    A.write(
        out / "source_program_seal.json",
        {
            "file": "source_program.npz",
            "sha256": A.digest(out / "source_program.npz"),
            "application_start_outcomes": 0,
            "destination_already_observed": True,
            "decomposition_error": float(np.max(abs(full - within - between))),
            "endpoint_difference_error": float(
                np.max(abs(full - (direct_end[1] - direct_start[1])))
            ),
            "scope": "source cross-moment action witness;not an identified conditional expectation",
        },
    )
    print(
        json.dumps(
            {
                "source_profiles": len(x),
                "action_accounts": len(coefficients),
                "application_start_outcomes": 0,
            }
        ),
        flush=True,
    )


def cold(out):
    """Execute a separate numeric-only scorer before any new application start."""
    check(out)
    R.artifact(out, "source_program_seal.json")
    if (out / "application_collection_status.json").exists():
        raise ValueError("New application start observation already attempted")
    command = [
        sys.executable,
        str(ROOT / "scripts/score_grain_response_function.py"),
        "--features",
        str(out / "application_features.npz"),
        "--program",
        str(out / "source_program.npz"),
        "--output",
        str(out / "cold_readings.npz"),
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    A.write(
        out / "cold_seal.json",
        {
            "file": "cold_readings.npz",
            "sha256": A.digest(out / "cold_readings.npz"),
            "application_start_outcomes": 0,
            "destination_already_observed": True,
            "command": command,
        },
    )


def analyze(out):
    """Retain matched-action and mixed-start comparisons without choosing a winner."""
    manifest = check(out)
    for seal in ("source_program_seal.json", "cold_seal.json"):
        R.artifact(out, seal)
    status = A.read(out / "application_collection_status.json")
    if status["status"] not in ("COMPLETE", "PARTIAL_BUDGET"):
        raise ValueError("New application observation not terminal")
    start, mass = P.profiles(out, "application")
    end, _ = P.profiles(ROOT / manifest["config"]["context_predecessor"], "application")
    articles = A.read(out / "application_articles.json")
    paired = {}
    for i, meta in enumerate(articles):
        paired.setdefault(meta["pair_id"], {})[meta["pole"]] = i
    complete = [
        (articles[ids[1]], ids[-1], ids[1])
        for ids in paired.values()
        if np.isfinite(start[[ids[-1], ids[1]]]).all()
    ]
    if not complete:
        raise ValueError("No complete pair; retain terminal partial observations")
    with np.load(out / "cold_readings.npz", allow_pickle=False) as saved:
        cold_y = saved["readings"]
    gaps = M.order_fields(
        np.array(
            [(end[hi] - start[hi]) - (end[lo] - start[lo]) for _, lo, hi in complete]
        )
    )
    cold_gaps = M.order_fields(
        np.array([cold_y[hi] - cold_y[lo] for _, lo, hi in complete])
    )
    accounts = []
    for meta, lo, hi in complete:
        for c, opportunity in enumerate(manifest["opportunities"]):
            for order in (0, 1):
                accounts.append(
                    {
                        k: meta[k]
                        for k in (
                            "pair_id",
                            "axis",
                            "role",
                            "situation",
                            "style",
                            "seed",
                        )
                    }
                    | {"opportunity": opportunity, "order": order}
                    | P.C.context_pair_account(
                        start[[lo, hi], 1, c, order], end[[lo, hi], 1, c, order]
                    )
                )
    alignments = []
    for role in ("development", "evaluation"):
        for axis in ("extraversion", "neuroticism"):
            ids = np.array(
                [m["role"] == role and m["axis"] == axis for m, _, _ in complete]
            )
            if not ids.any():
                continue
            for arm, name in enumerate(ARMS):
                for m, channel in enumerate(R.CHANNELS):
                    for c, opportunity in enumerate(manifest["opportunities"]):
                        for f, field in enumerate(("even", "odd")):
                            alignments.append(
                                {
                                    "role": role,
                                    "axis": axis,
                                    "arm": name,
                                    "channel": channel,
                                    "opportunity": opportunity,
                                    "order_field": field,
                                }
                                | M.signed_alignment(
                                    cold_gaps[ids, arm, m, c, f], gaps[ids, m, c, f]
                                )
                            )
    A.table(out / "transition_pair_accounts.csv", accounts)
    A.table(out / "action_alignment.csv", alignments)
    A.write(out / "complete_pairs.json", [m for m, _, _ in complete])
    R.npz(out / "paired_actions.npz", native=gaps, cold=cold_gaps)
    summary = {
        "status": status["status"],
        "complete_pairs": len(complete),
        "pair_order_accounts": len(accounts),
        "new_application_start_inputs": len(P.captures(out, "application")),
        "total_new_model_seconds": status["total_model_seconds"],
        "option_mass_min": float(np.nanmin(mass)),
        "log_odds_gap_reversals": sum(
            r["old_log_odds_gap"] * r["common_log_odds_gap"] < 0 for r in accounts
        ),
        "log_odds_gap_change_rms": float(
            np.sqrt(np.mean([r["log_odds_gap_change"] ** 2 for r in accounts]))
        ),
        "destination_already_observed": True,
        "original_scope_complete": False,
        "scope": "matched model-history action;not human behavior/trait accuracy or untouched two-endpoint validation",
    }
    A.write(out / "summary.json", summary)
    names = [
        "transition_pair_accounts.csv",
        "action_alignment.csv",
        "complete_pairs.json",
        "paired_actions.npz",
        "summary.json",
    ]
    A.write(out / "analysis_seal.json", {n: A.digest(out / n) for n in names})
    print(json.dumps(summary), flush=True)


def main():
    """Run a single bounded immutable stage; never automatically retry a collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "collect-source",
            "compile",
            "cold",
            "collect-application",
            "analyze",
        ],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage.startswith("collect-"):
        check(out)
        P.collect(out, args.stage.removeprefix("collect-"))
        check(out)
    else:
        {
            "prepare": prepare,
            "compile": compile_source,
            "cold": cold,
            "analyze": analyze,
        }[args.stage](out)


if __name__ == "__main__":
    main()
