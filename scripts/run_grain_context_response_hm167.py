#!/usr/bin/env python3
"""Finite same-body prior-task intervention and observation-indexed cold functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_context_response as C
from scripts import run_grain_response_function_transport_hm166 as R

A, H, M = R.A, R.H, R.M
DEFAULT = ROOT / "results/grain_math_response_v1/context_response_functions_r1"
CONFIG = ROOT / "configs/grain_context_response_hm167.json"
SCIENCE = (
    "configs/grain_context_response_hm167.json",
    "docs/GRAIN_CONTEXT_RESPONSE_PLAN_HM167.md",
    "grain_research/construction/math_context_response.py",
    "scripts/run_grain_context_response_hm167.py",
    "scripts/verify_grain_context_response_hm167.py",
    "tests/test_grain_context_response.py",
)
ARMS = (
    ["original_" + a for a in R.ARM_NAMES]
    + ["common_" + a for a in R.ARM_NAMES]
    + ["delta_full", "delta_within", "delta_between"]
)


def check(out):
    """Require original seals and every fixed new scientific dependency."""
    manifest = A.read(out / "manifest.json")
    parent = ROOT / manifest["config"]["predecessor"]
    R.check(parent)
    for section, base in (("science_hashes", ROOT), ("prepared_hashes", out)):
        for name, expected in manifest[section].items():
            if A.digest(base / name) != expected:
                raise ValueError("Frozen input changed: " + name)
    for name, expected in manifest["parent_hashes"].items():
        if A.digest(parent / name) != expected:
            raise ValueError("Original observation changed: " + name)
    return manifest


def prepare(out):
    """Bind exact original articles to one preselected common task, not new texts."""
    from transformers import AutoTokenizer

    config = A.read(CONFIG)
    parent, history = ROOT / config["predecessor"], ROOT / config["history_source"]
    old = R.check(parent)
    if A.read(parent / "collection_status.json")["status"] != "COMPLETE":
        raise ValueError("Original application observation must be terminal")
    for name, digest in A.read(parent / "analysis_seal.json").items():
        if A.digest(parent / name) != digest:
            raise ValueError("Original analysis changed")
    requests = A.read(history / "requests.json")
    tasks = {
        r["messages"][1]["content"]
        for r in requests
        if r["role"] == "discovery"
        and r["situation"] == config["common_task"]
        and r["style"] == config["common_task_style_for_address"]
        and r["policy"] == 0
    }
    if len(tasks) != 1:
        raise ValueError("Exactly one metadata-selected common task required")
    common_task = tasks.pop()
    lookup = {
        (r["pair_id"], r["history"], r["opportunity"], r["order"]): r
        for r in requests
        if r["policy"] == 0 and r["history"] != 0
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(H.OLD.MODEL), local_files_only=True, trust_remote_code=False
    )
    old_captures = H.results(history)
    native_by_hash = {r["unit_id"]: r for r in old_captures.values()}
    all_old_app = {r["unit_id"] for r in R.captures(parent).values()}
    stages, specs = {}, {}
    metadata = {
        "source": A.read(parent / "source_articles.json"),
        "application": A.read(parent / "application_articles.json"),
    }
    for stage, articles in metadata.items():
        units, addresses, seen = [], [], {}
        for article, meta in enumerate(articles):
            for c, opportunity in enumerate(old["config"]["opportunities"]):
                for order in (0, 1):
                    prior = lookup[meta["pair_id"], meta["pole"], opportunity, order]
                    messages = [r.copy() for r in prior["messages"]]
                    body = messages[2]["content"]
                    messages[1]["content"] = common_task
                    if any(messages[i] != prior["messages"][i] for i in (0, 2, 3)):
                        raise ValueError(
                            "The intervention changed more than prior user task"
                        )
                    ids = H.M.rendered_token_ids(
                        tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True
                        )
                    )
                    if len(ids) > config["max_prompt_tokens"]:
                        raise ValueError(
                            "Full history exceeds frozen support; no truncation"
                        )
                    key = hashlib.sha256(
                        np.asarray(ids, dtype="<i8").tobytes()
                    ).hexdigest()
                    if stage == "application" and (
                        key in native_by_hash or key in all_old_app
                    ):
                        raise ValueError("Application input is not genuinely new")
                    if key not in seen:
                        seen[key] = len(units)
                        reuse = native_by_hash.get(key) if stage == "source" else None
                        units.append(
                            {
                                "unit_id": key,
                                "prompt_ids": ids,
                                "reuse_old_unit": None
                                if reuse is None
                                else reuse["unit_index"],
                            }
                        )
                    addresses.append(
                        {
                            "article": article,
                            "opportunity": c,
                            "order": order,
                            "unit_index": seen[key],
                            "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                            "text_id": meta["text_id"],
                            "original_unit_index": prior["unit_index"],
                            "original_task": prior["messages"][1]["content"],
                        }
                    )
        stages[stage], specs[stage] = units, addresses
    source_new = sum(u["reuse_old_unit"] is None for u in stages["source"])
    if source_new > config["max_new_source_inputs"]:
        raise ValueError("Registered source budget/support exceeded")
    if len(stages["application"]) != config["application_inputs"]:
        raise ValueError("The full new application support changed")
    with np.load(parent / "source.npz", allow_pickle=False) as data:
        source_data = {k: data[k].copy() for k in data.files}
    with np.load(parent / "application_features.npz", allow_pickle=False) as data:
        application_data = data["features"].copy()
    out.mkdir(parents=True, exist_ok=False)
    names = []
    for stage, stage_units in stages.items():
        (out / (stage + "_batches")).mkdir()
        for ending, values in (
            ("units", stage_units),
            ("addresses", specs[stage]),
            ("articles", metadata[stage]),
        ):
            name = stage + "_" + ending + ".json"
            A.write(out / name, values)
            names.append(name)
    R.npz(out / "source.npz", **source_data)
    R.npz(out / "application_features.npz", features=application_data)
    names += ["source.npz", "application_features.npz"]
    A.write(
        out / "manifest.json",
        {
            "packet": "HM167",
            "config": config,
            "common_task": common_task,
            "source_new": source_new,
            "source_reused": len(stages["source"]) - source_new,
            "application_new": len(stages["application"]),
            "option_token_ids": old["option_token_ids"],
            "model_weights_sha256": old["model_weights_sha256"],
            "opportunities": old["config"]["opportunities"],
            "arms": ARMS,
            "science_hashes": {name: A.digest(ROOT / name) for name in SCIENCE},
            "prepared_hashes": {name: A.digest(out / name) for name in names},
            "parent_hashes": {
                name: A.digest(parent / name)
                for name in (
                    "manifest.json",
                    "source_program.npz",
                    "source_program_seal.json",
                    "cold_seal.json",
                    "analysis_seal.json",
                    "capture_seal.json",
                    "paired_accounts.npz",
                    "complete_pairs.json",
                )
            },
            "application_outcomes_at_prepare": 0,
            "intervention": "Only prior user task; all system/style,body,choice prompt/order retained",
            "original_scope_complete": False,
        },
    )
    print(
        json.dumps(
            {
                "source_new": source_new,
                "source_reused": len(stages["source"]) - source_new,
                "application_new": len(stages["application"]),
            }
        )
    )


def captures(out, stage):
    """Load unique new observations plus only exact-input source reuses."""
    manifest = check(out)
    seal = out / (stage + "_capture_seal.json")
    if seal.exists():
        for name, digest in A.read(seal).items():
            if A.digest(out / name) != digest:
                raise ValueError("New native capture changed: " + name)
    units = A.read(out / (stage + "_units.json"))
    result = {}
    if stage == "source":
        old = H.results(ROOT / manifest["config"]["history_source"])
        for i, unit in enumerate(units):
            if unit["reuse_old_unit"] is not None:
                row = old[unit["reuse_old_unit"]]
                if row["unit_id"] != unit["unit_id"]:
                    raise ValueError("Reuse is not the exact same native input")
                result[i] = row | {"unit_index": i}
    for path in sorted((out / (stage + "_batches")).glob("[0-9][0-9][0-9][0-9].json")):
        for row in A.read(path):
            if row["unit_index"] in result:
                raise ValueError("Duplicate capture/reuse")
            result[row["unit_index"]] = row
    return result


def stage_status(out, stage, **fields):
    """Update runtime-only state, preserving immutable batches and scientific outputs."""
    temporary = out / (stage + "_status.tmp")
    temporary.write_text(json.dumps(fields, indent=2) + "\n")
    temporary.replace(out / (stage + "_collection_status.json"))


def collect(out, stage):
    """Spend one shared finite model budget; never resume an attempted collection."""
    import torch

    manifest = check(out)
    if (out / (stage + "_collection_status.json")).exists():
        raise ValueError("This collection was already attempted; no automatic retry")
    spent = 0.0
    if stage == "application":
        R.artifact(out, "source_program_seal.json")
        R.artifact(out, "cold_seal.json")
        status = A.read(out / "source_collection_status.json")
        if status["status"] != "COMPLETE":
            raise ValueError("Source observation incomplete; do not start application")
        spent = status["model_seconds"]
    units = A.read(out / (stage + "_units.json"))
    new = [i for i, u in enumerate(units) if u["reuse_old_unit"] is None]
    observer = H.OLD.offline_model(manifest)
    A.write(out / (stage + "_model_manifest.json"), observer.manifest())
    elapsed, count = 0.0, 0
    config = manifest["config"]
    for batch, start in enumerate(range(0, len(new), config["batch_size"])):
        if spent + elapsed >= config["model_seconds_cap_total"]:
            stage_status(
                out,
                stage,
                status="PARTIAL_BUDGET",
                completed=count,
                total=len(new),
                model_seconds=elapsed,
                total_model_seconds=spent + elapsed,
                pid=os.getpid(),
            )
            return
        ids = new[start : start + config["batch_size"]]
        group = [units[i]["prompt_ids"] for i in ids]
        width, eos = max(map(len, group)), observer.tokenizer.eos_token_id
        padded = torch.tensor(
            [[eos] * (width - len(p)) + p for p in group], device="mps"
        )
        mask = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in group], device="mps"
        )
        stage_status(
            out,
            stage,
            status="RUNNING",
            completed=count,
            total=len(new),
            model_seconds=elapsed,
            total_model_seconds=spent + elapsed,
            pid=os.getpid(),
        )
        before = time.monotonic()
        with torch.inference_mode():
            hidden = observer.model.model(
                input_ids=padded, attention_mask=mask, use_cache=False, return_dict=True
            ).last_hidden_state[:, -1]
            logits = observer.model.lm_head(hidden)
            ab = logits[:, manifest["option_token_ids"]].cpu().numpy().astype(float)
            normalizer = torch.logsumexp(logits, dim=-1).cpu().numpy().astype(float)
        seconds = time.monotonic() - before
        elapsed += seconds
        rows = []
        for i, values, total in zip(ids, ab, normalizer, strict=True):
            q = H.M.choice_account(values, total, 0)["q_positive"]
            rows.append(
                {
                    "unit_index": i,
                    "unit_id": units[i]["unit_id"],
                    "ab_logits": values.tolist(),
                    "log_normalizer": float(total),
                    "draw_a": H.M.sample_protocol_choice(
                        q, units[i]["unit_id"], config["draw_seed"]
                    ),
                    "model_seconds_share": seconds / len(ids),
                }
            )
        A.write(out / (stage + "_batches") / f"{batch:04d}.json", rows)
        count += len(ids)
        print(
            json.dumps(
                {
                    "stage": stage,
                    "batch": batch,
                    "new_complete": count,
                    "new_total": len(new),
                    "total_model_seconds": spent + elapsed,
                }
            ),
            flush=True,
        )
    A.write(
        out / (stage + "_capture_seal.json"),
        {
            str(p.relative_to(out)): A.digest(p)
            for p in sorted(
                (out / (stage + "_batches")).glob("[0-9][0-9][0-9][0-9].json")
            )
        },
    )
    stage_status(
        out,
        stage,
        status="COMPLETE",
        completed=count,
        total=len(new),
        model_seconds=elapsed,
        total_model_seconds=spent + elapsed,
        pid=os.getpid(),
        all_batches_terminal=True,
    )


def profiles(out, stage):
    """Address complete native profiles; unobserved cells remain NaN."""
    check(out)
    cap = captures(out, stage)
    articles = A.read(out / (stage + "_articles.json"))
    y, mass = (
        np.full((len(articles), 2, 4, 2), np.nan),
        np.full((len(articles), 4, 2), np.nan),
    )
    for r in A.read(out / (stage + "_addresses.json")):
        if r["unit_index"] not in cap:
            continue
        row = cap[r["unit_index"]]
        account = H.M.choice_account(
            row["ab_logits"], row["log_normalizer"], r["order"]
        )
        i, c, o = r["article"], r["opportunity"], r["order"]
        y[i, :, c, o] = account["q_positive"], account["positive_log_odds"]
        mass[i, c, o] = account["unrestricted_option_mass"]
    return y, mass


def compile_source(out):
    """Freeze observation-indexed functions after source and before application outcomes."""
    manifest = check(out)
    if (
        A.read(out / "source_collection_status.json")["status"] != "COMPLETE"
        or (out / "application_collection_status.json").exists()
    ):
        raise ValueError("Complete source and absent application observation required")
    y, _ = profiles(out, "source")
    with np.load(out / "source.npz", allow_pickle=False) as data:
        x, old_y, g = data["features"], data["responses"], data["groups"]
    origin, full, within, between = M.response_components(x, y, g)
    common = [
        full,
        full * np.array([s.bit_count() <= 1 for s in R.S.feature_sectors()]),
        within,
        between,
    ]
    for shift in range(1, 6):
        common.append(M.response_components(x, M.relink_profiles(y, g, shift), g)[1])
    parent = ROOT / manifest["config"]["predecessor"]
    with np.load(parent / "source_program.npz", allow_pickle=False) as data:
        original = data["coefficients"].copy()
        np.testing.assert_array_equal(origin, data["origin"])
    delta = np.array(common)[[0, 2, 3]] - original[[0, 2, 3]]
    _, direct_delta, _, _ = M.response_components(x, y - old_y, g)
    R.npz(
        out / "source_program.npz",
        origin=origin,
        coefficients=np.concatenate((original, common, delta)),
    )
    R.npz(out / "source_observation.npz", responses=y)
    A.write(
        out / "source_program_seal.json",
        {
            "file": "source_program.npz",
            "sha256": A.digest(out / "source_program.npz"),
            "application_outcomes": 0,
            "source_only": True,
            "delta_covariance_error": float(np.max(abs(delta[0] - direct_delta))),
            "decomposition_error": float(np.max(abs(full - within - between))),
        },
    )
    print(
        json.dumps(
            {"source_profiles": len(y), "arms": len(ARMS), "application_outcomes": 0}
        )
    )


def cold(out):
    """Seal all numeric article readings before any new application input is evaluated."""
    check(out)
    R.artifact(out, "source_program_seal.json")
    if (out / "application_collection_status.json").exists():
        raise ValueError("New application outcome already exists")
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
    subprocess.run(command, check=True)
    A.write(
        out / "cold_seal.json",
        {
            "file": "cold_readings.npz",
            "sha256": A.digest(out / "cold_readings.npz"),
            "application_outcomes": 0,
            "command": command,
        },
    )


def analyze(out):
    """Retain all context effects and cold programs without selecting a winning score."""
    manifest = check(out)
    R.artifact(out, "source_program_seal.json")
    R.artifact(out, "cold_seal.json")
    status = A.read(out / "application_collection_status.json")
    if status["status"] not in ("COMPLETE", "PARTIAL_BUDGET"):
        raise ValueError("Application observation is not terminal")
    common, mass = profiles(out, "application")
    parent = ROOT / manifest["config"]["predecessor"]
    original, _, _, _ = R.application_responses(parent)
    articles = A.read(out / "application_articles.json")
    pair_addresses = {}
    for i, meta in enumerate(articles):
        pair_addresses.setdefault(meta["pair_id"], {})[meta["pole"]] = i
    old_pairs = {p["pair_id"]: p for p in A.read(parent / "complete_pairs.json")}
    complete = [
        (old_pairs[k], v[-1], v[1])
        for k, v in pair_addresses.items()
        if np.isfinite(common[[v[-1], v[1]]]).all()
    ]
    if not complete:
        raise ValueError("No complete new pair; preserve terminal captures")
    with np.load(out / "cold_readings.npz", allow_pickle=False) as data:
        cold_y = data["readings"]
    pairs = [p for p, _, _ in complete]
    old_gaps = M.order_fields(
        np.array([original[hi] - original[lo] for _, lo, hi in complete])
    )
    new_gaps = M.order_fields(
        np.array([common[hi] - common[lo] for _, lo, hi in complete])
    )
    cold_gaps = M.order_fields(
        np.array([cold_y[hi] - cold_y[lo] for _, lo, hi in complete])
    )
    accounts = []
    for p, lo, hi in complete:
        for c, opportunity in enumerate(manifest["opportunities"]):
            for o in (0, 1):
                accounts.append(
                    {
                        k: p[k]
                        for k in (
                            "pair_id",
                            "axis",
                            "role",
                            "situation",
                            "style",
                            "seed",
                        )
                    }
                    | {"opportunity": opportunity, "order": o}
                    | C.context_pair_account(
                        original[[lo, hi], 1, c, o], common[[lo, hi], 1, c, o]
                    )
                )
    alignments = []
    for role in ("all", "development", "evaluation"):
        for axis in ("all", "extraversion", "neuroticism"):
            ids = np.array(
                [
                    (role == "all" or p["role"] == role)
                    and (axis == "all" or p["axis"] == axis)
                    for p in pairs
                ]
            )
            for arm, name in enumerate(ARMS):
                for m, channel in enumerate(R.CHANNELS):
                    for c, opportunity in enumerate(manifest["opportunities"]):
                        for f, field in enumerate(("even", "odd")):
                            for response, values in (
                                ("original", old_gaps),
                                ("common", new_gaps),
                                ("change", new_gaps - old_gaps),
                            ):
                                alignments.append(
                                    {
                                        "role": role,
                                        "axis": axis,
                                        "arm": name,
                                        "channel": channel,
                                        "opportunity": opportunity,
                                        "order_field": field,
                                        "response_law": response,
                                    }
                                    | M.signed_alignment(
                                        cold_gaps[ids, arm, m, c, f],
                                        values[ids, m, c, f],
                                    )
                                )
    context_summary = []
    for role in ("development", "evaluation"):
        for axis in ("extraversion", "neuroticism"):
            for opportunity in manifest["opportunities"]:
                cell = [
                    r
                    for r in accounts
                    if r["role"] == role
                    and r["axis"] == axis
                    and r["opportunity"] == opportunity
                ]
                context_summary.append(
                    {
                        "role": role,
                        "axis": axis,
                        "opportunity": opportunity,
                        "pair_order_cells": len(cell),
                        "probability_gap_reversals": sum(
                            r["old_probability_gap"] * r["common_probability_gap"] < 0
                            for r in cell
                        )
                        if cell
                        else None,
                        "log_odds_gap_reversals": sum(
                            r["old_log_odds_gap"] * r["common_log_odds_gap"] < 0
                            for r in cell
                        )
                        if cell
                        else None,
                    }
                    | {
                        name + "_rms": float(
                            np.sqrt(np.mean([r[name] ** 2 for r in cell]))
                        )
                        if cell
                        else None
                        for name in (
                            "context_character",
                            "joint_character",
                            "log_odds_gap_change",
                            "probability_gap_change",
                            "offset_only_probability_gap_change",
                            "joint_probability_remainder",
                        )
                    }
                    | {
                        "body_only_two_context_floor_mean": float(
                            np.mean([r["body_only_two_context_floor"] for r in cell])
                        )
                        if cell
                        else None
                    }
                )
    R.npz(
        out / "paired_accounts.npz", original=old_gaps, common=new_gaps, cold=cold_gaps
    )
    A.write(out / "complete_pairs.json", pairs)
    A.table(out / "context_pair_accounts.csv", accounts)
    A.table(out / "alignment_summary.csv", alignments)
    A.table(out / "context_summary.csv", context_summary)
    A.write(
        out / "summary.json",
        {
            "status": status["status"],
            "source_new": manifest["source_new"],
            "source_reused": manifest["source_reused"],
            "application_new": len(captures(out, "application")),
            "complete_pairs": len(pairs),
            "model_seconds_total": status["total_model_seconds"],
            "option_mass_min": float(np.nanmin(mass)),
            "context_accounts": context_summary,
            "original_scope_complete": False,
            "scope": "model counterfactual-history law;not natural human behavior or accuracy promotion",
        },
    )
    names = (
        "paired_accounts.npz",
        "complete_pairs.json",
        "context_pair_accounts.csv",
        "alignment_summary.csv",
        "context_summary.csv",
        "summary.json",
    )
    A.write(out / "analysis_seal.json", {n: A.digest(out / n) for n in names})
    print(
        json.dumps(
            {
                "status": status["status"],
                "complete_pairs": len(pairs),
                "context_pair_order_accounts": len(accounts),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "collect-source",
            "compile",
            "cold",
            "collect-application",
            "analyze",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage.startswith("collect-"):
        collect(args.output, args.stage.removeprefix("collect-"))
    else:
        {
            "prepare": prepare,
            "compile": compile_source,
            "cold": cold,
            "analyze": analyze,
        }[args.stage](args.output)
