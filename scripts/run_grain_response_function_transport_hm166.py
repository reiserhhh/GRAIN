#!/usr/bin/env python3
"""Construct response functions on old source observations, then cold-test new choices."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_response_function as M
from grain_research.construction import math_semantic_anchor as S
from scripts import run_grain_observed_text_anchor_hm164 as A
from scripts import run_grain_writer_history_bridge_hm165 as H

DEFAULT = ROOT / "results/grain_math_response_v1/response_function_transport_r1"
CONFIG = ROOT / "configs/grain_response_function_transport_hm166.json"
ARM_NAMES = ["full", "single", "within", "between"] + [
    f"relink{k}" for k in range(1, 6)
]
CHANNELS = ["probability", "log_odds"]
SCIENCE = (
    "configs/grain_response_function_transport_hm166.json",
    "docs/GRAIN_RESPONSE_FUNCTION_TRANSPORT_PLAN_HM166.md",
    "grain_research/construction/math_response_function.py",
    "scripts/run_grain_response_function_transport_hm166.py",
    "scripts/score_grain_response_function.py",
    "scripts/verify_grain_response_function_hm166.py",
    "tests/test_grain_response_function.py",
    "grain_research/construction/math_semantic_anchor.py",
    "scripts/run_grain_writer_history_bridge_hm165.py",
    "grain_research/construction/math_writer_history_bridge.py",
    "scripts/run_grain_joint_persona_style_hm150.py",
    "grain_research/vector_mechanisms/trace.py",
)


def npz(path, **values):
    """Create an immutable numeric artifact rather than replacing an old stage."""
    with path.open("xb") as handle:
        np.savez_compressed(handle, **values)


def state(out, **fields):
    """Atomically update runtime state only; never replace scientific observations."""
    temporary = out / "collection_status.tmp"
    temporary.write_text(json.dumps(fields, indent=2) + "\n")
    temporary.replace(out / "collection_status.json")


def check(out):
    """Verify frozen science, source captures, prepared addresses and ancestor seals."""
    manifest = A.read(out / "manifest.json")
    H.check(ROOT / manifest["config"]["predecessor"])
    for section, base in (
        ("science_hashes", ROOT),
        ("input_hashes", ROOT),
        ("prepared_hashes", out),
    ):
        for name, expected in manifest[section].items():
            if A.digest(base / name) != expected:
                raise ValueError("Sealed dependency changed: " + name)
    return manifest


def artifact(out, seal):
    """Require a sealed source program/cold output before observing application answers."""
    receipt = A.read(out / seal)
    if A.digest(out / receipt["file"]) != receipt["sha256"]:
        raise ValueError("Frozen numeric stage changed: " + seal)
    return receipt


def prepare(out):
    """Freeze available source eligibility and every new neutral application request."""
    config = A.read(CONFIG)
    previous = ROOT / config["predecessor"]
    old_manifest = H.check(previous)
    if A.read(previous / "collection_status.json")["status"] != "PARTIAL_BUDGET":
        raise ValueError(
            "Expected terminal partial predecessor, not a live/retried stage"
        )
    captured = H.results(previous)
    requests = A.read(previous / "requests.json")
    articles, pairs = (
        A.read(previous / "articles.json"),
        A.read(previous / "pairs.json"),
    )
    opportunities = config["opportunities"]
    observed = {}
    for r in requests:
        if (
            r["role"] == "discovery"
            and r["policy"] == 0
            and r["history"]
            and r["unit_index"] in captured
        ):
            observed[r["pair_id"], r["history"], r["opportunity"], r["order"]] = r
    eligible = [
        p
        for p in pairs
        if p["role"] == "discovery"
        and all(
            (p["pair_id"], h, c, o) in observed
            for h in (-1, 1)
            for c in opportunities
            for o in (0, 1)
        )
    ]
    if len(eligible) != config["source_complete_pairs"]:
        raise ValueError("Available source changed; no outcome-dependent reselection")
    source_ids, source_y, groups = [], [], []
    for p in eligible:
        for h, index in ((1, p["high_article"]), (-1, p["low_article"])):
            response = np.empty((2, 4, 2))
            for c, name in enumerate(opportunities):
                for o in (0, 1):
                    request = observed[p["pair_id"], h, name, o]
                    row = captured[request["unit_index"]]
                    account = H.M.choice_account(
                        row["ab_logits"], row["log_normalizer"], o
                    )
                    response[:, c, o] = [
                        account["q_positive"],
                        account["positive_log_odds"],
                    ]
            source_ids.append(index)
            source_y.append(response)
            groups.append(p["situation"] + "|" + p["style"])
    app_ids = [i for i, r in enumerate(articles) if r["role"] != "discovery"]
    app_pairs = [p for p in pairs if p["role"] != "discovery"]
    if len(app_ids) != 108 or len(app_pairs) != 54 or len(source_ids) != 42:
        raise ValueError("Full new finite article support missing")
    old_units = A.read(previous / "units.json")
    captured_hashes = {r["unit_id"] for r in captured.values()}
    units, wanted, lookup = [], [], {}
    for r in requests:
        if r["role"] == "discovery" or r["policy"] != 0 or not r["history"]:
            continue
        unit = old_units[r["unit_index"]]
        if unit["unit_id"] in captured_hashes:
            raise ValueError("New application input already observed in source")
        if unit["unit_id"] not in lookup:
            lookup[unit["unit_id"]] = len(units)
            units.append(unit)
        item = {k: v for k, v in r.items() if k != "messages"}
        item["old_unit_index"] = r["unit_index"]
        item["unit_index"] = lookup[unit["unit_id"]]
        wanted.append(item)
    if len(wanted) != config["application_requests"]:
        raise ValueError("All864 application requests required")
    with np.load(previous / "features.npz", allow_pickle=False) as old:
        source_x, application_x = old["features"][source_ids], old["features"][app_ids]
    out.mkdir(parents=True, exist_ok=False)
    (out / "batches").mkdir()
    npz(
        out / "source.npz",
        features=source_x,
        responses=source_y,
        groups=groups,
        article_indices=source_ids,
    )
    npz(out / "application_features.npz", features=application_x)
    A.write(out / "source_articles.json", [articles[i] for i in source_ids])
    A.write(
        out / "application_articles.json",
        [articles[i] | {"old_article_index": i} for i in app_ids],
    )
    A.write(out / "application_pairs.json", app_pairs)
    A.write(out / "requests.json", wanted)
    A.write(out / "units.json", units)
    inputs = [
        previous / name
        for name in (
            "manifest.json",
            "features.npz",
            "articles.json",
            "requests.json",
            "units.json",
            "pairs.json",
            "collection_status.json",
        )
    ]
    inputs += sorted((previous / "batches").glob("[0-9][0-9][0-9][0-9].json"))
    names = (
        "source.npz",
        "application_features.npz",
        "source_articles.json",
        "application_articles.json",
        "application_pairs.json",
        "requests.json",
        "units.json",
    )
    A.write(
        out / "manifest.json",
        {
            "packet": "HM166",
            "config": config,
            "science_hashes": {name: A.digest(ROOT / name) for name in SCIENCE},
            "input_hashes": {str(p.relative_to(ROOT)): A.digest(p) for p in inputs},
            "prepared_hashes": {name: A.digest(out / name) for name in names},
            "option_token_ids": old_manifest["option_token_ids"],
            "model_weights_sha256": old_manifest["model_weights_sha256"],
            "source_articles": len(source_ids),
            "application_articles": len(app_ids),
            "logical_requests": len(wanted),
            "unique_histories": len(units),
            "source_selection": "completeness only; nonrandom E-only historical prefix",
            "application_outcomes_at_prepare": 0,
            "predecessor_full_scope_complete": False,
            "original_scope_complete": False,
            "arms": ARM_NAMES,
            "channels": CHANNELS,
        },
    )
    print(
        json.dumps(
            {"prepared": str(out), "source": len(source_ids), "unique": len(units)}
        )
    )


def compile_source(out):
    """Compile all predeclared observable functions without persona labels or target fit."""
    check(out)
    if list((out / "batches").glob("[0-9]*.json")):
        raise ValueError("Application outcomes exist before source freezing")
    with np.load(out / "source.npz", allow_pickle=False) as s:
        x, y, g = s["features"], s["responses"], s["groups"]
    m, c, w, b = M.response_components(x, y, g)
    additive = np.array([mask.bit_count() <= 1 for mask in S.feature_sectors()])
    coefficients = [c, c * additive, w, b]
    for shift in range(1, 6):
        _, relink, _, _ = M.response_components(x, M.relink_profiles(y, g, shift), g)
        coefficients.append(relink)
    npz(out / "source_program.npz", origin=m, coefficients=coefficients)
    A.write(
        out / "source_program_seal.json",
        {
            "file": "source_program.npz",
            "sha256": A.digest(out / "source_program.npz"),
            "source_only": True,
            "application_outcomes": 0,
            "decomposition_error": float(np.max(abs(c - w - b))),
            "meaning": "cross-moment response witness;not conditional expectation/probability",
        },
    )
    print(
        json.dumps({"source_functions": [9, 2, 4, 2, 625], "application_outcomes": 0})
    )


def cold(out):
    """Run a numeric-only subprocess and seal all prospective article readings."""
    check(out)
    artifact(out, "source_program_seal.json")
    if list((out / "batches").glob("[0-9]*.json")):
        raise ValueError("Cold application is no longer prospective")
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
            "command": command,
            "application_outcomes": 0,
        },
    )


def captures(out):
    """Read real completed batches once per exact input; no duplicate observations."""
    result = {}
    for p in sorted((out / "batches").glob("[0-9][0-9][0-9][0-9].json")):
        for row in A.read(p):
            if row["unit_index"] in result:
                raise ValueError("Duplicate native capture")
            result[row["unit_index"]] = row
    return result


def collect(out):
    """Observe a finite new application after cold outputs; no fit/retry/budget renewal."""
    import torch

    manifest = check(out)
    artifact(out, "source_program_seal.json")
    artifact(out, "cold_seal.json")
    if captures(out) or (out / "collection_status.json").exists():
        raise ValueError(
            "Collection already attempted; no rerun or automatic continuation"
        )
    config, units = manifest["config"], A.read(out / "units.json")
    observer = H.OLD.offline_model(manifest)
    A.write(out / "model_manifest.json", observer.manifest())
    elapsed, count = 0.0, 0
    for batch, start in enumerate(range(0, len(units), config["batch_size"])):
        if elapsed >= config["model_seconds_cap"]:
            state(
                out,
                status="PARTIAL_BUDGET",
                completed=count,
                total=len(units),
                model_seconds=elapsed,
                pid=os.getpid(),
            )
            return
        ids = list(range(start, min(start + config["batch_size"], len(units))))
        group = [units[i]["prompt_ids"] for i in ids]
        width, eos = max(map(len, group)), observer.tokenizer.eos_token_id
        padded = torch.tensor(
            [[eos] * (width - len(p)) + p for p in group], device="mps"
        )
        mask = torch.tensor(
            [[0] * (width - len(p)) + [1] * len(p) for p in group], device="mps"
        )
        state(
            out,
            status="RUNNING",
            batch=batch,
            completed=count,
            total=len(units),
            model_seconds=elapsed,
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
        A.write(out / "batches" / f"{batch:04d}.json", rows)
        count += len(ids)
        print(
            json.dumps(
                {
                    "batch": batch,
                    "completed": count,
                    "total": len(units),
                    "model_seconds": elapsed,
                }
            ),
            flush=True,
        )
    A.write(
        out / "capture_seal.json",
        {
            str(p.relative_to(out)): A.digest(p)
            for p in sorted((out / "batches").glob("[0-9][0-9][0-9][0-9].json"))
        },
    )
    state(
        out,
        status="COMPLETE",
        completed=count,
        total=len(units),
        model_seconds=elapsed,
        pid=os.getpid(),
        all_batches_terminal=True,
    )


def application_responses(out):
    """Address observed profiles, retaining incomplete pairs as missing rather than zero."""
    manifest = check(out)
    cap, req = captures(out), A.read(out / "requests.json")
    pairs = {p["pair_id"]: p for p in A.read(out / "application_pairs.json")}
    articles = A.read(out / "application_articles.json")
    address = {r["old_article_index"]: i for i, r in enumerate(articles)}
    y, mass = (
        np.full((len(articles), 2, 4, 2), np.nan),
        np.full((len(articles), 4, 2), np.nan),
    )
    for r in req:
        if r["unit_index"] not in cap:
            continue
        observed = cap[r["unit_index"]]
        p = pairs[r["pair_id"]]
        i = address[p["high_article"] if r["history"] == 1 else p["low_article"]]
        c, o = manifest["config"]["opportunities"].index(r["opportunity"]), r["order"]
        if np.isfinite(y[i, 0, c, o]):
            raise ValueError("Duplicate article response address")
        account = H.M.choice_account(
            observed["ab_logits"], observed["log_normalizer"], o
        )
        y[i, :, c, o] = [account["q_positive"], account["positive_log_odds"]]
        mass[i, c, o] = account["unrestricted_option_mass"]
    complete, missing = [], []
    for p in pairs.values():
        hi, lo = address[p["high_article"]], address[p["low_article"]]
        if np.isfinite(y[[hi, lo]]).all():
            complete.append((p, hi, lo))
        else:
            missing.append(
                {
                    "pair_id": p["pair_id"],
                    "available_responses": int(np.isfinite(y[[hi, lo], 0]).sum()),
                    "required": 16,
                }
            )
    return y, mass, complete, missing


def analyze(out):
    """Report frozen function transfer and signed condition/order comparisons without refit."""
    manifest = check(out)
    artifact(out, "source_program_seal.json")
    artifact(out, "cold_seal.json")
    current = A.read(out / "collection_status.json")
    if current["status"] not in ("COMPLETE", "PARTIAL_BUDGET"):
        raise ValueError("Application collection is not terminal")
    if current["status"] == "COMPLETE":
        for name, expected in A.read(out / "capture_seal.json").items():
            if A.digest(out / name) != expected:
                raise ValueError("Native application capture changed")
    y, mass, complete, missing = application_responses(out)
    if not complete:
        raise ValueError("No complete application pair; preserve terminal captures")
    with np.load(out / "cold_readings.npz", allow_pickle=False) as saved:
        reading = saved["readings"]
    response_gaps = M.order_fields(np.array([y[hi] - y[lo] for _, hi, lo in complete]))
    reading_gaps = M.order_fields(
        np.array([reading[hi] - reading[lo] for _, hi, lo in complete])
    )
    npz(out / "paired_accounts.npz", observed=response_gaps, cold=reading_gaps)
    pairs = [p for p, _, _ in complete]
    A.write(out / "complete_pairs.json", pairs)
    if missing:
        A.table(out / "missing_pairs.csv", missing)
    cells = []
    for i, p in enumerate(pairs):
        for arm, name in enumerate(ARM_NAMES):
            for m, channel in enumerate(CHANNELS):
                for c, opportunity in enumerate(manifest["config"]["opportunities"]):
                    for field, label in enumerate(("even", "odd")):
                        cells.append(
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
                            | {
                                "arm": name,
                                "channel": channel,
                                "opportunity": opportunity,
                                "order_field": label,
                                "cold_gap": float(reading_gaps[i, arm, m, c, field]),
                                "observed_gap": float(response_gaps[i, m, c, field]),
                            }
                        )
    A.table(out / "paired_response_accounts.csv", cells)
    summaries, cross = [], []
    for role in ("all", "development", "evaluation"):
        for axis in ("all", "extraversion", "neuroticism"):
            selected = np.array(
                [
                    (role == "all" or p["role"] == role)
                    and (axis == "all" or p["axis"] == axis)
                    for p in pairs
                ]
            )
            if not selected.any():
                continue
            for arm, name in enumerate(ARM_NAMES):
                for m, channel in enumerate(CHANNELS):
                    for c, opportunity in enumerate(
                        manifest["config"]["opportunities"]
                    ):
                        for field, label in enumerate(("even", "odd")):
                            left = reading_gaps[selected, arm, m, c, field]
                            base = {
                                "role": role,
                                "axis": axis,
                                "arm": name,
                                "channel": channel,
                                "source_function": opportunity,
                                "order_field": label,
                            }
                            for d, target in enumerate(
                                manifest["config"]["opportunities"]
                            ):
                                metric = M.signed_alignment(
                                    left, response_gaps[selected, m, d, field]
                                )
                                cross.append(
                                    base | {"observed_opportunity": target} | metric
                                )
                                if c == d:
                                    summaries.append(base | metric)
    A.table(out / "alignment_summary.csv", summaries)
    A.table(out / "opportunity_cross_readings.csv", cross)
    primary = [
        r
        for r in summaries
        if r["role"] == "evaluation"
        and r["axis"] != "all"
        and r["channel"] == "probability"
        and r["order_field"] == "even"
    ]
    A.write(
        out / "summary.json",
        {
            "status": current["status"],
            "source_articles": 42,
            "complete_application_pairs": len(pairs),
            "planned_pairs": 54,
            "new_native_captures": len(captures(out)),
            "new_native_batches": len(
                list((out / "batches").glob("[0-9][0-9][0-9][0-9].json"))
            ),
            "model_seconds": current["model_seconds"],
            "source_is_E_only_nonrandom_prefix": True,
            "application_program_was_frozen_before_outcomes": True,
            "option_mass_min": float(np.nanmin(mass)),
            "option_mass_mean": float(np.nanmean(mass)),
            "evaluation_probability_even": primary,
            "original_scope_complete": False,
            "scope": "model response witness;not probability/human trait accuracy;no tuning or inverse",
        },
    )
    names = (
        "paired_accounts.npz",
        "complete_pairs.json",
        "paired_response_accounts.csv",
        "alignment_summary.csv",
        "opportunity_cross_readings.csv",
        "summary.json",
    )
    A.write(out / "analysis_seal.json", {name: A.digest(out / name) for name in names})
    print(
        json.dumps(
            {
                "status": current["status"],
                "pairs": len(pairs),
                "native": len(captures(out)),
            }
        )
    )


def main():
    """Execute one explicit finite stage, never silently retry or continue old queues."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "compile", "cold", "collect", "analyze")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    {
        "prepare": prepare,
        "compile": compile_source,
        "cold": cold,
        "collect": collect,
        "analyze": analyze,
    }[args.stage](args.out)


if __name__ == "__main__":
    main()
