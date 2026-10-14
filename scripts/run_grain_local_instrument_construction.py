#!/usr/bin/env python3
"""Build two contextual instruments from source evidence, then cold-score new authors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.choice_design import learn_cell_moments
from grain_research.contextual_instrument import (
    PROFILE_TARGETS,
    evidence_rows,
    score_records,
    text_key,
)
from grain_research.pilot_transport import calibration, fit_pilot, restore_units
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_entailment_reference import MODEL, REVISION
from scripts.run_grain_query_moment_operator import read, sha, write

BASE = ROOT / "results/grain_vector_mechanisms/mechanism_todo_v1"
DEFAULT = BASE / "local_instrument_r1"
OLD = BASE / "condition_accumulation_r1"
BANK = OLD / "public_scored_bank.csv"
CONFIG = ROOT / "configs/grain_local_instrument_construction.json"
PLAN = ROOT / "docs/GRAIN_LOCAL_INSTRUMENT_CONSTRUCTION_PLAN_V1.md"
NATURAL = BASE / "relational_evidence_r1/natural/private_relations.jsonl"


def jsonlines(path, rows):
    """Write generated raw records or their separate evaluation-only latent targets."""
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def loadlines(path):
    """Read only the explicitly requested evidence partition."""
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latent_targets(spec, profile, state, z, partner):
    """Declare evaluation-only targets, never passed to source construction/scoring."""
    if spec["kind"] == "conditional_profile":
        cells = np.array([profile[0] - profile[1], profile[0] + profile[1]])
        return {name: float(t @ cells) for name, t in PROFILE_TARGETS.items()}
    weights = np.r_[0.5, np.ones(len(state) - 2), 0.5] / (len(state) - 1)
    return {
        "level": profile[0],
        "condition": profile[1],
        "partner": profile[2],
        "current_state": float(state[-1]),
        "current_response": float(
            profile @ np.array([1.0, z[-1], partner[-1]]) + state[-1]
        ),
        "window_state": float(weights @ state),
    }


def generate(cfg, spec, bank, seed, partition, wording):
    """Generate bounded latent worlds, independent events and raw text selections."""
    instrument_index = 0 if spec["kind"] == "conditional_profile" else 1
    code = 0 if partition == "pilot" else (1 if wording == "source" else 2)
    rng = np.random.default_rng(np.random.SeedSequence([seed, instrument_index, code]))
    n = cfg["pilot_authors"] if partition == "pilot" else cfg["evaluation_authors"]
    pool = bank[(bank["query"] == spec["query"]) & (bank.partition == wording)]
    phrases = {
        (z, sign): pool[(pool.condition == z) & (pool.sign == sign)].text.tolist()
        for z in (-1, 1)
        for sign in (-1, 1)
    }
    raw, truth = [], []
    for person in range(n):
        author = f"{spec['id']}:{seed}:{partition}:{wording}:{person:03d}"
        if spec["kind"] == "conditional_profile":
            profile = [float(rng.uniform(-0.35, 0.35)), float(rng.uniform(-0.20, 0.20))]
            count = (
                cfg["profile_pilot_events_per_cell"]
                if partition == "pilot"
                else cfg["profile_application_events_per_cell"]
            )
            z, u, state = np.array([-1, 1]), np.zeros(2), np.zeros(2)
            batches = [
                (ti, int(zi), 0.0, 0, count, profile[0] + profile[1] * zi)
                for ti, zi in enumerate(z)
            ]
        else:
            profile = [float(rng.uniform(-b, b)) for b in (0.12, 0.10, 0.08)]
            z = np.resize(np.array([-1, 1, -1, 1]), cfg["process_occasions"])
            u = np.resize(np.array([-1.0, -1.0, 1.0, 1.0]), cfg["process_occasions"])
            previous = 0.0
            for _ in range(256):
                previous = 0.7 * previous + 0.3 * rng.uniform(-0.45, 0.45)
            state = []
            for _ in z:
                previous = 0.7 * previous + 0.3 * rng.uniform(-0.45, 0.45)
                state.append(previous)
            state = np.asarray(state)
            batches = [
                (
                    ti,
                    int(zi),
                    float(ui),
                    rep,
                    cfg["process_events_per_replicate"],
                    profile[0] + profile[1] * zi + profile[2] * ui + state[ti],
                )
                for ti, (zi, ui) in enumerate(zip(z, u, strict=True))
                for rep in range(cfg["process_replicates"])
            ]
        for ti, zi, ui, rep, count, mu in batches:
            if abs(mu) >= 1:
                raise ValueError(
                    "Declared bounded world exceeded binary expectation range"
                )
            for event in range(count):
                sign = 1 if rng.random() < (1 + mu) / 2 else -1
                choices = phrases[zi, sign]
                text = choices[int(rng.integers(len(choices)))]
                raw.append(
                    {
                        "author": author,
                        "event": f"{ti}:{rep}:{event}",
                        "text": text,
                        "scope": "self_event_report",
                        "condition": zi,
                        "time": float(ti),
                        "partner": ui,
                        "replicate": rep,
                    }
                )
        truth.append(
            {
                "author": author,
                "nominal": latent_targets(spec, profile, state, z, u),
                "profile": profile,
                "state": state.tolist(),
                "condition": z.tolist(),
                "partner": u.tolist(),
            }
        )
    return raw, truth


def prepare(out):
    """Freeze material/parameters and separate source raw evidence from hidden values."""
    cfg, oldcfg = read(CONFIG), read(OLD / "config.json")
    bank = pd.read_csv(BANK)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("raw", "truth", "cold_inputs"):
        (out / name).mkdir()
    sources = [
        CONFIG,
        PLAN,
        BANK,
        OLD / "public_nli.csv",
        OLD / "reader_manifest.json",
        OLD / "config.json",
        Path(__file__),
        ROOT / "scripts/score_grain_local_instrument.py",
        ROOT / "scripts/audit_grain_local_instrument_construction.py",
        ROOT / "grain_research/contextual_instrument.py",
        ROOT / "grain_research/choice_design.py",
        ROOT / "grain_research/pilot_transport.py",
        ROOT / "grain_research/temporal_measurement.py",
        ROOT / "grain_research/vector_mechanisms/stage_file_guard.py",
        ROOT / "scripts/run_grain_evidence_calibration.py",
        BASE / "query_moment_finite_r1/audit.json",
        NATURAL,
    ]
    old_manifest = read(OLD / "reader_manifest.json")
    if sha(OLD / "public_nli.csv") != old_manifest["files"]["public_nli.csv"]:
        raise ValueError("Old NLI evidence cache changed")
    raw_nli = pd.read_csv(OLD / "public_nli.csv").pivot(
        index="case_id", columns="key", values="p_entailment"
    )
    expected = bank.case_id.map(raw_nli.positive - raw_nli.negative)
    np.testing.assert_allclose(bank.event_margin, expected, atol=1e-12)
    queries = {q["id"]: q for q in oldcfg["queries"]}
    for spec in cfg["instruments"]:
        spec["positive_hypothesis"] = queries[spec["query"]]["positive_hypothesis"]
        spec["negative_hypothesis"] = queries[spec["query"]]["negative_hypothesis"]
    write(out / "config.json", cfg)
    cache = {
        text_key(r.query, r.text): float(r.event_margin) for r in bank.itertuples()
    }
    write(
        out / "reader_cache.json",
        {"model": MODEL, "revision": REVISION, "margins": cache},
    )
    bank[bank.partition == "source"].to_csv(out / "source_anchors.csv", index=False)
    jobs = []
    for spec in cfg["instruments"]:
        for seed in cfg["seeds"]:
            for partition, wording in (
                ("pilot", "source"),
                ("evaluation", "source"),
                ("evaluation", "held"),
            ):
                key = f"{spec['id']}_{seed}_{partition}_{wording}"
                raw, truth = generate(cfg, spec, bank, seed, partition, wording)
                jsonlines(out / "raw" / (key + ".jsonl"), raw)
                jsonlines(out / "truth" / (key + ".jsonl"), truth)
                jobs.append(
                    {
                        "key": key,
                        "instrument": spec["id"],
                        "seed": seed,
                        "partition": partition,
                        "wording": wording,
                        "authors": len(truth),
                        "events": len(raw),
                    }
                )
                if (
                    seed == cfg["seeds"][0]
                    and partition == "evaluation"
                    and wording == "held"
                ):
                    author = truth[0]["author"]
                    write(
                        out / "cold_inputs" / (spec["id"] + ".json"),
                        {
                            "instrument_id": spec["id"],
                            "clock_unit": spec["clock_unit"],
                            "records": [r for r in raw if r["author"] == author],
                        },
                    )
    prepared = [
        p for p in out.rglob("*") if p.is_file() and not p.name.startswith("._")
    ]
    write(
        out / "contract.json",
        {
            "protocol": cfg["protocol"],
            "jobs": jobs,
            "files": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
            "prepared_time_ns": time.time_ns(),
            "new_neural_calls_in_simulation": 0,
        },
    )
    print(
        {
            "stage": "prepare",
            "jobs": len(jobs),
            "authors": sum(j["authors"] for j in jobs),
            "events": sum(j["events"] for j in jobs),
        },
        flush=True,
    )


def verify(out):
    """Verify current frozen code and raw/evaluation partition bytes before execution."""
    contract = read(out / "contract.json")
    for name, expected in contract["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Changed source: " + name)
    for name, expected in contract["prepared"].items():
        if sha(out / name) != expected:
            raise ValueError("Changed prepared material: " + name)
    return contract, read(out / "config.json")


def construct(out):
    """Fit only source semantic anchors and independent pilot observations."""
    contract, cfg = verify(out)
    dst = out / "instruments"
    dst.mkdir(exist_ok=False)
    jobs = [j for j in contract["jobs"] if j["partition"] == "pilot"]
    paths = [dst / (j["instrument"] + "_" + str(j["seed"]) + ".json") for j in jobs]
    probes = [out / "truth" / (j["key"] + ".jsonl") for j in contract["jobs"]]
    probes += [
        out / "raw" / (j["key"] + ".jsonl")
        for j in contract["jobs"]
        if j["partition"] != "pilot"
    ]
    reads = [out / "reader_cache.json", out / "source_anchors.csv"] + [
        out / "raw" / (j["key"] + ".jsonl") for j in jobs
    ]
    denied = install(
        reads,
        [*paths, dst / "freeze.json"],
        probes,
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    bank, margins = (
        pd.read_csv(out / "source_anchors.csv"),
        read(out / "reader_cache.json")["margins"],
    )
    start = time.process_time()
    fit_rows = []
    for job, path in zip(jobs, paths, strict=True):
        spec = next(s for s in cfg["instruments"] if s["id"] == job["instrument"])
        anchor = bank[bank["query"] == spec["query"]]
        fitted_units = calibration(
            anchor.sign, anchor.event_margin, anchor.condition, conditional=True
        )
        bundle = {
            "spec": spec,
            "calibration": fitted_units,
            "reader": {"model": MODEL, "revision": REVISION},
            "seed": job["seed"],
            "pilot_reports_per_occasion": cfg["process_replicates"]
            * cfg["process_events_per_replicate"],
        }
        rows = evidence_rows(
            loadlines(out / "raw" / (job["key"] + ".jsonl")), margins, bundle
        )
        groups = defaultdict(list)
        for r in rows:
            groups[r["author"]].append(r)
        members = sorted(groups)
        if spec["kind"] == "conditional_profile":
            y = np.array(
                [
                    [
                        [r["value"] for r in groups[a] if r["condition"] == z]
                        for z in (-1, 1)
                    ]
                    for a in members
                ]
            )
            fit = learn_cell_moments
        else:
            first = groups[members[0]]
            times = np.array(sorted({r["time"] for r in first}))
            z = np.array(
                [next(r["condition"] for r in first if r["time"] == t) for t in times]
            )
            u = np.array(
                [next(r["partner"] for r in first if r["time"] == t) for t in times]
            )
            y = np.array(
                [
                    [
                        [
                            np.mean(
                                [
                                    r["value"]
                                    for r in groups[a]
                                    if r["time"] == t and r["replicate"] == rep
                                ]
                            )
                            for rep in range(cfg["process_replicates"])
                        ]
                        for t in times
                    ]
                    for a in members
                ]
            )

            def fit(values, z=z, u=u, times=times):
                return fit_pilot(values, z, u, times)

        main = fit(y)
        rng = np.random.default_rng(job["seed"] + 80)
        boot = [
            fit(y[rng.integers(0, len(y), len(y))])
            for _ in range(cfg["bootstrap_constructions"])
        ]
        bundle.update(
            main=main,
            bootstrap=boot,
            pilot_authors=members,
            construction_input="raw pilot text through fixed reader; no author truth or evaluation material",
        )
        write(path, bundle)
        fit_rows.append(
            {
                "instrument": spec["id"],
                "seed": job["seed"],
                "main_success": main.get("success", True),
                "bootstrap_nonconverged": sum(not b.get("success", True) for b in boot),
                "main_floor_or_boundary": main.get(
                    "floor_count", main.get("bound_count", 0)
                ),
                "main_rate": main.get("rate"),
                "main_stationary": main.get("stationary"),
            }
        )
        if time.process_time() - start > cfg["max_numerical_cpu_seconds"]:
            raise TimeoutError(
                "Source construction CPU budget exhausted; preserve fits"
            )
    write(
        dst / "freeze.json",
        {
            "files": {p.name: sha(p) for p in paths},
            "fits": fit_rows,
            "cpu_seconds": time.process_time() - start,
            "frozen_time_ns": time.time_ns(),
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
            "human_labels": False,
        },
    )
    print(
        {
            "stage": "construct",
            "bundles": len(paths),
            "cpu_seconds": time.process_time() - start,
            "fits": fit_rows,
        },
        flush=True,
    )


def frozen_bundles(out):
    """Require unchanged source instruments for every new-person application."""
    freeze = read(out / "instruments/freeze.json")
    for name, expected in freeze["files"].items():
        if sha(out / "instruments" / name) != expected:
            raise ValueError("Source instrument changed")
    return freeze


def apply(out):
    """Apply frozen instruments and save source/raw-data ownership before evaluation."""
    contract, cfg = verify(out)
    frozen_bundles(out)
    dst = out / "application"
    dst.mkdir(exist_ok=False)
    jobs = [j for j in contract["jobs"] if j["partition"] == "evaluation"]
    bundles = {
        p.name: read(p)
        for p in (out / "instruments").glob("*.json")
        if p.name != "freeze.json" and not p.name.startswith("._")
    }
    margins = read(out / "reader_cache.json")["margins"]
    writes = [dst / (j["key"] + ".json") for j in jobs] + [
        dst / n for n in ("predictions.csv", "freeze.json", "ablations.json")
    ]
    reads = [out / "raw" / (j["key"] + ".jsonl") for j in jobs]
    probes = [out / "truth" / (j["key"] + ".jsonl") for j in contract["jobs"]]
    denied = install(reads, writes, probes, [ROOT / "scripts", ROOT / "grain_research"])
    start = time.process_time()
    records, ablations = [], []
    for job in jobs:
        bundle = bundles[f"{job['instrument']}_{job['seed']}.json"]
        spec = bundle["spec"]
        raw = loadlines(out / "raw" / (job["key"] + ".jsonl"))
        scores, evidence = score_records(
            raw, margins, bundle, clock_unit=spec["clock_unit"], explain=False
        )
        write(dst / (job["key"] + ".json"), scores)
        group = defaultdict(list)
        for row in evidence:
            group[row["author"]].append(row)
        for author, scored in scores.items():
            rows = group[author]
            if spec["kind"] == "conditional_profile":
                empirical = np.array(
                    [
                        np.mean([r["value"] for r in rows if r["condition"] == z])
                        for z in (-1, 1)
                    ]
                )
                simple = {n: float(t @ empirical) for n, t in PROFILE_TARGETS.items()}
            else:
                times = sorted({r["time"] for r in rows})
                yy = np.array(
                    [
                        np.mean([r["value"] for r in rows if r["time"] == t])
                        for t in times
                    ]
                )
                zz = [
                    next(r["condition"] for r in rows if r["time"] == t) for t in times
                ]
                uu = [next(r["partner"] for r in rows if r["time"] == t) for t in times]
                x = np.c_[np.ones(len(times)), zz, uu]
                beta = np.linalg.lstsq(x, yy, rcond=None)[0]
                state = yy - x @ beta
                simple = latent_targets(spec, beta, state, zz, uu)
            for name, result in scored["targets"].items():
                common = {
                    "instrument": spec["id"],
                    "seed": job["seed"],
                    "world": job["wording"],
                    "author": author,
                    "target": name,
                }
                records.append(
                    {
                        **common,
                        "arm": "source_constructed",
                        **{
                            k: v
                            for k, v in result.items()
                            if isinstance(v, (float, int)) or v is None
                        },
                    }
                )
                records.append(
                    {
                        **common,
                        "arm": "empirical_reference",
                        "score": simple[name],
                        "variance": None,
                    }
                )
        if job["seed"] == cfg["seeds"][0] and job["wording"] == "held":
            author = min(scores)
            sample = [r for r in raw if r["author"] == author]
            original = score_records(
                sample, margins, bundle, clock_unit=spec["clock_unit"]
            )[0][author]
            variants = {
                "exact_repeat": sample + [sample[0].copy()] * 5,
                "record_order_only": list(reversed(sample)),
                "missing_positive_condition": [
                    r for r in sample if r["condition"] == -1
                ],
                "unqualified_scope": [dict(r, scope="unknown") for r in sample],
            }
            if spec["kind"] == "occasion_process":
                largest = max(r["time"] for r in sample)
                variants["reverse_clock_assignment"] = [
                    dict(r, time=largest - r["time"]) for r in sample
                ]
            results = {
                k: score_records(v, margins, bundle, clock_unit=spec["clock_unit"])[0][
                    author
                ]
                for k, v in variants.items()
            }
            ablations.append(
                {
                    "instrument": spec["id"],
                    "author": author,
                    "original": original,
                    "variants": results,
                }
            )
        if time.process_time() - start > cfg["max_numerical_cpu_seconds"]:
            raise TimeoutError("Application CPU budget reached; preserve results")
    pd.DataFrame(records).to_csv(dst / "predictions.csv", index=False)
    write(dst / "ablations.json", ablations)
    write(
        dst / "freeze.json",
        {
            "files": {p.name: sha(p) for p in writes if p.name != "freeze.json"},
            "rows": len(records),
            "cpu_seconds": time.process_time() - start,
            "frozen_time_ns": time.time_ns(),
            "target_denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
        },
    )
    print(
        {
            "stage": "apply",
            "rows": len(records),
            "cpu_seconds": time.process_time() - start,
        },
        flush=True,
    )


def metrics(frame, keys):
    """Report target loss and working-interval coverage for independent authors."""
    rows = []
    for group, f in frame.groupby(keys):
        error = f.score.to_numpy() - f.truth.to_numpy()
        variance = f.variance.to_numpy()
        rows.append(
            {
                **dict(zip(keys, group, strict=True)),
                "n": len(f),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "r": float(np.corrcoef(f.score, f.truth)[0, 1])
                if min(f.score.std(), f.truth.std()) > 1e-12
                else None,
                "coverage95": float(
                    np.mean(abs(error) <= 1.959963984540054 * np.sqrt(variance))
                )
                if np.isfinite(variance).all()
                else None,
            }
        )
    return pd.DataFrame(rows)


def evaluate(out):
    """Open nominal latent and finite reader targets only after freezing predictions."""
    contract, _cfg = verify(out)
    frozen_bundles(out)
    freeze = read(out / "application/freeze.json")
    for name, expected in freeze["files"].items():
        if sha(out / "application" / name) != expected:
            raise ValueError("Application changed before evaluation")
    dst = out / "evaluation"
    dst.mkdir(exist_ok=False)
    pred = pd.read_csv(out / "application/predictions.csv")
    bank = pd.read_csv(BANK)
    truth_rows = []
    for job in contract["jobs"]:
        if job["partition"] != "evaluation":
            continue
        bundle = read(out / "instruments" / f"{job['instrument']}_{job['seed']}.json")
        spec = bundle["spec"]
        pool = bank[
            (bank["query"] == spec["query"]) & (bank.partition == job["wording"])
        ].copy()
        pool["reading"] = restore_units(
            pool.event_margin.to_numpy(),
            pool.condition.to_numpy(),
            bundle["calibration"],
        )
        means = {
            (z, sign): pool[(pool.condition == z) & (pool.sign == sign)].reading.mean()
            for z in (-1, 1)
            for sign in (-1, 1)
        }
        for row in loadlines(out / "truth" / (job["key"] + ".jsonl")):
            for target, value in row["nominal"].items():
                truth_rows.append(
                    {
                        "author": row["author"],
                        "target": target,
                        "truth_kind": "nominal_event_world",
                        "truth": value,
                    }
                )
            if spec["kind"] == "conditional_profile":
                cells = []
                for z in (-1, 1):
                    mu = row["profile"][0] + z * row["profile"][1]
                    cells.append(
                        (1 - mu) / 2 * means[z, -1] + (1 + mu) / 2 * means[z, 1]
                    )
                target_values = {
                    name: float(t @ cells) for name, t in PROFILE_TARGETS.items()
                }
            else:
                z = row["condition"][-1]
                mu = row["nominal"]["current_response"]
                target_values = {
                    "current_response": (1 - mu) / 2 * means[z, -1]
                    + (1 + mu) / 2 * means[z, 1]
                }
            truth_rows.extend(
                {
                    "author": row["author"],
                    "target": name,
                    "truth_kind": "fixed_reader_expectation",
                    "truth": value,
                }
                for name, value in target_values.items()
            )
    truth = pd.DataFrame(truth_rows)
    result = pred.merge(truth, on=["author", "target"], validate="many_to_many")
    result.to_csv(dst / "predictions_with_targets.csv", index=False)
    keys = ["instrument", "world", "truth_kind", "target", "arm"]
    metrics(result, keys).to_csv(dst / "metrics.csv", index=False)
    metrics(result, ["seed", *keys]).to_csv(dst / "metrics_by_seed.csv", index=False)
    ablations = read(out / "application/ablations.json")
    checks = []
    for example in ablations:
        original = example["original"]
        for kind, changed in example["variants"].items():
            diffs = [
                abs(v["score"] - original["targets"][name]["score"])
                for name, v in changed["targets"].items()
            ]
            checks.append(
                {
                    "instrument": example["instrument"],
                    "variant": kind,
                    "status": changed["status"],
                    "qualified_events": changed["qualified_events"],
                    "max_target_change": max(diffs) if diffs else None,
                    "missing_condition": changed.get("empirically_absent_condition"),
                }
            )
    pd.DataFrame(checks).to_csv(dst / "interface_checks.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "jobs": len(contract["jobs"]),
        "construction": read(out / "instruments/freeze.json"),
        "application_cpu_seconds": freeze["cpu_seconds"],
        "evaluation_rows": len(result),
        "nominal_target": "Known simulated event-propensity/occasion world, not human psychological truth",
        "reader_target": "Finite wording-conditioned expectation, not equivalent to nominal latent truth",
        "new_neural_calls": 0,
        "human_or_meps_fit": False,
        "independent_builder": False,
    }
    write(dst / "summary.json", summary)
    write(
        dst / "freeze.json",
        {
            "files": {
                p.name: sha(p)
                for p in dst.iterdir()
                if p.is_file() and not p.name.startswith("._")
            }
        },
    )
    print(
        {"stage": "evaluate", "rows": len(result), "interface_checks": checks},
        flush=True,
    )


def natural(out):
    """Audit addressable natural events against the examples, without fabricating design."""
    _, cfg = verify(out)
    dst = out / "natural"
    dst.mkdir(exist_ok=False)
    events = loadlines(NATURAL)
    rows = []
    for event in events:
        for spec in cfg["instruments"]:
            rows.append(
                {
                    "event_id": event["event_id"],
                    "instrument": spec["id"],
                    "source_sha256": event["source_sha256"],
                    "address_available": bool(
                        event["predicate_span"] and event["word_ids"]
                    ),
                    "symbolic_condition_links": len(event["condition_links"]),
                    "semantic_actor_available": event["semantic_actor"] is not None,
                    "assigned_instrument_condition_available": False,
                    "recorded_event_time_available": event["event_time"] is not None,
                    "status": "not_equated_to_this_controlled_instrument",
                    "score": None,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(dst / "private_applicability.csv", index=False)
    write(
        dst / "summary.json",
        {
            "events": len(events),
            "instrument_event_rows": len(rows),
            "addressable_rows": int(frame.address_available.sum()),
            "eligible_examples": 0,
            "new_neural_calls": 0,
            "new_text_or_scale_reads": 0,
            "scope": "This inherited event ledger does not supply the examples' controlled conditions or semantic qualification; not a claim that all raw corpora lack timestamps or all GRAIN measurement is unavailable",
        },
    )
    print({"stage": "natural", "events": len(events), "qualified": 0}, flush=True)


def main():
    """Use a separate process per irreversible source/application access guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "construct", "apply", "evaluate", "natural")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
