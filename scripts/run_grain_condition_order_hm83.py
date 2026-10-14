#!/usr/bin/env python3
"""Finite condition/observation experiment, with new-text cold functions."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import local_measure as L
from grain_research.construction import math_author_contrast as A
from grain_research.construction import math_cold_joint_author as J
from grain_research.construction import math_condition_order as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL
from scripts.run_grain_author_contrast_hm82 import load, seal, sha, table, write
from scripts.run_grain_math_construct_routing import write as mutable_write
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/condition_observation_order_r1"
SOURCE = ROOT / "results/grain_math_response_v1/author_contrast_observability_r1"
HM81 = ROOT / "results/grain_math_response_v1/cold_joint_author_r1"
HM79 = ROOT / "results/grain_math_response_v1/role_assignment_dealiasing_r1"
CONFIG = ROOT / "configs/grain_condition_observation_order_hm83.json"
CODE = (
    "docs/GRAIN_CONDITION_OBSERVATION_ORDER_PLAN_HM83.md",
    "configs/grain_condition_observation_order_hm83.json",
    "grain_research/construction/math_condition_order.py",
    "grain_research/construction/math_author_contrast.py",
    "grain_research/construction/math_cold_joint_author.py",
    "grain_research/local_measure.py",
    "grain_research/construction/math_answer_recoding_mechanism.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_math_relational_cut.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
    "scripts/run_grain_condition_order_hm83.py",
    "scripts/verify_grain_condition_order_hm83.py",
    "tests/test_grain_math_condition_order.py",
)


def materials(config):
    """Generate new deterministic public accounts with two distinct behavior targets."""
    bodies = []
    for focal, companion in itertools.permutations(J.TARGETS, 2):
        for x, c in itertools.product((-1, 1), repeat=2):
            body = (
                "At a neighbourhood workshop, the following completed actions were recorded.\n"
                f"Person A {config['behaviours'][focal][(x + 1) // 2]}.\n"
                f"Person B {config['behaviours'][companion][(c + 1) // 2]}.\n"
                "Person A opened a plain notebook on a table after the activity."
            )
            bodies.append(
                {
                    "event_id": f"new_{len(bodies):03d}",
                    "body": body,
                    "focal": focal,
                    "companion": companion,
                    "x": x,
                    "c": c,
                    "actor": -1,
                    "anchor": 1,
                }
            )
    if len(bodies) != 120 or len({b["body"] for b in bodies}) != 120:
        raise ValueError("120 distinct accounts required")
    return bodies


def prepare(out):
    """Freeze source projection coefficients and new stimuli before any inference."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("preserve existing packets")
    if (
        load(SOURCE / "verification.json")["status"]
        != "VERIFIED_FINITE_HM82_NOT_GLOBAL"
    ):
        raise ValueError("verified HM82 source required")
    source_manifest = load(SOURCE / "source_freeze.json")
    inputs = dict(source_manifest["inputs"])
    for name, expected in source_manifest["files"].items():
        path = SOURCE / name
        if sha(path) != expected:
            raise ValueError("source scientific seal drift: " + name)
        inputs[str(path.relative_to(ROOT))] = expected
    for name in ("source_freeze.json", "verification.json"):
        path = SOURCE / name
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for name, expected in inputs.items():
        if sha(ROOT / name) != expected:
            raise ValueError("inherited source drift: " + name)
    observation = load(HM81 / "00/bundle.json")
    J.verify_bundle(observation)
    with np.load(SOURCE / "source.npz") as a:
        p, t, c = a["reference"], a["targets"], a["conditions"]
        base = a["marginal10_features"]
        arms = M.matrices(base[:, :4], base[:, 4:], a["joint49_features"], c)
    functions, accounts, visibility = {}, [], []
    for arm, features in arms.items():
        functions[arm] = M.compile_function(features, t, p)
        accounts.append({"arm": arm} | M.order_account(features, p, c))
        projection = A.source_projection(features, p)
        retained = A.project(t, p, projection)
        for j, target in enumerate(M.TARGETS):
            visibility.append(
                {
                    "arm": arm,
                    "target": target,
                    "visible_norm2": A.norm2(retained[:, j], p),
                }
            )
    config = load(CONFIG)
    application = materials(config)
    prior_bodies = {r["body"] for r in load(HM79 / "00/rows.json")}
    prior_bodies |= {r["body"] for r in load(HM81 / "00/application.json")}
    if prior_bodies.intersection(b["body"] for b in application):
        raise ValueError("new material overlaps historical body")
    material_path = ROOT / "configs/grain_math_passive_event_composition_v1.json"
    inherited_material = load(material_path)
    alphabet = load(HM79 / "00/manifest.json")["alphabet"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    if [tokenizer.encode(s, add_special_tokens=False) for s in alphabet["labels"]] != [
        [t] for t in alphabet["token_ids"]
    ]:
        raise ValueError("answer alphabet changed")
    design = []
    for event in application:
        for query in J.TARGETS:
            text, ids, part, _, _ = render(
                event["body"], query, "A", 1, inherited_material["system"], tokenizer
            )
            design.append(
                event
                | {
                    "unit": len(design),
                    "query": query,
                    "rendered": text,
                    "input_ids": ids,
                    "partition": part,
                }
            )
    for path in (
        material_path,
        HM79 / "01/observer.json",
        HM81 / "00/application.json",
        MODEL / "model.safetensors",
    ):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    code = {}
    for name in CODE:
        target = out / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        code[name] = sha(target)
    write(
        out / "00/bundle.json",
        {
            "packet": "HM83",
            "observation": observation,
            "functions": functions,
            "targets": M.TARGETS,
            "scope": "synthetic design functions; source-frozen conditional observation",
        },
    )
    write(out / "00/application.json", application)
    write(out / "00/design.json", design)
    table(out / "00/order_accounts.csv", accounts)
    table(out / "00/source_target_visibility.csv", visibility)
    paths = [
        p for p in (out / "00").iterdir() if p.is_file() and not p.name.startswith("._")
    ]
    write(
        out / "00/manifest.json",
        {
            "packet": "HM83",
            "code": code,
            "inputs": inputs,
            "assets": seal(out, paths),
            "alphabet": alphabet,
            "max_forwards": config["max_forwards"],
            "max_model_seconds": config["max_model_seconds"],
            "new_bodies": 120,
            "source_design_function_calibrations": 4,
            "new_lexical_neural_or_teacher_fits": 0,
            "current_model_weights_hashed": True,
            "historical_HM79_weights_not_separately_hashed": True,
            "source_before_new_readings": True,
            "original_scope_complete": False,
        },
    )
    mutable_write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "calls": 0,
            "attempted": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "HM83 source-frozen:4 functions;120 unseen bodies;720 reads maximum", flush=True
    )


def check(out):
    """Verify bound inputs and code without inspecting application target outcomes."""
    manifest = load(out / "00/manifest.json")
    for name, expected in manifest["code"].items():
        if sha(ROOT / name) != expected or sha(out / "00/code" / name) != expected:
            raise ValueError("source-code drift: " + name)
    for field, base in (("inputs", ROOT), ("assets", out)):
        for name, expected in manifest[field].items():
            if sha(base / name) != expected:
                raise ValueError("bound input drift: " + name)
    return manifest


def collect(out):
    """Run finite actual reads once; uncertain calls never silently restart."""
    manifest = check(out)
    status = load(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] != "PREPARED" or status["in_flight"] is not None:
        raise RuntimeError("inspect incomplete collection; do not retry automatically")
    start = time.monotonic()
    try:
        observer = load_observer()
        old, actual = load(HM79 / "01/observer.json"), observer.manifest()
        for key in (
            "model",
            "revision",
            "dtype",
            "device",
            "layers",
            "hidden_size",
            "max_tokens",
            "attention",
            "use_cache",
            "training",
            "template_sha256",
        ):
            if actual.get(key) != old.get(key):
                raise ValueError("observer drift: " + key)
        write(out / "01/observer.json", actual)
        for row in load(out / "00/design.json"):
            if (
                status["attempted"] >= manifest["max_forwards"]
                or time.monotonic() - start >= manifest["max_model_seconds"]
            ):
                raise TimeoutError("HM83 finite model budget exhausted")
            status.update(
                status="COLLECTING",
                in_flight=row["unit"],
                attempted=status["attempted"] + 1,
            )
            mutable_write(out / "run_status.json", status)
            native, _ = observe_full(observer, row, manifest["alphabet"], 16)
            write(
                out / f"01/units/unit_{row['unit']:04d}.json",
                {k: row[k] for k in ("unit", "event_id", "query")} | {"native": native},
            )
            status.update(
                calls=status["calls"] + 1,
                in_flight=None,
                model_seconds=time.monotonic() - start,
            )
            mutable_write(out / "run_status.json", status)
            if status["calls"] % 120 == 0:
                print(
                    f"HM83 {status['calls']}/720; {status['model_seconds']:.1f}s",
                    flush=True,
                )
        if status["calls"] != 720:
            raise ValueError("incomplete actual reads")
        status["status"] = "COLLECTED"
        paths = sorted((out / "01/units").glob("unit_*.json"))
        write(out / "01/seal.json", {"files": seal(out, paths)})
    except Exception as error:
        status.update(
            status="STOPPED", error=repr(error), model_seconds=time.monotonic() - start
        )
        raise
    finally:
        mutable_write(out / "run_status.json", status)


def cold(out, expected_sha):
    """Run new-event functions with exactly four files and no target truth access."""
    paths = [
        out / n
        for n in (
            "00/bundle.json",
            "03/observations.jsonl",
            "03/scores.jsonl",
            "03/cold_audit.json",
        )
    ]
    allowed, opened = {os.path.realpath(p) for p in paths}, set()

    def guard(event, values):
        if event == "open" and not isinstance(values[0], int):
            path = os.path.realpath(os.fsdecode(values[0]))
            opened.add(path)
            if path not in allowed and not path.startswith("/dev/"):
                raise PermissionError("cold process refused " + path)

    sys.addaudithook(guard)
    if sha(paths[0]) != expected_sha:
        raise ValueError("compiled bundle authentication failed")
    bundle = load(paths[0])
    with paths[2].open("x") as stream:
        seen = set()
        for line in paths[1].read_text().splitlines():
            record = json.loads(line)
            if record["event_id"] in seen:
                raise ValueError("duplicate cold event identity")
            seen.add(record["event_id"])
            stream.write(
                json.dumps(M.cold_score(bundle, record), allow_nan=False) + "\n"
            )
    write(
        paths[3],
        {
            "allowed": sorted(allowed),
            "opened": sorted(opened | {str(paths[3].resolve())}),
            "source_or_application_truth_access": False,
        },
    )


def analyze(out):
    """Keep all frozen-arm signed actions on new bodies; no application selection."""
    manifest = check(out)
    if load(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("actual observations incomplete")
    for name, expected in load(out / "01/seal.json")["files"].items():
        if sha(out / name) != expected:
            raise ValueError("raw read seal drift")
    dest = out / "03"
    dest.mkdir(exist_ok=False)
    bundle = load(out / "00/bundle.json")
    events = load(out / "00/application.json")
    grain = L.score_documents(
        bundle["observation"]["lexical"], [e["body"] for e in events]
    )["z"]
    query = np.empty((120, 6))
    decoder_errors = []
    records = []
    for i, event in enumerate(events):
        for j, target in enumerate(J.TARGETS):
            meta = load(out / f"01/units/unit_{6 * i + j:04d}.json")
            if (meta["unit"], meta["event_id"], meta["query"]) != (
                6 * i + j,
                event["event_id"],
                target,
            ):
                raise ValueError("actual query/event alias mismatch")
            native = meta["native"]
            logits = np.asarray(native["candidate_logits"])
            prob = np.exp(logits - logits.max())
            prob /= prob.sum()
            score = prob @ manifest["alphabet"]["values"]
            decoder_errors.append(abs(score - native["conditional_score"]))
            query[i, j] = native["conditional_score"]
        records.append(
            {
                "event_id": event["event_id"],
                "observer_id": bundle["observation"]["observer_id"],
                "grain": grain[i].tolist(),
                "query": query[i].tolist(),
                "condition": event["c"],
            }
        )
    records.append(records[0] | {"event_id": "unknown_c_control", "condition": None})
    with (dest / "observations.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_grain_condition_order_hm83.py"),
            "cold",
            "--out",
            str(out),
            "--sha256",
            sha(out / "00/bundle.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    write(
        dest / "cold_process.json",
        {
            "exit_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        },
    )
    scores = [
        json.loads(line) for line in (dest / "scores.jsonl").read_text().splitlines()
    ]
    truth = np.array(
        [[e["c"], e["x"], e["x"] * e["c"], -e["x"] * e["c"]] for e in events]
    )
    np.savez_compressed(
        dest / "terminal_arrays.npz", grain=grain, query=query, truth=truth
    )
    summary_rows, action_rows = [], []
    for arm in M.ARMS:
        values = np.array([s["arms"][arm]["scores"] for s in scores[:120]])
        for j, target in enumerate(M.TARGETS):
            summary_rows.append(
                {
                    "arm": arm,
                    "target": target,
                    "new_text_RMSE_design_only": float(
                        np.sqrt(np.mean((values[:, j] - truth[:, j]) ** 2))
                    ),
                    "signed_target_action": float(truth[:, j] @ values[:, j] / 120),
                    "new_text_score_mean": float(values[:, j].mean()),
                    "new_text_score_RMS": float(np.sqrt(np.mean(values[:, j] ** 2))),
                }
            )
        for k in range(30):
            local = values[k * 4 : (k + 1) * 4]
            design = truth[k * 4 : (k + 1) * 4, :3]
            for j, target in enumerate(M.TARGETS):
                coeff = design.T @ local[:, j] / 4
                action_rows.append(
                    {
                        "arm": arm,
                        "focal": events[k * 4]["focal"],
                        "companion": events[k * 4]["companion"],
                        "target_function": target,
                        "x_action": float(coeff[1]),
                        "c_action": float(coeff[0]),
                        "cx_action": float(coeff[2]),
                        "offset": float(local[:, j].mean()),
                    }
                )
    if max(decoder_errors) > 1e-10:
        raise ArithmeticError("candidate decoding mismatch")
    table(dest / "new_text_function_summary.csv", summary_rows)
    table(dest / "new_text_signed_actions.csv", action_rows)
    summary = {
        "status": "FINITE_HM83_EXECUTED_AWAITING_ALTERNATE_VERIFICATION",
        "new_bodies": 120,
        "actual_reads": 720,
        "source_calibrations": 4,
        "new_lexical_neural_teacher_fits": 0,
        "unknown_condition_control": scores[-1],
        "decoder_max_error": max(decoder_errors),
        "not_human_personality_accuracy": True,
        "not_LLM_free_observation": True,
        "original_scope_complete": False,
        "no_application_repair_or_winner_selection": True,
    }
    write(dest / "summary.json", summary)
    write(
        dest / "seal.json",
        {
            "files": seal(
                out,
                [
                    p
                    for p in dest.iterdir()
                    if p.is_file()
                    and not p.name.startswith("._")
                    and p.name != "seal.json"
                ],
            )
        },
    )
    print(json.dumps(summary), flush=True)


def main():
    """Explicit finite stages; never continue unrelated queues or tune outcomes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "cold", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == "cold":
            cold(args.out, args.sha256)
        elif args.stage == "collect":
            with collector_guard(args.out):
                collect(args.out)
        else:
            globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
