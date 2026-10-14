#!/usr/bin/env python3
"""HM68: fixed observer, full ownership-by-cross-target composition experiment."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_ownership_composition as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_evidence_compatibility import bound_check as prior_check
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/ownership_composition_r1"
PRIOR = ROOT / "results/grain_math_response_v1/contextual_evidence_compatibility_r1"
MATERIAL = ROOT / "configs/grain_math_passive_event_composition_v1.json"
REFERENCE = ROOT / "results/grain_math_response_v1/heterogeneous_evidence_cube_r1"
MAX_CALLS, MAX_SECONDS = 3456, 1500
SEEDS = (20261009, 20261010, 20261011)
IDENTITY = ("unit", "doc_id", "family", "focal", "companion", "corner", "query")
CODE = (
    "docs/GRAIN_OWNERSHIP_GATED_COMPOSITION_PLAN_V1.md",
    "grain_research/construction/math_ownership_composition.py",
    "scripts/run_grain_math_ownership_composition.py",
    "tests/test_grain_math_ownership_composition.py",
    "grain_research/construction/math_heterogeneous_evidence_cube.py",
    "grain_research/construction/math_answer_recoding_mechanism.py",
    "grain_research/construction/math_passive_event_composition.py",
    "grain_research/construction/math_construct_routing.py",
    "grain_research/construction/math_relational_operation_specificity.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_math_relational_cut.py",
    "scripts/run_grain_math_construct_routing.py",
    "scripts/run_grain_math_evaluative_routing.py",
    "scripts/run_grain_math_response.py",
    "scripts/run_grain_upstream_routes.py",
)


def prepare(out):
    """Freeze all bodies, actual token swaps, dependencies and observation protocol."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing packet;no overwrite")
    prior_check(PRIOR)
    if (
        read_json(PRIOR / "verification.json")["status"]
        != "VERIFIED_FINITE_HM67_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM67 required")
    material = read_json(MATERIAL)
    alphabet = read_json(REFERENCE / "00/manifest.json")["alphabet"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    if [tokenizer.encode(s, add_special_tokens=False) for s in alphabet["labels"]] != [
        [t] for t in alphabet["token_ids"]
    ]:
        raise ValueError("Frozen candidate alphabet drift")
    rows, paired, proofs = [], {}, []
    for doc in M.documents(material):
        prefix = None
        for query in M.TARGETS:
            text, ids, part, _, _ = render(
                doc["body"], query, "A", 1, material["system"], tokenizer
            )
            current = ids[: part["body"][-1] + 1]
            if prefix is not None and prefix != current:
                raise ValueError("Evidence prefix depends on future query")
            prefix = current
            row = {
                **doc,
                "unit": len(rows),
                "query": query,
                "input_ids": ids,
                "rendered": text,
                "partition": part,
            }
            rows.append(row)
            key = (
                doc["family"],
                doc["focal"],
                doc["companion"],
                *doc["coordinates"][:2],
                query,
            )
            paired.setdefault(key, []).append(row)
    for key, pair in paired.items():
        a, b = pair
        ia, ib = a["input_ids"], b["input_ids"]
        differences = [i for i, (x, y) in enumerate(zip(ia, ib, strict=True)) if x != y]
        if Counter(ia) != Counter(ib) or len(differences) != 2:
            raise ValueError("Ownership pair is not exactly two swapped tokens")
        if any(i not in a["partition"]["body"] for i in differences):
            raise ValueError("Ownership swap outside body")
        proofs.append(
            {
                "key": list(key),
                "units": [a["unit"], b["unit"]],
                "swapped_positions": differences,
                "same_token_bag": True,
            }
        )
    if len(rows) != MAX_CALLS or len({r["body"] for r in rows}) != 576:
        raise ValueError("Incomplete declared material")
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/swap_proofs.json", proofs)
    code = {}
    for name in CODE:
        destination = out / "00/code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        code[name] = digest(destination)
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            MATERIAL,
            PRIOR / "manifest.json",
            PRIOR / "verification.json",
            REFERENCE / "00/manifest.json",
            REFERENCE / "01/observer.json",
        )
    }
    tokenizer_files = {p.name: digest(p) for p in MODEL.glob("*token*") if p.is_file()}
    write(
        out / "00/manifest.json",
        {
            "packet": "HM68",
            "code": code,
            "source_files": sources,
            "assets": {
                n: digest(out / n) for n in ("00/rows.json", "00/swap_proofs.json")
            },
            "alphabet": alphabet,
            "targets": M.TARGETS,
            "stages": M.STAGES,
            "worlds": M.WORLDS,
            "seeds": SEEDS,
            "layer": 16,
            "max_forwards": MAX_CALLS,
            "max_model_seconds": MAX_SECONDS,
            "model_config_sha256": digest(MODEL / "config.json"),
            "tokenizer_files": tokenizer_files,
            "model_weights_separately_hashed": False,
            "no_training_fitting_generation_API_private_labels": True,
            "new_bodies": 576,
            "directed_pairs_per_family": 30,
            "neutral_controls_per_family": 6,
            "human_units": 0,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "original_scope_complete": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "HM68 PREPARED:576 bodies/3456 fixed readings;1728 exact token swaps",
        flush=True,
    )


def check(out):
    """Reject any material/code/dependency/observer drift before collection or analysis."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift:" + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen material drift:" + name)
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source drift:" + name)
    for name, sha in m["tokenizer_files"].items():
        if digest(MODEL / name) != sha:
            raise ValueError("Tokenizer drift:" + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Model config drift")
    return m


def collect(out):
    """Acquire frozen readings once; persist actual attempts and every raw capture."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError(
            "Stopped/uncertain attempts require inspection;no automatic retry"
        )
    if len(list((out / "01/units").glob("unit_*.json"))) != status["completed_units"]:
        raise RuntimeError("Capture/status disagreement")
    rows = read_json(out / "00/rows.json")
    start, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual, old = observer.manifest(), read_json(REFERENCE / "01/observer.json")
        for key in (
            "model",
            "torch",
            "transformers",
            "dtype",
            "device",
            "hidden_size",
            "layers",
            "max_tokens",
            "attention",
            "training",
            "template_sha256",
        ):
            if actual[key] != old[key]:
                raise ValueError("Observer drift:" + key)
        write(out / "01/observer.json", actual)
        for row in rows:
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                meta, _ = load_arrays(path)
                if any(meta[k] != row[k] for k in IDENTITY):
                    raise ValueError("Resume identity mismatch")
                continue
            elapsed = previous + time.monotonic() - start
            if status["attempted_forward_calls"] >= MAX_CALLS or elapsed >= MAX_SECONDS:
                raise TimeoutError("Declared HM68 collector bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(observer, row, m["alphabet"], m["layer"])
            save(
                path,
                {k: row[k] for k in IDENTITY} | {"native": native},
                {
                    "body_end": arrays["full"][row["partition"]["body"][-1]],
                    "query_end": arrays["full"][-1],
                },
            )
            status["forward_calls"] += 1
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - start,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 144 == 0:
                print(
                    f"HM68 ACTUAL:{status['completed_units']}/3456;"
                    f"{status['model_seconds']:.1f}s",
                    flush=True,
                )
        if any(
            status[k] != MAX_CALLS
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        ):
            raise ValueError("Incomplete collection")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
        raw_files = [out / "run_status.json", out / "01/observer.json"]
        raw_files.extend(sorted((out / "01/units").glob("unit_*.*")))
        write(
            out / "01/raw_seal.json",
            {str(p.relative_to(out)): digest(p) for p in raw_files},
        )
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - start,
        )
        write(out / "run_status.json", status)
        raise


def raw_tables(out):
    """Rebuild every complete factorial stage from hash-bound raw observations."""
    for name, sha in read_json(out / "01/raw_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Raw capture drift:" + name)
    tables, seen, bodies = {}, set(), {}
    prefix_error = 0.0
    for row in read_json(out / "00/rows.json"):
        meta, a = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        if any(meta[k] != row[k] for k in IDENTITY):
            raise ValueError("Capture identity mismatch")
        key = (row["family"], row["focal"], row["companion"])
        group = tables.setdefault(
            key, {"logits": np.zeros((8, 6, 10)), "body_endpoint": np.zeros((8, 1536))}
        )
        q, c = M.TARGETS.index(row["query"]), row["corner"]
        if (key, c, q) in seen:
            raise ValueError("Duplicate factorial cell")
        seen.add((key, c, q))
        z = np.asarray(meta["native"]["candidate_logits"], float)
        if abs(M.decode(z)["score"] - meta["native"]["conditional_score"]) > 1e-10:
            raise ValueError("Candidate score drift")
        group["logits"][c, q] = z
        h = a["body_end"].astype(float)
        bkey = (key, c)
        if bkey in bodies:
            prefix_error = max(prefix_error, float(np.max(np.abs(h - bodies[bkey]))))
        else:
            bodies[bkey] = h
            group["body_endpoint"][c] = h
    if len(seen) != MAX_CALLS or len(tables) != 72 or prefix_error > 1e-5:
        raise ValueError("Missing complete design or question-dependent body capture")
    for group in tables.values():
        z = group.pop("logits")
        centered = z - z.mean(-1, keepdims=True)
        _, control = M.link_controls(centered)
        group.update(
            centered_logits=centered,
            **M.decode(z),
            **{u + "_link_control": v for u, v in control.items()},
        )
    return tables, prefix_error


def key_name(key, stage):
    """Address a stage by immutable lexical family and directed behaviour pair."""
    return f"f{key[0]}_{key[1]}_{key[2]}_{stage}"


def analyze(out):
    """Describe group-projected mixed actions without selecting a predictor or target."""
    check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Complete collection required")
    if (out / "03").exists():
        raise FileExistsError("Preserve existing scientific analysis")
    tables, prefix = raw_tables(out)
    (out / "03").mkdir()
    arrays, effects, interactions, actions = {}, {}, [], []
    for key, group in tables.items():
        for stage in M.STAGES:
            arrays[key_name(key, stage)] = group[stage]
            effects[key, stage] = M.group_effects(group[stage])
    for (key, stage), e in effects.items():
        f, j, k = key
        control = effects[(f, j, "neutral"), stage]["gate"]
        row = {
            "family": f,
            "focal": j,
            "companion": k,
            "stage": stage,
            "own_norm": float(np.linalg.norm(e["own"])),
            "other_norm": float(np.linalg.norm(e["other"])),
            "gate_norm": float(np.linalg.norm(e["gate"])),
            "neutral_gate_norm": float(np.linalg.norm(control)),
            "neutral_adjusted_norm": float(np.linalg.norm(e["gate"] - control)),
            "mixed_identity_error": e["identity_error"],
        }
        interactions.append(row)
        if tables[key][stage].shape == (8, 6):
            for q, name in enumerate(M.TARGETS):
                dx = e["focal_edges"][:, :, q]
                actions.append(
                    {
                        "family": f,
                        "focal": j,
                        "companion": k,
                        "stage": stage,
                        "query": name,
                        "own_focal_low_companion": float(dx[0, 1]),
                        "own_focal_high_companion": float(dx[1, 1]),
                        "other_focal_low_companion": float(dx[0, 0]),
                        "other_focal_high_companion": float(dx[1, 0]),
                        "own_mixed": float(e["own"][q]),
                        "other_mixed": float(e["other"][q]),
                        "gate": float(e["gate"][q]),
                        "neutral_gate": float(control[q]),
                        "neutral_adjusted": float(e["gate"][q] - control[q]),
                    }
                )
    summaries = []
    for family, stage in itertools.product((0, 1), M.STAGES):
        keys = [k for k in tables if k[0] == family and k[2] != "neutral"]
        e = [effects[k, stage] for k in keys]
        g = np.asarray([r["gate"] for r in e])
        neutral = np.asarray(
            [effects[(family, k[1], "neutral"), stage]["gate"] for k in keys]
        )
        summary = {"family": family, "stage": stage, "directed_pairs": len(keys)}
        for label, values in (
            ("own", [r["own"] for r in e]),
            ("other", [r["other"] for r in e]),
            ("gate", g),
            ("neutral", neutral),
            ("neutral_adjusted", g - neutral),
        ):
            summary[label + "_rms"] = float(np.sqrt(np.mean(np.square(values))))
        if tables[keys[0]][stage].shape == (8, 6):
            for label in ("own", "other", "gate"):
                summary[label + "_companion_query_mean"] = float(
                    np.mean(
                        [
                            r[label][M.TARGETS.index(k[2])]
                            for k, r in zip(keys, e, strict=True)
                        ]
                    )
                )
                summary[label + "_focal_query_mean"] = float(
                    np.mean(
                        [
                            r[label][M.TARGETS.index(k[1])]
                            for k, r in zip(keys, e, strict=True)
                        ]
                    )
                )
        summaries.append(summary)
    toy_rows = []
    for world, seed in itertools.product(M.WORLDS, SEEDS):
        w = M.toy_world(world, seed)
        for branch in ("main", "neutral"):
            z = w[branch + "_logits"]
            for stage, value in {"centered_logits": z, **M.decode(z)}.items():
                arrays[f"toy_{world}_{seed}_{branch}_{stage}"] = value
                e = M.group_effects(value)
                toy_rows.append(
                    {
                        "world": world,
                        "seed": seed,
                        "branch": branch,
                        "stage": stage,
                        "own_norm": float(np.linalg.norm(e["own"])),
                        "other_norm": float(np.linalg.norm(e["other"])),
                        "gate_norm": float(np.linalg.norm(e["gate"])),
                        "identity_error": e["identity_error"],
                    }
                )
    np.savez_compressed(out / "03/stages_and_worlds.npz", **arrays)
    write_csv(out / "03/interactions.csv", interactions)
    write_csv(out / "03/target_actions.csv", actions)
    write_csv(out / "03/stage_summary.csv", summaries)
    write_csv(out / "03/toy_mechanisms.csv", toy_rows)
    write(
        out / "03/summary.json",
        {
            "packet": "HM68",
            "body_prefix_max_error": prefix,
            "actual_new_bodies": 576,
            "actual_new_readings": MAX_CALLS,
            "interaction_stage_rows": len(interactions),
            "target_action_rows": len(actions),
            "toy_stage_rows": len(toy_rows),
            "stage_summary": summaries,
            "no_fitting_training_generation_API_private_labels": True,
            "template_readings_not_population_or_personality_accuracy": True,
            "original_scope_complete": False,
        },
    )
    write(
        out / "03/output_seal.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted((out / "03").iterdir())
            if p.is_file()
        },
    )
    print(json.dumps(summaries, indent=2), flush=True)


def verify(out):
    """Recompute observations and group contrasts using independent tensor arithmetic."""
    import csv

    check(out)
    for name, sha in read_json(out / "03/output_seal.json").items():
        if digest(out / name) != sha:
            raise ValueError("Scientific output drift:" + name)
    tables, prefix = raw_tables(out)
    with np.load(out / "03/stages_and_worlds.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    errors, comparisons = [], 0
    direct = {}
    for key, group in tables.items():
        for stage, value in group.items():
            stored = arrays[key_name(key, stage)]
            errors.append(float(np.max(np.abs(stored - value))))
            comparisons += value.size
            a = stored.reshape(2, 2, 2, *stored.shape[1:])
            other = a[1, 1, 0] - a[1, 0, 0] - a[0, 1, 0] + a[0, 0, 0]
            own = a[1, 1, 1] - a[1, 0, 1] - a[0, 1, 1] + a[0, 0, 1]
            signs = np.prod(M.grid(3), axis=1)
            c7 = np.asarray(
                [
                    math.fsum(
                        float(s) * float(v) for s, v in zip(signs, column, strict=True)
                    )
                    / 8
                    for column in stored.reshape(8, -1).T
                ]
            ).reshape(stored.shape[1:])
            errors.append(float(np.max(np.abs(own - other - 8 * c7))))
            comparisons += c7.size
            direct[key, stage] = {"own": own, "other": other, "gate": own - other}
    with (out / "03/interactions.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (int(row["family"]), row["focal"], row["companion"])
            stage = row["stage"]
            e, control = (
                direct[key, stage],
                direct[(key[0], key[1], "neutral"), stage]["gate"],
            )
            expected = {
                name + "_norm": np.linalg.norm(value) for name, value in e.items()
            }
            expected.update(
                neutral_gate_norm=np.linalg.norm(control),
                neutral_adjusted_norm=np.linalg.norm(e["gate"] - control),
            )
            for name, value in expected.items():
                errors.append(abs(float(row[name]) - float(value)))
                comparisons += 1
    for key, value in arrays.items():
        if "centered_logits" not in key:
            continue
        if key.startswith("toy_"):
            for unit in ("score", "log_odds"):
                expected = (
                    softmax(value, axis=-1) @ (np.arange(10) / 9)
                    if unit == "score"
                    else logsumexp(value[..., 5:], -1) - logsumexp(value[..., :5], -1)
                )
                stored = arrays[key.replace("centered_logits", unit)]
                errors.append(float(np.max(np.abs(expected - stored))))
                comparisons += stored.size
    maximum = max(errors)
    if maximum > 1e-9 or prefix > 1e-5:
        raise ValueError("Alternate arithmetic verification failed")
    report = {
        "status": "VERIFIED_FINITE_HM68_NOT_GLOBAL",
        "actual_comparisons": comparisons,
        "maximum_alternate_error": maximum,
        "body_prefix_max_error": prefix,
        "actual_calls": read_json(out / "run_status.json")["forward_calls"],
        "same_agent_not_independent_replication": True,
        "original_scope_complete": False,
    }
    write(out / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    """Expose bounded explicit stages; never launch old queues or automatic retries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "collect":
        with collector_guard(args.out):
            collect(args.out)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
