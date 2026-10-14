#!/usr/bin/env python3
"""HM51: finite source-only full-query relation transport, with no accuracy search."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_source_relational_transport as M
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_carrier import summarize
from scripts.run_grain_math_relational_cut import CODE as PARENT_CODE
from scripts.run_grain_math_relational_cut import check as parent_check
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_source_relational_transport_v1.json"
OUT = ROOT / "results/grain_math_response_v1/source_relational_transport_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_source_relational_transport_v1.json",
            "docs/GRAIN_SOURCE_RELATIONAL_TRANSPORT_PLAN_V1.md",
            "grain_research/construction/math_source_relational_transport.py",
            "scripts/run_grain_math_source_relational_transport.py",
            "scripts/verify_grain_math_source_relational_transport.py",
            "tests/test_grain_math_source_relational_transport.py",
            *PARENT_CODE,
        )
    )
)


def prepare(out):
    """Seal24source captures, actor sectors and all168planned application calls."""
    if out.exists():
        raise FileExistsError("Preserve existing result versions")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    prior = parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM50_NOT_GLOBAL"
        or spec["layer"] != prior["spec"]["layer"]
        or spec["targets"] != prior["spec"]["targets"]
        or spec["arms"] != list(M.ARMS)
    ):
        raise ValueError("Verified unchanged HM50 and complete planned scope required")
    old_design = read_json(parent / "00/design.json")
    captures = [e for e in old_design if e["kind"] == "capture"]
    index = {
        (e["role"], e["content"], e["person"], e["direction"]): e for e in captures
    }
    rows = {r["unit"]: r for r in read_json(parent / "00/receivers.json")}
    states, metas, hashes = {}, {}, {}
    for e in captures:
        path = parent / f"01/units/unit_{e['unit']:04d}.json"
        metas[e["receiver"]], arrays = load_arrays(path)
        states[e["receiver"]] = arrays["full"]
        for p in (path, path.with_suffix(".npz")):
            hashes[str(p.relative_to(parent))] = digest(p)
    source, application = (
        [e for e in captures if e["role"] == "source"],
        [e for e in captures if e["role"] == "application"],
    )
    if (
        len(index) != 48
        or len(source) != spec["expected_source_captures"]
        or len(application) != spec["expected_application_receivers"]
    ):
        raise ValueError("Require all24source and24application receiver captures")
    assets, maps, actors = {}, {}, {}
    for target in spec["targets"]:
        deltas = {}
        for person in ("A", "B"):
            zero, one = [index["source", target, person, d] for d in (0, 1)]
            p = zero["partition"]["suffix"]
            if M.suffix_ids(rows[zero["receiver"]], zero) != M.suffix_ids(
                rows[one["receiver"]], one
            ):
                raise ValueError("Source ownership exchange changes query IDs")
            deltas[person] = states[one["receiver"]][p].astype(float) - states[
                zero["receiver"]
            ][p].astype(float)
        sectors = M.source_sectors(deltas["A"], deltas["B"])
        assets.update({f"{target}__{k}": v for k, v in sectors.items()})
        a, b = [index["source", target, p, 0] for p in ("A", "B")]
        actors[target] = M.actor_slot_proof(
            M.suffix_ids(rows[a["receiver"]], a), M.suffix_ids(rows[b["receiver"]], b)
        )
    design = []
    for app in application:
        src = index["source", app["content"], app["person"], app["direction"]]
        maps[str(app["pair"])] = M.prove_transport(
            rows[src["receiver"]], src, rows[app["receiver"]], app
        )
        exact = next(
            e for e in old_design if e["pair"] == app["pair"] and e["arm"] == "suffix"
        )
        path = parent / f"01/units/unit_{exact['unit']:04d}.json"
        for p in (path, path.with_suffix(".npz")):
            hashes[str(p.relative_to(parent))] = digest(p)
        for arm in spec["arms"]:
            design.append(
                {
                    **app,
                    "kind": "patch",
                    "arm": arm,
                    "unit": len(design),
                    "parent_capture": app["unit"],
                    "parent_exact_suffix": exact["unit"],
                }
            )
    if len(design) != spec["max_forwards"]:
        raise ValueError("Incomplete168call design")
    for name in ("00/code", "01/units", "03"):
        (out / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "00/source_vectors.npz", **assets)
    write(out / "00/design.json", design)
    write(out / "00/mappings.json", {"same_person": maps, "actor_slot": actors})
    write(out / "00/receivers.json", list(rows.values()))
    write(
        out / "00/source_seal.json",
        {
            "reused_source_captures": 24,
            "new_source_forwards": 0,
            "before_all_new_application_calls": True,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_capture_units": [e["unit"] for e in source],
        },
    )
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM51",
            "spec": spec,
            "alphabet": prior["alphabet"],
            "code": code,
            "assets": {
                n: digest(out / n)
                for n in (
                    "00/design.json",
                    "00/mappings.json",
                    "00/receivers.json",
                    "00/source_vectors.npz",
                    "00/source_seal.json",
                )
            },
            "parent_files": {
                n: digest(parent / n)
                for n in (
                    "00/manifest.json",
                    "00/design.json",
                    "00/receivers.json",
                    "verification.json",
                    "01/observer.json",
                )
            },
            "parent_actual_files": hashes,
            "model_config_sha256": digest(MODEL / "config.json"),
            "known_material_followup": True,
            "model_weights_not_separately_hashed": True,
            "human_truth": False,
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
        "PREPARED:24 source captures reused,0 newsourcecalls;168 fixed app patches",
        flush=True,
    )


def check(out):
    """Check unchanged frozen programs, source-only assets, actual references and model."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Changed frozen code " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed source/design asset " + name)
    parent = ROOT / m["spec"]["parent"]
    for name, sha in {**m["parent_files"], **m["parent_actual_files"]}.items():
        if digest(parent / name) != sha:
            raise ValueError("Changed actual HM50 reference " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Local observer config drift")
    return m


def references(m, out):
    """Load only the specified hash-bound source and receiver captures, not private data."""
    parent = ROOT / m["spec"]["parent"]
    caps = [e for e in read_json(parent / "00/design.json") if e["kind"] == "capture"]
    states, scores = {}, {}
    for e in caps:
        meta, a = load_arrays(parent / f"01/units/unit_{e['unit']:04d}.json")
        states[e["receiver"]], scores[e["receiver"]] = (
            a["full"],
            meta["native"]["conditional_score"],
        )
    with np.load(out / "00/source_vectors.npz", allow_pickle=False) as archive:
        vectors = {k: archive[k].copy() for k in archive.files}
    return states, scores, vectors


def collect(out):
    """Run168 application interventions, never repeating uncertain or stopped attempts."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError(
            "Stopped/uncertain attempt; inspect,do not automatically retry"
        )
    states, _, vectors = references(m, out)
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    design, spec = read_json(out / "00/design.json"), m["spec"]
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual, prior = (
            observer.manifest(),
            read_json(ROOT / spec["parent"] / "01/observer.json"),
        )
        for k in (
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
            if actual[k] != prior[k]:
                raise ValueError("Observer drift " + k)
        write(out / "01/observer.json", actual)
        for e in design:
            path = out / f"01/units/unit_{e['unit']:04d}.json"
            if path.exists():
                load_arrays(path)
                continue
            elapsed = previous + time.monotonic() - started
            if not call_allowed(status, spec, elapsed):
                raise TimeoutError("168call/600second finite budget reached")
            sectors = {
                k: vectors[f"{e['content']}__{k}"]
                for k in ("A", "B", "shared", "bound")
            }
            patch = M.patch_values(
                e,
                states[e["receiver"]],
                states[e["donor"]],
                sectors,
                spec["random_seed"],
            )
            status.update(
                status="COLLECTING", in_flight=e["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(
                observer, rows[e["receiver"]], m["alphabet"], spec["layer"], patch
            )
            status["forward_calls"] += 1
            arrays.update(
                patch_positions=np.asarray(patch[0], int), patch_values=patch[1]
            )
            save(
                path,
                {
                    k: e[k]
                    for k in (
                        "unit",
                        "pair",
                        "role",
                        "kind",
                        "arm",
                        "receiver",
                        "donor",
                    )
                }
                | {"native": native},
                arrays,
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 24 == 0:
                print(
                    f"application:{status['completed_units']}/168 actual forwards",
                    flush=True,
                )
        if (
            any(
                status[k] != spec["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > spec["max_model_seconds"]
        ):
            raise ValueError("Incomplete or over-budget actual acquisition")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Retain all signed target/person effects and source-sector output interactions."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Actual complete acquisition required")
    states, scores, vectors = references(m, out)
    design, by_pair, effects, geometry = (
        read_json(out / "00/design.json"),
        defaultdict(dict),
        [],
        [],
    )
    max_before, exact_error, noop_error = 0.0, 0.0, 0.0
    for e in design:
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        native, donor, y = (
            scores[e["receiver"]],
            scores[e["donor"]],
            meta["native"]["conditional_score"],
        )
        by_pair[e["pair"]][e["arm"]] = y
        max_before = max(
            max_before,
            float(
                np.max(
                    abs(a["before"].astype(float) - states[e["receiver"]].astype(float))
                )
            ),
        )
        if e["arm"] == "exact_suffix":
            old = read_json(
                ROOT
                / m["spec"]["parent"]
                / f"01/units/unit_{e['parent_exact_suffix']:04d}.json"
            )["native"]["conditional_score"]
            exact_error = max(exact_error, abs(y - old))
        if e["arm"] == "noop":
            noop_error = max(noop_error, abs(y - native))
        effects.append(
            {k: e[k] for k in ("unit", "pair", "content", "person", "direction", "arm")}
            | {
                "native": native,
                "donor": donor,
                "patched": y,
                "candidate_mass": meta["native"]["candidate_mass"],
                "full_change": donor - native,
                "patch_change": y - native,
                "remaining": donor - y,
            }
        )
    for target in m["spec"]["targets"]:
        x = {k: vectors[f"{target}__{k}"] for k in ("A", "B", "shared", "bound")}
        total = float(np.sum(x["A"] ** 2) + np.sum(x["B"] ** 2))
        geometry.append(
            {
                "content": target,
                "full_square": total,
                "shared_square": float(2 * np.sum(x["shared"] ** 2)),
                "bound_square": float(2 * np.sum(x["bound"] ** 2)),
                "person_difference_cosine": float(
                    np.sum(x["A"] * x["B"])
                    / (np.linalg.norm(x["A"]) * np.linalg.norm(x["B"]))
                ),
            }
        )
    cells = []
    for pair, v in by_pair.items():
        e = next(e for e in design if e["pair"] == pair)
        cells.append(
            {k: e[k] for k in ("pair", "content", "person", "direction")}
            | M.four_cells(
                scores[e["receiver"]],
                v["source_shared"],
                v["source_bound"],
                v["source_full"],
            )
        )
    groups, targets, people = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in effects:
        groups[r["arm"]].append(r)
        targets[r["content"], r["arm"]].append(r)
        people[r["person"], r["arm"]].append(r)
    a = np.asarray(
        [
            [
                r[k]
                for k in (
                    "shared_action",
                    "bound_action",
                    "interaction",
                    "joint_action",
                )
            ]
            for r in cells
        ]
    )
    energy = {
        "shared_square": float(np.mean(a[:, 0] ** 2)),
        "bound_square": float(np.mean(a[:, 1] ** 2)),
        "interaction_square": float(np.mean(a[:, 2] ** 2)),
        "shared_bound_cross": float(2 * np.mean(a[:, 0] * a[:, 1])),
        "shared_interaction_cross": float(2 * np.mean(a[:, 0] * a[:, 2])),
        "bound_interaction_cross": float(2 * np.mean(a[:, 1] * a[:, 2])),
    }
    energy["reconstructed_square"] = sum(energy.values())
    energy["joint_square"] = float(np.mean(a[:, 3] ** 2))
    for name, data in (
        ("effects.csv", effects),
        ("four_cells.csv", cells),
        ("source_geometry.csv", geometry),
        ("signed_energy.csv", [energy]),
        ("arm_summary.csv", [{"arm": k, **summarize(v)} for k, v in groups.items()]),
        (
            "content_summary.csv",
            [
                {"content": k[0], "arm": k[1], **summarize(v)}
                for k, v in targets.items()
            ],
        ),
        (
            "person_summary.csv",
            [{"person": k[0], "arm": k[1], **summarize(v)} for k, v in people.items()],
        ),
    ):
        write_csv(out / "03" / name, data)
    write(
        out / "03/summary.json",
        {
            "packet": "HM51",
            "effects": len(effects),
            "four_cells": len(cells),
            "groups": {k: summarize(v) for k, v in groups.items()},
            "interaction_rms": float(np.sqrt(np.mean(a[:, 2] ** 2))),
            "max_fresh_prepatch_difference": max_before,
            "exact_suffix_replay_max_error": exact_error,
            "noop_max_error": noop_error,
            "new_source_calls": 0,
            "known_material_followup_not_blind_replication": True,
            "human_personality_accuracy": False,
            "original_scope_complete": False,
        },
    )
    paths = (
        [out / "01/observer.json"]
        + [
            p
            for e in design
            for p in (
                out / f"01/units/unit_{e['unit']:04d}.json",
                out / f"01/units/unit_{e['unit']:04d}.npz",
            )
        ]
        + list((out / "03").glob("[!._]*.csv"))
        + [out / "03/summary.json"]
    )
    write(
        out / "03/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    print("ANALYZED:168signed actions/24shared-bound interactions", flush=True)


def main():
    """Expose bounded explicit stages without implicit old acquisition restarts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.output):
        {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](
            args.output
        )


if __name__ == "__main__":
    main()
