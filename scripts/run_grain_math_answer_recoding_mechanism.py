#!/usr/bin/env python3
"""HM53: source/receiver-code crossing with a source-first acquisition barrier."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_answer_recoding_mechanism as M
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.construction.math_relational_operation_specificity import (
    orientation,
)
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_relational_operation_specificity import CODE as PARENT_CODE
from scripts.run_grain_math_relational_operation_specificity import (
    check as parent_check,
)
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_answer_recoding_mechanism_v1.json"
OUT = ROOT / "results/grain_math_response_v1/answer_recoding_mechanism_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_answer_recoding_mechanism_v1.json",
            "docs/GRAIN_ANSWER_RECODING_MECHANISM_PLAN_V1.md",
            "grain_research/construction/math_answer_recoding_mechanism.py",
            "scripts/run_grain_math_answer_recoding_mechanism.py",
            "scripts/verify_grain_math_answer_recoding_mechanism.py",
            "tests/test_grain_math_answer_recoding_mechanism.py",
            *PARENT_CODE,
        )
    )
)
IDENTITY = (
    "unit",
    "kind",
    "receiver",
    "donor",
    "arm",
    "role",
    "content",
    "class",
    "person",
    "direction",
    "codec",
)


def prepare(out):
    """Freeze all inputs/mappings and hashes; do not load application score/state payloads."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing experiment versions")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    prior = parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM52_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM52 required")
    if (
        spec["targets"] != prior["spec"]["targets"]
        or spec["layer"] != prior["spec"]["layer"]
        or spec["arms"] != list(M.ARMS)
        or spec["classes"] != ["own", "roster"]
        or spec["codecs"] != [1, -1]
    ):
        raise ValueError("Full unchanged planned design required")
    hm51 = ROOT / prior["spec"]["parent"]
    hm50 = ROOT / read_json(hm51 / "00/manifest.json")["spec"]["parent"]
    historical = {r["unit"]: r for r in read_json(hm50 / "00/receivers.json")}
    source_entries = [
        e
        for e in read_json(hm50 / "00/design.json")
        if e["kind"] == "capture" and e["role"] == "source"
    ]
    application = [
        r
        for r in read_json(parent / "00/receivers.json")
        if r["class"] in spec["classes"]
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows, reused = [], {}
    for role, originals, base in (
        ("source", source_entries, hm50),
        ("application", application, parent),
    ):
        for original in originals:
            old = historical[original["receiver"]] if role == "source" else original
            standard = None
            for codec in spec["codecs"]:
                rendered, ids, part, cues, actor = M.render(
                    old["body"],
                    old["content"],
                    old["person"],
                    codec,
                    prior["spec"]["system"],
                    tokenizer,
                )
                row = {
                    "unit": len(rows),
                    "role": role,
                    "class": "source" if role == "source" else old["class"],
                    "content": old["content"],
                    "query": old["content"],
                    "person": old["person"],
                    "direction": original["direction"],
                    "codec": codec,
                    "body": old["body"],
                    "rendered": rendered,
                    "input_ids": ids,
                    "partition": part,
                    "cue_slots": cues,
                    "actor_slot": actor,
                    "standard_reference_uid": original["unit"],
                    "reused": codec == 1,
                }
                if codec == 1:
                    if ids != old["input_ids"] or rendered != old["rendered"]:
                        raise ValueError(
                            "Standard recoding must exactly reuse the original prompt"
                        )
                    path = base / f"01/units/unit_{original['unit']:04d}.json"
                    row["reused_path"] = str(path.relative_to(ROOT))
                    for p in (path, path.with_suffix(".npz")):
                        reused[str(p.relative_to(ROOT))] = digest(p)
                    standard = row
                else:
                    row["code_proof"] = M.code_pair(standard, row)
                rows.append(row)
    if (
        len(rows) != 144
        or sum(r["role"] == "source" for r in rows) != 48
        or len(reused) != 144
    ):
        raise ValueError("48 source/96 receiver rows and72 reused captures required")
    index = {
        (
            r["role"],
            r["content"],
            r["class"],
            r["person"],
            r["direction"],
            r["codec"],
        ): r
        for r in rows
    }
    design = []
    for r in rows:
        if not r["reused"]:
            r["capture_unit"] = len(design)
            design.append(
                {
                    "unit": len(design),
                    "kind": "capture",
                    "receiver": r["unit"],
                    "donor": r["unit"],
                    "arm": "native",
                    **{
                        k: r[k]
                        for k in (
                            "role",
                            "content",
                            "class",
                            "person",
                            "direction",
                            "codec",
                        )
                    },
                }
            )
    for r in rows:
        if r["role"] != "application":
            continue
        donor = index[
            "application",
            r["content"],
            r["class"],
            r["person"],
            1 - r["direction"],
            r["codec"],
        ]
        mapping = M.prove_mapping(donor, r)
        for arm in spec["arms"]:
            src_code = (
                1
                if arm == "source_forward"
                else -1
                if arm == "source_reverse"
                else r["codec"]
            )
            src_actor = (
                ("B" if r["person"] == "A" else "A")
                if arm == "wrong_person"
                else r["person"]
            )
            src = index["source", r["content"], "source", src_actor, 0, src_code]
            proof = M.prove_mapping(src, r, allow_actor=arm == "wrong_person")
            design.append(
                {
                    "unit": len(design),
                    "kind": "patch",
                    "receiver": r["unit"],
                    "donor": donor["unit"],
                    "arm": arm,
                    **{
                        k: r[k]
                        for k in (
                            "role",
                            "content",
                            "class",
                            "person",
                            "direction",
                            "codec",
                        )
                    },
                    "source_code": src_code,
                    "source_actor": src_actor,
                    "source_mapping": proof,
                    "donor_positions": mapping["source_positions"],
                }
            )
    if (
        len(design) != spec["max_forwards"]
        or [e["role"] for e in design[:24]] != ["source"] * 24
        or any(e["kind"] != "capture" for e in design[:72])
    ):
        raise ValueError("24source/48application/576patches in fixed order required")
    for name in ("00/code", "01/units", "03"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write(out / "00/rows.json", rows)
    write(out / "00/design.json", design)
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM53",
            "spec": spec,
            "alphabet": prior["alphabet"],
            "code": code,
            "assets": {n: digest(out / n) for n in ("00/rows.json", "00/design.json")},
            "parent_files": {
                n: digest(parent / n)
                for n in ("00/manifest.json", "verification.json", "01/observer.json")
            },
            "reused_files": reused,
            "model_config_sha256": digest(MODEL / "config.json"),
            "model_weights_not_separately_hashed": True,
            "known_material_recoding_not_independent_replication": True,
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
    print("PREPARED:144 rows/72 reused captures/648 new calls maximum", flush=True)


def check(out):
    """Check immutable code/material/reuse identities before any staged operation."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code changed " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen material changed " + name)
    parent_check(ROOT / m["spec"]["parent"])
    for name, sha in m["parent_files"].items():
        if digest(ROOT / m["spec"]["parent"] / name) != sha:
            raise ValueError("Parent changed " + name)
    for name, sha in m["reused_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Reused capture changed " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Observer config drift")
    return m


def capture_path(out, row):
    """Resolve provenance without copying historical results into new evidence."""
    return (
        ROOT / row["reused_path"]
        if row["reused"]
        else out / f"01/units/unit_{row['capture_unit']:04d}.json"
    )


def seal_source(out, rows, status):
    """Construct and seal source-only deltas before loading any application outputs."""
    if status["forward_calls"] != 24 or status["in_flight"] is not None:
        raise ValueError("Exactly24completed source calls must precede source sealing")
    source_rows = [r for r in rows.values() if r["role"] == "source"]
    index = {
        (r["content"], r["person"], r["direction"], r["codec"]): r for r in source_rows
    }
    arrays, hashes = {}, {}
    for r in source_rows:
        path = capture_path(out, r)
        _, a = load_arrays(path)
        arrays[r["unit"]] = a["full"]
        for p in (path, path.with_suffix(".npz")):
            hashes[str(p)] = digest(p)
    vectors = {}
    for target, person, direction, codec in index:
        if direction != 0:
            continue
        zero, one = index[target, person, 0, codec], index[target, person, 1, codec]
        M.prove_mapping(zero, one)
        vectors[M.source_key(target, person, codec)] = arrays[one["unit"]][
            one["partition"]["suffix"]
        ].astype(float) - arrays[zero["unit"]][zero["partition"]["suffix"]].astype(
            float
        )
    with (out / "01/source_vectors.npz").open("wb") as stream:
        np.savez_compressed(stream, **vectors)
    write(
        out / "01/source_seal.json",
        {
            "completed_new_source_calls": 24,
            "before_application_calls": True,
            "before_application_payload_loading": True,
            "created_epoch": time.time(),
            "source_capture_hashes": hashes,
            "source_vectors_sha256": digest(out / "01/source_vectors.npz"),
            "source_rows": 48,
            "source_operator_count": len(vectors),
            "fit_or_answer_selection": False,
        },
    )


def load_source(out):
    """Load only hash-sealed source arrays; never fit them to application readouts."""
    seal = read_json(out / "01/source_seal.json")
    if digest(out / "01/source_vectors.npz") != seal["source_vectors_sha256"]:
        raise ValueError("Source vectors changed after barrier")
    for path, sha in seal["source_capture_hashes"].items():
        if digest(Path(path)) != sha:
            raise ValueError("Source capture changed")
    with np.load(out / "01/source_vectors.npz", allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def collect(out):
    """Commit all attempts, seal24source calls, then collect48native/576patch calls."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError("Stopped/uncertain acquisition cannot automatically retry")
    rows = {r["unit"]: r for r in read_json(out / "00/rows.json")}
    design = read_json(out / "00/design.json")
    spec, states, source = m["spec"], {}, None
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        old = read_json(ROOT / spec["parent"] / "01/observer.json")
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
                raise ValueError("Observer drift " + key)
        write(out / "01/observer.json", actual)
        for e in design:
            path = out / f"01/units/unit_{e['unit']:04d}.json"
            if e["unit"] == 24:
                if not (out / "01/source_seal.json").exists():
                    seal_source(out, rows, status)
                source = load_source(out)
                for r in rows.values():
                    if r["role"] == "application" and r["reused"]:
                        _, a = load_arrays(capture_path(out, r))
                        states[r["unit"]] = a["full"]
            if path.exists():
                _, a = load_arrays(path)
                if e["kind"] == "capture" and e["role"] == "application":
                    states[e["receiver"]] = a["full"]
                continue
            elapsed = previous + time.monotonic() - started
            if not call_allowed(status, spec, elapsed):
                raise TimeoutError("Finite648call/600second budget reached")
            patch = None
            if e["kind"] == "patch":
                patch = M.patch_values(
                    e,
                    rows[e["receiver"]],
                    states[e["receiver"]],
                    states[e["donor"]],
                    source,
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
            if patch is not None:
                arrays.update(
                    patch_positions=np.asarray(patch[0], int), patch_values=patch[1]
                )
            elif e["role"] == "application":
                states[e["receiver"]] = arrays["full"]
            save(
                path,
                {k: e[k] for k in IDENTITY}
                | {"native": native, "finished_epoch": time.time()},
                arrays,
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 72 == 0:
                print(f"ACTUAL:{status['completed_units']}/648 forwards", flush=True)
        if (
            any(
                status[k] != 648
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > 600
        ):
            raise ValueError("Incomplete or over-budget acquisition")
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
    """Preserve every codec/evidence action and exact paired finite character contrast."""
    check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Complete actual acquisition required")
    load_source(out)
    rows = {r["unit"]: r for r in read_json(out / "00/rows.json")}
    design = read_json(out / "00/design.json")
    captured, native_rows = {}, []
    for r in rows.values():
        meta, a = load_arrays(capture_path(out, r))
        captured[r["unit"]] = M.readout(meta["native"], r["codec"])
        native_rows.append(
            {
                k: r[k]
                for k in (
                    "unit",
                    "role",
                    "class",
                    "content",
                    "person",
                    "direction",
                    "codec",
                    "reused",
                )
            }
            | captured[r["unit"]]
            | {"tokens": len(a["full"])}
        )
    effects = []
    for e in design[72:]:
        meta, _ = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        n, d, p = (
            captured[e["receiver"]],
            captured[e["donor"]],
            M.readout(meta["native"], e["codec"]),
        )
        eta = orientation(e["person"], e["direction"])
        effects.append(
            {k: e[k] for k in IDENTITY}
            | {
                "source_code": e["source_code"],
                "source_actor": e["source_actor"],
                "candidate_mass": p["candidate_mass"],
                **{
                    prefix + "_" + k: v[k]
                    for prefix, v in (("native", n), ("donor", d), ("patched", p))
                    for k in (
                        "raw_score",
                        "semantic_score",
                        "raw_log_odds",
                        "semantic_log_odds",
                    )
                },
                "raw_action": eta * (p["raw_log_odds"] - n["raw_log_odds"]),
                "semantic_action": eta
                * (p["semantic_log_odds"] - n["semantic_log_odds"]),
                "raw_score_action": eta * (p["raw_score"] - n["raw_score"]),
                "semantic_score_action": eta
                * (p["semantic_score"] - n["semantic_score"]),
            }
        )
    ni = {
        (
            r["role"],
            r["content"],
            r["class"],
            r["person"],
            r["direction"],
            r["codec"],
        ): r
        for r in native_rows
    }
    pairs = []
    for key, a in ni.items():
        if key[-1] != 1:
            continue
        b = ni[key[:-1] + (-1,)]
        pairs.append(
            {k: a[k] for k in ("role", "content", "class", "person", "direction")}
            | {
                "standard_raw_score": a["raw_score"],
                "inverse_raw_score": b["raw_score"],
                "standard_semantic_score": a["semantic_score"],
                "inverse_semantic_score": b["semantic_score"],
                "semantic_score_difference": a["semantic_score"] - b["semantic_score"],
                "raw_complement_error": a["raw_score"] + b["raw_score"] - 1,
                "standard_raw_log_odds": a["raw_log_odds"],
                "inverse_raw_log_odds": b["raw_log_odds"],
                "semantic_log_odds_difference": a["semantic_log_odds"]
                - b["semantic_log_odds"],
            }
        )
    ei = {
        (r["content"], r["class"], r["person"], r["direction"], r["arm"], r["codec"]): r
        for r in effects
    }
    orbits = []
    for key, a in ei.items():
        if key[-1] != 1:
            continue
        b = ei[key[:-1] + (-1,)]
        orbits.append(
            {k: a[k] for k in ("content", "class", "person", "direction", "arm")}
            | {
                "standard_raw_action": a["raw_action"],
                "inverse_raw_action": b["raw_action"],
                **M.orbit(a["raw_action"], b["raw_action"]),
                "raw_score_even": (a["raw_score_action"] + b["raw_score_action"]) / 2,
                "raw_score_odd": (a["raw_score_action"] - b["raw_score_action"]) / 2,
            }
        )
    tables = [
        ("native.csv", [r for r in native_rows if r["role"] == "application"]),
        ("source_native.csv", [r for r in native_rows if r["role"] == "source"]),
        ("effects.csv", effects),
        ("native_code_pairs.csv", pairs),
        ("code_orbits.csv", orbits),
    ]
    for name, keys in (
        ("class_summary.csv", ("class", "codec", "arm")),
        ("target_summary.csv", ("content", "class", "codec", "arm")),
        ("person_summary.csv", ("person", "class", "codec", "arm")),
    ):
        grouped = defaultdict(list)
        for r in effects:
            grouped[tuple(r[k] for k in keys)].append(r)
        tables.append(
            (
                name,
                [
                    dict(zip(keys, key, strict=True)) | M.action_summary(v)
                    for key, v in grouped.items()
                ],
            )
        )
    for name, data in tables:
        write_csv(out / "03" / name, data)
    write(
        out / "03/summary.json",
        {
            "packet": "HM53",
            "source_rows": 48,
            "application_rows": 96,
            "effects": len(effects),
            "native_code_pairs": len(pairs),
            "code_orbits": len(orbits),
            "reused_captures": 72,
            "new_calls": 648,
            "groups": tables[-3][1],
            "minimum_candidate_mass": min(
                r["candidate_mass"] for r in native_rows + effects
            ),
            "no_accuracy_selection": True,
            "psychological_truth_or_hidden_group_representation": False,
            "original_scope_complete": False,
        },
    )
    paths = (
        list((out / "01").rglob("[!._]*.json"))
        + list((out / "01").rglob("[!._]*.npz"))
        + list((out / "03").glob("[!._]*.csv"))
        + [out / "03/summary.json"]
    )
    write(
        out / "03/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    print("ANALYZED:576 effects/72 native code pairs/288 finite orbits", flush=True)


def main():
    """Run explicit bounded stages; never recover old queues or uncertain calls."""
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
