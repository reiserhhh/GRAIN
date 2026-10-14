#!/usr/bin/env python3
"""HM54: fixed source delta x fresh evidence, with an explicit output-offset null."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_evidence_response_operation as M
from grain_research.construction.math_answer_recoding_mechanism import (
    prove_mapping,
    render,
)
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_answer_recoding_mechanism import CODE as PARENT_CODE
from scripts.run_grain_math_answer_recoding_mechanism import (
    capture_path,
    load_source,
)
from scripts.run_grain_math_answer_recoding_mechanism import (
    check as parent_check,
)
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_evidence_response_operation_v1.json"
OUT = ROOT / "results/grain_math_response_v1/evidence_response_operation_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_evidence_response_operation_v1.json",
            "docs/GRAIN_EVIDENCE_RESPONSE_OPERATION_PLAN_V1.md",
            "grain_research/construction/math_evidence_response_operation.py",
            "scripts/run_grain_math_evidence_response_operation.py",
            "scripts/verify_grain_math_evidence_response_operation.py",
            "tests/test_grain_math_evidence_response_operation.py",
            *PARENT_CODE,
        )
    )
)
IDENTITY = (
    "unit",
    "receiver",
    "donor",
    "arm",
    "kind",
    "content",
    "class",
    "person",
    "direction",
    "pair_id",
)


def prepare(out):
    """Freeze new text/token contracts and reconstruct source deltas before any call."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve prior results")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    prior = parent_check(parent)
    inherited_system = read_json(
        ROOT / prior["spec"]["parent"] / "00/manifest.json"
    )["spec"]["system"]
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM53_NOT_GLOBAL"
        or spec["targets"] != prior["spec"]["targets"]
        or spec["layer"] != prior["spec"]["layer"]
        or spec["system"] != inherited_system
        or spec["classes"] != list(M.CLASSES)
        or spec["arms"] != list(M.ARMS)
    ):
        raise ValueError("Verified unchanged reader and full fixed design required")
    src = [
        r
        for r in read_json(parent / "00/rows.json")
        if r["role"] == "source" and r["codec"] == 1
    ]
    source_index = {(r["content"], r["person"], r["direction"]): r for r in src}
    old = load_source(parent)
    source, source_rows, offsets, states = {}, [], {}, {}
    for r in src:
        path = capture_path(parent, r)
        meta, arrays = load_arrays(path)
        if meta["unit"] != r["standard_reference_uid"] or len(arrays["full"]) != len(
            r["input_ids"]
        ):
            raise ValueError("Source capture/input identity drift")
        states[r["unit"]] = arrays["full"]
        source_rows.append(
            {
                "row": r,
                "capture": str(path.relative_to(ROOT)),
                "json_sha256": digest(path),
                "npz_sha256": digest(path.with_suffix(".npz")),
                "readout": M.readout(meta["native"]),
            }
        )
    si = {
        (r["row"]["content"], r["row"]["person"], r["row"]["direction"]): r
        for r in source_rows
    }
    for target, person in itertools.product(spec["targets"], ("A", "B")):
        a, b = (source_index[target, person, d] for d in (0, 1))
        prove_mapping(a, b)
        key = f"{target}__{person}"
        delta = states[b["unit"]][b["partition"]["suffix"]].astype(float) - states[
            a["unit"]
        ][a["partition"]["suffix"]].astype(float)
        if not np.array_equal(delta, old[key + "__1"]):
            raise ValueError("Original standard source delta changed")
        source[key] = delta
        offsets[key] = (
            si[target, person, 1]["readout"]["log_odds"]
            - si[target, person, 0]["readout"]["log_odds"]
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    docs = M.documents(spec)
    di = {(r["content"], r["class"], r["direction"]): r for r in docs}
    rows = []
    for target, kind, person, direction in itertools.product(
        spec["targets"], M.CLASSES, ("A", "B"), (0, 1)
    ):
        doc = di[target, kind, direction]
        text, ids, part, cues, actor = render(
            doc["body"], target, person, 1, spec["system"], tokenizer
        )
        row = {
            **doc,
            "unit": len(rows),
            "pair_id": len(rows) // 2,
            "person": person,
            "query": target,
            "codec": 1,
            "rendered": text,
            "input_ids": ids,
            "partition": part,
            "cue_slots": cues,
            "actor_slot": actor,
        }
        row["source_mapping"] = prove_mapping(source_index[target, person, 0], row)
        row["wrong_source_mapping"] = prove_mapping(
            source_index[target, "B" if person == "A" else "A", 0],
            row,
            allow_actor=True,
        )
        rows.append(row)
    if (
        len(src) != 24
        or len(rows) != spec["expected_receivers"]
        or len({r["rendered"] for r in rows}) != 96
    ):
        raise ValueError("24 source captures and96 fresh rendered receivers required")
    historical_bodies = {r["body"] for r in read_json(parent / "00/rows.json")}
    if any(r["body"] in historical_bodies for r in rows):
        raise ValueError("Application body unexpectedly reuses HM53 material")
    index = {(r["content"], r["class"], r["person"], r["direction"]): r for r in rows}
    design = []
    for r in rows:
        donor = index[r["content"], r["class"], r["person"], 1 - r["direction"]]
        proof = prove_mapping(donor, r)
        design.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "unit",
                        "content",
                        "class",
                        "person",
                        "direction",
                        "pair_id",
                    )
                },
                "receiver": r["unit"],
                "donor": donor["unit"],
                "kind": "capture",
                "arm": "native",
                "donor_positions": proof["source_positions"],
            }
        )
    captures = design.copy()
    for e in captures:
        for arm in M.ARMS:
            design.append({**e, "unit": len(design), "kind": "patch", "arm": arm})
    if len(design) != spec["max_forwards"]:
        raise ValueError("Full672call design required")
    for name in ("00/code", "01/units", "03"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write(out / "00/documents.json", docs)
    write(out / "00/rows.json", rows)
    write(out / "00/design.json", design)
    write(out / "00/source_rows.json", source_rows)
    write(out / "00/source_offsets.json", offsets)
    with (out / "00/source_vectors.npz").open("wb") as stream:
        np.savez_compressed(stream, **source)
    write(
        out / "00/source_seal.json",
        {
            "source_capture_count": 24,
            "source_operator_count": 12,
            "created_epoch": time.time(),
            "before_all_new_calls": True,
            "new_source_calls": 0,
            "fit_or_application_answer_selection": False,
        },
    )
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    assets = (
        "00/documents.json",
        "00/rows.json",
        "00/design.json",
        "00/source_rows.json",
        "00/source_offsets.json",
        "00/source_vectors.npz",
        "00/source_seal.json",
    )
    write(
        out / "00/manifest.json",
        {
            "packet": "HM54",
            "spec": spec,
            "alphabet": prior["alphabet"],
            "code": code,
            "assets": {n: digest(out / n) for n in assets},
            "parent_files": {
                n: digest(parent / n)
                for n in (
                    "00/manifest.json",
                    "verification.json",
                    "01/observer.json",
                    "01/source_vectors.npz",
                    "01/source_seal.json",
                )
            },
            "model_config_sha256": digest(MODEL / "config.json"),
            "model_weights_not_separately_hashed": True,
            "unique_body_count": len({r["body"] for r in rows}),
            "independent_human_units": 0,
            "source_sign_selected_by_receiver_direction": False,
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
        "PREPARED:96 new receivers/38unique bodies/24source reused/672calls maximum",
        flush=True,
    )


def check(out):
    """Verify immutable live/frozen code, materials, source captures and observer config."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code changed " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen asset changed " + name)
    parent_check(ROOT / m["spec"]["parent"])
    for name, sha in m["parent_files"].items():
        if digest(ROOT / m["spec"]["parent"] / name) != sha:
            raise ValueError("Parent source changed " + name)
    for r in read_json(out / "00/source_rows.json"):
        p = ROOT / r["capture"]
        if (
            digest(p) != r["json_sha256"]
            or digest(p.with_suffix(".npz")) != r["npz_sha256"]
        ):
            raise ValueError("Original source capture changed")
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Observer config drift")
    return m


def vectors(out):
    """Load only already-sealed source operators."""
    with np.load(out / "00/source_vectors.npz", allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def collect(out):
    """Acquire96 native and576 fixed patches; never retry a stopped/uncertain call."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError("Uncertain/stopped acquisition cannot automatically retry")
    rows = {r["unit"]: r for r in read_json(out / "00/rows.json")}
    design = read_json(out / "00/design.json")
    source, states, spec = vectors(out), {}, m["spec"]
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual, old = (
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
            if actual[k] != old[k]:
                raise ValueError("Observer drift " + k)
        write(out / "01/observer.json", actual)
        for e in design:
            path = out / f"01/units/unit_{e['unit']:04d}.json"
            if path.exists():
                _, a = load_arrays(path)
                if e["kind"] == "capture":
                    states[e["receiver"]] = a["full"]
                continue
            elapsed = previous + time.monotonic() - started
            if not call_allowed(status, spec, elapsed):
                raise TimeoutError("Finite672call/600second budget reached")
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
            else:
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
            if status["completed_units"] % 96 == 0:
                print(f"ACTUAL:{status['completed_units']}/672 forwards", flush=True)
        if (
            any(
                status[k] != 672
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > 600
        ):
            raise ValueError("Incomplete or over-budget collection")
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
    """Compute raw paired responses and source-only output nulls without fitting."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Complete actual acquisition required")
    rows = {r["unit"]: r for r in read_json(out / "00/rows.json")}
    design = read_json(out / "00/design.json")
    offsets = read_json(out / "00/source_offsets.json")
    native, metadata, native_rows = {}, {}, []
    for r in rows.values():
        meta, _ = load_arrays(out / f"01/units/unit_{r['unit']:04d}.json")
        native[r["unit"]] = M.readout(meta["native"])
        metadata[r["unit"]] = meta["native"]
        native_rows.append(
            {
                k: r[k]
                for k in ("unit", "content", "class", "person", "direction", "pair_id")
            }
            | native[r["unit"]]
            | {"tokens": len(r["input_ids"])}
        )
    effects, patched = [], {}
    for e in design[96:]:
        meta, _ = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        v = M.readout(meta["native"])
        n = native[e["receiver"]]
        patched[e["receiver"], e["arm"]] = v
        effects.append(
            {k: e[k] for k in IDENTITY}
            | {
                **{
                    prefix + "_" + k: a[k]
                    for prefix, a in (
                        ("native", n),
                        ("donor", native[e["donor"]]),
                        ("patched", v),
                    )
                    for k in ("score", "log_odds")
                },
                "score_action": v["score"] - n["score"],
                "log_odds_action": v["log_odds"] - n["log_odds"],
                "candidate_mass": v["candidate_mass"],
            }
        )
    ri = {
        (r["content"], r["class"], r["person"], r["direction"]): r
        for r in rows.values()
    }
    pairs = []
    for target, kind, person, arm in itertools.product(
        m["spec"]["targets"], M.CLASSES, ("A", "B"), M.ARMS
    ):
        a, b = (ri[target, kind, person, d] for d in (0, 1))
        pairs.append(
            {
                "content": target,
                "class": kind,
                "person": person,
                "arm": arm,
                "pair_id": a["pair_id"],
            }
            | M.paired(
                native[a["unit"]],
                native[b["unit"]],
                patched[a["unit"], arm],
                patched[b["unit"], arm],
            )
        )
    null_rows, null_values = [], {}
    for r in rows.values():
        for sign in (1, -1):
            b = sign * offsets[f"{r['content']}__{r['person']}"]
            v = M.output_offset(metadata[r["unit"]]["candidate_logits"], b)
            null_values[r["unit"], sign] = v
            null_rows.append(
                {
                    k: r[k]
                    for k in (
                        "unit",
                        "content",
                        "class",
                        "person",
                        "direction",
                        "pair_id",
                    )
                }
                | {
                    "source_sign": sign,
                    "offset": b,
                    "score": v["score"],
                    "log_odds": v["log_odds"],
                    "score_action": v["score"] - native[r["unit"]]["score"],
                    "log_odds_action": v["log_odds"] - native[r["unit"]]["log_odds"],
                    **{f"p{k}": x for k, x in enumerate(v["probabilities"])},
                }
            )
    null_pairs = []
    for target, kind, person, sign in itertools.product(
        m["spec"]["targets"], M.CLASSES, ("A", "B"), (1, -1)
    ):
        a, b = (ri[target, kind, person, d] for d in (0, 1))
        null_pairs.append(
            {
                "content": target,
                "class": kind,
                "person": person,
                "source_sign": sign,
                "pair_id": a["pair_id"],
            }
            | M.paired(
                native[a["unit"]],
                native[b["unit"]],
                null_values[a["unit"], sign],
                null_values[b["unit"], sign],
            )
        )
    tables = [
        ("native.csv", native_rows),
        ("effects.csv", effects),
        ("evidence_pair_actions.csv", pairs),
        ("output_offset_null.csv", null_rows),
        ("output_offset_null_pairs.csv", null_pairs),
    ]
    for name, keys in (
        ("class_summary.csv", ("class", "arm")),
        ("target_class_summary.csv", ("content", "class", "arm")),
    ):
        groups = defaultdict(list)
        for p in pairs:
            groups[tuple(p[k] for k in keys)].append(p)
        tables.append(
            (
                name,
                [
                    dict(zip(keys, key, strict=True)) | M.paired_summary(v)
                    for key, v in groups.items()
                ],
            )
        )
    for name, table in tables:
        write_csv(out / "03" / name, table)
    write(
        out / "03/summary.json",
        {
            "packet": "HM54",
            "receivers": 96,
            "effects": 576,
            "paired_actions": 288,
            "output_null_rows": 192,
            "output_null_pairs": 96,
            "new_calls": 672,
            "source_captures_reused": 24,
            "unique_body_count": m["unique_body_count"],
            "groups": tables[-2][1],
            "maximum_null_log_interaction": max(
                abs(r["mixed_log_odds"]) for r in null_pairs
            ),
            "minimum_candidate_mass": min(
                r["candidate_mass"] for r in native_rows + effects
            ),
            "no_accuracy_selection": True,
            "nonzero_interaction_not_unique_reading_program": True,
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
    print("ANALYZED:576effects/288paired actions/192null rows/96null pairs", flush=True)


def main():
    """Run explicit bounded stages without starting historical queues."""
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
