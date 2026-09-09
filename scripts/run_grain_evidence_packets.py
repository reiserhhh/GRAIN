#!/usr/bin/env python3
"""NAV-04/05/07: immutable evidence, query and temporal intervention packets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.evidence_packets import (
    binary_query_factorial,
    construct_view_reference,
    natural_evidence_edits,
    temporal_worlds,
    tokenize_question,
)
from grain_research.vector_mechanisms.navigation import (
    BASE,
    KEYS,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    read_json,
    tokenize_world,
    write_json,
)
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard

PACKET_DIRS = {
    "identity": "NAV-05/query_identity_r1",
    "evidence": "NAV-04/natural_evidence_r1",
    "time": "NAV-07/temporal_order_r1",
}


def code_hashes():
    """Freeze experimental rendering and numerical implementation."""
    names = [
        "scripts/run_grain_evidence_packets.py",
        "grain_research/vector_mechanisms/evidence_packets.py",
        "grain_research/vector_mechanisms/navigation.py",
        "grain_research/vector_mechanisms/full_state_edit.py",
        "grain_research/vector_mechanisms/query_functional.py",
        "grain_research/vector_mechanisms/factor_functions.py",
    ]
    return {p: digest(ROOT / p) for p in names}


def natural_material():
    """Read exposed evaluation-author texts, never MEPS or human labels."""
    source = BASE / "multiquery_sensitivity_s1_r1"
    material_path = source / "private_material.json"
    roles_path = BASE / "shared_author_profiles_m25_v1/private_design.json"
    if digest(material_path) != read_json(source / "manifest.json")["material_sha256"]:
        raise ValueError("Source texts changed")
    roles = {}
    for d in read_json(roles_path):
        if d["author_id"] in roles and roles[d["author_id"]] != d["role"]:
            raise ValueError("Author role conflict")
        roles[d["author_id"]] = d["role"]
    material = [
        r
        for r in read_json(material_path)
        if roles[r["author_id"]] == "evaluation" and r["view"] == 0
    ]
    if len(material) != 12 or len({r["author_id"] for r in material}) != 12:
        raise ValueError("Expected the12 frozen held authors")
    worlds, audit = [], []
    for row in material:
        edited = natural_evidence_edits(row["texts"])
        audit.append(
            {
                "author_id": row["author_id"],
                "eligible": edited is not None,
                "reason": "first_person_proposition"
                if edited
                else "no_explicit_first_person_proposition",
            }
        )
        if edited is None:
            continue
        for condition, texts in edited["variants"].items():
            worlds.append(
                {
                    "world_id": f"author{row['panel_index']:02d}_{condition}",
                    "author_id": row["author_id"],
                    "scene_id": str(row["panel_index"]),
                    "condition": condition,
                    "split": "held_author",
                    "selected_excerpt": edited["selected_excerpt"],
                    "texts": texts,
                    "modified_excerpt_is_counterfactual": condition != "original",
                }
            )
    return (
        worlds,
        audit,
        {"source_text": digest(material_path), "source_roles": digest(roles_path)},
    )


def prepare(kind, out):
    """Freeze tokens, question coordinates and reference roles before inference."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Choose a fresh packet directory")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    definitions, references, sources = (
        {},
        {},
        {"parent_manifest": digest(OUTPUT / "manifest.json")},
    )
    if kind == "identity":
        definitions = binary_query_factorial()
        worlds = read_json(OUTPUT / "worlds.json")
        references = {
            q: ["s00_self_voluntary_r0", "s01_self_voluntary_r0"] for q in definitions
        }
        queries, audit = list(definitions), []
    elif kind == "time":
        worlds = temporal_worlds()
        original = read_json(BASE / "narrow_functions_m30b_v1/queries.json")[
            "adapted_dejected5"
        ]["question"]
        definitions = {
            "adapted_dejected5_now": {
                "question": original
                + " Focus on the participant's present state, distinguishing it from states described as past.",
                "style": "narrow",
                "labels": list("0123456789"),
                "values": (np.arange(10) / 9).tolist(),
            }
        }
        references = {"adapted_dejected5_now": ["t00_n1_r0", "t01_n0_r0"]}
        queries, audit = ["adapted_dejected5", "adapted_dejected5_now"], []
    else:
        worlds, audit, extra = natural_material()
        sources.update(extra)
        queries = list(QUERIES)
    design, question_tokens = [], {}
    for w in worlds:
        for q in queries:
            view = (
                tokenize_question(tokenizer, w["texts"], definitions[q])
                if q in definitions
                else tokenize_world(tokenizer, w["texts"], q)
            )
            _, _, encoded, mask, alphabet = view
            tokens = np.asarray(encoded["input_ids"])[mask].tolist()
            if q in question_tokens and question_tokens[q] != tokens:
                raise ValueError("Question coordinates changed across material")
            question_tokens[q] = tokens
            design.append(
                {
                    "index": len(design),
                    "world_id": w["world_id"],
                    "query": q,
                    "input_ids": encoded["input_ids"],
                    "mask": mask.tolist(),
                    "alphabet": alphabet,
                }
            )
    out.mkdir(parents=True)
    for sub in ("private_packets", "reference_units"):
        (out / sub).mkdir()
    write_json(out / "private_worlds.json", worlds)
    write_json(out / "private_design.json", design)
    write_json(out / "queries.json", definitions)
    write_json(out / "private_eligibility.json", audit)
    write_json(
        out / "manifest.json",
        {
            "protocol": "NAV_EVIDENCE_PACKETS_V1",
            "kind": kind,
            "units": len(design),
            "worlds": len(worlds),
            "queries": queries,
            "question_tokens": question_tokens,
            "synthetic_references": references,
            "source_sha256": sources,
            "code_sha256": code_hashes(),
            "model_sha256": digest(MODEL / "model.safetensors"),
            "input_sha256": {
                n: digest(out / n)
                for n in (
                    "private_worlds.json",
                    "private_design.json",
                    "queries.json",
                    "private_eligibility.json",
                )
            },
            "old_reference_sha256": {
                n: digest(OUTPUT / n)
                for n in ("real_reference.npz", "synthetic_reference.npz")
            },
            "new_query_reference_policy": "new query rows require own construction; no coordinate transplantation",
            "human_label_access": False,
            "meps_material_used": False,
            "old_lockbox_access": False,
            "conditional_score_not_parse_rate": True,
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "units": 0},
    )
    print(
        {
            "status": "PREPARED",
            "kind": kind,
            "units": len(design),
            "worlds": len(worlds),
            "queries": queries,
        },
        flush=True,
    )


def collect(out, seconds=1200):
    """Collect native/PQ states and own-coordinate references in restartable units."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, prior = read_json(out / "manifest.json"), read_json(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print(prior, flush=True)
        return
    if (
        any(digest(out / n) != sha for n, sha in m["input_sha256"].items())
        or code_hashes() != m["code_sha256"]
    ):
        raise ValueError("Frozen code or material changed")
    if digest(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Model changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    write_json(out / "observer.json", observer.manifest())
    design, start = read_json(out / "private_design.json"), time.monotonic()
    definitions = read_json(out / "queries.json")
    worlds = {w["world_id"]: w for w in read_json(out / "private_worlds.json")}
    complete = 0

    def status(name):
        value = {
            "status": name,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": complete,
        }
        write_json(out / "run_status.json", value)
        return value

    def check_time():
        if time.monotonic() - start >= seconds:
            raise TimeoutError

    try:
        for d in design:
            file = out / "private_packets" / f"unit_{d['index']:04d}.json"
            if file.exists():
                meta, _ = checked_arrays(file)
                if any(meta[k] != d[k] for k in ("index", "query", "world_id")):
                    raise ValueError("Packet identity changed")
            else:
                check_time()
                state, _, native = capture_boundary(
                    observer, d["input_ids"], d["alphabet"]
                )
                mask = np.asarray(d["mask"], bool)
                save_unit(
                    file,
                    {k: d[k] for k in ("index", "world_id", "query")}
                    | {"native": native},
                    P=state[0, mask].cpu().numpy(),
                    Q=state[0, -1].cpu().numpy(),
                )
            complete += 1
            if complete % 30 == 0:
                print(status("CAPTURING"), flush=True)
        arrays = {}
        for q, world_ids in m["synthetic_references"].items():
            pieces = []
            for i, wid in enumerate(world_ids):
                file = out / "reference_units" / f"{q}_{i}.json"
                if file.exists():
                    meta, z = checked_arrays(file)
                    if meta["query"] != q or meta["world_id"] != wid:
                        raise ValueError("Reference provenance changed")
                else:
                    check_time()
                    full = tokenize_question(
                        observer.tokenizer, worlds[wid]["texts"], definitions[q]
                    )
                    removed = tokenize_question(
                        observer.tokenizer, worlds[wid]["texts"][1:], definitions[q]
                    )
                    z, audit = construct_view_reference(observer, full, removed)
                    save_unit(
                        file,
                        {
                            "query": q,
                            "world_id": wid,
                            "alignment": audit,
                            "source_split": "construction",
                        },
                        **z,
                    )
                pieces.append(z)
            arrays.update(
                {
                    f"{q}__{k}": np.mean(
                        [z[k] for z in pieces], axis=0, dtype=np.float64
                    )
                    for k in KEYS
                }
            )
        if arrays:
            np.savez_compressed(out / "new_references.npz", **arrays)
            write_json(
                out / "reference_freeze.json",
                {
                    "sha256": digest(out / "new_references.npz"),
                    "query_tokens": {q: m["question_tokens"][q] for q in definitions},
                    "no_cross_query_coordinate_reuse": True,
                },
            )
        print(status("COMPLETE"), flush=True)
    except TimeoutError:
        print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
    except Exception as exc:
        status("ERROR")
        write_json(
            out / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise


def evaluate(out):
    """Read numerical states, retaining native and fixed-function components."""
    m = read_json(out / "manifest.json")
    if read_json(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete collection before evaluation")
    old_queries = [q for q in m["queries"] if q in QUERIES]
    fs = {}
    if old_queries:
        for name, file in (
            ("R_real", "real_reference.npz"),
            ("R_synth", "synthetic_reference.npz"),
        ):
            if digest(OUTPUT / file) != m["old_reference_sha256"][file]:
                raise ValueError("Original fixed reader changed")
            fs[name] = load_functions(OUTPUT / file, queries=old_queries)
    if m["synthetic_references"]:
        if (
            digest(out / "new_references.npz")
            != read_json(out / "reference_freeze.json")["sha256"]
        ):
            raise ValueError("New reference changed")
        fs["R_new"] = load_functions(
            out / "new_references.npz", queries=list(m["synthetic_references"])
        )
    rows = []
    for d in read_json(out / "private_design.json"):
        meta, z = checked_arrays(
            out / "private_packets" / f"unit_{d['index']:04d}.json"
        )
        base = {k: d[k] for k in ("index", "world_id", "query")}
        rows.append(
            base
            | {
                "readout": "native",
                "score": meta["native"]["conditional_score"],
                "candidate_mass": meta["native"]["candidate_mass"],
            }
        )
        for name, functions in fs.items():
            if d["query"] in functions:
                rows.append(
                    base
                    | {"readout": name, **functions[d["query"]].parts(z["P"], z["Q"])}
                )
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    write_json(
        out / "prediction_freeze.json",
        {
            "sha256": digest(out / "predictions.csv"),
            "human_criteria_used": False,
            "runtime": "full prefix P/Q for fixed function comparisons",
        },
    )
    print({"status": "FIXED_FUNCTIONS_SCORED", "rows": len(rows)}, flush=True)


def main(default_packet=None):
    """Expose bounded packages without changing any historical collector."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "collect", "evaluate"])
    p.add_argument(
        "--packet", choices=PACKET_DIRS, default=default_packet or "identity"
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--seconds", type=int, default=1200)
    a = p.parse_args()
    out = a.out or BASE / "navigation_v1" / PACKET_DIRS[a.packet]
    if not 0 < a.seconds <= 1200 or not out.resolve().is_relative_to(
        (BASE / "navigation_v1").resolve()
    ):
        p.error("Use bounded local navigation output")
    with collector_guard(out):
        if a.mode == "prepare":
            prepare(a.packet, out)
        elif a.mode == "collect":
            collect(out, a.seconds)
        else:
            evaluate(out)


if __name__ == "__main__":
    main()
