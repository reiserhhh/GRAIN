#!/usr/bin/env python3
"""CON01: collect matched conditional endpoints from source-reference authors only."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_inputs import (
    evidence_mask,
    evidence_summaries,
)
from grain_research.vector_mechanisms.evidence_packets import natural_evidence_edits
from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    read_json,
    tokenize_world,
    write_json,
)
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "conditional_navigation_v1/CON-01/source_endpoints_r1"
MATERIAL = BASE / "multiquery_sensitivity_s1_r1/private_material.json"
MEMBERS = BASE / "navigation_v1/NAV-02/source_span_r1/private_membership.json"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_NAVIGATION_PLAN_V1.md"


def code_hashes():
    """Pin collector, renderer, numerical inputs and full-model scoring implementation."""
    names = [
        "scripts/collect_grain_conditional_sources.py",
        "grain_research/vector_mechanisms/conditional_inputs.py",
        "grain_research/vector_mechanisms/evidence_packets.py",
        "grain_research/vector_mechanisms/navigation.py",
        "grain_research/vector_mechanisms/full_state_edit.py",
        "grain_research/vector_mechanisms/trace.py",
    ]
    return {n: digest(ROOT / n) for n in names}


def prepare(out):
    """Freeze twelve disjoint reference authors, no psychological criteria or evaluation texts."""
    from transformers import AutoTokenizer

    if out.exists() or not out.is_relative_to(
        (BASE / "conditional_navigation_v1").resolve()
    ):
        raise ValueError("Choose a fresh ignored construction directory")
    if (
        digest(MATERIAL)
        != read_json(MATERIAL.parent / "manifest.json")["material_sha256"]
    ):
        raise ValueError("Author material changed")
    member = read_json(MEMBERS)
    refs, held = set(member["reference_authors"]), set(member["evaluation_authors"])
    if len(refs) != 12 or len(held) != 12 or refs & held:
        raise ValueError("Expected twelve disjoint reference/evaluation authors")
    records = sorted(
        [r for r in read_json(MATERIAL) if r["author_id"] in refs and r["view"] == 0],
        key=lambda r: str(r["author_id"]),
    )
    if len(records) != 12 or {r["author_id"] for r in records} != refs:
        raise ValueError("Reference materials do not match frozen membership")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    expected_tokens = read_json(OUTPUT / "manifest.json")["question_tokens"]
    worlds, design = [], []
    for i, r in enumerate(records):
        edit = natural_evidence_edits(r["texts"])
        if edit is None:
            raise ValueError("Eligibility changed; do not silently replace an author")
        for condition, texts in edit["variants"].items():
            wid = f"reference{i:02d}_{condition}"
            worlds.append(
                {
                    "world_id": wid,
                    "author_id": r["author_id"],
                    "source_index": i,
                    "condition": condition,
                    "role": "reference",
                    "view": 0,
                    "selected_excerpt": edit["selected_excerpt"],
                    "texts": texts,
                    "counterfactual_wrapper": condition != "original",
                }
            )
            for q in QUERIES:
                _, spans, encoded, mask, alphabet = tokenize_world(tokenizer, texts, q)
                if (
                    np.asarray(encoded["input_ids"])[mask].tolist()
                    != expected_tokens[q]
                ):
                    raise ValueError("Fixed query coordinates changed")
                body = evidence_mask(encoded["offset_mapping"], spans, mask)
                design.append(
                    {
                        "index": len(design),
                        "world_id": wid,
                        "query": q,
                        "author_id": r["author_id"],
                        "source_index": i,
                        "condition": condition,
                        "input_ids": encoded["input_ids"],
                        "mask": mask.tolist(),
                        "evidence_mask": body.tolist(),
                        "alphabet": alphabet,
                    }
                )
    if len(design) != 288:
        raise ValueError("Unexpected source endpoint count")
    out.mkdir(parents=True)
    (out / "private_packets").mkdir()
    write_json(out / "private_worlds.json", worlds)
    write_json(out / "private_design.json", design)
    write_json(
        out / "manifest.json",
        {
            "protocol": "CON01_SOURCE_ENDPOINTS_V1",
            "units": 288,
            "authors": 12,
            "queries": list(QUERIES),
            "conditions": list(edit["variants"]),
            "code_sha256": code_hashes(),
            "source_sha256": {
                str(p.relative_to(ROOT)): digest(p)
                for p in (MATERIAL, MEMBERS, PLAN, OUTPUT / "manifest.json")
            },
            "input_sha256": {
                n: digest(out / n)
                for n in ("private_worlds.json", "private_design.json")
            },
            "model_sha256": digest(MODEL / "model.safetensors"),
            "human_labels_used": False,
            "MEPS_used": False,
            "evaluation_authors_used": False,
            "early_block": 3,
            "upper_block": 16,
            "summary": "excerpt-only mean and query-mean cosine softmax mean; temperature1",
        },
    )
    write_json(
        out / "run_status.json",
        {"status": "PREPARED", "units": 0, "calls": 0, "seconds": 0},
    )
    print({"status": "PREPARED", "units": 288, "reference_authors": 12}, flush=True)


def collect(out, seconds):
    """Save block3 and block16 states plus native score from one scoped forward."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, prior = read_json(out / "manifest.json"), read_json(out / "run_status.json")
    if (
        m["code_sha256"] != code_hashes()
        or any(digest(ROOT / n) != sha for n, sha in m["source_sha256"].items())
        or any(digest(out / n) != sha for n, sha in m["input_sha256"].items())
    ):
        raise ValueError("Frozen implementation/source/design changed")
    if digest(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Frozen observer changed")
    design = read_json(out / "private_design.json")
    if prior["status"] == "COMPLETE":
        for d in design:
            meta, _ = checked_arrays(
                out / "private_packets" / f"unit_{d['index']:04d}.json"
            )
            if any(meta[k] != d[k] for k in ("index", "world_id", "query")):
                raise ValueError("Completed unit identity changed")
        print(prior, flush=True)
        return
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start, complete = time.monotonic(), 0

    def status(name):
        value = {
            "status": name,
            "units": complete,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        }
        write_json(out / "run_status.json", value)
        return value

    try:
        for d in design:
            path = out / "private_packets" / f"unit_{d['index']:04d}.json"
            if path.exists():
                meta, _ = checked_arrays(path)
                if any(
                    meta[k] != d[k]
                    for k in ("index", "world_id", "query", "author_id", "condition")
                ):
                    raise ValueError("Source endpoint identity mismatch")
            else:
                if time.monotonic() - start >= seconds:
                    print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                    return
                early = {}
                mask, body = (
                    np.asarray(d["mask"], bool),
                    np.asarray(d["evidence_mask"], bool),
                )

                def capture_early(_module, _args, output, early=early, mask=mask, body=body):
                    """Capture only this endpoint's predeclared early coordinates."""
                    h = output[0] if isinstance(output, tuple) else output
                    value = h[0].detach().float().cpu().numpy()
                    early.update(
                        P_early=value[mask].copy(),
                        Q_early=value[-1].copy(),
                        evidence_summary=evidence_summaries(value, body, mask),
                    )

                handle = observer.model.model.layers[3].register_forward_hook(
                    capture_early
                )
                try:
                    upper, _, native = capture_boundary(
                        observer, d["input_ids"], d["alphabet"]
                    )
                finally:
                    handle.remove()
                if not early:
                    raise ValueError("Early hook did not run")
                save_unit(
                    path,
                    {
                        k: d[k]
                        for k in (
                            "index",
                            "world_id",
                            "query",
                            "author_id",
                            "source_index",
                            "condition",
                        )
                    }
                    | {"native": native},
                    **early,
                    P=upper[0, mask].cpu().numpy(),
                    Q=upper[0, -1].cpu().numpy(),
                )
                del upper
            complete += 1
            if complete % 24 == 0:
                print(status("CAPTURING"), flush=True)
        write_json(
            out / "collection_freeze.json",
            {
                "manifest_sha256": digest(out / "manifest.json"),
                "packets": {
                    p.name: digest(p) for p in (out / "private_packets").glob("*.json")
                },
                "units": complete,
            },
        )
        print(status("COMPLETE"), flush=True)
    except Exception as exc:
        status("ERROR")
        write_json(out / "error.json", {"type": type(exc).__name__})
        raise


def main():
    """Run one bounded source-only packet, without starting other research queues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "collect"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    with collector_guard(out):
        if args.phase == "prepare":
            prepare(out)
        else:
            collect(out, args.seconds)


if __name__ == "__main__":
    main()
