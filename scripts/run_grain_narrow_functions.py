#!/usr/bin/env python3
"""M30B: construct25 local derivative functions without using MEPS for training."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write

SOURCE = BASE / "multiquery_sensitivity_s1_r1"
M7 = BASE / "query_region_functional_v1"


def reference_view(tokenizer, bodies, question):
    """Keep question coordinates exact without misdescribing source posts as AI chat."""
    from grain_research.vector_mechanisms.context_carrier import query_region_mask
    from grain_research.vector_mechanisms.counseling_queries import SYSTEM

    user = "These are the anonymous participant's own prior online posts. They are not a transcript of an AI interaction.\n\n"
    spans = []
    for i, body in enumerate(bodies):
        user += f"Excerpt {i + 1}:\n"
        spans.append((len(user), len(user) + len(body)))
        user += body + "\n\n"
    user += "Evidence ends here.\n\nQuestion: " + question + "\nAnswer:"
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if text.count(user) != 1:
        raise ValueError("Ambiguous source template")
    offset = text.index(user)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start = offset + user.rindex("\nQuestion: ") + 1
    mask = query_region_mask(encoded["offset_mapping"], start)
    if len(encoded["input_ids"]) > 3056:
        raise ValueError("Frozen source reference exceeds query context budget")
    return text, [(a + offset, b + offset) for a, b in spans], encoded, mask


def prepare(out, counseling, mood):
    """Fix independent authors and all question/mapping bytes, no MEPS scores."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.full_state_edit import align_excerpts
    from scripts.analyze_grain_multiquery_sensitivity import load_panel
    from scripts.run_grain_real_author_path import MODEL

    refs = {
        d["author_id"]
        for d in read(M7 / "private_design.json")
        if d["role"] == "reference"
    }
    _, panel = load_panel(SOURCE)
    records = [r[0] for r in panel if r[0]["author_id"] in refs and r[0]["view"] == 0]
    if len(refs) != 2 or len(records) != 2:
        raise ValueError("Exactly the two fixed M7 reference authors required")
    definitions = read(counseling / "queries.json") | read(mood / "queries.json")
    cm, mm = read(counseling / "manifest.json"), read(mood / "manifest.json")
    if cm["alphabet"] != mm["alphabet"] or cm["model_sha256"] != mm["model_sha256"]:
        raise ValueError("Narrow observer alphabets/weights differ")
    tokens = cm["question_tokens"] | mm["question_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    design = []
    for q, definition in definitions.items():
        for row in records:
            full, fs, fe, fm = reference_view(
                tokenizer, row["texts"], definition["question"]
            )
            keep = [j for j in range(len(row["texts"])) if j not in row["e1"]]
            edited, es, ee, em = reference_view(
                tokenizer, [row["texts"][j] for j in keep], definition["question"]
            )
            for encoded, mask in ((fe, fm), (ee, em)):
                if np.asarray(encoded["input_ids"])[mask].tolist() != tokens[q]:
                    raise ValueError("Reference/MEPS question token identity changed")
            mapping, audit = align_excerpts(full, edited, fs, es, keep, fe, ee)
            if not np.array_equal(mapping[em], np.flatnonzero(fm)):
                raise ValueError("Question alignment failed")
            design.append(
                {
                    "index": len(design),
                    "query": q,
                    "author_id": row["author_id"],
                    "full_ids": fe["input_ids"],
                    "edit_ids": ee["input_ids"],
                    "full_mask": fm.tolist(),
                    "edit_mask": em.tolist(),
                    "mapping": mapping.tolist(),
                    "alignment": audit,
                }
            )
    if len(design) != 50:
        raise ValueError("Expected25 queries x2 source reference units")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_units").mkdir()
    write(out / "private_design.json", design)
    write(out / "queries.json", definitions)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_M30B_SOURCE_REFERENCED_FUNCTIONS_V1",
            "units": 50,
            "calls_planned": 450,
            "queries": list(definitions),
            "per_run_seconds": 1200,
            "alphabet": cm["alphabet"],
            "model_sha256": cm["model_sha256"],
            "question_tokens": tokens,
            "design_sha256": sha(out / "private_design.json"),
            "sources": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    SOURCE / "private_material.json",
                    M7 / "private_design.json",
                    counseling / "queries.json",
                    mood / "queries.json",
                )
            },
            "code_sha256": {
                p: sha(ROOT / p)
                for p in (
                    "scripts/run_grain_narrow_functions.py",
                    "grain_research/vector_mechanisms/factor_functions.py",
                )
            },
            "plan_sha256": sha(ROOT / "docs/GRAIN_NARROW_FUNCTION_TRANSPORT_M30B.md"),
            "meps_reference_authors": 0,
            "human_outcome_fit": False,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed_units": 0},
    )
    print(
        {"status": "PREPARED", "queries": 25, "units": 50, "calls_planned": 450},
        flush=True,
    )


def collect(out):
    """One saved reference-unit at a time, using the same M28 integration rule."""
    import torch

    from grain_research.vector_mechanisms.context_carrier import (
        cached_suffix_kwargs,
        split_context_state,
    )
    from grain_research.vector_mechanisms.factor_functions import joint_gradient
    from grain_research.vector_mechanisms.full_state_edit import (
        align_states,
        capture_boundary,
    )
    from grain_research.vector_mechanisms.query_functional import (
        integrated_matrix,
        path_state,
        region_gradient,
    )
    from grain_research.vector_mechanisms.trace import DecoderObserver
    from scripts.run_grain_real_author_path import MODEL

    m, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        return
    if (
        sha(out / "private_design.json") != m["design_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_sha256"]
    ):
        raise ValueError("Frozen input/model changed")
    for name, digest in (m["sources"] | m["code_sha256"]).items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen reference or implementation changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    write(out / "observer.json", observer.manifest())
    start, completed = time.monotonic(), 0

    def status(label):
        """Persist a restartable boundary without emitting source identities."""
        s = {
            "status": label,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": completed,
            "meps_used_for_construction": False,
        }
        write(out / "run_status.json", s)
        return s

    try:
        for d in read(out / "private_design.json"):
            path = out / "private_units" / f"unit_{d['index']:03d}.json"
            if path.exists():
                if sha(path.with_suffix(".npz")) != read(path)["array_sha256"]:
                    raise ValueError("Saved reference unit changed")
                completed += 1
                continue
            if time.monotonic() - start > m["per_run_seconds"]:
                print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                return
            full, full_kwargs, native = capture_boundary(
                observer, d["full_ids"], m["alphabet"]
            )
            fm, em = np.asarray(d["full_mask"], bool), np.asarray(d["edit_mask"], bool)
            _, _, g = joint_gradient(observer, full, fm, full_kwargs, m["alphabet"])
            edited, kwargs, _ = capture_boundary(observer, d["edit_ids"], m["alphabet"])
            aligned = align_states(full, edited, d["mapping"])
            kwargs = cached_suffix_kwargs(observer.model, edited)
            gradients = []
            for e in (0, 1):
                context = (
                    aligned if e == 0 else split_context_state(aligned, edited, em, "E")
                )
                gs = []
                for t in (0.0, 0.5, 1.0):
                    _, G = region_gradient(
                        observer,
                        context,
                        path_state(aligned[:, em], edited[:, em], t),
                        em,
                        kwargs,
                        m["alphabet"],
                    )
                    gs.append(G)
                gradients.append(gs)
            np.savez_compressed(
                path.with_suffix(".npz"),
                Z=full[0, fm].cpu().numpy(),
                h=full[0, -1].cpu().numpy(),
                G=integrated_matrix(gradients),
                g=g,
                b=native["conditional_score"],
            )
            write(
                path,
                {
                    "query": d["query"],
                    "index": d["index"],
                    "array_sha256": sha(path.with_suffix(".npz")),
                },
            )
            completed += 1
            print(status("COLLECTING"), flush=True)
            del full, full_kwargs, edited, kwargs, aligned, gradients
        values = {}
        for q in m["queries"]:
            rows = []
            for d in read(out / "private_design.json"):
                if d["query"] != q:
                    continue
                with np.load(
                    out / "private_units" / f"unit_{d['index']:03d}.npz",
                    allow_pickle=False,
                ) as z:
                    rows.append(dict(z))
            if len(rows) != 2:
                raise ValueError("Missing one of two reference functions")
            for key in ("Z", "h", "G", "g", "b"):
                values[f"{q}__{key}"] = np.mean(
                    [r[key] for r in rows], axis=0, dtype=np.float64
                )
        np.savez_compressed(out / "reference.npz", **values)
        write(
            out / "reference_manifest.json",
            {
                "sha256": sha(out / "reference.npz"),
                "queries": m["queries"],
                "meps_used_for_construction": False,
                "human_outcome_fit": False,
            },
        )
        print(status("COMPLETE"), flush=True)
    except Exception:
        status("ERROR")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counseling-dir", type=Path)
    parser.add_argument("--mood-dir", type=Path)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Use local ignored output")
    if a.mode == "prepare":
        if a.counseling_dir is None or a.mood_dir is None:
            parser.error("Fixed M30/M30C query sources required")
        prepare(
            a.output_dir.resolve(), a.counseling_dir.resolve(), a.mood_dir.resolve()
        )
    else:
        collect(a.output_dir.resolve())
