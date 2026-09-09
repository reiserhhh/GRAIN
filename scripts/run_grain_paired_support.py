#!/usr/bin/env python3
"""Acquire fixed source/transport reversals with early and upper states together."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.input_order import ordinal_vector
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.run_grain_reading_levels import load_inputs

OUT = BASE / "logic_paired_support_v1/r1"
PRIOR = BASE / "logic_input_order_v1/paired_r2"
SOURCE_TEXT = BASE / "source_remainder_s9_r1/private_material.json"
PLAN = ROOT / "docs/GRAIN_LOGIC_CONDITIONAL_SUPPORT_PLAN_V1.md"
CORE = ROOT / "grain_research/vector_mechanisms/paired_support.py"
QUERIES = ("TF_cont", "neuroticism")


def prepare(out):
    """Choose authors and bind inputs without looking up human criteria."""
    from transformers import AutoTokenizer

    from scripts.run_grain_level_response import question_mask

    meta, bags, paths = load_inputs()
    old = read(PRIOR / "manifest.json")
    if (
        sha(SOURCE_TEXT)
        != read(SOURCE_TEXT.parent / "manifest.json")["material_sha256"]
    ):
        raise ValueError("Source material identity changed")
    texts = {r["author_id"]: r for r in read(SOURCE_TEXT)}
    held = {
        (r["author_id"], r["query"]): r for r in read(PRIOR / "private_design.json")
    }
    tok = AutoTokenizer.from_pretrained(old["model_dir"], local_files_only=True)
    template = {}
    for q in QUERIES:
        template[q] = next(r for (a, k), r in held.items() if k == q)
    designs, means, ordinals = [], [], []
    for q in QUERIES:
        group = "mbti_only" if q == "TF_cont" else "big5_only"
        source = meta[(meta.group == group) & (meta.view == 0)]
        ids = sorted(
            source.author_id,
            key=lambda uid: hashlib.sha256(f"NEXT08c:{q}:{uid}".encode()).hexdigest(),
        )[:32]
        if len(ids) != 32 or set(ids) & {a for a, k in held}:
            raise ValueError("Insufficient or overlapping source authors")
        for role, selected in (
            ("source", ids),
            ("transport", sorted({a for a, k in held})),
        ):
            for number, uid in enumerate(selected):
                if role == "source":
                    row = source[source.author_id == uid]
                    bodies = texts[uid]["views"][0]["texts"]
                    inputs = {}
                    for order, ts in (("original", bodies), ("reverse", bodies[::-1])):
                        _, _, enc, mask = question_mask(tok, ts, q)
                        inputs[order] = {"ids": enc["input_ids"], "mask": mask.tolist()}
                    pc = paths[
                        (paths.author_id == uid)
                        & (paths["query"] == q)
                        & (paths.view == 0)
                    ]
                    if len(pc) != 1:
                        raise ValueError("Source configuration reference ambiguous")
                    reference = {"configuration": float(pc.configuration.iloc[0])}
                    repeat = False
                else:
                    row = meta[(meta.author_id == uid) & (meta.edit == "full")]
                    record = held[(uid, q)]
                    inputs = {key: record[key] for key in ("original", "reverse")}
                    reference = {**record["old_parts"], "native": record["old_native"]}
                    repeat = number < 2
                if len(row) != 1:
                    raise ValueError("Input vector binding ambiguous")
                bag = bags[row.index[0]]
                if bag.shape != (12, 1024):
                    raise ValueError("Expected12 full comment embeddings")
                qids = np.array(template[q]["original"]["ids"])[
                    template[q]["original"]["mask"]
                ]
                for value in inputs.values():
                    if len(value["ids"]) + 16 > 3072 or not np.array_equal(
                        np.array(value["ids"])[value["mask"]], qids
                    ):
                        raise ValueError("Token truncation or question identity")
                ca, cb = [Counter(inputs[o]["ids"]) for o in ("original", "reverse")]
                designs.append(
                    {
                        "index": len(designs),
                        "author_id": uid,
                        "query": q,
                        "role": role,
                        "repeat": repeat,
                        **inputs,
                        "alphabet": template[q]["alphabet"],
                        "reference": reference,
                        "length_delta": len(inputs["reverse"]["ids"])
                        - len(inputs["original"]["ids"]),
                        "token_bag_l1": sum(
                            abs(ca[k] - cb[k]) for k in ca.keys() | cb.keys()
                        ),
                    }
                )
                means.append(np.stack([bag.mean(0), bag.mean(0)]))
                v = ordinal_vector(bag)
                ordinals.append(np.stack([v, -v]))
    if len(designs) != 88 or sum(d["repeat"] for d in designs) != 4:
        raise ValueError("Fixed panel changed")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", designs)
    np.savez_compressed(out / "private_e5_inputs.npz", mean=means, ordinal=ordinals)
    write(
        out / "manifest.json",
        {
            "model_dir": old["model_dir"],
            "model_sha256": old["model_sha256"],
            "functions": old["functions"],
            "functions_sha256": old["functions_sha256"],
            "source_material_sha256": sha(SOURCE_TEXT),
            "prior_manifest_sha256": sha(PRIOR / "manifest.json"),
            "code_hashes": {
                str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), CORE, PLAN)
            },
            "design_sha256": sha(out / "private_design.json"),
            "e5_sha256": sha(out / "private_e5_inputs.npz"),
            "max_calls": 180,
            "max_seconds": 900,
            "criterion_selection": False,
        },
    )
    write(
        out / "status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed": 0},
    )
    print("Prepared64 source+24 transport pairs; no criteria read", flush=True)


def verify(out):
    """Verify fixed numerical inputs, plan, code and teacher identity."""
    m = read(out / "manifest.json")
    for name, digest in m["code_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen file changed: {name}")
    for name, key in (
        ("private_design.json", "design_sha256"),
        ("private_e5_inputs.npz", "e5_sha256"),
    ):
        if sha(out / name) != m[key]:
            raise ValueError("Input packet changed")
    if (
        sha(m["functions"]) != m["functions_sha256"]
        or sha(Path(m["model_dir"]) / "model.safetensors") != m["model_sha256"]
    ):
        raise ValueError("Fixed teacher changed")
    return m


def acquire(out):
    """Capture complete paired units, stopping on replay or resource boundaries."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m, prior = verify(out), read(out / "status.json")
    if prior["status"] == "COMPLETE":
        print("Already complete; no forwards")
        return
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        m["model_dir"], device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    current, previous = observer.manifest(), read(PRIOR / "observer.json")
    for key in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if current[key] != previous[key]:
            raise ValueError(f"Observer changed: {key}")
    write(out / "observer.json", current)
    with np.load(m["functions"], allow_pickle=False) as z:
        fs = {
            q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
            for q in QUERIES
        }
    start, completed = time.monotonic(), 0

    def status(state):
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed": completed,
        }
        write(out / "status.json", value)
        print(value, flush=True)

    def capture(spec, alphabet):
        saved = {}

        def hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            saved["P"] = h[0, spec["mask"]].detach().cpu().numpy().copy()
            saved["Q"] = h[0, -1].detach().cpu().numpy().copy()

        handle = observer.model.model.layers[3].register_forward_hook(hook)
        try:
            h, kw, native = capture_boundary(observer, spec["ids"], alphabet)
            result = (
                saved,
                h[0, spec["mask"]].cpu().numpy().astype(float),
                h[0, -1].cpu().numpy().astype(float),
                native["conditional_score"],
            )
            del h, kw
            return result
        finally:
            handle.remove()

    try:
        for d in read(out / "private_design.json"):
            path = out / "private_packets" / f"pair_{d['index']:03d}.json"
            if path.exists():
                cached = read(path)
                if (
                    cached["author_id"] != d["author_id"]
                    or sha(path.with_suffix(".npz")) != cached["arrays_sha256"]
                ):
                    raise ValueError("Cached pair changed")
                completed += 1
                continue
            if (
                prior["calls"] + observer.calls + 2 + d["repeat"] > m["max_calls"]
                or prior["seconds"] + time.monotonic() - start > m["max_seconds"]
            ):
                status("PAUSED_RESOURCE_BOUNDARY")
                return
            values, arrays = {}, {}
            for order in ("original", "reverse"):
                early, p, q, native = capture(d[order], d["alphabet"])
                values[order] = {**fs[d["query"]].parts(p, q), "native": native}
                arrays.update(
                    {
                        order + "_earlyP": early["P"],
                        order + "_earlyQ": early["Q"],
                        order + "_P": p,
                        order + "_Q": q,
                    }
                )
            error = max(
                abs(values["original"][k] - v) for k, v in d["reference"].items()
            )
            if error > 2e-6:
                raise ValueError(
                    f"Prior endpoint replay failed at pair {d['index']}: {error}"
                )
            values["old_replay_error"] = error
            if d["repeat"]:
                early, p, q, native = capture(d["original"], d["alphabet"])
                repeat_error = max(
                    abs(native - values["original"]["native"]),
                    float(np.max(np.abs(early["P"] - arrays["original_earlyP"]))),
                    float(np.max(np.abs(early["Q"] - arrays["original_earlyQ"]))),
                )
                if repeat_error > 2e-6:
                    raise ValueError("Same-run repeat changed")
                values["repeat_error"] = repeat_error
            np.savez_compressed(path.with_suffix(".npz"), **arrays)
            write(
                path,
                {
                    "author_id": d["author_id"],
                    "query": d["query"],
                    "role": d["role"],
                    **values,
                    "arrays_sha256": sha(path.with_suffix(".npz")),
                },
            )
            completed += 1
            status("RUNNING")
            gc.collect()
            torch.mps.empty_cache()
        status("COMPLETE")
    except Exception:
        status("STOP_ERROR")
        raise


def main():
    """Dispatch explicit prepare/run phases without starting another queue."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "run"))
    p.add_argument("--output-dir", type=Path, default=OUT)
    args = p.parse_args()
    (prepare if args.stage == "prepare" else acquire)(args.output_dir)


if __name__ == "__main__":
    main()
