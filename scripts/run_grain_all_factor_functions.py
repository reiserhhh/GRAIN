#!/usr/bin/env python3
"""M28: resumable seven-query extension of the frozen M7/M26 construction."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.context_carrier import split_context_state
from grain_research.vector_mechanisms.direct_traits import AXES, BIG5, candidates
from grain_research.vector_mechanisms.factor_functions import joint_gradient
from grain_research.vector_mechanisms.full_state_edit import (
    align_excerpts,
    align_states,
    capture_boundary,
)
from grain_research.vector_mechanisms.query_functional import (
    integrated_matrix,
    path_state,
    region_gradient,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_multiquery_sensitivity import load_panel
from scripts.run_grain_level_response import question_mask
from scripts.run_grain_real_author_path import MODEL

SOURCE = BASE / "multiquery_sensitivity_s1_r1"
M25 = BASE / "shared_author_profiles_m25_v1"
M7 = BASE / "query_region_functional_v1"
M8 = BASE / "complete_level_response_v1"
NEW = tuple(q for q in (*BIG5, *AXES) if q not in {"TF_cont", "neuroticism"})


def prepare(out):
    """Freeze all tokenized material and reference roles before neural inference."""
    _, records = load_panel(SOURCE)
    reference = {
        r["author_id"]
        for r in read(M7 / "private_design.json")
        if r["role"] == "reference"
    }
    old = read(M25 / "private_design.json")
    roles = {d["author_id"]: d["role"] for d in old}
    if (
        len(reference) != 2
        or len(roles) != 24
        or not reference <= {u for u, r in roles.items() if r == "reference"}
    ):
        raise ValueError(
            "Original two-function-reference and 12/12 profile roles required"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("private_packets", "private_nodes"):
        (out / sub).mkdir()
    design, tokens = [], {}
    for q in NEW:
        for row, _, _ in records:
            uid, view = row["author_id"], row["view"]
            full, fs, fe, fm = question_mask(tokenizer, row["texts"], q)
            qids = np.asarray(fe["input_ids"])[fm].tolist()
            if q in tokens and tokens[q] != qids:
                raise ValueError("Query token identities vary by author")
            tokens[q] = qids
            for condition in ("full", "e1", "e2"):
                kept = [
                    j
                    for j in range(len(row["texts"]))
                    if condition == "full" or j not in row[condition]
                ]
                text, spans, encoded, mask = question_mask(
                    tokenizer, [row["texts"][j] for j in kept], q
                )
                ids = np.asarray(encoded["input_ids"])
                if len(ids) > 3056 or ids[mask].tolist() != qids:
                    raise ValueError(
                        "Question changed or frozen observer input too long"
                    )
                is_ref = uid in reference and view == 0
                mapping = None
                if is_ref and condition == "e1":
                    mapping, _ = align_excerpts(
                        full, text, fs, spans, kept, fe, encoded
                    )
                    if not np.array_equal(mapping[mask], np.flatnonzero(fm)):
                        raise ValueError("Reference question alignment failed")
                    mapping = mapping.tolist()
                design.append(
                    {
                        "index": len(design),
                        "author_id": uid,
                        "view": view,
                        "query": q,
                        "condition": condition,
                        "role": roles[uid],
                        "function_reference": is_ref,
                        "input_ids": ids.tolist(),
                        "question_mask": mask.tolist(),
                        "mapping": mapping,
                        "alphabet": candidates(q, tokenizer),
                    }
                )
    write(out / "private_design.json", design)
    sources = (
        SOURCE / "private_material.json",
        M25 / "manifest.json",
        M25 / "private_design.json",
        M7 / "private_design.json",
        M8 / "reference.npz",
        M8 / "observer.json",
    )
    code = (
        "scripts/run_grain_all_factor_functions.py",
        "grain_research/vector_mechanisms/factor_functions.py",
    )
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_ALL_FACTOR_FUNCTIONS_M28_V1",
            "new_queries": list(NEW),
            "units": len(design),
            "reference_nodes": 98,
            "per_run_seconds": 1200,
            "human_labels_read": False,
            "edits": "legacy_three_query_common_edits",
            "question_tokens": tokens,
            "input_sha256": sha(out / "private_design.json"),
            "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
            "code_sha256": {p: sha(ROOT / p) for p in code},
            "plan_sha256": sha(ROOT / "docs/GRAIN_ALL_FACTOR_FUNCTIONS_M28.md"),
            "model_sha256": sha(MODEL / "model.safetensors"),
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "calls": 0, "seconds": 0, "completed_units": 0},
    )
    print(
        {"status": "PREPARED", "new_queries": list(NEW), "units": len(design)},
        flush=True,
    )


def collect(out):
    """Save every bounded query unit; never regenerate completed observations."""
    m, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        print({"status": "COMPLETE", "new_calls": 0}, flush=True)
        return
    if sha(out / "private_design.json") != m["input_sha256"]:
        raise ValueError("Design changed")
    for p, digest in {**m["code_sha256"], **m["inputs_sha256"]}.items():
        if sha(ROOT / p) != digest:
            raise ValueError("Frozen implementation/source changed")
    if sha(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Observer weights changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    observed, expected = observer.manifest(), read(M8 / "observer.json")
    for k in (
        "torch",
        "transformers",
        "dtype",
        "device",
        "attention",
        "template_sha256",
        "pad_to_length",
    ):
        if observed[k] != expected[k]:
            raise ValueError(f"Observer changed: {k}")
    write(out / "observer.json", observed)
    design, start, complete = read(out / "private_design.json"), time.monotonic(), 0

    def status(state):
        """Write atomic-unit progress without private identifiers."""
        value = {
            "status": state,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "completed_units": complete,
            "human_labels_read": False,
        }
        write(out / "run_status.json", value)
        return value

    def check_time():
        """Pause at a saved unit boundary, not an unbounded background queue."""
        if time.monotonic() - start >= m["per_run_seconds"]:
            raise TimeoutError("bounded neural packet reached")

    try:
        for d in design:
            path = out / "private_packets" / f"cell_{d['index']:04d}.json"
            if path.exists():
                packet = read(path)
                if (
                    packet["array_sha256"] != sha(path.with_suffix(".npz"))
                    or packet["index"] != d["index"]
                ):
                    raise ValueError("Completed unit changed")
                complete += 1
                continue
            check_time()
            state, kwargs, native = capture_boundary(
                observer, d["input_ids"], d["alphabet"]
            )
            mask = np.asarray(d["question_mask"], bool)
            array = {"P": state[0, mask].cpu().numpy(), "Q": state[0, -1].cpu().numpy()}
            if d["function_reference"] and d["condition"] in {"full", "e1"}:
                array["state"] = state[0].cpu().numpy()
                if d["condition"] == "full":
                    _, G, g = joint_gradient(
                        observer, state, mask, kwargs, d["alphabet"]
                    )
                    array.update(G_origin=G, g=g)
            np.savez_compressed(path.with_suffix(".npz"), **array)
            write(
                path,
                {
                    k: d[k]
                    for k in (
                        "index",
                        "author_id",
                        "view",
                        "query",
                        "condition",
                        "role",
                        "function_reference",
                    )
                }
                | {
                    "native_score": native["conditional_score"],
                    "candidate_mass": native["candidate_mass"],
                    "array_sha256": sha(path.with_suffix(".npz")),
                },
            )
            complete += 1
            if complete % 24 == 0:
                print(status("COLLECTING"), flush=True)
            del state, kwargs, array
        result = {}
        for q in NEW:
            values = []
            for d in (
                d
                for d in design
                if d["query"] == q
                and d["function_reference"]
                and d["condition"] == "e1"
            ):
                f = next(
                    f
                    for f in design
                    if f["author_id"] == d["author_id"]
                    and f["view"] == 0
                    and f["query"] == q
                    and f["condition"] == "full"
                )
                fp = out / "private_packets" / f"cell_{f['index']:04d}"
                ep = out / "private_packets" / f"cell_{d['index']:04d}"
                with np.load(fp.with_suffix(".npz"), allow_pickle=False) as z:
                    za = dict(z)
                with np.load(ep.with_suffix(".npz"), allow_pickle=False) as z:
                    zb = dict(z)
                device = next(observer.model.parameters()).device
                full = torch.tensor(za["state"][None], device=device)
                edited = torch.tensor(zb["state"][None], device=device)
                aligned = align_states(full, edited, d["mapping"])
                from grain_research.vector_mechanisms.context_carrier import (
                    cached_suffix_kwargs,
                )

                kwargs = cached_suffix_kwargs(observer.model, edited)
                mask = np.asarray(d["question_mask"], bool)
                gradients = []
                for e in (0, 1):
                    context = (
                        aligned
                        if e == 0
                        else split_context_state(aligned, edited, mask, "E")
                    )
                    gs = []
                    for n, t in enumerate((0, 0.5, 1)):
                        node = (
                            out
                            / "private_nodes"
                            / f"cell_{d['index']:04d}_e{e}_n{n}.json"
                        )
                        if node.exists():
                            if (
                                sha(node.with_suffix(".npz"))
                                != read(node)["array_sha256"]
                            ):
                                raise ValueError("Reference node changed")
                            with np.load(
                                node.with_suffix(".npz"), allow_pickle=False
                            ) as z:
                                G = z["G"].copy()
                        else:
                            check_time()
                            score, G = region_gradient(
                                observer,
                                context,
                                path_state(aligned[:, mask], edited[:, mask], t),
                                mask,
                                kwargs,
                                d["alphabet"],
                            )
                            np.savez_compressed(node.with_suffix(".npz"), G=G)
                            write(
                                node,
                                {
                                    "score": score,
                                    "array_sha256": sha(node.with_suffix(".npz")),
                                },
                            )
                        gs.append(G)
                    gradients.append(gs)
                values.append(
                    {
                        "Z": za["P"],
                        "h": za["Q"],
                        "G": integrated_matrix(gradients),
                        "g": za["g"],
                        "b": read(fp.with_suffix(".json"))["native_score"],
                    }
                )
            if len(values) != 2:
                raise ValueError(
                    "Expected two independently fixed functional reference authors"
                )
            for name in ("Z", "h", "G", "g", "b"):
                result[f"{q}__{name}"] = np.mean(
                    [v[name] for v in values], axis=0, dtype=np.float64
                )
            print({"reference_function_complete": q}, flush=True)
        np.savez_compressed(out / "reference.npz", **result)
        write(
            out / "reference_manifest.json",
            {
                "array_sha256": sha(out / "reference.npz"),
                "construction": "two_author_M7_symmetric_P_paths_plus_full_Q_gradient",
                "queries": list(NEW),
            },
        )
        print(status("COMPLETE"), flush=True)
    except TimeoutError:
        print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
    except Exception:
        status("ERROR")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else collect)(args.output_dir)
