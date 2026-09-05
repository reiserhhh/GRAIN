#!/usr/bin/env python3
"""Frozen capability comparison: existing Ollama observer or pinned whitebox Qwen."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.capability import (
    capability_cases,
    parse_binary_answer,
)
from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from scripts.run_grain_vector_mechanism_controls import write_json

DIGEST = "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def request_json(endpoint, route, payload=None, timeout=180):
    """One bounded local request; an uncertain timeout is never automatically retried."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if "error" in value:
        raise RuntimeError(str(value["error"]))
    return value


def ollama_identity(endpoint):
    """Verify the exact installed model rather than fetching or trusting a tag."""
    tags = request_json(endpoint, "/api/tags", timeout=8)
    found = [r for r in tags["models"] if r["name"] == "llama3.1:8b"]
    if len(found) != 1 or found[0]["digest"] != DIGEST:
        raise ValueError("Installed second observer does not match frozen digest")
    show = request_json(endpoint, "/api/show", {"model": "llama3.1:8b"}, timeout=8)
    return {
        "model": "llama3.1:8b",
        "digest": DIGEST,
        "details": found[0]["details"],
        "version": request_json(endpoint, "/api/version", timeout=8),
        "template_sha256": material_hash(show.get("template")),
        "template": show.get("template"),
        "model_info": show.get("model_info"),
        "default_parameters": show.get("parameters"),
        "runtime_positions_unobserved": True,
    }


def greedy_answer(observer, ids):
    """Use complete-vocabulary argmax while also recording the restricted first margin."""
    eos = observer.tokenizer.eos_token_id
    yes, no = [
        observer.tokenizer.encode(x, add_special_tokens=False) for x in ("Yes", "No")
    ]
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Frozen Qwen binary candidates changed")
    generated, margin, first_token = [], None, None
    for step in range(16):
        _, logits = observer.capture(ids + generated)
        token = int(np.argmax(logits))
        if step == 0:
            margin = float(logits[yes[0]] - logits[no[0]])
            first_token = token
        if token == eos:
            break
        generated.append(token)
    return {
        "content": observer.tokenizer.decode(generated, skip_special_tokens=True),
        "restricted_margin": margin,
        "first_token_id": first_token,
        "generated_token_ids": generated,
        "generated_tokens": len(generated),
        "eos_reached": token == eos,
        "prompt_tokens": len(ids),
    }


def main():
    """Run 208 fixed conditions plus eight repeats without natural text or labels."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("ollama", "qwen"), required=True)
    p.add_argument("--endpoint", default="http://192.168.1.30:11434")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, out / "source_snapshot.zip")
    cases = capability_cases()
    observer = None
    if a.backend == "ollama":
        identity = ollama_identity(a.endpoint)
        ps = request_json(a.endpoint, "/api/ps", timeout=8)
        if any(m["name"] != "llama3.1:8b" for m in ps["models"]):
            raise RuntimeError(
                "Other loaded model: do not evict another project workload"
            )
        identity["ps_before"] = ps
    else:
        import torch

        from grain_research.vector_mechanisms.trace import DecoderObserver

        torch.manual_seed(731)
        torch.set_num_threads(4)
        observer = DecoderObserver.load(
            "Qwen/Qwen2.5-1.5B-Instruct",
            revision=REVISION,
            device="cuda",
            dtype="bfloat16",
            max_tokens=1552,
            pad_to_length=1536,
            local_files_only=True,
        )
        identity = observer.manifest()
        for c in cases:
            c["input_ids"] = observer.prompt_ids(c["system"], c["user"])
            if len(c["input_ids"]) + 16 > 1536:
                raise ValueError("Generation exceeds frozen execution budget")
    manifest = {
        "protocol": "GRAIN_SECOND_OBSERVER_CAPABILITY_V1",
        "protocol_commit": "cd99ee7",
        "backend": a.backend,
        "observer": identity,
        "cases": cases,
        "max_seconds": 1200,
        "code_sha256": code_fingerprint(ROOT),
        "decode": {
            "temperature": 0,
            "seed": 731,
            "max_new_tokens": 16,
            "context": 1536,
            "keep_alive": "2m",
        },
        "labels": "synthetic text facts only",
        "natural_text_used": False,
    }
    manifest["identity"] = material_hash(manifest)
    write_json(out / "manifest.json", manifest)
    rows, payloads = [], []
    stage = "RUNNING"
    start = time.monotonic()
    write_json(out / "run_status.json", {"status": stage, "completed": 0})
    try:
        for index, case in enumerate(cases):
            if time.monotonic() - start > 1200:
                stage = "PARTIAL_BUDGET"
                break
            tick = time.monotonic()
            if a.backend == "ollama":
                response = request_json(
                    a.endpoint,
                    "/api/chat",
                    {
                        "model": "llama3.1:8b",
                        "messages": [
                            {"role": "system", "content": case["system"]},
                            {"role": "user", "content": case["user"]},
                        ],
                        "stream": False,
                        "keep_alive": "2m",
                        "options": {
                            "temperature": 0,
                            "seed": 731,
                            "num_predict": 16,
                            "num_ctx": 1536,
                        },
                    },
                )
                if response.get("done") is not True:
                    raise RuntimeError("Incomplete non-streaming reply")
                result = {
                    "content": response.get("message", {}).get("content", ""),
                    "generated_tokens": response.get("eval_count"),
                    "prompt_tokens": response.get("prompt_eval_count"),
                    "eos_reached": response.get("done_reason") != "length",
                }
                payloads.append(response)
            else:
                result = greedy_answer(observer, case["input_ids"])
                payloads.append(result)
            parsed = parse_binary_answer(result["content"])
            row = {
                k: v
                for k, v in case.items()
                if k not in ("system", "user", "input_ids")
            }
            row.update(result)
            row.update(
                {
                    "parsed_yes": parsed,
                    "parse_ok": parsed is not None,
                    "correct": parsed is not None and parsed == case["expected_yes"],
                    "seconds": time.monotonic() - tick,
                }
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "responses.csv", index=False)
            write_json(out / "raw_responses.json", payloads)
            write_json(
                out / "run_status.json",
                {
                    "status": "RUNNING",
                    "completed": len(rows),
                    "seconds": time.monotonic() - start,
                },
            )
            if (index + 1) % 8 == 0:
                print(f"CAPABILITY {a.backend} {index + 1}/{len(cases)}", flush=True)
        else:
            stage = "COMPLETE"
        if a.backend == "ollama":
            after = ollama_identity(a.endpoint)
            if (
                after["digest"] != identity["digest"]
                or after["template_sha256"] != identity["template_sha256"]
            ):
                raise RuntimeError("Observer changed during run")
            write_json(
                out / "ps_after.json", request_json(a.endpoint, "/api/ps", timeout=8)
            )
    except Exception as exc:
        write_json(
            out / "run_status.json",
            {
                "status": "UNCERTAIN_REQUEST_OR_FAILED_RUNTIME",
                "completed": len(rows),
                "error": type(exc).__name__ + ": " + str(exc),
                "automatic_retry": False,
            },
        )
        raise
    status = {
        "status": stage,
        "completed": len(rows),
        "seconds": time.monotonic() - start,
        "unique_cases": len({r["case_id"] for r in rows}),
        "generated_tokens": sum(r.get("generated_tokens") or 0 for r in rows),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in rows),
        "whitebox_forwards": observer.calls if observer else None,
        "psychological_labels": False,
    }
    write_json(out / "run_status.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
