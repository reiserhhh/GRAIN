#!/usr/bin/env python3
"""Run the frozen no-label MEPS current-response context mechanism pilot locally."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.meps_material import load_text_material
from grain_research.vector_mechanisms.response_context import (
    ResponseObserver,
    context_derangements,
    cosine,
    native_parts,
    paired_interval,
    select_responses,
)
from scripts.audit_grain_meps_material import verify_outputs
from scripts.run_grain_vector_mechanism_controls import write_json

EXPECTED_FILES = {
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def file_hash(path):
    """Hash large local weights without allocating another multi-GB buffer."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(8 * 1024**2):
            h.update(data)
    return h.hexdigest()


def technical_gate(observer):
    """Six nonprivate forwards test numerical repeat, patch and positional invariants."""
    examples = [
        (
            "You can list possible options and consider one small step.",
            "I would write down the options first, then choose a small step that seems manageable today.",
        ),
        (
            "まず選択肢を書き出して、小さな一歩を考えてみてください。",
            "私はまず選択肢を書き出してから、今日できそうな小さな一歩を選んで試してみたいと思います。",
        ),
    ]
    rows = []
    for language, (context, response) in zip(("en", "ja"), examples):
        prefix, body = native_parts(observer.tokenizer, response, context)
        _, alone = native_parts(observer.tokenizer, response)
        if body != alone:
            raise ValueError("Technical response tokens changed")
        fixed = observer.forward(prefix, body)
        compact = observer.forward(prefix, body, compact=True)
        sham = observer.forward(
            prefix, body, patch=fixed["trace"]["block_10"]["states"]
        )
        repeat = float(np.max(abs(sham["token_logp"] - fixed["token_logp"])))
        gap = abs(compact["mean_logp"] - fixed["mean_logp"])
        similarities = {
            s: cosine(fixed["trace"][s]["mean"], compact["trace"][s]["mean"])
            for s in observer.sites
        }
        rows.append(
            {
                "language": language,
                "response_tokens": len(body),
                "self_patch_max_logp_error": repeat,
                "compact_fixed_mean_logp_difference": gap,
                "minimum_site_cosine": min(similarities.values()),
                "passed": repeat == 0
                and gap <= 0.02
                and min(similarities.values()) >= 0.999,
            }
        )
    return rows


def main():
    """Execute a bounded fixed-material context contrast, preserving every failure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--technical-only", action="store_true")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(p for p in out.iterdir() if not p.name.startswith("._")):
        raise ValueError("Fresh execution directory required")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    start = time.monotonic()
    paths = [
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/response_context.py",
        ROOT / "grain_research/vector_mechanisms/meps_material.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "suica_core/meps_ai_conv.py",
        ROOT / "suica_core/meps_layout.py",
        ROOT / "scripts/audit_grain_meps_material.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "docs/GRAIN_VM06_RESPONSE_CONTEXT_PROTOCOL_V1.md",
        ROOT / "tests/test_grain_response_context.py",
    ]
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT))
    try:
        actual = {name: file_hash(a.model_dir / name) for name in EXPECTED_FILES}
        if actual != EXPECTED_FILES:
            raise ValueError("Copied model differs from verified Windows cache")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.backends.mps.is_available():
            raise ValueError("Local MPS backend unavailable; no remote data fallback")
        torch.manual_seed(731)
        torch.set_num_threads(2)
        transformers.utils.logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(
            str(a.model_dir), local_files_only=True, trust_remote_code=False
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(a.model_dir),
                local_files_only=True,
                trust_remote_code=False,
                dtype=getattr(torch, a.dtype),
                attn_implementation="eager",
            )
            .eval()
            .to("mps")
        )
        observer = ResponseObserver(model, tokenizer)
        manifest = {
            "protocol": "GRAIN_VM06_RESPONSE_CONTEXT_V1",
            "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "model_files": actual,
            "template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode()
            ).hexdigest(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "device": "mps",
            "dtype": a.dtype,
            "attention": "eager",
            "use_cache": False,
            "max_seconds": 1800,
            "max_forwards": 282,
            "max_mps_allocated_gib": 9 if a.dtype == "float32" else 7,
            "source_code": {str(x.relative_to(ROOT)): file_hash(x) for x in paths},
            "technical_only": a.technical_only,
            "labels_read": False,
            "raw_text_or_activations_persisted": False,
        }
        manifest["identity"] = material_hash(manifest)
        write_json(out / "manifest.json", manifest)
        technical = technical_gate(observer)
        write_json(out / "technical_gate.json", technical)
        if not all(row["passed"] for row in technical):
            raise ValueError(
                "Technical context instrument gate failed before private text access"
            )
        if a.technical_only:
            write_json(
                out / "summary.json",
                {
                    "status": "COMPLETE_TECHNICAL_ONLY",
                    "forwards": observer.calls,
                    "labels_read": False,
                    "private_text_read": False,
                    "seconds": time.monotonic() - start,
                },
            )
            print(json.dumps(technical), flush=True)
            return
        pre = verify_outputs(a.preflight_dir)
        source_manifest = json.loads((a.preflight_dir / "manifest.json").read_text())
        views, integrity = load_text_material(a.input_root)
        if (
            source_manifest["identity"] != pre["parent_identity"]
            or material_hash(integrity["source_files"])
            != source_manifest["source_identity"]
        ):
            raise ValueError("Changed frozen MEPS source")
        cases, coverage = select_responses(views, tokenizer)
        pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
        if not 4 <= len(cases) <= 46:
            raise ValueError("Insufficient or unexpectedly expanded response material")
        matched, random = context_derangements([c["context_tokens"] for c in cases])
        safe = [
            {k: v for k, v in c.items() if k not in {"response", "context"}}
            | {
                "matched_donor_code": cases[matched[i]]["participant_code"],
                "random_donor_code": cases[random[i]]["participant_code"],
            }
            for i, c in enumerate(cases)
        ]
        write_json(
            out / "private_material_manifest.json",
            {
                "preflight_identity": pre["parent_identity"],
                "cases": safe,
                "identity": material_hash(safe),
            },
        )
        rows, geometry, peak = [], [], 0
        for i, case in enumerate(cases):
            if time.monotonic() - start > 1800:
                raise TimeoutError("Bounded response-context budget reached")
            arms = {
                "actual": case["context"],
                "matched": cases[matched[i]]["context"],
                "random": cases[random[i]]["context"],
                "none": None,
            }
            _, same_body = native_parts(tokenizer, case["response"])
            observed = {}
            for arm, context in arms.items():
                prefix, body = native_parts(tokenizer, case["response"], context)
                if body != same_body:
                    raise ValueError("Body token sequence differs across contexts")
                observed[arm] = observer.forward(prefix, body)
            prefix, body = native_parts(tokenizer, case["response"], arms["matched"])
            sham = observer.forward(
                prefix, body, patch=observed["matched"]["trace"]["block_10"]["states"]
            )
            patched = observer.forward(
                prefix, body, patch=observed["actual"]["trace"]["block_10"]["states"]
            )
            repeat = float(
                np.max(abs(sham["token_logp"] - observed["matched"]["token_logp"]))
            )
            if repeat != 0:
                raise ValueError("Private self-patch did not exactly replay")
            base = observed["actual"]["trace"]["input_embedding"]["mean"]
            if any(
                not np.array_equal(v["trace"]["input_embedding"]["mean"], base)
                for v in observed.values()
            ):
                raise ValueError("Changed current-token input embeddings")
            row = {
                "participant_code": case["participant_code"],
                "response_tokens": len(body),
                "self_patch_max_error": repeat,
            }
            for arm, value in observed.items():
                row[arm + "_mean_logp"] = value["mean_logp"]
                row[arm + "_tail_logp"] = value["tail_logp"]
            row["patch_tail_logp"] = patched["tail_logp"]
            for arm in ("matched", "random", "none"):
                row["actual_minus_" + arm] = (
                    observed["actual"]["mean_logp"] - observed[arm]["mean_logp"]
                )
                for site in observer.sites:
                    h = observed["actual"]["trace"][site]["mean"]
                    g = observed[arm]["trace"][site]["mean"]
                    geometry.append(
                        {
                            "participant_code": case["participant_code"],
                            "site": site,
                            "comparison": arm,
                            "cosine": cosine(h, g),
                            "relative_l2": float(
                                np.linalg.norm(h - g) / max(np.linalg.norm(h), 1e-12)
                            ),
                        }
                    )
            row["patch_minus_matched_tail"] = (
                patched["tail_logp"] - observed["matched"]["tail_logp"]
            )
            row["actual_minus_matched_tail"] = (
                observed["actual"]["tail_logp"] - observed["matched"]["tail_logp"]
            )
            rows.append(row)
            peak = max(peak, torch.mps.current_allocated_memory())
            if (
                peak > manifest["max_mps_allocated_gib"] * 1024**3
                or observer.calls > 282
            ):
                raise ValueError("Fixed GPU/call resource guard exceeded")
            pd.DataFrame(rows).to_csv(out / "private_response_metrics.csv", index=False)
            pd.DataFrame(geometry).to_csv(
                out / "private_geometry_metrics.csv", index=False
            )
            status = {
                "status": "RUNNING",
                "participants": len(rows),
                "planned": len(cases),
                "forwards": observer.calls,
                "mps_peak_sampled_bytes": peak,
                "seconds": time.monotonic() - start,
                "labels_read": False,
            }
            write_json(out / "run_status.json", status)
            print(json.dumps(status), flush=True)
        frame = pd.DataFrame(rows)
        effects = {
            c: paired_interval(frame[c])
            for c in [
                "actual_minus_matched",
                "actual_minus_random",
                "actual_minus_none",
                "patch_minus_matched_tail",
                "actual_minus_matched_tail",
            ]
        }
        status = status | {"status": "COMPLETE"}
        write_json(out / "run_status.json", status)
        result = status | {
            "parent_identity": manifest["identity"],
            "effects": effects,
            "scope": "fixed-material conditional context use, not personality validity",
            "bootstrap_scope": "conditional on responses and fixed donor assignments, not population inference",
            "outputs_sha256": {
                x.name: file_hash(x)
                for x in out.iterdir()
                if x.is_file() and not x.name.startswith("._")
            },
        }
        write_json(out / "summary.json", result)
        print(json.dumps(result), flush=True)
    except Exception as error:
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "error_type": type(error).__name__,
                "error": str(error),
                "seconds": time.monotonic() - start,
            },
        )
        raise


if __name__ == "__main__":
    main()
