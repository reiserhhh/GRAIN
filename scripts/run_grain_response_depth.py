#!/usr/bin/env python3
"""Run fixed-layer state interventions on the already-frozen RC1 responses."""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from itertools import pairwise
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
    native_parts,
    paired_interval,
    select_responses,
)
from scripts.audit_grain_meps_material import verify_outputs
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import (
    EXPECTED_FILES,
    file_hash,
    technical_gate,
)

BLOCKS = (4, 10, 16, 22, 27)
SITES = tuple(f"block_{i:02d}" for i in BLOCKS) + ("final_norm",)


def summarize(frame):
    """Describe fixed-material layer effects without unstable individual ratios."""
    effects = {}
    for site in SITES:
        part = frame.loc[frame.site == site]
        delta = part.patch_tail_logp - part.matched_tail_logp
        gap = part.actual_tail_logp - part.matched_tail_logp
        effects[site] = {
            "patch_minus_matched": paired_interval(delta),
            "actual_minus_matched": paired_interval(gap),
            "actual_minus_patch": paired_interval(gap - delta),
            "ratio_of_group_means": float(delta.mean() / gap.mean())
            if abs(gap.mean()) > 1e-12
            else None,
            "maximum_self_patch_error": float(part.self_patch_max_error.max()),
            "maximum_tail_replay_error": float(part.patch_actual_tail_max_error.max()),
            "maximum_first_token_change": float(part.patch_first_token_change.max()),
        }
    pivot = frame.pivot(
        index="participant_code", columns="site", values="patch_tail_logp"
    )
    adjacent = {
        f"{b}_minus_{a}": paired_interval(pivot[b] - pivot[a])
        for a, b in pairwise(SITES)
    }
    return {"effects": effects, "adjacent_layer_contrasts": adjacent}


def main():
    """Verify material, run positive controls, then all fixed sites with hard guards."""
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model-dir",
        "input-root",
        "preflight-dir",
        "parent-dir",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--technical-only", action="store_true")
    a = p.parse_args()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Fresh output directory required")
    start, peak = time.monotonic(), 0
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source_paths = [
        Path(__file__),
        ROOT / "docs/GRAIN_VM06_RESPONSE_DEPTH_PROTOCOL_V1.md",
        ROOT / "tests/test_grain_response_depth.py",
        ROOT / "tests/test_grain_response_context.py",
        ROOT / "grain_research/vector_mechanisms/response_context.py",
        ROOT / "grain_research/vector_mechanisms/meps_material.py",
        ROOT / "grain_research/vector_mechanisms/controls.py",
        ROOT / "suica_core/meps_ai_conv.py",
        ROOT / "suica_core/meps_layout.py",
        ROOT / "scripts/audit_grain_meps_material.py",
        ROOT / "scripts/run_grain_vector_mechanism_controls.py",
        ROOT / "scripts/run_grain_vector_mechanism_response_context.py",
    ]
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in source_paths:
            z.write(path, path.relative_to(ROOT))
    try:
        parent = verify_outputs(a.parent_dir)
        if parent["status"] != "COMPLETE":
            raise ValueError("RC1 parent must be complete")
        model_hashes = {k: file_hash(a.model_dir / k) for k in EXPECTED_FILES}
        if model_hashes != EXPECTED_FILES:
            raise ValueError("Model identity changed")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable; private text has no remote fallback")
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
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .eval()
            .to("mps")
        )
        if len(model.model.layers) != 28:
            raise ValueError("Unexpected number of decoder blocks")
        observer = ResponseObserver(
            model, tokenizer, blocks=BLOCKS, capture_sites=SITES
        )
        manifest = {
            "protocol": "GRAIN_VM06_RESPONSE_DEPTH_V1",
            "parent_identity": parent["parent_identity"],
            "model_files": model_hashes,
            "template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode()
            ).hexdigest(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "mps",
            "dtype": "float32",
            "attention": "eager",
            "sites": SITES,
            "max_forwards": 656,
            "max_seconds": 2400,
            "max_mps_allocated_gib": 9,
            "technical_only": a.technical_only,
            "labels_read": False,
            "raw_text_or_activations_persisted": False,
            "source_code": {
                str(x.relative_to(ROOT)): file_hash(x) for x in source_paths
            },
        }
        manifest["identity"] = material_hash(manifest)
        write_json(out / "manifest.json", manifest)

        def forward(prefix, body, **kwargs):
            nonlocal peak
            if time.monotonic() - start > 2400 or observer.calls >= 656:
                raise TimeoutError("Frozen execution budget exhausted")
            result = observer.forward(prefix, body, **kwargs)
            peak = max(peak, torch.mps.current_allocated_memory())
            if peak > 9 * 1024**3:
                raise MemoryError("MPS sampled allocation exceeds frozen guard")
            return result

        numerical = technical_gate(observer)
        write_json(out / "technical_numerical_gate.json", numerical)
        if not all(x["passed"] for x in numerical):
            raise ValueError("Numerical gate failed before private text access")
        examples = [
            (
                "en",
                "Please describe the next small step.",
                "Explain a recipe for vegetable soup.",
                "I would write down the options first and try one small step today.",
            ),
            (
                "ja",
                "次にできる小さな一歩を教えてください。",
                "野菜のスープの作り方を説明してください。",
                "私はまず選択肢を書き出して、今日できそうな小さな一歩を試したいと思います。",
            ),
        ]
        controls = []
        for language, actual, mismatch, response in examples:
            pa, body = native_parts(tokenizer, response, actual)
            pm, bm = native_parts(tokenizer, response, mismatch)
            if body != bm:
                raise ValueError("Synthetic current tokens changed")
            va, vm = forward(pa, body), forward(pm, body)
            vp = forward(
                pm,
                body,
                patch_site="final_norm",
                patch=va["trace"]["final_norm"]["states"],
            )
            tail = float(np.max(abs(vp["token_logp"][1:] - va["token_logp"][1:])))
            first = float(abs(vp["token_logp"][0] - vm["token_logp"][0]))
            controls.append(
                {
                    "language": language,
                    "tail_max_error": tail,
                    "first_token_change": first,
                    "passed": tail <= 1e-5 and first == 0,
                }
            )
        write_json(out / "technical_readout_gate.json", controls)
        if not all(x["passed"] for x in controls):
            raise ValueError(
                "Final readout positive control failed before private text"
            )
        if a.technical_only:
            write_json(
                out / "summary.json",
                {
                    "status": "COMPLETE_TECHNICAL_ONLY",
                    "forwards": observer.calls,
                    "private_text_read": False,
                    "labels_read": False,
                    "controls": controls,
                },
            )
            print(json.dumps(controls), flush=True)
            return
        pre = verify_outputs(a.preflight_dir)
        source = json.loads((a.preflight_dir / "manifest.json").read_text())
        views, integrity = load_text_material(a.input_root)
        if (
            source["identity"] != pre["parent_identity"]
            or material_hash(integrity["source_files"]) != source["source_identity"]
        ):
            raise ValueError("Frozen text source changed")
        cases, _ = select_responses(views, tokenizer)
        matched, random = context_derangements([c["context_tokens"] for c in cases])
        safe = [
            {k: v for k, v in c.items() if k not in {"response", "context"}}
            | {
                "matched_donor_code": cases[matched[i]]["participant_code"],
                "random_donor_code": cases[random[i]]["participant_code"],
            }
            for i, c in enumerate(cases)
        ]
        frozen = json.loads(
            (a.parent_dir / "private_material_manifest.json").read_text()
        )
        if safe != frozen["cases"] or material_hash(safe) != frozen["identity"]:
            raise ValueError("RC1 material/assignments did not reproduce exactly")
        write_json(out / "private_material_manifest.json", frozen)
        old = pd.read_csv(a.parent_dir / "private_response_metrics.csv").set_index(
            "participant_code"
        )
        rows, baseline_rows = [], []
        for i, case in enumerate(cases):
            pa, body = native_parts(tokenizer, case["response"], case["context"])
            pm, bm = native_parts(
                tokenizer, case["response"], cases[matched[i]]["context"]
            )
            if body != bm:
                raise ValueError("Current body differs between native templates")
            va, vm = forward(pa, body), forward(pm, body)
            if not np.array_equal(
                va["trace"]["input_embedding"]["mean"],
                vm["trace"]["input_embedding"]["mean"],
            ):
                raise ValueError("Current input embeddings changed")
            code = case["participant_code"]
            replay = max(
                abs(v[key] - old.loc[code, f"{arm}_{key}"])
                for arm, v in (("actual", va), ("matched", vm))
                for key in ("mean_logp", "tail_logp")
            )
            if replay > 1e-5:
                raise ValueError("Unmodified RC1 values failed replay")
            baseline_rows.append(
                {
                    "participant_code": code,
                    "response_tokens": len(body),
                    "actual_tail_logp": va["tail_logp"],
                    "matched_tail_logp": vm["tail_logp"],
                    "parent_replay_max_error": replay,
                }
            )
            for site in SITES:
                sham = forward(
                    pm, body, patch_site=site, patch=vm["trace"][site]["states"]
                )
                vp = forward(
                    pm, body, patch_site=site, patch=va["trace"][site]["states"]
                )
                repeat = float(np.max(abs(sham["token_logp"] - vm["token_logp"])))
                tail = float(np.max(abs(vp["token_logp"][1:] - va["token_logp"][1:])))
                first = float(abs(vp["token_logp"][0] - vm["token_logp"][0]))
                if (
                    repeat != 0
                    or first != 0
                    or (site in {"block_27", "final_norm"} and tail > 1e-5)
                ):
                    raise ValueError("Private self/causal/positive control failed")
                rows.append(
                    {
                        "participant_code": code,
                        "site": site,
                        "actual_tail_logp": va["tail_logp"],
                        "matched_tail_logp": vm["tail_logp"],
                        "patch_tail_logp": vp["tail_logp"],
                        "self_patch_max_error": repeat,
                        "patch_actual_tail_max_error": tail,
                        "patch_first_token_change": first,
                    }
                )
                pd.DataFrame(rows).to_csv(
                    out / "private_depth_metrics.csv", index=False
                )
            pd.DataFrame(baseline_rows).to_csv(
                out / "private_baseline_metrics.csv", index=False
            )
            status = {
                "status": "RUNNING",
                "participants": i + 1,
                "planned": len(cases),
                "forwards": observer.calls,
                "mps_peak_sampled_bytes": peak,
                "seconds": time.monotonic() - start,
                "labels_read": False,
            }
            write_json(out / "run_status.json", status)
            print(json.dumps(status), flush=True)
        status = status | {"status": "COMPLETE"}
        write_json(out / "run_status.json", status)
        result = (
            status
            | summarize(pd.DataFrame(rows))
            | {
                "parent_identity": manifest["identity"],
                "rc1_identity": parent["parent_identity"],
                "scope": "fixed-response body-state sufficiency, not personality validity or exclusive mediation",
                "outputs_sha256": {
                    x.name: file_hash(x)
                    for x in out.iterdir()
                    if x.is_file() and not x.name.startswith("._")
                },
            }
        )
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
                "labels_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
