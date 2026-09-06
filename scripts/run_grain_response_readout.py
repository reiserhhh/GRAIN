#!/usr/bin/env python3
"""Run bounded final-readout geometry on the original RC1 response pairs."""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.meps_material import load_text_material
from grain_research.vector_mechanisms.response_context import (
    ResponseObserver,
    context_derangements,
    native_parts,
    select_responses,
)
from grain_research.vector_mechanisms.response_readout import pair_readout, summarize
from scripts.audit_grain_meps_material import verify_outputs
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import (
    EXPECTED_FILES,
    file_hash,
    technical_gate,
)


def main():
    """Check model/source identity and exact algebra before describing geometry."""
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
    paths = [
        Path(__file__),
        ROOT / "docs/GRAIN_VM06_RESPONSE_READOUT_PROTOCOL_V1.md",
        ROOT / "tests/test_grain_response_readout.py",
        *[
            ROOT / f"grain_research/vector_mechanisms/{x}.py"
            for x in (
                "response_readout",
                "response_context",
                "meps_material",
                "controls",
            )
        ],
        *[ROOT / f"suica_core/{x}.py" for x in ("meps_ai_conv", "meps_layout")],
        *[
            ROOT / f"scripts/{x}.py"
            for x in (
                "audit_grain_meps_material",
                "run_grain_vector_mechanism_controls",
                "run_grain_vector_mechanism_response_context",
            )
        ],
    ]
    with zipfile.ZipFile(out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(ROOT))
    try:
        parent = verify_outputs(a.parent_dir)
        if parent["status"] != "COMPLETE":
            raise ValueError("RC1 parent must be complete")
        hashes = {k: file_hash(a.model_dir / k) for k in EXPECTED_FILES}
        if hashes != EXPECTED_FILES:
            raise ValueError("Changed model file identity")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.backends.mps.is_available():
            raise ValueError("Local MPS unavailable; no private remote fallback")
        torch.manual_seed(731)
        torch.set_num_threads(2)
        transformers.utils.logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(
            str(a.model_dir), local_files_only=True, trust_remote_code=False
        )
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        if (
            template_hash
            != json.loads((a.parent_dir / "manifest.json").read_text())[
                "template_sha256"
            ]
        ):
            raise ValueError("Changed native template")
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
        observer = ResponseObserver(
            model, tokenizer, capture_sites=("block_10", "final_norm")
        )
        manifest = {
            "protocol": "GRAIN_VM06_RESPONSE_READOUT_V1",
            "rc1_identity": parent["parent_identity"],
            "model_files": hashes,
            "template_sha256": template_hash,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "mps",
            "dtype": "float32",
            "attention": "eager",
            "use_cache": False,
            "readout_reduction_dtype": "float64",
            "head_batch_rows": 8,
            "max_forwards": 110,
            "max_seconds": 1800,
            "max_mps_allocated_gib": 9,
            "technical_only": a.technical_only,
            "labels_read": False,
            "raw_text_or_activations_persisted": False,
            "source_code": {str(x.relative_to(ROOT)): file_hash(x) for x in paths},
        }
        manifest["identity"] = material_hash(manifest)
        write_json(out / "manifest.json", manifest)

        def guard():
            nonlocal peak
            peak = max(peak, torch.mps.current_allocated_memory())
            if (
                peak > 9 * 1024**3
                or time.monotonic() - start > 1800
                or observer.calls > 110
            ):
                raise RuntimeError("Frozen readout resource budget exceeded")

        def pair(pa, pm, body):
            guard()
            va, vm = observer.forward(pa, body), observer.forward(pm, body)
            rows, checks = pair_readout(model, va, vm, body)
            guard()
            return rows, checks, va, vm

        numerical = technical_gate(observer)
        write_json(out / "technical_numerical_gate.json", numerical)
        if not all(x["passed"] for x in numerical):
            raise ValueError("Numerical gate failed before private access")
        tech, tech_checks = [], []
        examples = (
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
        )
        for lang, actual, mismatch, response in examples:
            pa, body = native_parts(tokenizer, response, actual)
            pm, bm = native_parts(tokenizer, response, mismatch)
            if body != bm:
                raise ValueError("Technical body mismatch")
            rows, checks, _, _ = pair(pa, pm, body)
            tech.extend({"language": lang, **x} for x in rows)
            tech_checks.append(
                {"language": lang, "response_tokens": len(body), **checks}
            )
        pd.DataFrame(tech).to_csv(out / "technical_readout_metrics.csv", index=False)
        write_json(out / "technical_readout_checks.json", tech_checks)
        if a.technical_only:
            result = {
                "status": "COMPLETE_TECHNICAL_ONLY",
                "forwards": observer.calls,
                "head_only_batches": sum(x["head_batches"] for x in tech_checks),
                "private_text_read": False,
                "labels_read": False,
                "checks": tech_checks,
            }
            write_json(out / "summary.json", result)
            print(json.dumps(result), flush=True)
            return
        pre = verify_outputs(a.preflight_dir)
        source = json.loads((a.preflight_dir / "manifest.json").read_text())
        views, integrity = load_text_material(a.input_root)
        if (
            source["identity"] != pre["parent_identity"]
            or material_hash(integrity["source_files"]) != source["source_identity"]
        ):
            raise ValueError("Frozen MEPS source changed")
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
        if (
            safe != frozen["cases"]
            or material_hash(safe) != frozen["identity"]
            or len(cases) != 44
        ):
            raise ValueError("Changed fixed replies/assignments")
        write_json(out / "private_material_manifest.json", frozen)
        old = pd.read_csv(a.parent_dir / "private_response_metrics.csv").set_index(
            "participant_code"
        )
        all_rows, all_checks = [], []
        for i, case in enumerate(cases):
            pa, body = native_parts(tokenizer, case["response"], case["context"])
            pm, bm = native_parts(
                tokenizer, case["response"], cases[matched[i]]["context"]
            )
            if body != bm:
                raise ValueError("Changed private response IDs")
            rows, checks, va, vm = pair(pa, pm, body)
            code = case["participant_code"]
            replay = max(
                abs(v[k] - old.loc[code, f"{role}_{k}"])
                for role, v in (("actual", va), ("matched", vm))
                for k in ("mean_logp", "tail_logp")
            )
            if replay > 1e-5:
                raise ValueError("Native RC1 scores did not replay")
            all_rows.extend({"participant_code": code, **r} for r in rows)
            all_checks.append(
                {
                    "participant_code": code,
                    "response_tokens": len(body),
                    "parent_replay_max_error": replay,
                    **checks,
                }
            )
            pd.DataFrame(all_rows).to_csv(
                out / "private_readout_metrics.csv", index=False
            )
            write_json(out / "private_readout_checks.json", all_checks)
            status = {
                "status": "RUNNING",
                "participants": i + 1,
                "planned": 44,
                "forwards": observer.calls,
                "head_only_batches": sum(
                    x["head_batches"] for x in (*tech_checks, *all_checks)
                ),
                "mps_peak_sampled_bytes": peak,
                "seconds": time.monotonic() - start,
                "labels_read": False,
            }
            write_json(out / "run_status.json", status)
            print(json.dumps(status), flush=True)
        persons, effects = summarize(pd.DataFrame(all_rows), all_checks)
        persons.to_csv(out / "private_person_readout_means.csv")
        status = status | {"status": "COMPLETE"}
        write_json(out / "run_status.json", status)
        result = status | {
            "parent_identity": manifest["identity"],
            "rc1_identity": parent["parent_identity"],
            "tail_token_rows": len(all_rows),
            "effects": effects,
            "scope": "exact model readout geometry, not personality validity or a learned score",
            "outputs_sha256": {
                x.name: file_hash(x)
                for x in out.iterdir()
                if x.is_file() and not x.name.startswith("._")
            },
        }
        write_json(out / "summary.json", result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "participants",
                        "forwards",
                        "head_only_batches",
                        "seconds",
                        "parent_identity",
                    )
                }
            ),
            flush=True,
        )
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
