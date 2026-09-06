#!/usr/bin/env python3
"""Test native context advantage while preserving exact assistant token bags."""

import argparse
import hashlib
import json
import os
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
    select_responses,
)
from grain_research.vector_mechanisms.response_order import (
    MODES,
    SEEDS,
    assistant_span,
    permute_prefix,
    summarize_order,
)
from scripts.audit_grain_meps_material import verify_outputs
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import (
    EXPECTED_FILES,
    file_hash,
    technical_gate,
)


def technical_order(tokenizer):
    """Check native EN/JA boundaries and invertibility without private neural input."""
    records = []
    for lang, context, response in [
        (
            "en",
            "Write down several options, consider the evidence, and then choose a small step.",
            "I will write down the options and choose one small step today.",
        ),
        (
            "ja",
            "選択肢をいくつか書き出して、根拠を確かめてから、小さな一歩を選んでください。",
            "私は選択肢を書き出してから、今日できそうな小さな一歩を選びたいと思います。",
        ),
    ]:
        prefix, body, indices, info = assistant_span(tokenizer, response, context)
        for mode in MODES:
            for seed in SEEDS:
                changed, record = permute_prefix(prefix, indices, lang, mode, seed)
                if (
                    len(changed) != len(prefix)
                    or record["inverse_max_token_error"] != 0
                ):
                    raise ValueError("Synthetic order instrument failed")
                records.append(
                    {
                        "language": lang,
                        "mode": mode,
                        "seed": seed,
                        "response_tokens": len(body),
                        **info,
                        **record,
                    }
                )
    return records


def main():
    """Verify frozen cases and execute every preregistered token-order condition."""
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
        ROOT / "docs/GRAIN_VM06_RESPONSE_ORDER_PROTOCOL_V1.md",
        ROOT / "tests/test_grain_response_order.py",
        ROOT / "grain_research/vector_mechanisms/response_order.py",
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
        for path in paths:
            z.write(path, path.relative_to(ROOT))
    try:
        parent = verify_outputs(a.parent_dir)
        old_manifest = json.loads((a.parent_dir / "manifest.json").read_text())
        if parent["status"] != "COMPLETE":
            raise ValueError("RC1 parent incomplete")
        hashes = {k: file_hash(a.model_dir / k) for k in EXPECTED_FILES}
        if hashes != EXPECTED_FILES:
            raise ValueError("Observer model identity changed")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.backends.mps.is_available():
            raise ValueError("No MPS; private material has no remote fallback")
        torch.manual_seed(731)
        torch.set_num_threads(2)
        transformers.utils.logging.disable_progress_bar()
        tokenizer = AutoTokenizer.from_pretrained(
            str(a.model_dir), local_files_only=True, trust_remote_code=False
        )
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        if template_hash != old_manifest["template_sha256"]:
            raise ValueError("Native template differs from RC1")
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
        observer = ResponseObserver(model, tokenizer)
        manifest = {
            "protocol": "GRAIN_VM06_RESPONSE_ORDER_V1",
            "rc1_identity": parent["parent_identity"],
            "model_files": hashes,
            "template_sha256": template_hash,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "mps",
            "dtype": "float32",
            "attention": "eager",
            "use_cache": False,
            "modes": MODES,
            "seeds": SEEDS,
            "max_forwards": 466,
            "max_seconds": 1800,
            "max_mps_allocated_gib": 9,
            "technical_only": a.technical_only,
            "labels_read": False,
            "raw_text_or_activations_persisted": False,
            "source_code": {str(x.relative_to(ROOT)): file_hash(x) for x in paths},
        }
        manifest["identity"] = material_hash(manifest)
        write_json(out / "manifest.json", manifest)
        numerical = technical_gate(observer)
        write_json(out / "technical_numerical_gate.json", numerical)
        if not all(x["passed"] for x in numerical):
            raise ValueError("Numerical gate failed before private text access")
        instrument = technical_order(tokenizer)
        write_json(out / "technical_order_gate.json", instrument)
        if a.technical_only:
            result = {
                "status": "COMPLETE_TECHNICAL_ONLY",
                "forwards": observer.calls,
                "order_checks": len(instrument),
                "private_text_read": False,
                "labels_read": False,
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
        if safe != frozen["cases"] or material_hash(safe) != frozen["identity"]:
            raise ValueError("Changed RC1 responses or donors")
        write_json(out / "private_material_manifest.json", frozen)
        prepared, meta = [], []
        for i, case in enumerate(cases):
            conditions = {}
            body_same = None
            for role, owner in (("actual", i), ("matched", matched[i])):
                context_code = cases[owner]["participant_code"]
                prefix, body, indices, info = assistant_span(
                    tokenizer, case["response"], cases[owner]["context"]
                )
                if body_same is not None and body != body_same:
                    raise ValueError("Current response IDs differ across contexts")
                body_same = body
                conditions[role + "_native"] = (prefix, body)
                for mode in MODES:
                    for seed in SEEDS:
                        changed, record = permute_prefix(
                            prefix, indices, context_code, mode, seed
                        )
                        conditions[f"{role}_{mode}_{seed}"] = (changed, body)
                        meta.append(
                            {
                                "participant_code": case["participant_code"],
                                "context_code": context_code,
                                "role": role,
                                "mode": mode,
                                "seed": seed,
                                "prefix_tokens": len(prefix),
                                "response_tokens": len(body),
                                **info,
                                **record,
                            }
                        )
            prepared.append(conditions)
        pd.DataFrame(meta).to_csv(out / "private_order_manifest.csv", index=False)
        old = pd.read_csv(a.parent_dir / "private_response_metrics.csv").set_index(
            "participant_code"
        )
        rows = []
        for i, (case, conditions) in enumerate(zip(cases, prepared)):
            code = case["participant_code"]
            body_embedding, replay = None, 0.0
            for condition, (prefix, body) in conditions.items():
                if time.monotonic() - start > 1800 or observer.calls >= 466:
                    raise TimeoutError("Frozen runtime/forward budget exhausted")
                value = observer.forward(prefix, body)
                peak = max(peak, torch.mps.current_allocated_memory())
                if peak > 9 * 1024**3:
                    raise MemoryError("Frozen MPS memory guard exceeded")
                current = value["trace"]["input_embedding"]["mean"]
                if body_embedding is not None and not np.array_equal(
                    current, body_embedding
                ):
                    raise ValueError(
                        "Current input embedding changed under permutation"
                    )
                body_embedding = current
                if condition.endswith("_native"):
                    role = condition.split("_")[0]
                    error = max(
                        abs(value[k] - old.loc[code, f"{role}_{k}"])
                        for k in ("mean_logp", "tail_logp")
                    )
                    replay = max(replay, error)
                    if error > 1e-5:
                        raise ValueError("RC1 native scores failed replay")
                rows.append(
                    {
                        "participant_code": code,
                        "condition": condition,
                        "mean_logp": value["mean_logp"],
                        "tail_logp": value["tail_logp"],
                        "response_tokens": len(body),
                        "input_embedding_max_error": 0.0,
                    }
                )
            pd.DataFrame(rows).to_csv(out / "private_order_metrics.csv", index=False)
            status = {
                "status": "RUNNING",
                "participants": i + 1,
                "planned": len(cases),
                "forwards": observer.calls,
                "mps_peak_sampled_bytes": peak,
                "last_parent_replay_max_error": replay,
                "seconds": time.monotonic() - start,
                "labels_read": False,
            }
            write_json(out / "run_status.json", status)
            print(json.dumps(status), flush=True)
        status = status | {"status": "COMPLETE"}
        write_json(out / "run_status.json", status)
        result = status | {
            "parent_identity": manifest["identity"],
            "rc1_identity": parent["parent_identity"],
            "effects": summarize_order(pd.DataFrame(rows)),
            "scope": "fixed-material token-order context use; not semantic or personality validity",
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
                "labels_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
