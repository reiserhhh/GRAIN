#!/usr/bin/env python3
"""Run bounded Qwen source-context/order functions on frozen Essays/X cases."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.domain_context import user_parts
from grain_research.vector_mechanisms.response_context import (
    ResponseObserver,
    cosine,
    paired_interval,
)
from grain_research.vector_mechanisms.response_order import permute_prefix
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_response_context import (
    EXPECTED_FILES,
    file_hash,
)

CONDITIONS = ("actual", "matched", "random", "none", "actual_block8", "matched_block8")


def summarize(frame):
    """Keep language/corpus separate and report all frozen function endpoints."""
    output = []
    for (corpus, language), group in frame.groupby(["corpus", "language"], sort=True):
        for score in ("tail_logp", "mean_logp"):
            p = group.pivot(index="case_code", columns="condition", values=score)
            if set(p.columns) != set(CONDITIONS) or p.isna().any().any():
                raise ValueError("Incomplete domain condition family")
            natural = p.actual - p.matched
            differences = {
                "actual_minus_matched": natural,
                "actual_minus_random": p.actual - p.random,
                "actual_minus_none": p.actual - p.none,
                "block8_actual_minus_matched": p.actual_block8 - p.matched_block8,
                "order_interaction": natural - (p.actual_block8 - p.matched_block8),
                "common_block8_penalty": (
                    (p.actual - p.actual_block8) + (p.matched - p.matched_block8)
                )
                / 2,
            }
            for endpoint, values in differences.items():
                output.append(
                    {
                        "corpus": corpus,
                        "language": language,
                        "score": score,
                        "endpoint": endpoint,
                        **paired_interval(values),
                    }
                )
    return pd.DataFrame(output)


def public_gate(observer):
    """New user/user template numerical checks precede private neural extraction."""
    examples = [
        (
            "en",
            "Earlier I walked around the park and observed the trees near a quiet pond.",
            "Later I wrote a short description of the trees and the quiet path next to the water.",
        ),
        (
            "ja",
            "公園を散歩しながら、静かな池の近くにある木々を観察しました。",
            "そのあと、木々の様子や池のそばの静かな道について、短い文章を書きました。",
        ),
    ]
    rows = []
    for language, context, current in examples:
        prefix, body, ix, _ = user_parts(observer.tokenizer, current, context)
        _, alone, _, _ = user_parts(observer.tokenizer, current)
        assert body == alone
        altered, meta = permute_prefix(prefix, ix, language, "block8", 731)
        assert sorted(altered) == sorted(prefix)
        actual = observer.forward(prefix, body)
        compact = observer.forward(prefix, body, compact=True)
        sham = observer.forward(
            prefix, body, patch=actual["trace"]["block_10"]["states"]
        )
        repeat = float(np.max(abs(sham["token_logp"] - actual["token_logp"])))
        gap = abs(compact["mean_logp"] - actual["mean_logp"])
        similarity = min(
            cosine(actual["trace"][s]["mean"], compact["trace"][s]["mean"])
            for s in observer.sites
        )
        rows.append(
            {
                "language": language,
                "self_patch_max_logp_error": repeat,
                "compact_fixed_logp_difference": gap,
                "minimum_site_cosine": similarity,
                "outside_context_changed": meta["outside_span_changed"],
                "inverse_token_error": meta["inverse_max_token_error"],
                "passed": repeat == 0 and gap <= 0.02 and similarity >= 0.999,
            }
        )
    return {
        "passed": all(r["passed"] for r in rows),
        "checks": rows,
        "model_forwards": observer.calls,
    }


def main():
    """Save per-case full conditions with immutable model/material provenance."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--technical-only", action="store_true")
    a = p.parse_args()
    out = a.output_dir.resolve()
    if platform.system() != "Darwin" or not out.is_relative_to(ROOT / "results"):
        raise ValueError("This packet uses the local Mac and ignored results only")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Use a fresh neural output directory")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    start = time.monotonic()
    completed = 0
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(2)
        torch.manual_seed(731)
        if not torch.backends.mps.is_available():
            raise ValueError("Frozen MPS device unavailable")
        hashes = {name: file_hash(a.model_dir / name) for name in EXPECTED_FILES}
        if hashes != EXPECTED_FILES:
            raise ValueError("Frozen observer files changed")
        material = json.loads((a.material_dir / "manifest.json").read_text())
        if (
            material["identity"]
            != material_hash({k: v for k, v in material.items() if k != "identity"})
            or material["status"] != "COMPLETE_DOMAIN_CONTEXT_MATERIAL"
        ):
            raise ValueError("Unsealed or incomplete material manifest")
        paths = [
            Path(__file__),
            ROOT / "scripts/prepare_grain_domain_context.py",
            ROOT / "grain_research/vector_mechanisms/domain_context.py",
            ROOT / "grain_research/vector_mechanisms/response_context.py",
            ROOT / "grain_research/vector_mechanisms/response_order.py",
            ROOT / "grain_research/vector_mechanisms/controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_response_context.py",
            ROOT / "docs/GRAIN_VM06_DOMAIN_CONTEXT_V1.md",
            ROOT / "tests/test_grain_domain_context.py",
        ]
        manifest = {
            "protocol": "GRAIN_VM06_DOMAIN_CONTEXT_V1",
            "material_identity": material["identity"],
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "model_files": hashes,
            "conditions": list(CONDITIONS),
            "dtype": "float32",
            "device": "mps",
            "fixed_body_origin": 768,
            "fixed_physical_tokens": 1024,
            "context_role": "user",
            "current_role": "user",
            "shuffle": "exact_token_block8_seed731",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "max_seconds": 7200,
            "max_mps_allocated_bytes": 9 * 1024**3,
            "labels_read": False,
            "new_generation": False,
            "technical_only": a.technical_only,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "code": {str(x.relative_to(ROOT)): file_hash(x) for x in paths},
        }
        manifest["identity"] = material_hash(manifest)
        write_json(out / "manifest.json", manifest)
        with zipfile.ZipFile(
            out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
        ) as z:
            for path in paths:
                z.write(path, path.relative_to(ROOT))
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
            .to("mps")
            .eval()
        )
        observer = ResponseObserver(model, tokenizer)
        technical = public_gate(observer)
        write_json(out / "technical_checks.json", technical)
        if not technical["passed"]:
            raise ValueError("Public domain template gate failed")
        if a.technical_only:
            write_json(
                out / "summary.json", {"status": "COMPLETE_TECHNICAL_ONLY", **technical}
            )
            print(json.dumps(technical), flush=True)
            return
        observer.capture_sites = set()
        cases = json.loads((a.material_dir / "private_cases.json").read_text())
        if (
            material_hash(cases) != material["case_material_sha256"]
            or len(cases) != material["cases"]
        ):
            raise ValueError("Changed frozen domain cases")
        lookup = {c["case_code"]: c for c in cases}
        if len(lookup) != len(cases):
            raise ValueError("Duplicate case code")
        directory = out / "private_cases"
        directory.mkdir()
        rows, peak = [], 0
        for case in cases:
            code = case["case_code"]
            arrays, metadata = {}, {}
            reference_body = None
            for condition in CONDITIONS:
                donor = (
                    case
                    if condition.startswith("actual")
                    else lookup[
                        case["random_case_code"]
                        if condition == "random"
                        else case["matched_case_code"]
                    ]
                )
                if condition == "none":
                    prefix, body, ix, record = user_parts(tokenizer, case["current"])
                else:
                    if donor["stratum"] != case["stratum"] or (
                        not condition.startswith("actual")
                        and donor["author_id"] == case["author_id"]
                    ):
                        raise ValueError("Invalid frozen donor assignment")
                    prefix, body, ix, record = user_parts(
                        tokenizer, case["current"], donor["context"]
                    )
                if reference_body is None:
                    reference_body = body
                if body != reference_body or len(body) != case["current_tokens"]:
                    raise ValueError("Changed current tokens across conditions")
                if condition.endswith("block8"):
                    prefix, order = permute_prefix(
                        prefix, ix, donor["case_code"], "block8", 731
                    )
                    record.update(order)
                result = observer.forward(prefix, body)
                arrays[f"{condition}__logp"] = result["token_logp"]
                arrays[f"{condition}__prefix"] = np.asarray(prefix, dtype=np.int64)
                for site in observer.sites:
                    arrays[f"{condition}__{site}"] = result["trace"][site]["mean"]
                metadata[condition] = {
                    "context_case_code": None
                    if condition == "none"
                    else donor["case_code"],
                    "prefix_tokens": len(prefix),
                    "body_sha256": material_hash(body),
                    **record,
                }
                row = {
                    "case_code": code,
                    "corpus": case["corpus"],
                    "language": case["language"],
                    "stratum": case["stratum"],
                    "condition": condition,
                    "mean_logp": result["mean_logp"],
                    "tail_logp": result["tail_logp"],
                    "current_tokens": len(body),
                }
                for site in observer.sites:
                    same = arrays[f"actual__{site}"]
                    other = result["trace"][site]["mean"]
                    row[f"{site}_cosine_to_actual"] = cosine(same, other)
                    row[f"{site}_norm_delta_to_actual"] = float(
                        np.linalg.norm(same - other)
                    )
                if row["input_embedding_norm_delta_to_actual"] != 0:
                    raise ValueError("Current input-embedding identity failed")
                rows.append(row)
                peak = max(peak, torch.mps.current_allocated_memory())
                if (
                    peak > manifest["max_mps_allocated_bytes"]
                    or time.monotonic() - start > 7200
                ):
                    raise RuntimeError("Frozen memory or time budget reached")
            arrays["body"] = np.asarray(reference_body, dtype=np.int64)
            np.savez_compressed(directory / f"{code}.npz", **arrays)
            write_json(directory / f"{code}.json", metadata)
            completed += 1
            pd.DataFrame(rows).to_csv(out / "private_conditions.csv", index=False)
            status = {
                "status": "RUNNING",
                "completed_cases": completed,
                "planned_cases": len(cases),
                "model_forwards": observer.calls,
                "seconds": time.monotonic() - start,
                "mps_peak_sampled_bytes": peak,
            }
            write_json(out / "run_status.json", status)
            if completed % 8 == 0 or completed == len(cases):
                print(json.dumps(status), flush=True)
        frame = pd.DataFrame(rows)
        summary = summarize(frame)
        summary.to_csv(out / "domain_sensitivity.csv", index=False)
        geometry = (
            frame.groupby(["corpus", "language", "condition"])
            .mean(numeric_only=True)
            .reset_index()
        )
        geometry.to_csv(out / "domain_state_geometry.csv", index=False)
        result = {
            "status": "COMPLETE_DOMAIN_CONTEXT_FUNCTION",
            "identity": manifest["identity"],
            "material_identity": material["identity"],
            "cases": len(cases),
            "conditions": len(frame),
            "model_forwards": observer.calls,
            "seconds": time.monotonic() - start,
            "mps_peak_sampled_bytes": peak,
            "labels_read": False,
            "whole_program_complete": False,
            "outputs_sha256": {
                str(p.relative_to(out)): file_hash(p)
                for p in out.rglob("*")
                if p.is_file()
                and not p.name.startswith("._")
                and p.name != "run_status.json"
            },
        }
        write_json(out / "summary.json", result)
        write_json(
            out / "run_status.json",
            {k: v for k, v in result.items() if k != "outputs_sha256"},
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "outputs_sha256"}),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 -- protect source text from arbitrary exception messages
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
                "completed_cases": completed,
                "exception_type": type(exc).__name__,
                "seconds": time.monotonic() - start,
            },
        )
        (out / "error_log.txt").write_text(
            "".join(traceback.format_list(traceback.extract_tb(exc.__traceback__)))
            + "\nException type: "
            + type(exc).__name__
            + "; private message withheld.\n"
        )
        print(
            json.dumps({"status": "FAILED_STOP", "exception_type": type(exc).__name__}),
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
