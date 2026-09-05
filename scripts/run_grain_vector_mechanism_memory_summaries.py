#!/usr/bin/env python3
"""Generate bounded local evidence summaries; audit citations before readout tests."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.inventory import archive_source, code_fingerprint
from grain_research.vector_mechanisms.memory import (
    SUMMARY_REQUEST,
    SUMMARY_SYSTEM,
    parse_summary,
    render_records,
)
from scripts.run_grain_vector_mechanism_controls import write_json


def main():
    """Six greedy generations, no cloud calls, no retries selected for a score."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    source = json.loads(a.sources.read_text(encoding="utf-8"))
    if not 1 <= len(source) <= 6:
        raise ValueError("At most six natural author histories")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise ValueError("Use a new output directory")
    archive_source(ROOT, a.output_dir / "source_snapshot.zip")
    name, rev = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    torch.manual_seed(731)
    torch.set_num_threads(4)
    transformers.utils.logging.disable_progress_bar()
    tok = AutoTokenizer.from_pretrained(name, revision=rev, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            name,
            revision=rev,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .eval()
        .to("cuda")
    )
    write_json(
        a.output_dir / "manifest.json",
        {
            "model": name,
            "revision": rev,
            "source_sha256": material_hash(source),
            "max_new_tokens": 1024,
            "do_sample": False,
            "dtype": "bfloat16",
            "use_cache": True,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "code_sha256": code_fingerprint(ROOT),
            "system": SUMMARY_SYSTEM,
            "request": SUMMARY_REQUEST,
            "labels_used": False,
            "scope": "Summary generator, not numerical activation instrument or independent rater",
        },
    )
    results, start = [], time.monotonic()
    for item in source:
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": SUMMARY_REQUEST + render_records(item["history"]),
            },
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch = tok(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        length = batch.input_ids.shape[1]
        if length > 2300:
            raise ValueError("Summary input budget exceeded; no truncation")
        with torch.inference_mode():
            output = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=1024,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        tokens = output[0, length:].tolist()
        raw = tok.decode(tokens, skip_special_tokens=True)
        audit = parse_summary(raw, item["history"])
        result = {
            "author_id": item["author_id"],
            "raw": raw,
            "audit": audit,
            "prompt_tokens": length,
            "generated_tokens": len(tokens),
            "hit_token_limit": len(tokens) == 1024,
        }
        results.append(result)
        write_json(a.output_dir / "generated_summaries.json", results)
        print(
            json.dumps(
                {
                    "completed": len(results),
                    "status": audit["status"],
                    "facts": len(audit["facts"]),
                    "citation_valid": sum(
                        f["status"] == "EXACT_QUOTE_SUPPORTED" for f in audit["facts"]
                    ),
                    "seconds": time.monotonic() - start,
                }
            ),
            flush=True,
        )
    write_json(
        a.output_dir / "run_status.json",
        {
            "status": "COMPLETE",
            "generations": len(results),
            "seconds": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    main()
