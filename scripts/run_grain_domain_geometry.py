#!/usr/bin/env python3
"""Observe independent embeddings and join the frozen domain Qwen function gap."""

from __future__ import annotations

import argparse
import hashlib
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
from grain_research.vector_mechanisms.domain_geometry import (
    association_bootstrap,
    shared_windows,
)
from grain_research.vector_mechanisms.response_context import paired_interval
from scripts.run_grain_vector_mechanism_controls import write_json
from scripts.run_grain_vector_mechanism_meps_preflight import MODELS, tokenizer_setup
from scripts.run_grain_vector_mechanism_response_context import file_hash

WEIGHTS = {
    "bge_large": "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7",
    "e5_large": "d741c1a688a6169af0ecb5a047c44645cd992c31e1bf431269f98bba9ae2911a",
    "multilingual_e5_large": "020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6",
}


class Encoder:
    """One fixed512-position FP32 observer with verified legacy position buffer."""

    def __init__(self, snapshot, tokenizer, prefix, kind):
        import torch
        from safetensors import safe_open
        from transformers import AutoModel

        self.torch = torch
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.kind = kind
        model, info = AutoModel.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if (
            info.get("missing_keys")
            or info.get("mismatched_keys")
            or info.get("error_msgs")
            or set(info.get("unexpected_keys", [])) != {"embeddings.position_ids"}
        ):
            raise ValueError("Unexpected encoder load mismatch")
        with safe_open(
            str(snapshot / "model.safetensors"), framework="pt", device="cpu"
        ) as f:
            pos = f.get_tensor("embeddings.position_ids")
        expected = 514 if kind == "multilingual_e5_large" else 512
        if pos.shape != (1, expected) or not torch.equal(
            pos, torch.arange(expected)[None, :]
        ):
            raise ValueError("Legacy position buffer is not the nonlearned arange")
        self.model = model.to("mps").eval()
        self.calls = 0
        self.peak = 0

    def encode(self, text, *, compact=False, poison=False):
        """Use native CLS/mean pooling, never drop complete source tokens."""
        torch = self.torch
        ids = self.tokenizer(self.prefix + text, truncation=False, verbose=False)[
            "input_ids"
        ]
        if len(ids) > 512:
            raise ValueError("Complete common span exceeds fixed shape")
        n = len(ids)
        pad = self.tokenizer.pad_token_id
        mask = [1] * n + ([] if compact else [0] * (512 - n))
        ids = ids + (
            []
            if compact
            else [(self.tokenizer.unk_token_id if poison else pad)] * (512 - n)
        )
        batch = {
            "input_ids": torch.tensor([ids], device="mps"),
            "attention_mask": torch.tensor([mask], device="mps"),
        }
        with torch.inference_mode():
            h = self.model(**batch).last_hidden_state
            if self.kind == "bge_large":
                v = h[:, 0]
            else:
                v = (
                    h.masked_fill(~batch["attention_mask"][..., None].bool(), 0).sum(1)
                    / batch["attention_mask"].sum(1)[:, None]
                )
            v = torch.nn.functional.normalize(v, dim=1)[0].cpu().numpy()
        self.calls += 1
        self.peak = max(self.peak, torch.mps.current_allocated_memory())
        if (
            not np.isfinite(v).all()
            or abs(np.linalg.norm(v) - 1) > 1e-5
            or self.peak > 6 * 1024**3
        ):
            raise ValueError("Invalid or overbudget encoder output")
        return v


def main():
    """Run all predeclared geometry observers and preserve per-case comparison."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material-dir", type=Path, required=True)
    p.add_argument("--inventory-dir", type=Path, required=True)
    p.add_argument("--function-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if platform.system() != "Darwin" or not out.is_relative_to(ROOT / "results"):
        raise ValueError("Local ignored research output required")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Use a fresh geometry output directory")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    start = time.monotonic()
    try:
        import torch
        import transformers

        torch.set_num_threads(2)
        torch.manual_seed(731)
        material = json.loads((a.material_dir / "manifest.json").read_text())
        inventory = json.loads((a.inventory_dir / "manifest.json").read_text())
        inventory_summary = json.loads((a.inventory_dir / "summary.json").read_text())
        if (
            inventory["identity"]
            != material_hash({k: v for k, v in inventory.items() if k != "identity"})
            or inventory_summary["identity"] != inventory["identity"]
            or material["inventory_identity"] != inventory["identity"]
            or material["source_files"] != inventory["source_files"]
        ):
            raise ValueError("Frozen domain inventory lineage changed")
        function = json.loads((a.function_dir / "summary.json").read_text())
        if (
            material["identity"]
            != material_hash({k: v for k, v in material.items() if k != "identity"})
            or function["material_identity"] != material["identity"]
            or function["status"] != "COMPLETE_DOMAIN_CONTEXT_FUNCTION"
        ):
            raise ValueError("Need the complete same-material frozen function packet")
        for name, sha in function["outputs_sha256"].items():
            if file_hash(a.function_dir / name) != sha:
                raise ValueError("Function packet artifact changed")
        cases = json.loads((a.material_dir / "private_cases.json").read_text())
        if material_hash(cases) != material["case_material_sha256"]:
            raise ValueError("Material case hash changed")
        cache = Path.home() / ".cache/huggingface/hub"
        tokenizers, specs = tokenizer_setup(cache)
        if specs != inventory_summary["tokenizers"]:
            raise ValueError("Tokenizer files differ from the frozen inventory")
        snapshots = {
            name: cache
            / ("models--" + model.replace("/", "--"))
            / "snapshots"
            / revision
            for name, model, revision, prefix in MODELS
        }
        hashes = {
            name: file_hash(path / "model.safetensors")
            for name, path in snapshots.items()
        }
        if hashes != WEIGHTS:
            raise ValueError("Frozen cached encoder weights changed")
        paths = [
            Path(__file__),
            ROOT / "grain_research/vector_mechanisms/domain_geometry.py",
            ROOT / "grain_research/vector_mechanisms/response_context.py",
            ROOT / "grain_research/vector_mechanisms/controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_meps_preflight.py",
            ROOT / "scripts/run_grain_vector_mechanism_controls.py",
            ROOT / "scripts/run_grain_vector_mechanism_response_context.py",
            ROOT / "docs/GRAIN_VM06_DOMAIN_GEOMETRY_V1.md",
            ROOT / "tests/test_grain_domain_geometry.py",
        ]
        manifest = {
            "protocol": "GRAIN_VM06_DOMAIN_GEOMETRY_V1",
            "material_identity": material["identity"],
            "inventory_identity": inventory["identity"],
            "inventory_summary_sha256": file_hash(a.inventory_dir / "summary.json"),
            "function_identity": function["identity"],
            "function_summary_sha256": file_hash(a.function_dir / "summary.json"),
            "model_weights": hashes,
            "tokenizers": specs,
            "dtype": "float32",
            "device": "mps",
            "window_pooling": "shared_character_weighted_normalized_vectors",
            "max_seconds": 7200,
            "max_mps_bytes": 6 * 1024**3,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "labels_read": False,
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
        windows = []
        texts = []
        for case in cases:
            for arm in ("current", "context"):
                text = case[arm]
                for span in shared_windows(text, tokenizers):
                    piece = text[span["start"] : span["end"]]
                    windows.append(
                        {
                            "case_code": case["case_code"],
                            "corpus": case["corpus"],
                            "language": case["language"],
                            "arm": arm,
                            "start": span["start"],
                            "end": span["end"],
                            "text_sha256": hashlib.sha256(piece.encode()).hexdigest(),
                            **{"tokens_" + k: v for k, v in span["tokens"].items()},
                        }
                    )
                    texts.append(piece)
        frame = pd.DataFrame(windows)
        frame.to_csv(out / "private_windows.csv", index=False)
        f = pd.read_csv(a.function_dir / "private_conditions.csv").pivot(
            index="case_code", columns="condition", values="tail_logp"
        )
        effects = []
        all_technical = {}
        completed = []
        lookup = {c["case_code"]: c for c in cases}
        for name in ("multilingual_e5_large", "bge_large", "e5_large"):
            tokenizer, prefix = tokenizers[name]
            encoder = Encoder(snapshots[name], tokenizer, prefix, name)
            public = [
                (
                    "en",
                    "A visitor described a walk through the park and the quiet pond near the trees.",
                )
            ]
            if name == "multilingual_e5_large":
                public.append(
                    (
                        "ja",
                        "公園を散歩してから、木々と静かな池について短い文章を書きました。",
                    )
                )
            checks = []
            for language, text in public:
                fixed = encoder.encode(text)
                repeat = encoder.encode(text)
                compact = encoder.encode(text, compact=True)
                poisoned = encoder.encode(text, poison=True)
                errors = [
                    float(np.max(abs(fixed - v))) for v in (repeat, compact, poisoned)
                ]
                checks.append(
                    {
                        "language": language,
                        "repeat_compact_padding_max_errors": errors,
                        "passed": max(errors) <= 1e-5,
                    }
                )
            all_technical[name] = checks
            write_json(out / "technical_checks.json", all_technical)
            if not all(c["passed"] for c in checks):
                raise ValueError("Public encoder numerical gate failed")
            active = (
                frame
                if name == "multilingual_e5_large"
                else frame[frame.language == "en"]
            )
            vectors = []
            for i in active.index:
                vectors.append(encoder.encode(texts[i]))
                if time.monotonic() - start > 7200:
                    raise RuntimeError("Geometry packet time budget reached")
            v = np.stack(vectors)
            active = active.reset_index().rename(columns={"index": "source_window_row"})
            active.to_csv(out / f"private_{name}_windows.csv", index=False)
            np.save(out / f"private_{name}_vectors.npy", v)
            summaries = {}
            for (code, arm), g in active.groupby(["case_code", "arm"], sort=True):
                weighted = (
                    v[g.index].astype(float) * (g.end - g.start).to_numpy()[:, None]
                ).sum(0)
                summaries[(code, arm)] = weighted / np.linalg.norm(weighted)
            for case in cases:
                code = case["case_code"]
                if (code, "current") not in summaries:
                    continue
                current = summaries[(code, "current")]
                actual = float(current @ summaries[(code, "context")])
                matched = float(
                    current @ summaries[(case["matched_case_code"], "context")]
                )
                random = float(
                    current @ summaries[(case["random_case_code"], "context")]
                )
                other = [
                    float(current @ summaries[(c["case_code"], "context")])
                    for c in cases
                    if c["stratum"] == case["stratum"] and c["case_code"] != code
                ]
                effects.append(
                    {
                        "model": name,
                        "corpus": case["corpus"],
                        "language": case["language"],
                        "case_code": code,
                        "stratum": case["stratum"],
                        "actual_cosine": actual,
                        "matched_cosine": matched,
                        "random_cosine": random,
                        "delta_cosine": actual - matched,
                        "within_cell_percentile": float(
                            np.mean([(actual > o) + 0.5 * (actual == o) for o in other])
                        ),
                        "delta_log_context_length": float(
                            np.log(
                                case["context_tokens"]
                                / lookup[case["matched_case_code"]]["context_tokens"]
                            )
                        ),
                        "delta_tail_logp": float(
                            f.loc[code, "actual"] - f.loc[code, "matched"]
                        ),
                    }
                )
            completed.append(
                {
                    "model": name,
                    "windows": len(active),
                    "cases": len(summaries) // 2,
                    "forwards": encoder.calls,
                    "mps_peak_sampled_bytes": encoder.peak,
                }
            )
            del encoder
            torch.mps.empty_cache()
            write_json(
                out / "run_status.json",
                {
                    "status": "RUNNING",
                    "completed": completed,
                    "seconds": time.monotonic() - start,
                },
            )
            print(json.dumps(completed[-1]), flush=True)
        e = pd.DataFrame(effects)
        e.to_csv(out / "private_geometry_by_case.csv", index=False)
        rows = []
        for (model, corpus, language), g in e.groupby(
            ["model", "corpus", "language"], sort=True
        ):
            gap = paired_interval(g.delta_cosine)
            rank = paired_interval(g.within_cell_percentile - 0.5)
            rows.append(
                {
                    "model": model,
                    "corpus": corpus,
                    "language": language,
                    "cases": len(g),
                    "delta_cosine": gap["mean"],
                    "delta_cosine_low": gap["lower"],
                    "delta_cosine_high": gap["upper"],
                    "same_source_percentile": rank["mean"] + 0.5,
                    "same_source_percentile_low": rank["lower"] + 0.5,
                    "same_source_percentile_high": rank["upper"] + 0.5,
                    **association_bootstrap(
                        g.delta_cosine.to_numpy(),
                        g.delta_tail_logp.to_numpy(),
                        g.delta_log_context_length.to_numpy(),
                    ),
                }
            )
        pd.DataFrame(rows).to_csv(out / "domain_geometry_function.csv", index=False)
        result = {
            "status": "COMPLETE_DOMAIN_GEOMETRY_FUNCTION",
            "identity": manifest["identity"],
            "material_identity": material["identity"],
            "function_identity": function["identity"],
            "completed": completed,
            "case_observer_pairs": len(e),
            "seconds": time.monotonic() - start,
            "labels_read": False,
            "new_neural_training": False,
            "causal_association_claim": False,
            "outputs_sha256": {
                str(p.relative_to(out)): file_hash(p)
                for p in out.iterdir()
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
    except Exception as exc:  # noqa: BLE001 -- never print arbitrary private exception messages
        write_json(
            out / "run_status.json",
            {
                "status": "FAILED_STOP",
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
