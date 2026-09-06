#!/usr/bin/env python3
"""Run the frozen full-view MEPS observer locally, without psychological labels."""

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
from grain_research.vector_mechanisms.meps_fixed_free import (
    aggregate_windows,
    compare_views,
)
from grain_research.vector_mechanisms.meps_material import load_text_material
from scripts.run_grain_vector_mechanism_controls import write_json

REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
WEIGHTS = "020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6"
PREFLIGHT = "2a1a13248f7bc74984ad1e4eab1f72aa80fd53993f4f26d146530bfa8121bebb"
PREFIX = "query: "


def sha(path):
    """Hash potentially large local weights with bounded host RAM."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    """Read a local JSON artifact."""
    return json.loads(Path(path).read_text())


class WindowObserver:
    """Fixed float32 masked-mean multilingual E5 observer, without truncation."""

    def __init__(self, snapshot):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        self.model, loading = AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        from safetensors import safe_open

        with safe_open(
            str(Path(snapshot) / "model.safetensors"), framework="pt", device="cpu"
        ) as f:
            positions = f.get_tensor("embeddings.position_ids")
        if (
            loading.get("missing_keys")
            or loading.get("mismatched_keys")
            or loading.get("error_msgs")
            or set(loading.get("unexpected_keys", [])) != {"embeddings.position_ids"}
        ):
            raise ValueError("Unexpected model-load discrepancy")
        if positions.shape != (1, 514) or not torch.equal(
            positions, torch.arange(514)[None, :]
        ):
            raise ValueError(
                "Legacy nonlearned position buffer is not the standard integer sequence"
            )
        self.model = self.model.to("mps").eval()
        self.loading_audit = {
            "only_unexpected_key": "embeddings.position_ids",
            "nonlearned_arange_verified": True,
            "missing_keys": [],
        }
        self.calls = 0
        self.peak = 0

    def encode(self, text, *, fixed=True, poison_padding=False):
        """Return one normalized vector and own-text token count; no IDs persisted."""
        torch = self.torch
        tokens = self.tokenizer(
            PREFIX + text,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        if len(tokens["input_ids"]) > 512:
            raise ValueError("Frozen span exceeds observer budget")
        content_count = sum(
            end > len(PREFIX) for start, end in tokens["offset_mapping"]
        )
        if content_count <= 0:
            raise ValueError("Span has no own-text tokens")
        ids = tokens["input_ids"]
        mask = [1] * len(ids)
        if fixed:
            ids = ids + [self.tokenizer.pad_token_id] * (512 - len(ids))
            mask = mask + [0] * (512 - len(mask))
        if poison_padding:
            ids = [i if m else self.tokenizer.unk_token_id for i, m in zip(ids, mask)]
        batch = {
            "input_ids": torch.tensor([ids], device="mps"),
            "attention_mask": torch.tensor([mask], device="mps"),
        }
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            hidden = hidden.masked_fill(~batch["attention_mask"][..., None].bool(), 0)
            mean = hidden.sum(1) / batch["attention_mask"].sum(1)[:, None]
            vector = torch.nn.functional.normalize(mean, p=2, dim=1)[0].cpu().numpy()
        self.calls += 1
        self.peak = max(self.peak, torch.mps.current_allocated_memory())
        if not np.all(np.isfinite(vector)) or abs(np.linalg.norm(vector) - 1) > 1e-5:
            raise ValueError("Nonfinite or nonunit observer output")
        if self.peak > 6 * 1024**3:
            raise MemoryError("Observer exceeded frozen MPS allocation budget")
        return vector, content_count


def technical(observer):
    """Verify public bilingual numerical instruments before private material access."""
    rows = []
    for language, text in (
        (
            "en",
            "A visitor described a walk through the park and asked about the weather.",
        ),
        ("ja", "公園を散歩したあと、明日の天気について質問しました。"),
    ):
        fixed, _ = observer.encode(text)
        repeat, _ = observer.encode(text)
        compact, _ = observer.encode(text, fixed=False)
        poisoned, _ = observer.encode(text, poison_padding=True)
        result = {
            "language": language,
            "repeat_max_error": float(np.max(np.abs(fixed - repeat))),
            "compact_max_error": float(np.max(np.abs(fixed - compact))),
            "masked_padding_max_error": float(np.max(np.abs(fixed - poisoned))),
        }
        rows.append(result)
    passed = all(
        max(v for k, v in row.items() if k != "language") <= 1e-5 for row in rows
    )
    return {
        "passed": passed,
        "threshold": 1e-5,
        "public_checks": rows,
        "model_forwards": observer.calls,
        "loading_audit": observer.loading_audit,
    }


def main():
    """Seal lineage, extract bounded windows, pool by role and estimate correspondence."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--preflight-dir", type=Path, required=True)
    p.add_argument("--windows-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--technical-only", action="store_true")
    a = p.parse_args()
    out = a.output_dir.resolve()
    if platform.system() != "Darwin":
        raise ValueError("This private packet is authorized for local Mac only")
    if not out.is_relative_to(ROOT / "results"):
        raise ValueError("Private results must stay in the ignored local results tree")
    out.mkdir(parents=True, exist_ok=True)
    if any(x for x in out.iterdir() if not x.name.startswith("._")):
        raise ValueError("Use a fresh packet directory")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    start = time.monotonic()
    try:
        import torch
        import transformers

        torch.set_num_threads(2)
        torch.manual_seed(731)
        if not torch.backends.mps.is_available():
            raise ValueError("Frozen MPS device unavailable")
        snapshot = (
            Path.home()
            / ".cache/huggingface/hub/models--intfloat--multilingual-e5-large/snapshots"
            / REVISION
        )
        files = [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
        ]
        hashes = {name: sha(snapshot / name) for name in files}
        if hashes["model.safetensors"] != WEIGHTS:
            raise ValueError("Frozen observer weights changed")
        pre = read(a.preflight_dir / "manifest.json")
        window = read(a.windows_dir / "summary.json")
        if pre["identity"] != PREFLIGHT or pre["identity"] != material_hash(
            {k: v for k, v in pre.items() if k != "identity"}
        ):
            raise ValueError("Source preflight is not the frozen artifact")
        if (
            window["preflight_identity"] != PREFLIGHT
            or window["status"] != "COMPLETE_WINDOW_DESIGN_ONLY"
        ):
            raise ValueError("Window design does not belong to the frozen material")
        for name, digest in window["outputs_sha256"].items():
            if sha(a.windows_dir / name) != digest:
                raise ValueError("Window design artifact checksum changed")
        frozen_tokenizer = window["tokenizer"]
        if any(hashes.get(k) != v for k, v in frozen_tokenizer["files"].items()):
            raise ValueError("Frozen window tokenizer changed")
        source_paths = [
            Path(__file__),
            ROOT / "grain_research/vector_mechanisms/meps_fixed_free.py",
            ROOT / "grain_research/vector_mechanisms/meps_material.py",
            ROOT / "grain_research/vector_mechanisms/controls.py",
            ROOT / "suica_core/meps_ai_conv.py",
            ROOT / "suica_core/meps_layout.py",
            ROOT / "scripts/run_grain_vector_mechanism_controls.py",
            ROOT / "docs/GRAIN_VM06_FIXED_FREE_V1.md",
            ROOT / "tests/test_grain_meps_fixed_free.py",
        ]
        manifest = {
            "protocol": "GRAIN_VM06_FIXED_FREE_V1",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "preflight_identity": PREFLIGHT,
            "window_summary_sha256": sha(a.windows_dir / "summary.json"),
            "model": "intfloat/multilingual-e5-large",
            "revision": REVISION,
            "model_files": hashes,
            "prefix": PREFIX,
            "max_tokens": 512,
            "precision": "float32",
            "device": "mps",
            "batch_size": 1,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "max_seconds": 7200,
            "max_mps_bytes": 6 * 1024**3,
            "labels_read": False,
            "mixed_markdown_containers_scanned": True,
            "private_text_leaves_mac": False,
            "technical_only": a.technical_only,
            "code": {str(x.relative_to(ROOT)): sha(x) for x in source_paths},
        }
        manifest["identity"] = material_hash(manifest)
        write_json(out / "manifest.json", manifest)
        with zipfile.ZipFile(
            out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in source_paths:
                archive.write(path, path.relative_to(ROOT))
        observer = WindowObserver(str(snapshot))
        checks = technical(observer)
        write_json(out / "technical_checks.json", checks)
        if not checks["passed"]:
            raise ValueError(
                "Public numerical gate failed before private source loading"
            )
        if a.technical_only:
            write_json(
                out / "summary.json",
                {
                    "status": "COMPLETE_TECHNICAL_ONLY",
                    "identity": manifest["identity"],
                    **checks,
                },
            )
            print(json.dumps(checks), flush=True)
            return
        views, integrity = load_text_material(a.input_root)
        if material_hash(integrity["source_files"]) != pre["source_identity"]:
            raise ValueError("Private source changed since frozen preflight")
        turns = {}
        for view in views:
            meta = view["metadata"]
            for turn in view["turns"]:
                key = tuple(
                    meta[k] for k in ("participant_code", "view_kind", "condition")
                ) + (turn["role"], turn["source_sequence"])
                if key in turns:
                    raise ValueError("Duplicate source turn key")
                turns[key] = turn["text"]
        frame = pd.read_csv(a.windows_dir / "private_window_metadata.csv")
        turn_frame = pd.read_csv(a.windows_dir / "private_turn_integrity.csv")
        if (
            len(views) != 264
            or len(turns) != 1600
            or len(frame) != 1609
            or len(turn_frame) != len(turns)
        ):
            raise ValueError("Frozen material scope changed")
        for key, spans in frame.groupby(
            ["participant_code", "view_kind", "condition", "role", "turn_sequence"],
            sort=False,
        ):
            text = turns[key]
            ordered = spans.sort_values("window")
            if (
                ordered.start.iloc[0] != 0
                or ordered.end.iloc[-1] != len(text)
                or not np.array_equal(
                    ordered.end.to_numpy()[:-1], ordered.start.to_numpy()[1:]
                )
            ):
                raise ValueError("Nonlossless source window coverage")
        vectors, counts = [], []
        for i, row in enumerate(frame.itertuples(index=False)):
            text = turns[
                (
                    row.participant_code,
                    row.view_kind,
                    row.condition,
                    row.role,
                    row.turn_sequence,
                )
            ][row.start : row.end]
            if hashlib.sha256(text.encode()).hexdigest() != row.text_sha256:
                raise ValueError("Window text checksum changed")
            n = len(
                observer.tokenizer(PREFIX + text, truncation=False, verbose=False)[
                    "input_ids"
                ]
            )
            if n != row.tokens:
                raise ValueError("Window tokenization changed")
            vector, count = observer.encode(text)
            vectors.append(vector)
            counts.append(count)
            if time.monotonic() - start > 7200:
                raise TimeoutError("Frozen time budget reached")
            if (i + 1) % 50 == 0 or i + 1 == len(frame):
                np.save(out / "private_partial_vectors.npy", np.stack(vectors))
                status = {
                    "status": "RUNNING",
                    "windows_completed": i + 1,
                    "windows_planned": len(frame),
                    "seconds": time.monotonic() - start,
                    "mps_peak_sampled_bytes": observer.peak,
                    "model_forwards": observer.calls,
                }
                write_json(out / "run_status.json", status)
                print(json.dumps(status), flush=True)
        frame["content_tokens"] = counts
        frame.to_csv(out / "private_window_metadata.csv", index=False)
        np.save(out / "private_window_vectors.npy", np.stack(vectors))
        (out / "private_partial_vectors.npy").unlink()
        meta, pooled = aggregate_windows(frame, np.stack(vectors))
        meta.to_csv(out / "private_view_metadata.csv", index=False)
        np.save(out / "private_view_vectors.npy", pooled)
        results, matrices, pair_meta = compare_views(meta, pooled)
        results.to_csv(out / "meps_fixed_free_mechanisms.csv", index=False)
        np.savez_compressed(out / "private_similarity_matrices.npz", **matrices)
        write_json(out / "private_pair_metadata.json", pair_meta)
        contamination = []
        for (person, kind, condition, pooling), group in meta.groupby(
            ["participant_code", "view_kind", "condition", "pooling"]
        ):
            if not {"user", "assistant", "mixed"} <= set(group.role):
                continue
            ids = {row.role: i for i, row in group.iterrows()}
            ui, ai, mi = (ids[x] for x in ("user", "assistant", "mixed"))
            contamination.append(
                {
                    "participant_code": person,
                    "view_kind": kind,
                    "condition": condition,
                    "pooling": pooling,
                    "mixed_user_cosine": float(pooled[mi] @ pooled[ui]),
                    "mixed_assistant_cosine": float(pooled[mi] @ pooled[ai]),
                    "assistant_token_share": float(
                        meta.loc[ai, "content_tokens"] / meta.loc[mi, "content_tokens"]
                    ),
                }
            )
        pd.DataFrame(contamination).to_csv(
            out / "private_role_contamination.csv", index=False
        )
        summary = {
            "status": "COMPLETE_FIXED_FREE_COMPARISON",
            "identity": manifest["identity"],
            "participants": integrity["participants"],
            "views": len(views),
            "turns": len(turns),
            "windows": len(frame),
            "source_chars": int((frame.end - frame.start).sum()),
            "model_forwards": observer.calls,
            "private_model_forwards": len(frame),
            "mps_peak_sampled_bytes": observer.peak,
            "seconds": time.monotonic() - start,
            "labels_read": False,
            "full_conversation_context_encoding": False,
            "primary": json.loads(results[results.primary].to_json(orient="records")),
            "outputs_sha256": {
                x.name: sha(x)
                for x in out.iterdir()
                if x.is_file() and not x.name.startswith("._")
            },
        }
        write_json(out / "summary.json", summary)
        write_json(
            out / "run_status.json",
            {
                k: v
                for k, v in summary.items()
                if k not in {"primary", "outputs_sha256"}
            },
        )
        # Status is mutable and intentionally excluded from sealed artifact checksums.
        summary["outputs_sha256"].pop("run_status.json", None)
        write_json(out / "summary.json", summary)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in summary.items()
                    if k not in {"primary", "outputs_sha256"}
                }
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 -- never expose private text in exception messages
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
            + "\nMessage withheld to protect source text.\n"
        )
        print(
            json.dumps({"status": "FAILED_STOP", "exception_type": type(exc).__name__}),
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
