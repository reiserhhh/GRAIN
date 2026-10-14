#!/usr/bin/env python
"""R15 hedge-commitment local NLI reader (docs/GRAIN_R15_HEDGE_COMMITMENT_READER_PROTOCOL_V1.md). Local only."""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/GRAIN_R15_HEDGE_COMMITMENT_READER_PROTOCOL_V1.md"
PROTOCOL_SHA = "0ed32104956f98bf319c07f93be0b60fae2a0d8421a927ee8e0c13eaad04fcac"
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1"
MAT_SHA = "1b07fb5336e4909e267d97e844b47d6cc5330abc62a00fefeca20adbc53d12ac"
NLI_MODEL, NLI_REV = "cross-encoder/nli-deberta-v3-base", "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
CAP, BATCH, WALL = 576, 16, 600


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = snapshot_download(NLI_MODEL, revision=NLI_REV, local_files_only=True, allow_patterns=["*.json", "*.txt", "spm.model", "model.safetensors"])
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, use_safetensors=True, dtype=torch.float32).eval()
    mapping = {int(i): str(v).lower() for i, v in model.config.id2label.items()}
    if set(mapping.values()) != {"contradiction", "entailment", "neutral"}:
        raise ValueError("unknown NLI labels")
    return tok, model, mapping, path


def main() -> int:
    if sha(PROTOCOL) != PROTOCOL_SHA or sha(OUT / "materials_frozen.json") != MAT_SHA:
        raise RuntimeError("protocol or materials changed")
    if (OUT / "readings.jsonl").exists():
        raise FileExistsError("readings exist; preserve earlier runs")
    import torch
    mat = json.loads((OUT / "materials_frozen.json").read_text())
    prem = {t["text_id"]: t for t in mat["texts"]}
    pairs = mat["pairs"]
    order = [p for p in pairs if prem[p["text_id"]]["split"] == "source"] + [p for p in pairs if prem[p["text_id"]]["split"] == "test"]
    t0 = time.time()
    tok, model, mapping, path = setup()
    log = {"model": NLI_MODEL, "revision": NLI_REV, "snapshot_path": str(path), "load_seconds": time.time() - t0, "batches": 0, "pairs_scored": 0,
           "invocations": 0, "failures": 0, "boundary": None}
    out = (OUT / "readings.jsonl").open("w")
    for bi in range(0, len(order), BATCH):
        if time.time() - t0 > WALL:
            log["stopped"] = "wall-clock cap"
            break
        batch = order[bi:bi + BATCH]
        if log["invocations"] + len(batch) > CAP:
            log["stopped"] = "invocation cap"
            break
        log["invocations"] += len(batch)
        try:
            enc = tok([prem[p["text_id"]]["premise"] for p in batch], [p["hypothesis"] for p in batch], return_tensors="pt", padding=True, truncation=False)
            if enc["input_ids"].shape[1] > 512:
                raise ValueError("input exceeds 512")
            with torch.inference_mode():
                probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
        except Exception as e:  # noqa: BLE001
            log["failures"] += len(batch)
            log.setdefault("errors", []).append(repr(e)[:300])
            continue
        for p, pr in zip(batch, probs):
            rec = {**p, **{k: v for k, v in prem[p["text_id"]].items() if k != "premise"}, **{f"p_{mapping[j]}": float(pr[j]) for j in mapping}}
            out.write(json.dumps(rec) + "\n")
        out.flush()
        log["batches"] += 1
        log["pairs_scored"] += len(batch)
        if log["pairs_scored"] == 384 and log["boundary"] is None:
            log["boundary"] = {"pairs": 384, "seconds": time.time() - t0}
            (OUT / "boundary_checkpoint.json").write_text(json.dumps(log, indent=1) + "\n")
    out.close()
    log["seconds"] = time.time() - t0
    (OUT / "run_log.json").write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps({k: v for k, v in log.items() if k != "errors"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
