#!/usr/bin/env python3
"""Complete one interrupted fixed graph with full-context row-blocked attention."""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_full_attention_rows as F
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_author_comparison_geometry_hm226 as G
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_author_comparison_recovery_hm227.json"
DEFAULT = ROOT / "results/grain_math_response_v1/author_comparison_recovery_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_AUTHOR_COMPARISON_RECOVERY_PLAN_HM227.md",
    "grain_research/construction/math_full_attention_rows.py",
    "scripts/run_grain_author_comparison_recovery_hm227.py",
    "tests/test_grain_full_attention_rows.py",
)


def parent_check(parent):
    """Verify the old partial receipt without changing its failed terminal state."""
    manifest = R.check(parent)
    receipt = R.read(parent / "resource_stop_receipt.json")
    runtime = R.read(parent / "runtime.json")
    if runtime["status"] != "FAILED" or receipt["complete_batches"] != 27:
        raise ValueError("Only the registered terminal partial observation")
    if (parent / "ACTIVE.lock").exists():
        raise ValueError("Cannot recover a locked original worker")
    for file, digest in receipt["files"].items():
        if R.digest(parent / file) != digest:
            raise ValueError("Changed old partial observation: " + file)
    expected = [f"{i:03d}.npz" for i in range(27)]
    actual = sorted(p.name for p in (parent / "batches").glob("[0-9]*.npz"))
    if actual != expected:
        raise ValueError("Unexpected old batch inventory")
    return manifest, receipt


def prepare(out):
    """Bind unchanged scientific scope and immutable partial results before capture."""
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    old, receipt = parent_check(parent)
    if out.exists():
        raise FileExistsError("No overwriting an uncertain or completed recovery")
    files = ("units.json", "bindings.json", "observations.npz", "input_quality.json")
    inputs = dict(old["inputs"])
    for file in (
        *files,
        "manifest.json",
        "prepared_seal.json",
        "resource_stop_receipt.json",
    ):
        p = parent / file
        inputs[str(p.relative_to(ROOT))] = R.digest(p)
    for file, digest in receipt["files"].items():
        inputs[str((parent / file).relative_to(ROOT))] = digest
    (out / "batches").mkdir(parents=True, exist_ok=False)
    (out / "compatibility").mkdir(exist_ok=False)
    for file in (*files, "head.npz"):
        shutil.copyfile(parent / file, out / file)
    for i in range(cfg["reuse_batches"]):
        shutil.copyfile(
            parent / "batches" / f"{i:03d}.npz", out / "batches" / f"{i:03d}.npz"
        )
    config = old["config"] | {"recovery_packet": "HM227", "recovery": cfg}
    R.write(
        out / "manifest.json",
        {
            "config": config,
            "code": old["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "model_files": old["model_files"],
            "alphabet": old["alphabet"],
            "parent_remains_FAILED": True,
            "uncertain_batch_027_is_explicit_reobservation": True,
            "original_scope_complete": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        files + ("head.npz",) + tuple(f"batches/{i:03d}.npz" for i in range(27)),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 27, "new_batches": 0})
    print(
        "HM227 PREPARED:same335 endpoints;27 reused;42 new including2 equivalence",
        flush=True,
    )


def collect(out):
    """Capture the fixed remainder once after two finite equivalence observations."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg, recovery = manifest["config"], manifest["config"]["recovery"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No automatic retry of terminal or uncertain recovery")
    parent_check(ROOT / recovery["parent"])
    for file, digest in manifest["model_files"].items():
        if R.digest(MODEL / file) != digest:
            raise ValueError("Changed fixed model")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed, new_batches, pending = 0.0, 27, 0, None
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    try:
        torch.set_num_threads(2)
        stamp = time.monotonic()
        observer = DecoderObserver.load(
            str(MODEL),
            device=cfg["device"],
            dtype=cfg["dtype"],
            max_tokens=cfg["max_tokens"],
        )
        F.install(observer.model, recovery["query_rows"])
        load_seconds = time.monotonic() - stamp
        alpha = manifest["alphabet"]
        head = observer.model.lm_head.weight.detach()[alpha].float().cpu().numpy()
        if not np.array_equal(
            head, H.selected(out / "head.npz", ("weights",))["weights"]
        ):
            raise ValueError("Changed selected model head")
        units = R.read(out / "units.json")
        schedule = [("compatibility", b) for b in recovery["compatibility_batches"]]
        schedule += [("recovery", b) for b in range(27, cfg["batches"])]
        checks = []
        for mode, bi in schedule:
            if elapsed >= recovery["max_model_seconds"]:
                raise RuntimeError("Finite recovery cap reached;preserve partial graph")
            pending = {"mode": mode, "batch": bi}
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "new_batches": new_batches,
                    "pending": pending,
                    "model_seconds": elapsed,
                },
            )
            batch = units[bi * 5 : (bi + 1) * 5]
            if (
                len(batch) != 5
                or [b["target"] for b in batch] != list(range(5))
                or len({(b["edge"], b["order"]) for b in batch}) != 1
            ):
                raise ValueError("Changed endpoint binding")
            width = max(len(b["input_ids"]) for b in batch)
            pad = observer.tokenizer.pad_token_id
            if pad is None:
                pad = observer.tokenizer.eos_token_id
            ids = torch.full((5, width), pad, dtype=torch.long, device=cfg["device"])
            mask = torch.zeros_like(ids)
            last = []
            for j, b in enumerate(batch):
                n = len(b["input_ids"])
                ids[j, :n] = torch.tensor(b["input_ids"], device=cfg["device"])
                mask[j, :n] = 1
                last.append(n - 1)
            stamp = time.monotonic()
            try:
                with torch.inference_mode():
                    body = observer.model.model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    state = body.last_hidden_state[
                        torch.arange(5, device=cfg["device"]),
                        torch.tensor(last, device=cfg["device"]),
                    ]
                    logits = (
                        observer.model.lm_head(state)
                        .float()
                        .cpu()
                        .numpy()
                        .astype(float)
                    )
                    values = {
                        "logits": logits[:, alpha],
                        "normalizer": logsumexp(logits, -1),
                        "states": state.float().cpu().numpy(),
                        "greedy": logits.argmax(-1),
                    }
                if any(not np.isfinite(v).all() for v in values.values()):
                    raise ValueError("Nonfinite full observation")
                target = (
                    out
                    / ("compatibility" if mode == "compatibility" else "batches")
                    / f"{bi:03d}.npz"
                )
                R.archive(target, **values)
            finally:
                elapsed += time.monotonic() - stamp
            new_batches += 1
            if mode == "compatibility":
                old = H.selected(
                    out / "batches" / f"{bi:03d}.npz",
                    ("logits", "normalizer", "states"),
                )
                row = {
                    "batch": bi,
                    **{
                        k + "_max": float(
                            np.max(abs(values[k].astype(float) - old[k].astype(float)))
                        )
                        for k in old
                    },
                }
                row["log_odds_max"] = float(
                    np.max(
                        abs(
                            np.diff(values["logits"], axis=1)
                            - np.diff(old["logits"], axis=1)
                        )
                    )
                )
                row["PASS"] = all(
                    v <= recovery["compatibility_absolute_tolerance"]
                    for k, v in row.items()
                    if k != "batch"
                )
                R.write(out / "compatibility" / f"{bi:03d}.json", row)
                checks.append(row)
                if not row["PASS"]:
                    raise ValueError("Execution equivalence failed;no backend search")
                if len(checks) == 2:
                    R.write(
                        out / "compatibility.json",
                        {
                            "checks": checks,
                            "PASS": True,
                            "not_prediction_accuracy": True,
                        },
                    )
            else:
                completed += 1
            pending = None
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "new_batches": new_batches,
                    "pending": None,
                    "model_seconds": elapsed,
                },
            )
            print(
                f"HM227 {mode}:{bi:03d};full={completed}/67;new={new_batches}/42;seconds={elapsed:.3f}",
                flush=True,
            )
            del body, state, logits, values, ids, mask
            torch.mps.empty_cache()
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_batches": completed,
                "new_batches": new_batches,
                "model_seconds": elapsed,
                "load_seconds": load_seconds,
                "reused_endpoints": 135,
                "new_endpoints_including_compatibility": 210,
                "explicit_interrupted_batch_reobservation": 27,
                "new_API_train_private_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json", "compatibility.json")
            + tuple(f"batches/{i:03d}.npz" for i in range(67))
            + tuple(
                f"compatibility/{i:03d}.{ext}"
                for i in recovery["compatibility_batches"]
                for ext in ("npz", "json")
            ),
        )
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "pid": None,
                "completed_batches": completed,
                "new_batches": new_batches,
                "pending": pending,
                "model_seconds": elapsed,
                "error": repr(exc),
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    """Use the unchanged graph analysis after full observation; never rerun science."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, G.STAGES)
        parent_check(
            ROOT / R.read(args.out / "manifest.json")["config"]["recovery"]["parent"]
        )
        print("HM227 ALL SEALS PASS;parent failure unchanged")
    elif args.stage == "analyze":
        G.analyze(args.out)
    elif args.stage == "verify":
        G.verify(args.out)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
