#!/usr/bin/env python3
"""Complete the original graph under its actual log-odds observation contract."""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_full_attention_rows as F
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_author_comparison_geometry_hm226 as G
from scripts import run_grain_author_comparison_recovery_hm227 as C
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_comparison_observable_contract_hm228.json"
DEFAULT = ROOT / "results/grain_math_response_v1/comparison_observable_contract_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_COMPARISON_OBSERVABLE_CONTRACT_PLAN_HM228.md",
    "scripts/run_grain_comparison_observable_contract_hm228.py",
)


def prepare(out):
    """Retain failed state gate and reuse existing output checks without recapture."""
    cfg = R.read(ROOT / CONFIG)
    parent, failed = ROOT / cfg["parent"], ROOT / cfg["failed_compatibility_parent"]
    old, _ = C.parent_check(parent)
    fm = R.check(failed)
    runtime = R.read(failed / "runtime.json")
    if (
        runtime["status"] != "FAILED"
        or runtime["new_batches"] != 2
        or (failed / "ACTIVE.lock").exists()
    ):
        raise ValueError("Expected stopped HM227 with exactly two stored checks")
    outputs = []
    for i in (0, 26):
        a = H.selected(
            failed / "compatibility" / f"{i:03d}.npz",
            ("logits", "normalizer", "states"),
        )
        b = H.selected(parent / "batches" / f"{i:03d}.npz", tuple(a))
        row = {
            "batch": i,
            **{
                k + "_max": float(np.max(abs(a[k].astype(float) - b[k].astype(float))))
                for k in a
            },
        }
        row["log_odds_max"] = float(
            np.max(abs(np.diff(a["logits"], axis=1) - np.diff(b["logits"], axis=1)))
        )
        row["observable_PASS"] = all(
            row[k] <= cfg["output_tolerance"]
            for k in ("logits_max", "normalizer_max", "log_odds_max")
        )
        row["original_state_gate_PASS"] = R.read(
            failed / "compatibility" / f"{i:03d}.json"
        )["PASS"]
        if not row["observable_PASS"]:
            raise ValueError("Existing output check fails;no precision search")
        outputs.append(row)
    if out.exists():
        raise FileExistsError("No uncertain/completed output replacement")
    files = (
        "units.json",
        "bindings.json",
        "observations.npz",
        "input_quality.json",
        "head.npz",
    )
    extra = ("manifest.json", "runtime.json", "prepared_seal.json") + tuple(
        f"compatibility/{i:03d}.{ext}" for i in (0, 26) for ext in ("json", "npz")
    )
    inputs = dict(fm["inputs"])
    inputs.update(
        {str((failed / p).relative_to(ROOT)): R.digest(failed / p) for p in extra}
    )
    (out / "batches").mkdir(parents=True, exist_ok=False)
    for file in files:
        shutil.copyfile(parent / file, out / file)
    for i in range(27):
        shutil.copyfile(
            parent / "batches" / f"{i:03d}.npz", out / "batches" / f"{i:03d}.npz"
        )
    config = old["config"] | {"observable_contract": cfg}
    R.write(
        out / "manifest.json",
        {
            "config": config,
            "code": fm["code"] | {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": inputs,
            "model_files": old["model_files"],
            "alphabet": old["alphabet"],
            "remaining_model_seconds": cfg["model_seconds_total_recovery_cap"]
            - runtime["model_seconds"],
            "HM226_and_HM227_remain_FAILED": True,
            "post_instrument_outcome_protocol_revision": True,
            "original_scope_complete": False,
        },
    )
    R.write(
        out / "observable_contract.json",
        {
            "checks": outputs,
            "output_contract_PASS": True,
            "full_state_contract_PASS": False,
            "state_failure_not_hidden": True,
            "prior_model_seconds": runtime["model_seconds"],
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        files
        + ("observable_contract.json",)
        + tuple(f"batches/{i:03d}.npz" for i in range(27)),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 27, "new_batches": 0})
    print(
        "HM228 PREPARED:335 endpoints unchanged;200 remaining;old state gate FAILED retained",
        flush=True,
    )


def collect(out):
    """Observe remaining full histories once, with no further compatibility replay."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    manifest = R.check(out)
    cfg, contract = manifest["config"], manifest["config"]["observable_contract"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No automatic retry")
    for file, digest in manifest["model_files"].items():
        if R.digest(MODEL / file) != digest:
            raise ValueError("Changed fixed model")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed, pending = 0.0, 27, None
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
        F.install(observer.model, contract["query_rows"])
        load_seconds = time.monotonic() - stamp
        alpha = manifest["alphabet"]
        if not np.array_equal(
            observer.model.lm_head.weight.detach()[alpha].float().cpu().numpy(),
            H.selected(out / "head.npz", ("weights",))["weights"],
        ):
            raise ValueError("Changed head")
        units = R.read(out / "units.json")
        for bi in range(27, 67):
            if elapsed >= manifest["remaining_model_seconds"]:
                raise RuntimeError("Combined recovery cap reached;no extension")
            pending = bi
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "new_batches": completed - 27,
                    "pending_batch": bi,
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
                    states = state.float().cpu().numpy()
                if not np.isfinite(logits).all() or not np.isfinite(states).all():
                    raise ValueError("Nonfinite reading")
                R.archive(
                    out / "batches" / f"{bi:03d}.npz",
                    logits=logits[:, alpha],
                    normalizer=logsumexp(logits, -1),
                    states=states,
                    greedy=logits.argmax(-1),
                )
            finally:
                elapsed += time.monotonic() - stamp
            completed += 1
            pending = None
            H.status(
                out,
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "completed_batches": completed,
                    "new_batches": completed - 27,
                    "pending_batch": None,
                    "model_seconds": elapsed,
                },
            )
            print(
                f"HM228 {completed}/67;new={completed - 27}/40;model_seconds={elapsed:.3f}",
                flush=True,
            )
            del body, state, logits, states, ids, mask
            torch.mps.empty_cache()
        prior = R.read(out / "observable_contract.json")["prior_model_seconds"]
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_batches": completed,
                "new_batches": 40,
                "new_endpoints": 200,
                "reused_endpoints": 135,
                "prior_compatibility_batches": 2,
                "model_seconds": elapsed,
                "total_recovery_model_seconds": prior + elapsed,
                "load_seconds": load_seconds,
                "original_attempted027_reobserved": True,
                "new_API_private_train_calls": 0,
            },
        )
        R.seal(
            out,
            "collected_seal.json",
            ("head.npz", "runtime.json", "observable_contract.json")
            + tuple(f"batches/{i:03d}.npz" for i in range(67)),
        )
    except BaseException as exc:
        H.status(
            out,
            {
                "status": "FAILED",
                "pid": None,
                "completed_batches": completed,
                "new_batches": completed - 27,
                "pending_batch": pending,
                "model_seconds": elapsed,
                "error": repr(exc),
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def ordinal(out):
    """Check strict-order feasibility on the full graph, not only triangle signs."""
    cfg = R.check(out, G.STAGES[:2])["config"]["observable_contract"]
    obs = R.prepared(out)
    d = H.selected(out / "analysis.npz", ("odd",))["odd"]
    rows = []
    for q, target in enumerate(("E", "N", "A", "C", "O")):
        for margin in (0.0, cfg["ordinal_robust_margin"]):
            adj = np.zeros((24, 24), int)
            for (i, j), value in zip(obs["edges"], d[:, q], strict=True):
                if value > margin:
                    adj[i, j] = 1
                elif value < -margin:
                    adj[j, i] = 1
            _, labels = connected_components(adj, directed=True, connection="strong")
            components = [
                np.flatnonzero(labels == k).tolist() for k in np.unique(labels)
            ]
            cyclic = [c for c in components if len(c) > 1]
            rows.append(
                {
                    "target": target,
                    "margin": margin,
                    "directed_edges": int(adj.sum()),
                    "strict_scalar_order_feasible_on_observed_edges": not cyclic,
                    "cyclic_components": cyclic,
                    "not_personality_validity": True,
                }
            )
    R.write(
        out / "ordinal_feasibility.json",
        {"checks": rows, "no_new_model_or_fitted_coefficients": True},
    )
    R.seal(out, "ordinal_seal.json", ("ordinal_feasibility.json",))
    print("HM228 ordinal feasibility retained for all5targets", flush=True)


def main():
    """Explicit stages keep failures and the original scientific analysis intact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "collect", "analyze", "verify", "ordinal", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(args.out, G.STAGES + ("ordinal_seal.json",))
        print("HM228 COMPLETE SEALS PASS;HM226/227 failures retained")
    elif args.stage == "analyze":
        G.analyze(args.out)
    elif args.stage == "verify":
        G.verify(args.out)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
