#!/usr/bin/env python3
"""Finite capture/export/cold-reader construction, not a prediction search."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_criterion_capsule as C
from grain_research.construction import math_target_contract as T
from grain_research.construction.math_prospective_evidence import bootstrap
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_criterion_capsule_hm223.json"
DEFAULT = ROOT / "results/grain_math_response_v1/criterion_capsule_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CRITERION_CAPSULE_PLAN_HM223.md",
    "grain_research/construction/math_criterion_capsule.py",
    "scripts/run_grain_criterion_capsule_hm223.py",
    "tests/test_grain_criterion_capsule.py",
    "grain_research/construction/math_contract_routes.py",
    "grain_research/construction/math_target_contract.py",
    "grain_research/construction/math_prospective_evidence.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
)


def load_npz(path):
    """Load numeric packet artifacts without raw questionnaires or credentials."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prepare(out):
    """Bind existing evidence and excerpt-free recipient before any new forward."""
    if out.exists():
        raise FileExistsError("Prior or uncertain output must be preserved")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    manifest = R.check(
        parent, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    grand = ROOT / manifest["config"]["parent"]
    gm = R.check(
        grand, ("collected_seal.json", "analyzed_seal.json", "verification_seal.json")
    )
    units = R.read(parent / "units.json")
    empty = R.read(grand / "units.json")[-6:]
    if len(units) != 48 or any(u["author"] != -1 or u["arm"] != -1 for u in empty):
        raise ValueError("Fixed author set or empty calibration changed")
    receiver = {"input_ids": [u["input_ids"] for u in empty]}
    allrows = [r[:-1] for u in units for r in u["input_ids"]] + [
        r[:-1] for r in receiver["input_ids"]
    ]
    count = 0
    for offset in range(1, min(map(len, allrows)) + 1):
        if len({r[-offset] for r in allrows}) != 1:
            break
        count = offset
    if (
        count < 1
        or max(len(r) for u in units for r in u["input_ids"]) + 16 > cfg["max_tokens"]
    ):
        raise ValueError("Finite suffix or token cap unsupported")
    for u in units:
        suffix = C.suffix_support(u["input_ids"], count)
        if min(suffix[0]) <= max(u["body"]):
            raise ValueError("Suffix overlaps an evidence event")
    C.suffix_support(receiver["input_ids"], count)
    jobs = [
        {"author": u, "mode": mode, "sign": sign}
        for u in range(24)
        for mode in C.MODES
        for sign in (1, -1)
    ]
    jobs += [
        {"author": -1, "mode": "native", "sign": 0},
        {"author": -1, "mode": "zero", "sign": 0},
    ]
    (out / "capture").mkdir(parents=True, exist_ok=False)
    (out / "cold").mkdir(exist_ok=False)
    R.write(out / "units.json", units)
    R.write(out / "receiver.json", receiver)
    R.write(out / "jobs.json", jobs)
    files = [
        parent / "manifest.json",
        parent / "units.json",
        parent / "head.npz",
        grand / "manifest.json",
        grand / "units.json",
    ]
    files += [parent / "batches" / f"{3 * b:03d}.npz" for b in range(48)]
    files += [grand / "batches" / f"{b:03d}.npz" for b in range(48)]
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "suffix_tokens": count,
            "alphabet": gm["alphabet"],
            "model_files": gm["model_files"],
            "grand": str(grand.relative_to(ROOT)),
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
            "neural_reader_retained": True,
            "source_outcomes_not_used_for_capsule": True,
        },
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "catalogue_events": 288,
            "actual_selected_events": 192,
            "source_bodies": 48,
            "suffix_tokens": count,
            "source_lengths": [list(map(len, u["input_ids"])) for u in units],
            "recipient_lengths": list(map(len, receiver["input_ids"])),
            "recipient_has_author_excerpts": False,
            "source_capture_batches": 96,
            "cold_batches": len(jobs),
            "fit_or_application_labels": False,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        ("units.json", "receiver.json", "jobs.json", "input_quality.json"),
    )
    H.status(out, {"status": "PREPARED", "completed_batches": 0, "model_seconds": 0.0})
    print(f"PREPARED HM223:96capture+146cold;suffix={count}", flush=True)


def observer_for(manifest):
    """Load the same offline observer; model time and loading time are separate."""
    import torch

    from grain_research.vector_mechanisms.trace import DecoderObserver

    for name, digest in manifest["model_files"].items():
        if R.digest(MODEL / name) != digest:
            raise ValueError("Changed frozen model")
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    stamp = time.monotonic()
    cfg = manifest["config"]
    observer = DecoderObserver.load(
        str(MODEL),
        device=cfg["device"],
        dtype=cfg["dtype"],
        max_tokens=cfg["max_tokens"],
    )
    return observer, time.monotonic() - stamp


def capture(out):
    """Collect source-observed banks once; preserve matched native/closed controls."""
    manifest = R.check(out)
    cfg, count = manifest["config"], manifest["suffix_tokens"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No terminal or uncertain capture retry")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    elapsed, completed, files = 0.0, 0, []
    try:
        observer, load = observer_for(manifest)
        alphabet = manifest["alphabet"]["token_ids"]
        head = observer.model.lm_head.weight.detach()[alphabet].float().cpu().numpy()
        if not np.array_equal(
            head, load_npz(ROOT / cfg["parent"] / "head.npz")["weights"]
        ):
            raise ValueError("Changed digit head")
        R.archive(out / "head.npz", weights=head)
        units = R.read(out / "units.json")
        pair_banks, capsules, replay = [], [], []
        for b, unit in enumerate(units):
            banks = []
            for mode in ("native", "closed"):
                if elapsed >= cfg["max_model_seconds"]:
                    raise RuntimeError("Finite model cap; no expansion")
                H.status(
                    out,
                    {
                        "status": "CAPTURING",
                        "pid": os.getpid(),
                        "completed_batches": completed,
                        "pending": [b, mode],
                        "model_seconds": elapsed,
                        "capture_load_seconds": load,
                    },
                )
                stamp = time.monotonic()
                logits, states, values, receipts = C.native_batch(
                    observer, unit, count, closed=mode == "closed"
                )
                elapsed += time.monotonic() - stamp
                selected = logits[:, alphabet]
                oldpath = (
                    ROOT / manifest["grand"] / "batches" / f"{b:03d}.npz"
                    if mode == "native"
                    else ROOT / cfg["parent"] / "batches" / f"{3 * b:03d}.npz"
                )
                old = load_npz(oldpath)
                error = float(np.max(np.abs(selected - old["logits"])))
                if error > 8e-5:
                    raise ValueError("Source capture does not replay frozen endpoint")
                replay.append(error)
                stem = f"capture/{b:03d}_{mode}"
                R.archive(
                    out / (stem + ".npz"),
                    logits=selected,
                    states=states,
                    normalizer=logsumexp(logits, -1),
                    values=values,
                )
                R.write(
                    out / (stem + ".json"),
                    {
                        "batch": b,
                        "mode": mode,
                        "receipts": receipts,
                        "old_endpoint_error": error,
                    },
                )
                files.extend((stem + ".npz", stem + ".json"))
                banks.append(values)
                completed += 1
            pair_banks.append(banks)
            if b % 2 == 1:
                previous = pair_banks[-2]
                capsules.append(
                    C.capsule(
                        np.stack([previous[0], banks[0]]),
                        np.stack([previous[1], banks[1]]),
                    )
                )
                pair_banks.clear()
            print(f"CAPTURE {b + 1}/48;modelsec={elapsed:.3f}", flush=True)
        R.archive(out / "capsules.npz", values=np.stack(capsules))
        R.write(
            out / "capture_summary.json",
            {
                "batches": completed,
                "model_seconds": elapsed,
                "load_seconds": load,
                "native_closed_replay_max": max(replay),
                "capsule_shape": list(np.stack(capsules).shape),
                "source_final_probabilities_exported": False,
            },
        )
        R.seal(
            out,
            "captured_seal.json",
            (*files, "head.npz", "capsules.npz", "capture_summary.json"),
        )
        H.status(
            out,
            {
                "status": "CAPTURED",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "capture_load_seconds": load,
            },
        )
    except Exception as exc:
        H.status(
            out,
            {
                "status": "FAILED_CAPTURE",
                "pid": None,
                "completed_batches": completed,
                "model_seconds": elapsed,
                "error": str(exc),
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def cold(out):
    """Invoke capsules in a fresh process without loading source unit token IDs."""
    manifest = R.check(out, ("captured_seal.json",))
    cfg, count = manifest["config"], manifest["suffix_tokens"]
    runtime = R.read(out / "runtime.json")
    if runtime["status"] != "CAPTURED":
        raise ValueError("No terminal or uncertain cold retry")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    elapsed, completed, files = runtime["model_seconds"], 0, []
    try:
        observer, load = observer_for(manifest)
        alphabet = manifest["alphabet"]["token_ids"]
        receiver, jobs = R.read(out / "receiver.json"), R.read(out / "jobs.json")
        bank = load_npz(out / "capsules.npz")["values"]
        for i, job in enumerate(jobs):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Shared finite model cap; no expansion")
            H.status(
                out,
                {
                    "status": "COLD_RUNNING",
                    "pid": os.getpid(),
                    "completed_capture_batches": 96,
                    "completed_cold_batches": completed,
                    "pending": i,
                    "model_seconds": elapsed,
                    "cold_load_seconds": load,
                },
            )
            donor = None
            if job["mode"] in C.MODES:
                donor = C.condition_values(bank[job["author"]], job["mode"]) * (
                    cfg["dose"] * job["sign"]
                )
            elif job["mode"] == "zero":
                donor = np.zeros_like(bank[0])
            stamp = time.monotonic()
            logits, states, _values, receipts = C.native_batch(
                observer, receiver, count, values=donor
            )
            elapsed += time.monotonic() - stamp
            stem = f"cold/{i:03d}"
            R.archive(
                out / (stem + ".npz"),
                logits=logits[:, alphabet],
                states=states,
                normalizer=logsumexp(logits, -1),
            )
            R.write(out / (stem + ".json"), {"job": job, "receipts": receipts})
            files.extend((stem + ".npz", stem + ".json"))
            completed += 1
            if i % 12 == 0 or i >= 144:
                print(f"COLD {completed}/146;totalmodelsec={elapsed:.3f}", flush=True)
        R.write(
            out / "cold_summary.json",
            {
                "batches": completed,
                "total_model_seconds": elapsed,
                "load_seconds": load,
                "source_token_ids_interpreted": False,
                "source_attention_probabilities_loaded": False,
                "recipient_has_author_excerpts": False,
            },
        )
        R.seal(out, "cold_seal.json", (*files, "cold_summary.json"))
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "completed_capture_batches": 96,
                "completed_cold_batches": completed,
                "model_seconds": elapsed,
                "cold_load_seconds": load,
                "capture_load_seconds": runtime["capture_load_seconds"],
            },
        )
    except Exception as exc:
        H.status(
            out,
            {
                "status": "FAILED_COLD",
                "pid": None,
                "completed_cold_batches": completed,
                "model_seconds": elapsed,
                "error": str(exc),
            },
        )
        raise
    finally:
        lock.unlink(missing_ok=True)


def analyze(out):
    """Report all names, criteria and signed counterfactuals without score selection."""
    manifest = R.check(out, ("captured_seal.json", "cold_seal.json"))
    cfg = manifest["config"]
    logits = np.stack(
        [load_npz(out / "cold" / f"{i:03d}.npz")["logits"] for i in range(144)]
    ).reshape(24, 3, 2, 3, 2, 10)
    y, linear, curvature = T.responses(logits)
    z = y[:, :, 0] - y[:, :, 1]
    m = z[..., 0] - z[..., 1]
    primary = m[:, 0, 2] - m[:, 1, 2]
    excess = m[:, 0, 2] - m[:, 2, 2]
    fields = np.column_stack((primary, excess, m.reshape(24, -1)))
    ci = bootstrap(fields, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    rows = []
    for mode, label in enumerate(C.MODES):
        for name, text in enumerate(T.NAMES):
            index = 2 + mode * 3 + name
            rows.append(
                {
                    "mode": label,
                    "name": text,
                    "mean_definition_action": float(m[:, mode, name].mean()),
                    "ci_low": float(ci[index, 0]),
                    "ci_high": float(ci[index, 1]),
                    "positive_authors": int((m[:, mode, name] > 0).sum()),
                }
            )
    R.table(out / "criterion_by_mode.csv", rows)
    R.table(
        out / "author_actions.csv",
        [
            {
                "author": u,
                "aligned_K": float(m[u, 0, 2]),
                "swapped_K": float(m[u, 1, 2]),
                "common_K": float(m[u, 2, 2]),
                "primary": float(primary[u]),
                "aligned_minus_common": float(excess[u]),
            }
            for u in range(24)
        ],
    )
    bank = load_npz(out / "capsules.npz")["values"]
    common = np.stack([C.condition_values(v, "common") for v in bank])
    specific = bank - common
    source_logits = np.stack(
        [load_npz(out / "capture" / f"{b:03d}_native.npz")["logits"] for b in range(48)]
    ).reshape(24, 2, 3, 2, 10)
    source = T.action_accounts(T.responses(source_logits)[0])["anonymous_definition"]
    lm = (linear[:, :, 0] - linear[:, :, 1])[..., 0] - (
        linear[:, :, 0] - linear[:, :, 1]
    )[..., 1]
    cm = (curvature[:, :, 0] - curvature[:, :, 1])[..., 0] - (
        curvature[:, :, 0] - curvature[:, :, 1]
    )[..., 1]
    summary = {
        "authors": 24,
        "primary": {
            "mean": float(primary.mean()),
            "ci": ci[0].tolist(),
            "positive_authors": int((primary > 0).sum()),
            "negative_authors": int((primary < 0).sum()),
        },
        "aligned_minus_common": {"mean": float(excess.mean()), "ci": ci[1].tolist()},
        "all_mode_name_means": m.mean(0).tolist(),
        "primary_linear_curvature": [
            float((lm[:, 0, 2] - lm[:, 1, 2]).mean()),
            float((cm[:, 0, 2] - cm[:, 1, 2]).mean()),
        ],
        "source_cold_centered_covariance": float(
            np.mean((source - source.mean()) * (m[:, 0, 2] - m[:, 0, 2].mean()))
        ),
        "source_cold_uncentered_product": float(np.mean(source * m[:, 0, 2])),
        "capsule_common_specific_energy": [
            float(np.mean(common**2)),
            float(np.mean(specific**2)),
        ],
        "capsule_signed_cross": float(np.mean(common * specific)),
        "units": "nats;NOT_personality_r_or_accuracy",
        "natural_recipient_attention": True,
        "original_scope_complete": False,
    }
    R.archive(
        out / "analysis.npz",
        logits=logits,
        y=y,
        linear=linear,
        curvature=curvature,
        z=z,
        m=m,
        primary=primary,
        fields=fields,
        ci=ci,
        source=source,
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        ("analysis.npz", "summary.json", "criterion_by_mode.csv", "author_actions.csv"),
    )
    print(summary, flush=True)


def verify(out):
    """Independently re-sum digit odds, source banks, bootstrap and graph receipts."""
    manifest = R.check(
        out, ("captured_seal.json", "cold_seal.json", "analyzed_seal.json")
    )
    cfg = manifest["config"]
    analysis, errors = load_npz(out / "analysis.npz"), []
    head = load_npz(out / "head.npz")["weights"].astype(float)
    head_error, guards, prefix_checks = 0.0, 0.0, 0
    exported = load_npz(out / "capsules.npz")["values"]
    for u in range(24):
        banks = {}
        for arm in range(2):
            for mode in ("native", "closed"):
                b = 2 * u + arm
                data = load_npz(out / "capture" / f"{b:03d}_{mode}.npz")
                banks[arm, mode] = data["values"]
                head_error = max(
                    head_error,
                    float(
                        np.max(
                            np.abs(
                                data["states"].astype(float) @ head.T - data["logits"]
                            )
                        )
                    ),
                )
        literal = (
            banks[0, "native"].astype(float)
            - banks[0, "closed"].astype(float)
            - banks[1, "native"].astype(float)
            + banks[1, "closed"].astype(float)
        )
        errors.append(float(np.max(np.abs(literal - exported[u]))))
    for b in range(48):
        native = R.read(out / "capture" / f"{b:03d}_native.json")["receipts"]
        closed = R.read(out / "capture" / f"{b:03d}_closed.json")["receipts"]
        guards = max(guards, max(r["closed_body_weight_max"] for r in closed))
        for r, base in zip(closed, native, strict=True):
            if r["source_body_KV_hash"] != base["source_body_KV_hash"]:
                raise ValueError("Closing source edges changed body KV")
    baseline = load_npz(out / "cold/144.npz")
    base_receipts = R.read(out / "cold/144.json")["receipts"]
    for i in range(146):
        data = load_npz(out / "cold" / f"{i:03d}.npz")
        receipts = R.read(out / "cold" / f"{i:03d}.json")["receipts"]
        head_error = max(
            head_error,
            float(
                np.max(np.abs(data["states"].astype(float) @ head.T - data["logits"]))
            ),
        )
        for r, base in zip(receipts, base_receipts, strict=True):
            if r["prefix_KV_hash"] != base["prefix_KV_hash"]:
                raise ValueError("Recipient prefix was changed by injection")
            prefix_checks += 1
    zero = load_npz(out / "cold/145.npz")
    zero_error = float(np.max(np.abs(zero["logits"] - baseline["logits"])))
    yy = np.empty_like(analysis["y"])
    for index in np.ndindex(yy.shape):
        z = analysis["logits"][index].astype(float)
        shift = max(z)
        mass = [
            sum(np.exp(float(z[j]) - shift) for j in range(a, b))
            for a, b in ((0, 5), (5, 10))
        ]
        yy[index] = np.log(mass[1]) - np.log(mass[0])
    zz = yy[:, :, 0] - yy[:, :, 1]
    mm = zz[..., 0] - zz[..., 1]
    errors.extend(
        (
            float(np.max(np.abs(yy - analysis["y"]))),
            float(np.max(np.abs(mm - analysis["m"]))),
            float(np.max(np.abs((mm[:, 0, 2] - mm[:, 1, 2]) - analysis["primary"]))),
        )
    )
    ix = np.random.default_rng(cfg["bootstrap_seed"]).integers(
        24, size=(cfg["bootstrap_draws"], 24)
    )
    intervals = np.quantile(
        np.asarray([sum(analysis["fields"][j] for j in draw) / 24 for draw in ix]),
        [0.025, 0.975],
        axis=0,
    ).T
    errors.append(float(np.max(np.abs(intervals - analysis["ci"]))))
    error = max(errors)
    passed = error < 1e-10 and head_error < 8e-5 and guards == 0 and zero_error == 0
    result = {
        "PASS": bool(passed),
        "alternate_arithmetic_max_error": error,
        "native_head_max_error": head_error,
        "source_closed_edge_max": guards,
        "recipient_prefix_checks": prefix_checks,
        "zero_injection_error": zero_error,
        "same_agent_not_blind": True,
        "actual_new_batches": 242,
        "actual_new_endpoints": 1452,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not passed:
        raise ValueError("Keep failed scientific verification; no rescue rerun")
    print(result, flush=True)


def main():
    """Run one explicit stage; no automatic next packet or retry queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "capture", "cold", "analyze", "verify", "check")
    )
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.stage == "check":
        R.check(
            args.out,
            (
                "captured_seal.json",
                "cold_seal.json",
                "analyzed_seal.json",
                "verification_seal.json",
            ),
        )
        print("ALL HM223 SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
