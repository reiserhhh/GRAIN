#!/usr/bin/env python3
"""Finite source-block re-encoding and fresh observer assembly, without fitting."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_capsule_assembly as A
from grain_research.construction import math_criterion_capsule as C
from grain_research.construction import math_target_contract as T
from grain_research.construction.math_prospective_evidence import bootstrap
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_criterion_capsule_hm223 as P
from scripts import run_grain_observed_family_hm170 as R
from scripts import run_grain_prospective_evidence_hm218 as H

CONFIG = "configs/grain_capsule_assembly_hm224.json"
DEFAULT = ROOT / "results/grain_math_response_v1/capsule_assembly_r1"
SCIENCE = (
    CONFIG,
    "docs/GRAIN_CAPSULE_ASSEMBLY_PLAN_HM224.md",
    "grain_research/construction/math_capsule_assembly.py",
    "scripts/run_grain_capsule_assembly_hm224.py",
    "tests/test_grain_capsule_assembly.py",
    "grain_research/construction/math_criterion_capsule.py",
    "grain_research/construction/math_contract_routes.py",
    "grain_research/construction/math_target_contract.py",
    "grain_research/construction/math_prospective_evidence.py",
    "scripts/run_grain_criterion_capsule_hm223.py",
    "scripts/run_grain_observed_family_hm170.py",
    "scripts/run_grain_prospective_evidence_hm218.py",
)


def prepare(out):
    """Freeze one exact public-token partition, not a new sample or split search."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve all prior or uncertain outputs")
    cfg = R.read(ROOT / CONFIG)
    parent = ROOT / cfg["parent"]
    pm = R.check(
        parent,
        (
            "captured_seal.json",
            "cold_seal.json",
            "analyzed_seal.json",
            "verification_seal.json",
        ),
    )
    full = R.read(parent / "units.json")
    grand = ROOT / pm["grand"]
    bindings = R.read(grand / "bindings.json")
    if len(full) != 48 or len(bindings) != 48:
        raise ValueError("Fixed author/event scope changed")
    tok = AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False
    )
    count = pm["suffix_tokens"]
    receiver = R.read(parent / "receiver.json")
    units, provenance, lengths = [], [], []
    for u in range(24):
        for block in range(2):
            for arm in range(2):
                root = full[2 * u + arm]
                binding = bindings[2 * u + arm]
                if (root["author"], root["arm"]) != (u, arm):
                    raise ValueError("Author/arm address changed")
                if (binding["author"], binding["arm"]) != (u, arm):
                    raise ValueError("Event attribution changed")
                events = A.event_partition(binding["event_ids"])
                rows, supports = [], []
                for ids in root["input_ids"]:
                    decoded = tok.decode(
                        ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if tok.encode(decoded, add_special_tokens=False) != ids:
                        raise ValueError("Exact full token round trip failed")
                    text, _pieces = A.partition_prompt(decoded, block)
                    encoded = tok(
                        text, add_special_tokens=False, return_offsets_mapping=True
                    )
                    row = encoded["input_ids"]
                    start = text.index(f"Excerpt {2 * block + 1}:\n")
                    end = text.index("Evidence ends here.")
                    body = [
                        i
                        for i, (a, b) in enumerate(encoded["offset_mapping"])
                        if b > start and a < end
                    ]
                    if not body or len(row) + 16 > cfg["max_tokens"]:
                        raise ValueError("No unsupported subset truncation or omission")
                    rows.append(row)
                    supports.append(body)
                    lengths.append(len(row))
                if any(s != supports[0] for s in supports):
                    raise ValueError("Contract changed physical evidence support")
                suffix = C.suffix_support(rows, count)
                if min(suffix[0]) <= max(supports[0]):
                    raise ValueError("Carrier overlaps source events")
                unit = {
                    "author": u,
                    "block": block,
                    "arm": arm,
                    "input_ids": rows,
                    "body": supports[0],
                }
                units.append(unit)
                provenance.append(
                    {
                        "author": u,
                        "block": block,
                        "arm": arm,
                        "events": events[block],
                        "root_events": binding["event_ids"],
                        "body_hashes": [
                            binding["body_hashes"][j] for j in A.BLOCKS[block]
                        ],
                    }
                )
    # Transport uses the same suffix tokens, not merely the same suffix length.
    rs = C.suffix_support(receiver["input_ids"], count)
    ref = [receiver["input_ids"][0][i] for i in rs[0]]
    for unit in units:
        ss = C.suffix_support(unit["input_ids"], count)
        if [unit["input_ids"][0][i] for i in ss[0]] != ref:
            raise ValueError("Source/receiver suffix identity changed")
    jobs = [
        {"author": u, "mode": mode, "sign": sign}
        for u in range(24)
        for mode in A.MODES
        for sign in (1, -1)
    ]
    jobs += [
        {"author": -1, "mode": "native", "sign": 0},
        {"author": -1, "mode": "zero", "sign": 0},
    ]
    files = [
        parent / p
        for p in (
            "manifest.json",
            "units.json",
            "receiver.json",
            "capsules.npz",
            "head.npz",
            "verification.json",
        )
    ]
    files += [
        parent / "cold" / f"{6 * u + s:03d}.npz" for u in range(24) for s in range(2)
    ]
    files += [parent / "cold/144.npz", grand / "bindings.json"]
    (out / "capture").mkdir(parents=True, exist_ok=False)
    (out / "cold").mkdir(exist_ok=False)
    R.write(out / "units.json", units)
    R.write(out / "event_partition.json", provenance)
    R.write(out / "receiver.json", receiver)
    R.write(out / "jobs.json", jobs)
    R.write(
        out / "manifest.json",
        {
            "config": cfg,
            "suffix_tokens": count,
            "alphabet": pm["alphabet"],
            "model_files": pm["model_files"],
            "code": {p: R.digest(ROOT / p) for p in SCIENCE},
            "inputs": {str(p.relative_to(ROOT)): R.digest(p) for p in files},
            "original_scope_complete": False,
            "neural_compiler_and_reader_retained": True,
        },
    )
    R.write(
        out / "input_quality.json",
        {
            "authors": 24,
            "selected_distinct_events": 192,
            "new_subset_bodies": 96,
            "contracts_per_event_not_independent": 6,
            "source_capture_batches": 192,
            "cold_batches": len(jobs),
            "reused_joint_recipient_batches": 48,
            "new_subset_lengths": lengths,
            "partition_search": False,
            "labels_or_raw_private_files_read": False,
            "reencoding_not_lossless_vector_partition": True,
        },
    )
    R.seal(
        out,
        "prepared_seal.json",
        (
            "units.json",
            "event_partition.json",
            "receiver.json",
            "jobs.json",
            "input_quality.json",
        ),
    )
    H.status(out, {"status": "PREPARED", "model_seconds": 0.0, "completed_batches": 0})
    print("PREPARED HM224:192capture+146cold/2028newendpoints", flush=True)


def capture(out):
    """Capture all fixed subsets once; source outcomes never fit the assembly."""
    manifest = R.check(out)
    cfg, count = manifest["config"], manifest["suffix_tokens"]
    if R.read(out / "runtime.json")["status"] != "PREPARED":
        raise ValueError("No uncertain or terminal capture retry")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed, files, pieces, pair = 0.0, 0, [], [], []
    try:
        observer, load = P.observer_for(manifest)
        alphabet = manifest["alphabet"]["token_ids"]
        head = observer.model.lm_head.weight.detach()[alphabet].float().cpu().numpy()
        if not np.array_equal(
            head, P.load_npz(ROOT / cfg["parent"] / "head.npz")["weights"]
        ):
            raise ValueError("Changed source digit head")
        R.archive(out / "head.npz", weights=head)
        for b, unit in enumerate(R.read(out / "units.json")):
            bank = []
            for mode in ("native", "closed"):
                if elapsed >= cfg["max_model_seconds"]:
                    raise RuntimeError("Finite shared model cap reached")
                H.status(
                    out,
                    {
                        "status": "CAPTURING",
                        "pid": os.getpid(),
                        "pending": [b, mode],
                        "completed_batches": completed,
                        "model_seconds": elapsed,
                        "capture_load_seconds": load,
                    },
                )
                stamp = time.monotonic()
                logits, states, values, receipts = C.native_batch(
                    observer, unit, count, closed=mode == "closed"
                )
                elapsed += time.monotonic() - stamp
                stem = f"capture/{b:03d}_{mode}"
                R.archive(
                    out / (stem + ".npz"),
                    logits=logits[:, alphabet],
                    states=states,
                    values=values,
                )
                R.write(
                    out / (stem + ".json"),
                    {"batch": b, "mode": mode, "receipts": receipts},
                )
                files.extend((stem + ".npz", stem + ".json"))
                bank.append(values)
                completed += 1
            pair.append(bank)
            if b % 2 == 1:
                pieces.append(
                    C.capsule(
                        np.stack([pair[0][0], pair[1][0]]),
                        np.stack([pair[0][1], pair[1][1]]),
                    )
                )
                pair.clear()
            if b % 8 == 7 or b == 95:
                print(f"CAPTURE {b + 1}/96;modelsec={elapsed:.3f}", flush=True)
        parts = np.stack(pieces).reshape(24, 2, *pieces[0].shape)
        joint = P.load_npz(ROOT / cfg["parent"] / "capsules.npz")["values"]
        summed, coupling = A.assemble(np.moveaxis(parts, 1, 0), joint)
        R.archive(out / "assembly.npz", parts=parts, summed=summed, coupling=coupling)
        R.write(
            out / "capture_summary.json",
            {
                "batches": completed,
                "model_seconds": elapsed,
                "load_seconds": load,
                "part_shape": list(parts.shape),
                "joint_reconstruction_is_by_definition": True,
            },
        )
        R.seal(
            out,
            "captured_seal.json",
            (*files, "head.npz", "assembly.npz", "capture_summary.json"),
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
    """Read only numerical part capsules in an excerpt-free fresh process."""
    manifest = R.check(out, ("captured_seal.json",))
    cfg, count = manifest["config"], manifest["suffix_tokens"]
    runtime = R.read(out / "runtime.json")
    if runtime["status"] != "CAPTURED":
        raise ValueError("No uncertain or terminal cold retry")
    lock = out / "ACTIVE.lock"
    with lock.open("x") as f:
        f.write(str(os.getpid()))
    elapsed, completed, files = runtime["model_seconds"], 0, []
    try:
        observer, load = P.observer_for(manifest)
        alphabet = manifest["alphabet"]["token_ids"]
        receiver = R.read(out / "receiver.json")
        banks = P.load_npz(out / "assembly.npz")
        for i, job in enumerate(R.read(out / "jobs.json")):
            if elapsed >= cfg["max_model_seconds"]:
                raise RuntimeError("Finite shared model cap reached")
            H.status(
                out,
                {
                    "status": "COLD_RUNNING",
                    "pid": os.getpid(),
                    "completed_capture_batches": 192,
                    "completed_cold_batches": completed,
                    "pending": i,
                    "model_seconds": elapsed,
                    "cold_load_seconds": load,
                },
            )
            donor = None
            if job["mode"] in A.MODES:
                u = job["author"]
                value = (
                    banks["summed"][u]
                    if job["mode"] == "sum"
                    else banks["parts"][u, A.MODES.index(job["mode"])]
                )
                donor = value * (cfg["dose"] * job["sign"])
            elif job["mode"] == "zero":
                donor = np.zeros_like(banks["summed"][0])
            stamp = time.monotonic()
            logits, states, _values, receipts = C.native_batch(
                observer, receiver, count, values=donor
            )
            elapsed += time.monotonic() - stamp
            stem = f"cold/{i:03d}"
            R.archive(out / (stem + ".npz"), logits=logits[:, alphabet], states=states)
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
                "source_tokens_interpreted": False,
                "source_p_loaded": False,
            },
        )
        R.seal(out, "cold_seal.json", (*files, "cold_summary.json"))
        H.status(
            out,
            {
                "status": "COLLECTED",
                "pid": None,
                "model_seconds": elapsed,
                "completed_capture_batches": 192,
                "completed_cold_batches": completed,
                "capture_load_seconds": runtime["capture_load_seconds"],
                "cold_load_seconds": load,
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


def raw_observations(out):
    """Load new and inherited exact recipient endpoints, not source labels."""
    parent = ROOT / R.read(out / "manifest.json")["config"]["parent"]
    new = np.stack(
        [P.load_npz(out / "cold" / f"{i:03d}.npz")["logits"] for i in range(144)]
    ).reshape(24, 3, 2, 3, 2, 10)
    old = np.stack(
        [
            P.load_npz(parent / "cold" / f"{6 * u + sign:03d}.npz")["logits"]
            for u in range(24)
            for sign in range(2)
        ]
    ).reshape(24, 2, 3, 2, 10)
    return old, new


def observed_actions(logits, axis):
    """Return signed dose action and definition contrast, preserving all names."""
    fields = T.responses(logits)
    actions = [np.take(x, 0, axis=axis) - np.take(x, 1, axis=axis) for x in fields]
    return actions, [x[..., 0] - x[..., 1] for x in actions]


def analyze(out):
    """Report all finite source-compiler and recipient accounts without selection."""
    manifest = R.check(out, ("captured_seal.json", "cold_seal.json"))
    cfg = manifest["config"]
    joint_logits, new_logits = raw_observations(out)
    joint_z, joint_m = observed_actions(joint_logits, 1)
    part_z, part_m = observed_actions(new_logits, 2)
    all_accounts, contracts = [], []
    for j, p in zip(joint_m, part_m, strict=True):
        all_accounts.append(A.action_decomposition(j, p[:, 0], p[:, 1], p[:, 2]))
    contracts = A.action_decomposition(
        joint_z[0], part_z[0][:, 0], part_z[0][:, 1], part_z[0][:, 2]
    )
    accounts = all_accounts[0]
    fields = np.column_stack([accounts[k] for k in ("compiler", "reader", "total")])
    ci = bootstrap(fields, cfg["bootstrap_draws"], cfg["bootstrap_seed"])
    rows = []
    for i, account in enumerate(("compiler", "reader", "total")):
        for name, text in enumerate(T.NAMES):
            a = accounts[account][:, name]
            rows.append(
                {
                    "account": account,
                    "name": text,
                    "mean": float(a.mean()),
                    "ci_low": float(ci[3 * i + name, 0]),
                    "ci_high": float(ci[3 * i + name, 1]),
                    "rms": float(np.sqrt(np.mean(a**2))),
                    "positive": int((a > 0).sum()),
                    "negative": int((a < 0).sum()),
                }
            )
    R.table(out / "assembly_accounts.csv", rows)
    author_rows = [
        {
            "author": u,
            "joint_K": float(joint_m[0][u, 2]),
            "first_K": float(part_m[0][u, 0, 2]),
            "second_K": float(part_m[0][u, 1, 2]),
            "sum_K": float(part_m[0][u, 2, 2]),
            **{k + "_K": float(accounts[k][u, 2]) for k in accounts},
        }
        for u in range(24)
    ]
    R.table(out / "author_assembly.csv", author_rows)
    bank = P.load_npz(out / "assembly.npz")
    joint_bank = P.load_npz(ROOT / cfg["parent"] / "capsules.npz")["values"]
    summary = {
        "authors": 24,
        "primary": rows[2],
        "accounts": rows,
        "joint_first_second_sum_means": [
            joint_m[0].mean(0).tolist(),
            *part_m[0].mean(0).tolist(),
        ],
        "compiler_vector_relative_rms": float(
            np.linalg.norm(bank["coupling"]) / np.linalg.norm(joint_bank)
        ),
        "compiler_vector_ratio_not_psych_information": True,
        "anonymous_linear_curvature": {
            k: [
                float(all_accounts[1][k][:, 2].mean()),
                float(all_accounts[2][k][:, 2].mean()),
            ]
            for k in accounts
        },
        "units": "nats;NOT_personality_r_or_accuracy",
        "new_batches": 338,
        "new_endpoints": 2028,
        "original_scope_complete": False,
    }
    R.archive(
        out / "analysis.npz",
        joint_logits=joint_logits,
        new_logits=new_logits,
        joint_actions=joint_z[0],
        new_actions=part_z[0],
        joint_m=joint_m[0],
        part_m=part_m[0],
        fields=fields,
        ci=ci,
        **accounts,
        **{k + "_contracts": v for k, v in contracts.items()},
    )
    R.write(out / "summary.json", summary)
    R.seal(
        out,
        "analyzed_seal.json",
        (
            "analysis.npz",
            "summary.json",
            "assembly_accounts.csv",
            "author_assembly.csv",
        ),
    )
    print(summary, flush=True)


def verify(out):
    """Alternate scalar arithmetic, event partition, source and cold boundary audit."""
    manifest = R.check(
        out, ("captured_seal.json", "cold_seal.json", "analyzed_seal.json")
    )
    cfg = manifest["config"]
    z, bank = P.load_npz(out / "analysis.npz"), P.load_npz(out / "assembly.npz")
    errors, head_error, forbidden, prefix_checks = [], 0.0, 0.0, 0
    head = P.load_npz(out / "head.npz")["weights"].astype(float)
    for b in range(96):
        records = {}
        for mode in ("native", "closed"):
            data = P.load_npz(out / "capture" / f"{b:03d}_{mode}.npz")
            head_error = max(
                head_error,
                float(
                    np.max(
                        np.abs(data["states"].astype(float) @ head.T - data["logits"])
                    )
                ),
            )
            records[mode] = R.read(out / "capture" / f"{b:03d}_{mode}.json")["receipts"]
        for n, c in zip(records["native"], records["closed"], strict=True):
            if n["source_body_KV_hash"] != c["source_body_KV_hash"]:
                raise ValueError("Closing post-body graph changed source body")
            forbidden = max(forbidden, c["closed_body_weight_max"])
    for u in range(24):
        for block in range(2):
            arrays = [
                P.load_npz(
                    out / "capture" / f"{4 * u + 2 * block + arm:03d}_{mode}.npz"
                )["values"].astype(float)
                for arm, mode in (
                    (0, "native"),
                    (0, "closed"),
                    (1, "native"),
                    (1, "closed"),
                )
            ]
            literal = arrays[0] - arrays[1] - arrays[2] + arrays[3]
            errors.append(float(np.max(np.abs(literal - bank["parts"][u, block]))))
    joint = P.load_npz(ROOT / cfg["parent"] / "capsules.npz")["values"]
    errors += [
        float(
            np.max(np.abs(bank["summed"] - bank["parts"][:, 0] - bank["parts"][:, 1]))
        ),
        float(np.max(np.abs(joint - bank["summed"] - bank["coupling"]))),
    ]
    ref = R.read(out / "cold/144.json")["receipts"]
    for i in range(146):
        data = P.load_npz(out / "cold" / f"{i:03d}.npz")
        head_error = max(
            head_error,
            float(
                np.max(np.abs(data["states"].astype(float) @ head.T - data["logits"]))
            ),
        )
        for r, n in zip(
            R.read(out / "cold" / f"{i:03d}.json")["receipts"], ref, strict=True
        ):
            if r["prefix_KV_hash"] != n["prefix_KV_hash"]:
                raise ValueError("Capsule assembly changed recipient prefix")
            prefix_checks += 1
    zero = float(
        np.max(
            np.abs(
                P.load_npz(out / "cold/144.npz")["logits"]
                - P.load_npz(out / "cold/145.npz")["logits"]
            )
        )
    )
    replay = float(
        np.max(
            np.abs(
                P.load_npz(out / "cold/144.npz")["logits"]
                - P.load_npz(ROOT / cfg["parent"] / "cold/144.npz")["logits"]
            )
        )
    )

    def scalar_y(logits):
        out_y = np.empty(logits.shape[:-1])
        for index in np.ndindex(out_y.shape):
            row = logits[index]
            shift = max(row)
            mass = [
                sum(np.exp(float(row[k]) - shift) for k in range(a, b))
                for a, b in ((0, 5), (5, 10))
            ]
            out_y[index] = np.log(mass[1]) - np.log(mass[0])
        return out_y

    jy, py = scalar_y(z["joint_logits"]), scalar_y(z["new_logits"])
    jm = (jy[:, 0, :, 0] - jy[:, 1, :, 0]) - (jy[:, 0, :, 1] - jy[:, 1, :, 1])
    pm = (py[:, :, 0, :, 0] - py[:, :, 1, :, 0]) - (
        py[:, :, 0, :, 1] - py[:, :, 1, :, 1]
    )
    literal = {
        "compiler": jm - pm[:, 2],
        "reader": pm[:, 2] - pm[:, 0] - pm[:, 1],
        "total": jm - pm[:, 0] - pm[:, 1],
    }
    errors.extend(float(np.max(np.abs(literal[k] - z[k]))) for k in literal)
    errors.append(float(np.max(np.abs(z["compiler"] + z["reader"] - z["total"]))))
    fields = np.concatenate(
        [literal[k] for k in ("compiler", "reader", "total")], axis=1
    )
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(
        24, size=(cfg["bootstrap_draws"], 24)
    )
    intervals = np.quantile(
        np.asarray([sum(fields[i] for i in draw) / 24 for draw in draws]),
        [0.025, 0.975],
        axis=0,
    ).T
    errors.append(float(np.max(np.abs(intervals - z["ci"]))))
    provenance = R.read(out / "event_partition.json")
    for u in range(24):
        for arm in range(2):
            a, b = provenance[4 * u + arm], provenance[4 * u + 2 + arm]
            if a["events"] + b["events"] != a["root_events"] or set(a["events"]) & set(
                b["events"]
            ):
                raise ValueError("Split source event provenance changed")
    maximum = max(errors)
    passed = (
        maximum < 1e-10
        and head_error < 8e-5
        and forbidden == 0
        and zero == 0
        and replay < 8e-5
    )
    result = {
        "PASS": bool(passed),
        "alternate_arithmetic_max": maximum,
        "CPU64_head_max": head_error,
        "closed_edge_max": forbidden,
        "zero_injection_error": zero,
        "inherited_empty_replay_error": replay,
        "recipient_prefix_checks": prefix_checks,
        "same_agent_not_blind": True,
    }
    R.write(out / "verification.json", result)
    R.seal(out, "verification_seal.json", ("verification.json",))
    if not passed:
        raise ValueError("Keep failed verification; no rescue rerun")
    print(result, flush=True)


def main():
    """Run only one requested finite stage, never auto-retry or continue a queue."""
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
        print("ALL HM224 SEALS PASS")
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
