#!/usr/bin/env python3
"""Construct a finite response observation package, not a prediction search."""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_expression_response as E
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_recursive_collector as C
from grain_research.construction import math_response_observability as M
from grain_research.vector_mechanisms.navigation import MODEL
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_expression_response_hm119 as P
from scripts import run_grain_native_remainder_hm124 as F

SPEC = ROOT / "configs/grain_response_observability_hm127.json"
CODE = (
    "configs/grain_response_observability_hm127.json",
    "docs/GRAIN_RESPONSE_OBSERVABILITY_PLAN_HM127.md",
    "grain_research/construction/math_response_observability.py",
    "scripts/run_grain_response_observability_hm127.py",
    "scripts/score_grain_response_observability_cold.py",
    "scripts/readback_grain_response_observability_hm127.py",
    "tests/test_grain_response_observability.py",
    "grain_research/construction/math_expression_response.py",
    "grain_research/construction/math_history_routing.py",
    "grain_research/construction/math_recursive_collector.py",
    "scripts/run_grain_expression_response_hm119.py",
    "scripts/run_grain_math_span_binding.py",
)


def save(path, **arrays):
    """Preserve every prior artifact by exclusively creating numeric files."""
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def timeout(_signum, _frame):
    """Stop rather than renew the fixed packet CPU allowance."""
    raise TimeoutError("HM127 original CPU allowance exhausted")


@contextlib.contextmanager
def budget(out, phase):
    """Record terminal errors and cumulative process CPU at every phase."""
    state = H.load(out / "status.json")
    remaining = H.load(SPEC)["max_cpu_seconds"] - state["cpu_seconds"]
    if remaining <= 0 or state["state"] == "FAILED":
        raise RuntimeError("Failed/exhausted package is not replayed")
    before = time.process_time()
    old = signal.signal(signal.SIGPROF, timeout)
    signal.setitimer(signal.ITIMER_PROF, remaining)
    try:
        yield state
    except BaseException as error:
        state.update(state="FAILED", failed_phase=phase, error=repr(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, old)
        state["cpu_seconds"] += time.process_time() - before
        state["last_phase"] = phase
        H.write(out / "status.json", state)


def check(out):
    """Authenticate construction rules and source/input bytes before use."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Seal fresh controls and fixed readers before their new observations."""
    import torch

    if out.exists():
        raise FileExistsError("Do not replace previous/failed packets")
    (out / "00").mkdir(parents=True)
    H.write(out / "status.json", {"state": "PREPARING", "cpu_seconds": 0.0})
    with budget(out, "prepare") as state:
        spec = H.load(SPEC)
        old, fresh = ROOT / spec["source"], ROOT / spec["fresh"]
        previous = P.check(old)
        F.check(fresh)
        H.verify_hashes(old, H.load(old / "02_corrected/construction_seal.json"))
        rows = H.load(fresh / "00/rows.json")
        reader = H.load(old / "00/reader.json")
        if (
            len(rows) != spec["max_new_history_calls"]
            or reader["targets"] != spec["targets"]
        ):
            raise ValueError("All fresh controls and all six fixed targets required")
        with np.load(old / "00/coordinates.npz", allow_pickle=False) as z:
            seeds = z["seeds"]
        focus_ids = previous["focus_ids"]
        config = previous["config"]
        length = len(rows[0]["input_ids"])
        current = rows[0]["current"]
        if any(
            len(r["input_ids"]) != length
            or r["current"] != current
            or [r["input_ids"][i] for i in current] != focus_ids[0]
            for r in rows
        ):
            raise ValueError("Registered current positions and identities required")
        dim = config["hidden_size"] // config["num_attention_heads"]
        inv = 1 / (
            config["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        frequency = torch.outer(torch.arange(length, dtype=torch.float32), inv)
        angles = torch.cat((frequency, frequency), dim=1)
        save(
            out / "00/coordinates.npz",
            seeds=seeds,
            cos=angles.cos().numpy(),
            sin=angles.sin().numpy(),
        )
        H.write(out / "00/fresh_rows.json", rows)
        H.write(out / "00/reader.json", reader)
        names = [
            old / n
            for n in (
                "00/manifest.json",
                "00/rows.json",
                "00/reader.json",
                "00/coordinates.npz",
                "02_corrected/constructed.npz",
                "02_corrected/construction_seal.json",
            )
        ]
        names += [
            fresh / n for n in ("00/manifest.json", "00/rows.json", "status.json")
        ]
        H.write(
            out / "00/manifest.json",
            {
                "spec": spec,
                "config": config,
                "weights": previous["weights"],
                "focus_ids": focus_ids,
                "history_length": min(current),
                "endpoint": max(current),
                "code": {name: H.sha(ROOT / name) for name in CODE},
                "source": {str(p.relative_to(ROOT)): H.sha(p) for p in names},
                "inputs": H.seals(out, out / "00"),
                "HM119_native_compatibility_failure_preserved": True,
                "original_scope_complete": False,
            },
        )
        (out / "01/units").mkdir(parents=True)
        H.write(
            out / "01/capture_status.json",
            {
                "state": "PREPARED",
                "attempts": 0,
                "completed": 0,
                "pending": None,
                "model_seconds": 0.0,
                "constructed_focus_states": 0,
                "current_seconds": 0.0,
            },
        )
        state["state"] = "PREPARED"
    print("HM127 sealed48fresh public controls;no model call yet", flush=True)


def collect(out):
    """Capture fresh histories then construct both focuses once, with receipts."""
    import torch

    from scripts.run_grain_math_span_binding import load_observer

    manifest = check(out)
    if H.load(out / "status.json")["state"] != "PREPARED":
        raise ValueError("Only one new finite capture is permitted")
    for name, digest in manifest["weights"].items():
        if C.stream_sha(MODEL / name) != digest:
            raise ValueError("Frozen encoder changed")
    with budget(out, "collect") as state:
        spec = manifest["spec"]
        rows = H.load(out / "00/fresh_rows.json")
        with np.load(out / "00/coordinates.npz", allow_pickle=False) as z:
            coords = dict(z)
        observer = load_observer()
        receipt = H.load(out / "01/capture_status.json")
        for i, row in enumerate(rows):
            if receipt["model_seconds"] >= spec["max_history_model_seconds"]:
                raise TimeoutError("Original history allowance reached")
            receipt.update(
                state="CAPTURING", pending=i, attempts=receipt["attempts"] + 1
            )
            H.write(out / "01/capture_status.json", receipt)
            started = time.perf_counter()
            try:
                arrays, trace = P.capture_history(observer, row, coords, spec["blocks"])
            except BaseException as error:
                receipt.update(
                    state="FAILED",
                    error=repr(error),
                    model_seconds=receipt["model_seconds"]
                    + time.perf_counter()
                    - started,
                )
                H.write(out / "01/capture_status.json", receipt)
                raise
            receipt["model_seconds"] += time.perf_counter() - started
            path = out / f"01/units/history_{i:03d}.npz"
            save(path, **arrays)
            H.write(path.with_suffix(".json"), {"sha256": H.sha(path), "trace": trace})
            receipt.update(completed=i + 1, pending=None)
            H.write(out / "01/capture_status.json", receipt)
        del observer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        weights = C.load_weights(
            str(MODEL / "model.safetensors"), spec["blocks"], spec["device"]
        )
        keys, values, readouts = [], [], []
        with torch.no_grad():
            for start in range(0, len(rows), spec["batch_size"]):
                if receipt["current_seconds"] >= spec["max_fresh_current_seconds"]:
                    raise TimeoutError(
                        "Original current-construction allowance reached"
                    )
                chunk = rows[start : start + spec["batch_size"]]
                histories = {name: [] for name in ("k", "v")}
                for row in chunk:
                    with np.load(
                        out / f"01/units/history_{row['id']:03d}.npz",
                        allow_pickle=False,
                    ) as z:
                        for name, buffer in histories.items():
                            buffer.append(z[name])
                valid = np.ones((len(chunk), manifest["history_length"]), bool)
                for i, row in enumerate(chunk):
                    valid[i, row["padding"]] = False
                begun = time.perf_counter()
                k, v = E.compile_histories(
                    {n: np.stack(a) for n, a in histories.items()},
                    coords["seeds"],
                    coords["cos"],
                    coords["sin"],
                    valid,
                    weights,
                    manifest["config"],
                    spec["device"],
                )
                keys.extend(k)
                values.extend(v)
                readouts.extend(
                    np.array(
                        [
                            [
                                R.read_current(
                                    kk,
                                    vv,
                                    manifest["endpoint"],
                                    H.load(out / "00/reader.json"),
                                )
                                for kk, vv in zip(kr, vr, strict=True)
                            ]
                            for kr, vr in zip(k, v, strict=True)
                        ]
                    )
                )
                receipt["current_seconds"] += time.perf_counter() - begun
                receipt["constructed_focus_states"] += len(chunk) * 2
                H.write(out / "01/capture_status.json", receipt)
                print("HM127 fresh currents", start + len(chunk), flush=True)
        if (
            receipt["model_seconds"] > spec["max_history_model_seconds"]
            or receipt["current_seconds"] > spec["max_fresh_current_seconds"]
        ):
            raise TimeoutError("Allowance exceeded at final atomic boundary")
        save(
            out / "01/constructed.npz",
            k=np.stack(keys),
            v=np.stack(values),
            readings=np.stack(readouts),
        )
        receipt["state"] = "COLLECTED"
        H.write(out / "01/capture_status.json", receipt)
        H.write(out / "01/output_seal.json", H.seals(out, out / "01"))
        state["state"] = "COLLECTED"
    print("HM12748histories/96fresh focuses complete", flush=True)


def bank_fields(rows, raw, conditions):
    """Address every actual observation, without inventing missing conditions."""
    banks = {}
    channels = {
        "value": raw["v"].reshape(len(rows), 2, -1),
        "uniform": raw["readings"][:, :, 0],
        "weighted": raw["readings"][:, :, 1],
    }
    for wording in (0, 1):
        fields = {
            name: np.empty((6, len(conditions), 2, 2, 2, x.shape[-1]))
            for name, x in channels.items()
        }
        seen = set()
        for i, row in enumerate(rows):
            if row["wording"] != wording:
                continue
            index = (
                row["family"],
                conditions.index(row["condition"]),
                (row["polarity"] + 1) // 2,
                (row["owner"] + 1) // 2,
            )
            if index in seen:
                raise ValueError("Duplicated observation address")
            seen.add(index)
            for name, x in channels.items():
                fields[name][index] = x[i]
        if len(seen) != 6 * len(conditions) * 4:
            raise ValueError("Incomplete observed support")
        banks[wording] = fields
    return banks


def load_banks(out):
    """Join old and fresh observations from the same explicit constructor."""
    manifest = check(out)
    H.verify_hashes(out, H.load(out / "01/output_seal.json"))
    old = ROOT / manifest["spec"]["source"]
    rows = H.load(old / "00/rows.json")
    with np.load(old / "02_corrected/constructed.npz", allow_pickle=False) as z:
        raw = dict(z)
    banks = {}
    for variant in range(3):
        ids = [i for i, row in enumerate(rows) if row["variant"] == variant]
        for wording, field in bank_fields(
            [rows[i] for i in ids], {n: x[ids] for n, x in raw.items()}, [-1, 1]
        ).items():
            banks[variant, wording] = field
    with np.load(out / "01/constructed.npz", allow_pickle=False) as z:
        raw = dict(z)
    for wording, field in bank_fields(
        H.load(out / "00/fresh_rows.json"), raw, [-1]
    ).items():
        banks[3, wording] = field
    return banks


def analyze(out):
    """Compare geometry and observability without fitting an application reader."""
    if H.load(out / "status.json")["state"] != "COLLECTED":
        raise ValueError("Complete new observations required")
    with budget(out, "analyze") as state:
        banks = load_banks(out)
        sectors = {
            key: {n: M.sectors(x) for n, x in field.items()}
            for key, field in banks.items()
        }
        summaries, witness_reports, saved = {}, {}, {}
        for stratum, indices in (("all_conditions", [0, 1]), ("common_minus", [0])):
            for key, target in sectors.items():
                if stratum == "all_conditions" and key[0] == 3:
                    continue
                name = f"{stratum}__v{key[0]}_w{key[1]}"
                comparison = {}
                for channel, source_parts in sectors[0, 0].items():
                    for sector, source in source_parts.items():
                        comparison[f"{channel}__{sector}"] = M.correspondence(
                            source[:, indices], target[channel][sector][:, indices]
                        )
                summaries[name] = comparison
        source = sectors[0, 0]
        maps = M.observation_bundle(
            source["uniform"]["own_A"],
            source["weighted"]["own_A"],
            source["value"]["own_A"],
        )
        info, theta = M.ambiguous_law(
            maps["weighted"], H.load(SPEC)["profile_amplitude"]
        )
        saved["theta"] = theta
        saved["positive_law"] = M.law_probabilities(theta)
        saved["negative_law"] = M.law_probabilities(-theta)
        for key, target in sectors.items():
            if key[0] == 3:
                continue
            name = f"v{key[0]}_w{key[1]}"
            observation = M.observation_bundle(
                target["uniform"]["own_A"],
                target["weighted"]["own_A"],
                target["value"]["own_A"],
            )
            witness_reports[name] = M.contrast_report(observation, theta)
            for channel, r in observation.items():
                saved[name + "__" + channel] = r
        (out / "02").mkdir()
        save(out / "02/witness.npz", **saved)
        H.write(out / "02/correspondence.json", summaries)
        H.write(
            out / "02/ambiguity.json",
            {
                "source_only_witness": info,
                "applications": witness_reports,
                "fresh_12cell_application": "UNOBSERVED_CONDITION_PLUS_NOT_IMPUTED",
            },
        )
        old = ROOT / H.load(SPEC)["source"]
        rows = H.load(old / "00/rows.json")
        ids = [
            i
            for i, r in enumerate(rows)
            if r["variant"] == 0 and r["wording"] == 0 and r["owner"] == 1
        ]
        rows_own = [rows[i] for i in ids]
        with np.load(old / "02_corrected/constructed.npz", allow_pickle=False) as z:
            keys, values = z["k"][ids, 0], z["v"][ids, 0]
        probabilities = np.array(
            [
                [
                    M.law_probabilities(t)[
                        r["family"] * 2 + (r["condition"] + 1) // 2,
                        (r["polarity"] + 1) // 2,
                    ]
                    for r in rows_own
                ]
                for t in (theta, -theta)
            ]
        )
        save(
            out / "02/cold_input.npz",
            k=keys,
            v=values,
            probabilities=probabilities,
            condition_index=np.array([(r["condition"] + 1) // 2 for r in rows_own]),
            endpoint=np.array(max(rows_own[0]["current"])),
        )
        H.write(out / "02/cold_bundle.json", H.load(out / "00/reader.json"))
        H.write(
            out / "summary.json",
            {
                "status": "FINITE_RESPONSE_GEOMETRY_AND_OBSERVABILITY_MEASURED_NOT_GLOBAL",
                "source_witness": info,
                "case_count": len(summaries),
                "channel_sector_comparisons": sum(len(x) for x in summaries.values()),
                "capture": H.load(out / "01/capture_status.json"),
                "new_native_answers_or_fits": 0,
                "virtual_laws_not_humans": True,
                "original_scope_complete": False,
            },
        )
        H.write(
            out / "02/output_seal.json",
            H.seals(out, out / "02") | {"summary.json": H.sha(out / "summary.json")},
        )
        state["state"] = "ANALYZED"
    print(
        "HM12714cases/336channel-sector comparisons and source witness complete",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/response_observability_r1",
    )
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze}[args.phase](args.out)
