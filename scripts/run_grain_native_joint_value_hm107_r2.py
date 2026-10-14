#!/usr/bin/env python3
"""Native-native calibration, retaining the original failed short-prefix gate."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_native_joint_batch as B
from grain_research.construction import math_native_joint_value as M
from grain_research.vector_mechanisms.navigation import write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_native_joint_value_hm107 as R
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_native_joint_value_hm107_r2.json"
CODE = R.CODE + (
    "configs/grain_native_joint_value_hm107_r2.json",
    "docs/GRAIN_NATIVE_JOINT_VALUE_RECOVERY_HM107_R2.md",
    "grain_research/construction/math_native_joint_batch.py",
    "scripts/run_grain_native_joint_value_hm107_r2.py",
    "tests/test_grain_math_native_joint_batch.py",
)


def check(out):
    """Authenticate the old failure and actual native calibration independently."""
    m = R.check(out)
    R.check(ROOT / m["spec"]["prior"])
    if (out / "00/calibration_seal.json").exists():
        H.verify_hashes(out, H.load(out / "00/calibration_seal.json"))
    return m


def prepare(out):
    """Freeze all native batches; inherit rather than reset the model budget."""
    if out.exists():
        raise FileExistsError("Preserve all prior and partial research roots")
    spec = H.load(SPEC)
    prior = ROOT / spec["prior"]
    old = R.check(prior)
    state = H.load(prior / "status.json")
    if state["state"] != "FAILED" or state["attempts"] != 45:
        raise ValueError("The retained R1 failure must be explicitly reconciled")
    groups, design, queries = [
        H.load(prior / f"00/{n}.json") for n in ("groups", "design", "queries")
    ]
    grouped = {}
    for r in design:
        grouped.setdefault((r["group"], r["corner"], r["arm"]), []).append(r)
    order = [k for k in grouped if k[2] == "baseline"] + [
        k for k in grouped if k[2] != "baseline"
    ]
    batches = [
        {
            "batch": i,
            "group": k[0],
            "corner": k[1],
            "arm": k[2],
            "units": sorted(grouped[k], key=lambda r: r["target"]),
        }
        for i, k in enumerate(order)
    ]
    if len(batches) != 868 or any(
        [r["target"] for r in b["units"]] != list(range(6)) for b in batches
    ):
        raise ValueError("Complete native batch design required")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in (
        ("groups", groups),
        ("design", design),
        ("queries", queries),
        ("batches", batches),
    ):
        H.write(out / f"00/{name}.json", value)
    with np.load(prior / "00/edits.npz", allow_pickle=False) as z:
        np.savez_compressed(
            out / "00/inherited_edits.npz", **{k: z[k] for k in z.files}
        )
    failed_files = [prior / "00/manifest.json", prior / "status.json"] + list(
        (prior / "01/units").glob("unit_*.*")
    )
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {k: H.sha(ROOT / k) for k in CODE},
            "source": old["source"]
            | {str(p.relative_to(ROOT)): H.sha(p) for p in failed_files},
            "inputs": H.seals(out, out / "00"),
            "prior_failed_actual_calls": state["attempts"],
            "prior_model_seconds": state["model_seconds"],
            "new_fit": False,
            "unit": "six target suffixes in one native shape;actual native corner calibration",
        },
    )
    write_json(
        out / "status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "completed_batches": 0,
            "attempts": 0,
            "prior_attempts": state["attempts"],
            "model_seconds": state["model_seconds"],
            "prior_model_seconds": state["model_seconds"],
            "pending": None,
        },
    )
    print(
        "HM107_R2 PREPARED:868 batches; retained R1 seconds",
        state["model_seconds"],
        flush=True,
    )


def unit_path(out, item):
    """Keep the predeclared endpoint addresses across serial and batched readers."""
    return out / f"01/units/unit_{item['unit']:05d}"


def calibrate(out, groups, batches):
    """Build joint edits from observed values; no answer scores enter construction."""
    if (out / "00/edits.npz").exists():
        if not (out / "00/calibration_seal.json").exists():
            raise RuntimeError("Uncertain calibration must not regenerate")
        H.verify_hashes(out, H.load(out / "00/calibration_seal.json"))
        return
    lookup = {(b["group"], b["corner"]): b for b in batches if b["arm"] == "baseline"}
    edits, hashes = {}, {}
    with np.load(out / "00/inherited_edits.npz", allow_pickle=False) as old:
        for group in groups:
            g = group["group"]
            keys, values = [], []
            for corner in range(4):
                path = unit_path(out, lookup[g, corner]["units"][0])
                if H.load(path.with_suffix(".json"))["sha256"] != H.sha(
                    path.with_suffix(".npz")
                ):
                    raise ValueError("Actual native calibration changed")
                with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
                    keys.append(z["k"])
                    values.append(z["before"])
            k, v = np.stack(keys), np.stack(values)
            joint, random = M.joint_edits(v, group["coordinates"])
            tag = f"g{g}__"
            for name, value in (
                ("old_k", k),
                ("old_v", v),
                ("joint", joint),
                ("random", random),
                ("q", old[tag + "q"]),
                ("d", old[tag + "d"]),
            ):
                edits[tag + name] = value
    for batch in batches[:288]:
        for item in batch["units"]:
            path = unit_path(out, item)
            for p in (path.with_suffix(".json"), path.with_suffix(".npz")):
                hashes[str(p.relative_to(out))] = H.sha(p)
    np.savez_compressed(out / "00/edits.npz", **edits)
    H.write(
        out / "00/calibration_seal.json",
        hashes | {"00/edits.npz": H.sha(out / "00/edits.npz")},
    )
    print("HM107_R2 actual native calibration SEALED", flush=True)


def collect(out):
    """Run the fixed calibration/intervention order with one shared cumulative cap."""
    m = check(out)
    spec, status = m["spec"], H.load(out / "status.json")
    if status["state"] not in ("PREPARED", "RUNNING") or status["pending"] is not None:
        raise RuntimeError("Terminal or uncertain run cannot automatically restart")
    groups, batches, queries = [
        H.load(out / f"00/{n}.json") for n in ("groups", "batches", "queries")
    ]
    with np.load(out / "00/inherited_edits.npz", allow_pickle=False) as z:
        inherited = {k: z[k] for k in z.files}
    start = time.monotonic()
    observer = load_observer()
    status["model_seconds"] += time.monotonic() - start
    write_json(out / "model_manifest.json", observer.manifest())
    actual = None
    with collector_guard(out):
        for batch in batches[status["completed_batches"] :]:
            if batch["batch"] >= 288 and actual is None:
                calibrate(out, groups, batches)
                with np.load(out / "00/edits.npz", allow_pickle=False) as z:
                    actual = {k: z[k] for k in z.files}
            if status["model_seconds"] >= spec["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                write_json(out / "status.json", status)
                raise TimeoutError("Original R1+R2 cumulative model cap exhausted")
            row = groups[batch["group"]]["rows"][batch["corner"]]
            tag = f"g{batch['group']}__"
            delta = (
                actual[
                    tag + ("joint" if batch["arm"] == "joint_removed" else "random")
                ][batch["corner"]]
                if batch["arm"] in ("joint_removed", "random")
                else None
            )
            if any(
                unit_path(out, item).with_suffix(ext).exists()
                for item in batch["units"]
                for ext in (".json", ".npz")
            ):
                raise RuntimeError("Existing uncertain batch must not replay")
            status.update(
                state="RUNNING", pending=batch, attempts=status["attempts"] + 1
            )
            write_json(out / "status.json", status)
            clock = time.monotonic()
            try:
                scores, raw = B.native_batch(
                    observer,
                    row,
                    queries["suffixes"],
                    queries["alphabet"],
                    delta,
                    spec["layer"],
                )
                for b, item in enumerate(batch["units"]):
                    observed = {k: v[b] for k, v in raw.items()}
                    error = max(
                        float(
                            np.max(
                                abs(
                                    observed[k]
                                    - (
                                        actual[tag + old][batch["corner"]]
                                        if actual is not None
                                        else raw[k][0]
                                    )
                                )
                            )
                        )
                        for k, old in (("k", "old_k"), ("before", "old_v"))
                    )
                    edit_error = float(
                        np.max(
                            abs(
                                observed["after"]
                                - observed["before"]
                                - (delta if delta is not None else 0)
                            )
                        )
                    )
                    inherited_error = max(
                        float(
                            np.max(
                                abs(observed[k] - inherited[tag + old][batch["corner"]])
                            )
                        )
                        for k, old in (("k", "old_k"), ("before", "old_v"))
                    )
                    path = unit_path(out, item)
                    np.savez_compressed(path.with_suffix(".npz"), **observed)
                    H.write(
                        path.with_suffix(".json"),
                        {
                            **item,
                            **scores[b],
                            "batch": batch["batch"],
                            "sha256": H.sha(path.with_suffix(".npz")),
                            "prefix_error": error,
                            "inherited_short_prefix_error_diagnostic": inherited_error,
                            "edit_error": edit_error,
                            "execution_length": int(observed["execution_length"]),
                            "endpoint": int(observed["endpoint"]),
                            "seconds_whole_batch": time.monotonic() - clock,
                        },
                    )
                    if (
                        error > spec["prefix_tolerance"]
                        or edit_error > spec["edit_tolerance"]
                    ):
                        raise ValueError(
                            "Same-native prefix/edit failed unchanged gate"
                        )
            except Exception as exc:
                status.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
                status["model_seconds"] += time.monotonic() - clock
                write_json(out / "status.json", status)
                raise
            status["model_seconds"] += time.monotonic() - clock
            status.update(
                completed_batches=batch["batch"] + 1,
                completed=status["completed"] + 6,
                pending=None,
            )
            write_json(out / "status.json", status)
            if status["completed_batches"] % 24 == 0:
                print(
                    "HM107_R2",
                    status["completed_batches"],
                    "/868 batches; cumulative seconds",
                    round(status["model_seconds"], 2),
                    flush=True,
                )
    status["state"] = "CAPTURED"
    write_json(out / "status.json", status)
    H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    print("HM107_R2 CAPTURED", status, flush=True)


def verify(out):
    """Check true endpoints, complete support, native calibration and full arithmetic."""
    m, status = check(out), H.load(out / "status.json")
    groups, queries = [H.load(out / f"00/{n}.json") for n in ("groups", "queries")]
    if (
        status["completed_batches"] != 868
        or status["completed"] != 5208
        or status["attempts"] != 868
        or status["model_seconds"] > m["spec"]["max_model_seconds"]
    ):
        raise ValueError("Incomplete native battery or original budget exceeded")
    maximum = 0.0
    with np.load(out / "00/edits.npz", allow_pickle=False) as edits:
        for batch in H.load(out / "00/batches.json"):
            row = groups[batch["group"]]["rows"][batch["corner"]]
            lengths = [len(row["input_ids"]) + len(s) for s in queries["suffixes"]]
            length = max(lengths)
            for item in batch["units"]:
                with np.load(
                    unit_path(out, item).with_suffix(".npz"), allow_pickle=False
                ) as raw:
                    n = lengths[item["target"]]
                    support = np.tri(length, dtype=bool)
                    support[:, row["padding"]] = False
                    support[:, n:] = False
                    digest = np.frombuffer(
                        hashlib.sha256(support.tobytes()).digest(), dtype=np.uint8
                    )
                    if (
                        int(raw["endpoint"]) != n - 1
                        or int(raw["execution_length"]) != length
                        or not np.array_equal(raw["execution_support_hash"], digest)
                    ):
                        raise ValueError("Actual full native support/endpoint changed")
                    if batch["arm"] == "baseline":
                        maximum = max(
                            maximum,
                            float(
                                np.max(
                                    abs(
                                        raw["before"]
                                        - edits[f"g{batch['group']}__old_v"][
                                            batch["corner"]
                                        ]
                                    )
                                )
                            ),
                            float(
                                np.max(
                                    abs(
                                        raw["k"]
                                        - edits[f"g{batch['group']}__old_k"][
                                            batch["corner"]
                                        ]
                                    )
                                )
                            ),
                        )
    if maximum > m["spec"]["prefix_tolerance"]:
        raise ValueError("Calibration varies across targets in the native batch")
    R.verify(out)
    H.write(
        out / "batch_verification.json",
        {
            "status": "VERIFIED_SAME_NATIVE_UNIT_HM107_R2",
            "calibration_error": maximum,
            "actual_batch_forwards": 868,
            "actual_native_endpoints": 5208,
            "prior_failed_single_forwards": status["prior_attempts"],
            "cumulative_model_seconds_including_R1": status["model_seconds"],
            "no_tolerance_or_budget_increase": True,
        },
    )
    print("HM107_R2 EXECUTION UNIT VERIFIED", maximum, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / H.load(SPEC)["output"])
    args = parser.parse_args()
    (R.analyze if args.phase == "analyze" else globals()[args.phase])(args.output)
