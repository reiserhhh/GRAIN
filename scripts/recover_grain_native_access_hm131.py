#!/usr/bin/env python3
"""One mechanical HM131 guard recovery with inherited captures and elapsed cap."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_native_access_hm131 as R

SOURCE = ROOT / "results/grain_math_response_v1/native_access_r1"
DEFAULT = ROOT / "results/grain_math_response_v1/native_access_r2"
EXTRA = (
    "scripts/recover_grain_native_access_hm131.py",
    "docs/GRAIN_NATIVE_ACCESS_HM131_RECOVERY_NOTE.md",
    "tests/test_grain_native_access_recovery.py",
)


def suffix_support(row, suffixes, width):
    """Compare question rows, not untouched body padding between histories."""
    body = len(row["input_ids"])
    return np.stack([R.M.permissions(row, len(s), width, 0)[body:] for s in suffixes])


def prepare(out):
    """Preserve and authenticate r1; admit only its specific terminal guard fault."""
    from grain_research.vector_mechanisms.navigation import write_json

    if out.exists():
        raise FileExistsError("No failed-root replacement")
    m, old = R.check(SOURCE), R.H.load(SOURCE / "status.json")
    if (
        old["state"] != "FAILED"
        or old["completed"] != 54
        or old["attempts"] != 55
        or old["pending"] != {"batch": 54, "row": 2, "route": 0, "noop": False}
        or old["error"]
        != "ValueError: All-off native judgment retained a history dependency"
    ):
        raise ValueError(
            "Only the known post-forward mask-hash guard fault is recoverable"
        )
    source_files = [
        SOURCE / "status.json",
        SOURCE / "00/manifest.json",
        SOURCE / "model_manifest.json",
    ]
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name in ("rows.json", "design.json", "queries.json"):
        p = SOURCE / "00" / name
        shutil.copyfile(p, out / "00" / name)
        source_files.append(p)
    for index in range(old["completed"]):
        R.raw_unit(SOURCE, index)
        for suffix in (".npz", ".json"):
            p = SOURCE / f"01/units/b{index:03d}{suffix}"
            shutil.copyfile(p, out / f"01/units/b{index:03d}{suffix}")
            source_files.append(p)
    R.H.write(
        out / "00/manifest.json",
        {
            **m,
            "code": {**m["code"], **{p: R.H.sha(ROOT / p) for p in EXTRA}},
            "source": {
                **m["source"],
                **{str(p.relative_to(ROOT)): R.H.sha(p) for p in source_files},
            },
            "inputs": R.H.seals(out, out / "00"),
            "recovery": {
                "source": str(SOURCE.relative_to(ROOT)),
                "inherited_committed": 54,
                "inherited_attempts": 55,
                "inherited_model_seconds": old["model_seconds"],
                "known_lost_forward": old["pending"],
                "original_cumulative_cap_unchanged": True,
                "extra_forward_deviation": 1,
                "scientific_protocol_unchanged": True,
            },
        },
    )
    write_json(
        out / "status.json",
        {
            "state": "PREPARED",
            "completed": old["completed"],
            "attempts": old["attempts"],
            "pending": None,
            "model_seconds": old["model_seconds"],
            "inherited_committed": 54,
            "known_failed_forward_included": True,
        },
    )
    print(
        "PREPARED HM131 recovery:54 inherited units,one known lost capture", flush=True
    )


def collect(out):
    """Collect only uncommitted design units under the original cumulative cap."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    m, status = R.check(out), R.H.load(out / "status.json")
    if (
        status["state"] != "PREPARED"
        or status["completed"] != 54
        or status["attempts"] != 55
        or status["pending"] is not None
    ):
        raise RuntimeError("No uncertain or terminal retry")
    R.H.verify_hashes(MODEL, m["model"])
    rows = {r["id"]: r for r in R.H.load(out / "00/rows.json")}
    design = R.H.load(out / "00/design.json")
    query = R.H.load(out / "00/queries.json")
    bases = {d["row"]: d["batch"] for d in design[:48]}
    reference = R.raw_unit(out, 52)
    reference_support = suffix_support(
        rows[0], query["suffixes"], int(reference["execution_length"])
    )
    tol = m["spec"]["isolation_tolerance"]
    try:
        with collector_guard(out):
            start = time.monotonic()
            try:
                observer = load_observer()
            finally:
                status["model_seconds"] += time.monotonic() - start
                write_json(out / "status.json", status)
            R.H.write(out / "model_manifest.json", observer.manifest())
            if len(observer.model.model.layers) != m["model_blocks"]:
                raise ValueError("Decoder depth changed")
            for d in design[54:]:
                if status["model_seconds"] >= m["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Original cumulative model budget exhausted")
                r = rows[d["row"]]
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=d
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    readings, arrays = R.M.native_batch(
                        observer,
                        r,
                        query["suffixes"],
                        query["alphabet"],
                        d["route"],
                        m["spec"]["capture_layer"],
                        noop=d["noop"],
                    )
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    write_json(out / "status.json", status)
                arrays["logits"] = np.array([v["candidate_logits"] for v in readings])
                arrays["normalizer"] = np.array(
                    [v["full_vocab_log_normalizer"] for v in readings]
                )
                arrays["readings"] = R.M.units(arrays["logits"], arrays["normalizer"])
                base = R.raw_unit(out, bases[r["id"]])
                for name in ("body_final", "k_raw", "v"):
                    if np.max(abs(arrays[name] - base[name])) > tol:
                        raise ValueError("A suffix cut changed body states:" + name)
                if d["route"] == 0 and (
                    not np.array_equal(
                        suffix_support(
                            r, query["suffixes"], int(arrays["execution_length"])
                        ),
                        reference_support,
                    )
                    or np.max(abs(arrays["readings"] - reference["readings"])) > tol
                ):
                    raise ValueError(
                        "All-off question support/output retained a history dependency"
                    )
                stem = out / f"01/units/b{d['batch']:03d}"
                tmp = stem.with_suffix(".tmp")
                with tmp.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                tmp.replace(stem.with_suffix(".npz"))
                R.H.write(
                    stem.with_suffix(".json"),
                    {
                        **d,
                        "sha256": R.H.sha(stem.with_suffix(".npz")),
                        "seconds": seconds,
                        "endpoints": 12,
                        "recovery_new_capture": True,
                    },
                )
                status.update(completed=d["batch"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 24 == 0:
                    print("HM131 recovery", status, flush=True)
            if status["model_seconds"] > m["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Original cumulative cap crossed")
            status["state"] = "COLLECTED"
            write_json(out / "status.json", status)
            R.H.write(out / "01/raw_seal.json", R.H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}:{exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM131 recovery", status, flush=True)


def main():
    """Run only the explicit guarded recovery or read-only verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument(
        "--phase",
        choices=("prepare", "collect", "analyze", "verify", "all"),
        default="all",
    )
    args = parser.parse_args()
    for phase, fn in (
        ("prepare", prepare),
        ("collect", collect),
        ("analyze", R.analyze),
        ("verify", R.verify),
    ):
        if args.phase in (phase, "all"):
            fn(args.out)


if __name__ == "__main__":
    main()
