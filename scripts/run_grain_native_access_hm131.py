#!/usr/bin/env python3
"""HM131: native qualified evidence under exhaustive carrier-access programs."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_native_access as M
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_qualified_transport_hm105 as P

SPEC = ROOT / "configs/grain_native_access_hm131.json"
CODE = (
    "configs/grain_native_access_hm131.json",
    "docs/GRAIN_NATIVE_ACCESS_PLAN_HM131.md",
    "grain_research/construction/math_native_access.py",
    "scripts/run_grain_native_access_hm131.py",
    "tests/test_grain_math_native_access.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/run_grain_qualified_transport_hm105.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
)
UNITS = ("score", "log_odds", "candidate_mass", "entropy")


def check(out):
    """Authenticate the sealed new program and all admitted addressed inputs."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Freeze the complete intervention cube without reading target outcomes."""
    from grain_research.vector_mechanisms.navigation import write_json

    if out.exists():
        raise FileExistsError("Preserve prior and failed result roots")
    spec = H.load(SPEC)
    material, queries = ROOT / spec["material"], ROOT / spec["queries"]
    P.check(material)
    qm = H.load(queries / "00/manifest.json")
    H.verify_hashes(queries, qm["inputs"])
    query = H.load(queries / "00/queries.json")
    targets = H.load(material / "00/reader.json")["targets"]
    if targets != spec["targets"] or len(query["suffixes"]) != 12:
        raise ValueError("All six targets and both queried people required")
    originals = [
        r
        for r in H.load(material / "00/rows.json")
        if r["bank"] in spec["banks"]
        and r["bridge"] == spec["bridge"]
        and r["carrier"] == spec["carrier"]
        and r["axis"] >= 0
        and r["level"] != 1
        and r["owner"] != 1
    ]
    rows = []
    shared = None
    for original in originals:
        r = {
            **original,
            "id": len(rows),
            "family": original["axis"],
            "polarity": original["level"] - 1,
            "acting_person": original["owner"] - 1,
        }
        r["groups"] = M.partition(r)
        signatures = [
            (r["groups"][name], [r["input_ids"][i] for i in r["groups"][name]])
            for name in ("S", "B", "C")
        ]
        if shared is not None and signatures != shared:
            raise ValueError("S/B/C IDs and absolute support must be shared")
        shared = signatures
        rows.append(r)
    keys = {(r["bank"], r["family"], r["polarity"], r["acting_person"]) for r in rows}
    wanted = set(itertools.product(spec["banks"], range(6), (-1, 1), (-1, 1)))
    if len(rows) != 48 or keys != wanted:
        raise ValueError("Complete all-family two-bank evidence support required")
    design = []

    def append(row, route, noop=False):
        design.append(
            {"batch": len(design), "row": row["id"], "route": route, "noop": noop}
        )

    for r in rows:
        append(r, 7)
    for r in rows:
        if r["bank"] == 0 and r["family"] == 0:
            append(r, 7, True)
    for route in spec["route_order"][1:]:
        for r in rows:
            append(r, route)
    if len(design) != spec["max_batches"]:
        raise ValueError("Complete access cube and four early controls required")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    H.write(out / "00/rows.json", rows)
    H.write(out / "00/design.json", design)
    H.write(out / "00/queries.json", query)
    source_files = (
        material / "00/manifest.json",
        material / "00/rows.json",
        material / "00/reader.json",
        queries / "00/manifest.json",
        queries / "00/queries.json",
    )
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in source_files},
            "inputs": H.seals(out, out / "00"),
            "model": {p: h for p, h in qm["model"].items() if not p.startswith("._")},
            "model_blocks": qm["model_blocks"],
            "history_material_previously_observed": True,
            "new_fits": 0,
        },
    )
    write_json(
        out / "status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print("PREPARED HM131", len(rows), "bodies", len(design), "batches", flush=True)


def raw_unit(out, index):
    """Read a committed capture, never a pending or unauthenticated array."""
    stem = out / f"01/units/b{index:03d}"
    meta = H.load(stem.with_suffix(".json"))
    if H.sha(stem.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Raw native capture changed")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def collect(out):
    """Collect the full cube once; preserve all uncertain or terminal attempts."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    m, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No uncertain replay, terminal resume or budget renewal")
    H.verify_hashes(MODEL, m["model"])
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    design = H.load(out / "00/design.json")
    query = H.load(out / "00/queries.json")
    bases = {}
    reference_off = None
    tol = m["spec"]["isolation_tolerance"]
    try:
        with collector_guard(out):
            start = time.monotonic()
            try:
                observer = load_observer()
            finally:
                status["model_seconds"] += time.monotonic() - start
                write_json(out / "status.json", status)
            H.write(out / "model_manifest.json", observer.manifest())
            if len(observer.model.model.layers) != m["model_blocks"]:
                raise ValueError("Frozen decoder depth changed")
            for d in design:
                if status["model_seconds"] >= m["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Finite original budget exhausted")
                r = rows[d["row"]]
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=d
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    readings, arrays = M.native_batch(
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
                arrays["readings"] = M.units(arrays["logits"], arrays["normalizer"])
                if r["id"] in bases:
                    base = raw_unit(out, bases[r["id"]])
                    for name in ("body_final", "k_raw", "v"):
                        if np.max(abs(arrays[name] - base[name])) > tol:
                            raise ValueError(
                                "Suffix mask changed a body state: " + name
                            )
                    if (
                        d["noop"]
                        and np.max(abs(arrays["readings"] - base["readings"])) > tol
                    ):
                        raise ValueError("Early unmodified-mask no-op failed")
                else:
                    bases[r["id"]] = d["batch"]
                if d["route"] == 0:
                    if reference_off is not None and (
                        not np.array_equal(
                            arrays["support_hash"], reference_off["support_hash"]
                        )
                        or np.max(abs(arrays["readings"] - reference_off["readings"]))
                        > tol
                    ):
                        raise ValueError(
                            "All-off native judgment retained a history dependency"
                        )
                    reference_off = arrays
                stem = out / f"01/units/b{d['batch']:03d}"
                temporary = stem.with_suffix(".tmp")
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(stem.with_suffix(".npz"))
                H.write(
                    stem.with_suffix(".json"),
                    {
                        **d,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "seconds": seconds,
                        "endpoints": 12,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 24 == 0 or d["noop"]:
                    print("HM131", status, flush=True)
            if status["model_seconds"] > m["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Finite cap crossed")
            status["state"] = "COLLECTED"
            write_json(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM131", status, flush=True)


def observations(out):
    """Authenticate actual masks/body isolation and reconstruct the complete cube."""
    m = check(out)
    status = H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["pending"] is not None
        or status["completed"] != m["spec"]["max_batches"]
    ):
        raise ValueError("Incomplete battery cannot supply complete-route conclusions")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    query = H.load(out / "00/queries.json")
    native = np.full((2, 6, 8, 2, 2, 2, 6, 4), np.nan)
    bases, off, errors = (
        {},
        [],
        {n: 0.0 for n in ("body_final", "k_raw", "v", "noop", "units")},
    )
    for d in H.load(out / "00/design.json"):
        r, raw = rows[d["row"]], raw_unit(out, d["batch"])
        if not np.array_equal(raw["blocks"], np.arange(m["model_blocks"])):
            raise ValueError("An actual decoder cut is missing")
        width = int(raw["execution_length"])
        import hashlib

        expected = np.stack(
            [
                np.frombuffer(
                    hashlib.sha256(
                        M.permissions(r, len(s), width, d["route"]).tobytes()
                    ).digest(),
                    dtype=np.uint8,
                )
                for s in query["suffixes"]
            ]
        )
        if not np.array_equal(expected, raw["support_hash"]):
            raise ValueError(
                "Actual permission receipt differs from reconstructed support"
            )
        if not np.array_equal(
            raw["endpoints"],
            [len(r["input_ids"]) + len(s) - 1 for s in query["suffixes"]],
        ):
            raise ValueError("Native endpoints shifted")
        z = raw["logits"]
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        local = np.log(e.sum(axis=1)) + z.max(axis=1)
        alternate = np.column_stack(
            (
                p @ np.arange(10) / 9,
                np.log(e[:, 5:].sum(axis=1) / e[:, :5].sum(axis=1)),
                np.exp(local - raw["normalizer"]),
                -(p * np.log(p)).sum(axis=1),
            )
        )
        errors["units"] = max(
            errors["units"], float(np.max(abs(alternate - raw["readings"])))
        )
        if d["route"] == 7 and not d["noop"]:
            bases[r["id"]] = raw
        for name in ("body_final", "k_raw", "v"):
            errors[name] = max(
                errors[name], float(np.max(abs(raw[name] - bases[r["id"]][name])))
            )
        if d["noop"]:
            errors["noop"] = max(
                errors["noop"],
                float(np.max(abs(raw["readings"] - bases[r["id"]]["readings"]))),
            )
            continue
        bi = m["spec"]["banks"].index(r["bank"])
        b, o = (r["polarity"] + 1) // 2, (r["acting_person"] + 1) // 2
        if not np.isnan(native[bi, r["family"], d["route"], b, o]).all():
            raise ValueError("Duplicate support cell")
        native[bi, r["family"], d["route"], b, o] = raw["readings"].reshape(2, 6, 4)[
            ::-1
        ]
        if d["route"] == 0:
            off.append(raw["readings"])
    errors["all_off"] = float(np.max(np.ptp(np.stack(off), axis=0)))
    if (
        not np.isfinite(native).all()
        or max(errors.values()) > m["spec"]["isolation_tolerance"]
    ):
        raise ValueError("Missing support or isolation/readback check failed")
    return m, native, errors


def summarize(m, native):
    """Retain all signed routes, role sectors and positive evidence accounts."""
    routes, chars, authors, overview = [], [], [], []
    max_error = 0.0
    amplitude = m["spec"]["author_amplitude"]
    for bank, bank_id in enumerate(m["spec"]["banks"]):
        # Route coordinate first; all other coordinates remain fully explicit.
        fields = np.moveaxis(native[bank], 1, 0)
        components = M.mobius(fields)
        alternative = fields.copy()
        for bit in (1, 2, 4):
            for mask in range(8):
                if mask & bit:
                    alternative[mask] -= alternative[mask ^ bit]
        max_error = max(
            max_error,
            float(np.max(abs(alternative - components))),
            float(np.max(abs(M.reconstruct(components) - fields))),
        )
        coefficients = np.stack(
            [[M.characters(native[bank, f, r]) for r in range(8)] for f in range(6)]
        )
        for family, route, sector, target, unit in itertools.product(
            range(6), range(8), range(8), range(6), range(4)
        ):
            value = float(coefficients[family, route, sector, target, unit])
            chars.append(
                {
                    "bank": bank_id,
                    "family": family,
                    "route": route,
                    "role_sector": sector,
                    "target": m["spec"]["targets"][target],
                    "unit": UNITS[unit],
                    "value": value,
                }
            )
        for family, component, target, unit in itertools.product(
            range(6), range(8), range(6), range(4)
        ):
            value = M.characters(components[component, family])
            routes.append(
                {
                    "bank": bank_id,
                    "family": family,
                    "component": component,
                    "target": m["spec"]["targets"][target],
                    "unit": UNITS[unit],
                    "qualified_bop": float(value[7, target, unit]),
                    "common_bo": float(value[3, target, unit]),
                    "content_b": float(value[1, target, unit]),
                }
            )
        sign_product = np.outer([-1, 1], [-1, 1])
        plus, minus = (
            (1 + amplitude * sign_product) / 4,
            (1 - amplitude * sign_product) / 4,
        )
        for family, route, person, target, unit in itertools.product(
            range(6), range(8), range(2), range(6), range(4)
        ):
            corner = native[bank, family, route, :, :, person, target, unit]
            high, low = float(np.sum(plus * corner)), float(np.sum(minus * corner))
            expected = (
                2
                * amplitude
                * (
                    coefficients[family, route, 3, target, unit]
                    + (2 * person - 1) * coefficients[family, route, 7, target, unit]
                )
            )
            max_error = max(max_error, abs(high - low - expected))
            authors.append(
                {
                    "bank": bank_id,
                    "family": family,
                    "route": route,
                    "queried_person": "B" if person == 0 else "A",
                    "target": m["spec"]["targets"][target],
                    "unit": UNITS[unit],
                    "plus": high,
                    "minus": low,
                    "difference": high - low,
                }
            )
        for unit in range(4):
            full = coefficients[:, 7, 7, :, unit]
            for route in range(8):
                qualified = coefficients[:, route, 7, :, unit]
                common = coefficients[:, route, 3, :, unit]
                rms = float(np.sqrt(np.mean(qualified**2)))
                cosine = float(
                    np.sum(qualified * full)
                    / max(np.linalg.norm(qualified) * np.linalg.norm(full), 1e-30)
                )
                overview.append(
                    {
                        "bank": bank_id,
                        "unit": UNITS[unit],
                        "route": route,
                        "qualified_RMS": rms,
                        "common_RMS": float(np.sqrt(np.mean(common**2))),
                        "qualified_cosine_to_full": cosine,
                        "diagonal_positive_of6": int(np.sum(np.diag(qualified) > 0)),
                        "diagonal": np.diag(qualified).tolist(),
                    }
                )
    return routes, chars, authors, overview, max_error


def analyze(out):
    """Write full mechanism observations only after the entire actual cube passes."""
    m, native, errors = observations(out)
    routes, chars, authors, overview, error = summarize(m, native)
    folder = out / "02"
    folder.mkdir()
    with (folder / "native.npz").open("xb") as stream:
        np.savez_compressed(stream, native=native)
    H.table(folder / "role_characters.csv", chars)
    H.table(folder / "route_components.csv", routes)
    H.table(folder / "author_accounts.csv", authors)
    H.write(
        folder / "summary.json",
        {
            "packet": "HM131",
            "status": H.load(out / "status.json"),
            "checks": errors,
            "composition_error": error,
            "overview": overview,
            "table_rows": [len(chars), len(routes), len(authors)],
            "new_fits": 0,
            "original_scope_complete": False,
        },
    )
    H.write(out / "02/output_seal.json", H.seals(out, folder))
    print("ANALYZED HM131", errors, "composition", error, flush=True)


def verify(out):
    """Repeat numeric verification read-only; do not regenerate frozen artifacts."""
    m, native, errors = observations(out)
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    with np.load(out / "02/native.npz", allow_pickle=False) as z:
        if not np.array_equal(z["native"], native):
            raise ValueError("Complete native field changed")
    routes, chars, authors, overview, error = summarize(m, native)
    import csv
    import json

    for name, expected in (
        ("role_characters.csv", chars),
        ("route_components.csv", routes),
        ("author_accounts.csv", authors),
    ):
        with (out / "02" / name).open() as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected) or any(
            any(record[k] != str(v) for k, v in wanted.items())
            for record, wanted in zip(actual, expected, strict=True)
        ):
            raise ValueError("Complete table replay failed: " + name)
    summary = H.load(out / "02/summary.json")
    if (
        summary["overview"] != overview
        or summary["checks"] != errors
        or summary["composition_error"] != error
    ):
        raise ValueError("Summary differs from independent primitive reads")
    print(
        json.dumps(
            {
                "status": "VERIFIED_FINITE_HM131_NOT_GLOBAL",
                "checks": errors,
                "composition_error": error,
                "batches": 388,
                "endpoints": 4656,
                "table_rows": [len(chars), len(routes), len(authors)],
            }
        ),
        flush=True,
    )


def main():
    """Run each finite phase explicitly or all phases in order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--phase",
        choices=("prepare", "collect", "analyze", "verify", "all"),
        default="all",
    )
    args = parser.parse_args()
    out = args.out or ROOT / H.load(SPEC)["out"]
    for phase, function in (
        ("prepare", prepare),
        ("collect", collect),
        ("analyze", analyze),
        ("verify", verify),
    ):
        if args.phase in (phase, "all"):
            function(out)


if __name__ == "__main__":
    main()
