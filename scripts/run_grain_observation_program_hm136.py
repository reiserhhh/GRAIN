#!/usr/bin/env python3
"""One fixed access-program comparison, never reader-fidelity optimization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observation_program as M
from scripts import run_grain_causal_valuation_hm134 as R
from scripts import run_grain_choice_set_hm135 as T

H, A = R.H, R.A
SPEC = ROOT / "configs/grain_observation_program_hm136.json"
CODE = tuple(
    dict.fromkeys(
        (
            *R.CODE,
            *T.CODE,
            "configs/grain_observation_program_hm136.json",
            "docs/GRAIN_OBSERVATION_PROGRAM_PLAN_HM136.md",
            "grain_research/construction/math_observation_program.py",
            "scripts/run_grain_observation_program_hm136.py",
            "tests/test_grain_observation_program.py",
        )
    )
)
UNITS = ("score", "logodds", "digit_mass", "entropy")


def authenticate_source(source, name):
    """Check old inputs/code/results without changing a previous packet."""
    if name == "opportunity":
        with T.engine():
            manifest = R.check(source)
    elif name == "cause":
        manifest = R.check(source)
    else:
        raise ValueError("Unknown registered source")
    H.verify_hashes(source, H.load(source / "01/raw_seal.json"))
    H.verify_hashes(source, H.load(source / "02/output_seal.json"))
    status = H.load(source / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 196
        or status["pending"] is not None
    ):
        raise ValueError("Complete original observation field required")
    return manifest


def fixed_design(sources, spec):
    """Freeze all96 full reads and12 early/late compatibility controls."""
    result = []

    def add(source, row, route, noop=False):
        result.append(
            {
                "batch": len(result),
                "source": source,
                "row": row,
                "route": route,
                "noop": noop,
            }
        )

    for source in range(len(sources)):
        for row in spec["noop_rows"]:
            add(source, row, 7)
            add(source, row, 1)
            add(source, row, 7, True)
        for row in range(48):
            if row not in spec["noop_rows"]:
                add(source, row, 7)
        for row in spec["replay_rows"]:
            if row not in spec["noop_rows"]:
                add(source, row, 1)
    if len(sources) != 2 or len(result) != spec["max_batches"]:
        raise ValueError("Registered108-batch support changed")
    return result


def prepare(out):
    """Freeze identical materials, queries, model and all program contrasts."""
    if out.exists():
        raise FileExistsError("Preserve prior/failed result roots")
    spec = H.load(SPEC)
    admitted, files, query, prior = [], {}, None, None
    for definition in spec["sources"]:
        source = ROOT / definition["path"]
        manifest = authenticate_source(source, definition["name"])
        if prior is not None and (
            manifest["model"] != prior["model"]
            or manifest["model_blocks"] != prior["model_blocks"]
        ):
            raise ValueError("The two packets require an identical frozen model")
        prior = manifest
        rows, q = H.load(source / "00/rows.json"), H.load(source / "00/queries.json")
        R.validate_rows(rows)
        if query is not None and q != query:
            raise ValueError("Trait/fact queries must remain identical between corpora")
        query = q
        if len(q["suffixes"]) != 14:
            raise ValueError("All fourteen fixed queries required")
        old = [d for d in H.load(source / "00/design.json") if d["mode"] == "capture"]
        if len(old) != 48 or {d["row"] for d in old} != set(range(48)):
            raise ValueError("Missing baseline capture")
        captures = {str(d["row"]): d for d in old}
        admitted.append({**definition, "rows": rows, "captures": captures})
        names = [
            "00/manifest.json",
            "00/rows.json",
            "00/queries.json",
            "00/design.json",
            "01/raw_seal.json",
            "02/output_seal.json",
            "status.json",
        ]
        for d in old:
            R.committed(source, d)
            names.extend(
                [f"01/units/b{d['batch']:03d}.json", f"01/units/b{d['batch']:03d}.npz"]
            )
        files.update(
            {str((source / n).relative_to(ROOT)): H.sha(source / n) for n in names}
        )
    design = fixed_design(admitted, spec)
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in (("sources", admitted), ("queries", query), ("design", design)):
        H.write(out / f"00/{name}.json", value)
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "model": prior["model"],
            "model_blocks": prior["model_blocks"],
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": files,
            "inputs": H.seals(out, out / "00"),
            "old_H_readings_known": True,
            "new_fits": 0,
            "new_materials": 0,
            "original_scope_complete": False,
        },
    )
    H.write(
        out / "status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print("PREPARED HM136", len(design), "batches", flush=True)


def check(out):
    """Authenticate all admitted code, input and original primitive captures."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def old_capture(sources, d):
    """Get the actual H-only baseline of the identical registered body."""
    source = sources[d["source"]]
    return R.committed(ROOT / source["path"], source["captures"][str(d["row"])])


def raw_unit(out, d):
    """Read a committed exact address, never an uncertain attempted batch."""
    stem = out / f"01/units/b{d['batch']:03d}"
    meta = H.load(stem.with_suffix(".json"))
    if (
        any(meta[n] != v for n, v in d.items())
        or meta["endpoints"] != 14
        or H.sha(stem.with_suffix(".npz")) != meta["sha256"]
    ):
        raise ValueError("Address receipt or primitive bytes changed")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        return {n: z[n] for n in z.files}


def collect(out):
    """Run one bounded packet; preserve failures and never automatically retry."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No automatic uncertain/terminal replay or cap renewal")
    H.verify_hashes(MODEL, manifest["model"])
    sources, query = H.load(out / "00/sources.json"), H.load(out / "00/queries.json")
    design, full, tol = (
        H.load(out / "00/design.json"),
        {},
        manifest["spec"]["tolerance"],
    )
    try:
        with collector_guard(out):
            start = time.monotonic()
            try:
                observer = load_observer()
            finally:
                status["model_seconds"] += time.monotonic() - start
                write_json(out / "status.json", status)
            H.write(out / "model_manifest.json", observer.manifest())
            if len(observer.model.model.layers) != manifest["model_blocks"]:
                raise ValueError("Frozen model depth changed")
            for d in design:
                if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Original cumulative model cap reached")
                row = sources[d["source"]]["rows"][d["row"]]
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=d
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    reads, arrays = A.native_batch(
                        observer,
                        row,
                        query["suffixes"],
                        query["alphabet"],
                        d["route"],
                        manifest["spec"]["capture_layer"],
                        noop=d["noop"],
                    )
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    write_json(out / "status.json", status)
                arrays["logits"] = np.array([v["candidate_logits"] for v in reads])
                arrays["normalizer"] = np.array(
                    [v["full_vocab_log_normalizer"] for v in reads]
                )
                arrays["readings"] = A.units(arrays["logits"], arrays["normalizer"])
                old = old_capture(sources, d)
                for name in ("body_final", "k_raw", "v"):
                    if np.max(abs(arrays[name] - old[name])) > tol:
                        raise ValueError("Program change altered body: " + name)
                if d["route"] == 1:
                    for name in ("logits", "normalizer", "readings"):
                        if np.max(abs(arrays[name] - old[name])) > tol:
                            raise ValueError(
                                "H-only compatibility replay failed: " + name
                            )
                if d["noop"]:
                    base = raw_unit(out, full[d["source"], d["row"]])
                    if (
                        np.max(abs(arrays["readings"] - base["readings"])) > tol
                        or np.max(abs(arrays["logits"] - base["logits"])) > tol
                    ):
                        raise ValueError("Full original-mask no-op failed")
                stem = out / f"01/units/b{d['batch']:03d}"
                with stem.with_suffix(".tmp").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                stem.with_suffix(".tmp").replace(stem.with_suffix(".npz"))
                H.write(
                    stem.with_suffix(".json"),
                    {
                        **d,
                        "seconds": seconds,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "endpoints": 14,
                    },
                )
                if d["route"] == 7 and not d["noop"]:
                    full[d["source"], d["row"]] = d
                status.update(completed=status["completed"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 12 == 0 or d["noop"]:
                    print("HM136", status, flush=True)
            if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Original cap crossed")
            status["state"] = "COLLECTED"
            write_json(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM136", status, flush=True)


def person_units(raw):
    """Preserve person and factual slots rather than treating queries as people."""
    y = raw["readings"]
    if y.shape != (14, 4):
        raise ValueError("All registered query units required")
    return np.stack(
        (np.concatenate((y[:6], y[12:13])), np.concatenate((y[6:12], y[13:14])))
    )


def observations(out):
    """Rebuild the full program cube with original H data and all actual guards."""
    manifest, status = check(out), H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 108
        or status["pending"] is not None
    ):
        raise ValueError("Incomplete packet stays incomplete")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    sources, query = H.load(out / "00/sources.json"), H.load(out / "00/queries.json")
    cube = np.full((2, 2, 6, 2, 2, 2, 2, 7, 4), np.nan)
    checks = dict.fromkeys(
        (
            "body_final",
            "k_raw",
            "v",
            "H_logits",
            "H_normalizer",
            "H_units",
            "noop",
            "units",
        ),
        0.0,
    )
    full, filled, controls = {}, set(), {"H": 0, "noop": 0}
    for d in H.load(out / "00/design.json"):
        row, raw, old = (
            sources[d["source"]]["rows"][d["row"]],
            raw_unit(out, d),
            old_capture(sources, d),
        )
        if not np.array_equal(raw["blocks"], np.arange(manifest["model_blocks"])):
            raise ValueError("Missed an attention block")
        masks = [
            A.permissions(row, len(s), int(raw["execution_length"]), d["route"])
            for s in query["suffixes"]
        ]
        hashes = np.stack(
            [
                np.frombuffer(hashlib.sha256(m.tobytes()).digest(), dtype=np.uint8)
                for m in masks
            ]
        )
        if not np.array_equal(hashes, raw["support_hash"]) or not np.array_equal(
            raw["endpoints"], [187 + len(s) - 1 for s in query["suffixes"]]
        ):
            raise ValueError("Registered actual permissions/endpoints changed")
        for name in ("body_final", "k_raw", "v"):
            checks[name] = max(checks[name], float(np.max(abs(raw[name] - old[name]))))
        checks["units"] = max(
            checks["units"],
            float(
                np.max(abs(A.units(raw["logits"], raw["normalizer"]) - raw["readings"]))
            ),
        )
        if d["route"] == 1:
            controls["H"] += 1
            for name, guard in (
                ("logits", "H_logits"),
                ("normalizer", "H_normalizer"),
                ("readings", "H_units"),
            ):
                checks[guard] = max(
                    checks[guard], float(np.max(abs(raw[name] - old[name])))
                )
        elif d["noop"]:
            controls["noop"] += 1
            base = full[d["source"], d["row"]]
            checks["noop"] = max(
                checks["noop"],
                float(np.max(abs(raw["readings"] - base["readings"]))),
                float(np.max(abs(raw["logits"] - base["logits"]))),
            )
        else:
            key = (
                d["source"],
                row["bank"],
                row["family"],
                (row["b"] + 1) // 2,
                (row["c"] + 1) // 2,
            )
            if key in filled:
                raise ValueError("Duplicated full body substituted for missing cell")
            filled.add(key)
            cube[key] = np.stack((person_units(old), person_units(raw)))
            full[d["source"], d["row"]] = raw
    if (
        len(filled) != 96
        or controls != {"H": 8, "noop": 4}
        or not np.isfinite(cube).all()
        or max(checks.values()) > manifest["spec"]["tolerance"]
    ):
        raise ValueError("Complete scope or primitive compatibility guard failed")
    return cube, checks


def derive(cube):
    """Compute every field and descriptive comparison with no success selection."""
    effects, chars, summary = [], [], []
    for source, bank, family in itertools.product(range(2), range(2), range(6)):
        x = np.moveaxis(cube[source, bank, family], 2, 0)
        effects.append(M.effects(x))
        chars.append(M.characters(x))
    effects = np.stack(effects).reshape(2, 2, 6, 2, 4, 2, 7, 4)
    chars = np.stack(chars).reshape(2, 2, 6, 8, 2, 7, 4)
    for source, bank, person, unit in itertools.product(
        range(2), range(2), range(2), range(4)
    ):
        old, full = (effects[source, bank, :, r, 3, person, :6, unit] for r in range(2))
        entry = {
            "corpus": ("cause", "opportunity")[source],
            "bank": bank,
            "person": "AB"[person],
            "unit": UNITS[unit],
            "interaction": M.field_comparison(old, full),
            "diagonal_H": np.diag(old).tolist(),
            "diagonal_full": np.diag(full).tolist(),
        }
        for r, name in enumerate(("H", "full")):
            y = cube[source, bank, :, :, :, r, person, 6, unit]
            delta = y[:, :, 1] - y[:, :, 0]
            entry[f"fact_{name}_condition_mean_abs"] = float(np.mean(abs(delta)))
            entry[f"fact_{name}_condition_max_abs"] = float(np.max(abs(delta)))
            entry[f"fact_{name}_mean"] = float(np.mean(y))
        summary.append(entry)
    return effects, chars, summary


def tables(effects, chars):
    """Export all program/b/c/person/target/unit sectors, including facts."""
    targets = H.load(SPEC)["targets"] + ["action_fact"]
    result = {"program_effects": [], "program_characters": []}
    for source, bank, family, person, target, unit in itertools.product(
        range(2), range(2), range(6), range(2), range(7), range(4)
    ):
        base = {
            "corpus": ("cause", "opportunity")[source],
            "bank": bank,
            "family": targets[family],
            "person": "AB"[person],
            "target": targets[target],
            "unit": UNITS[unit],
        }
        for r, program in enumerate(("H", "full")):
            values = effects[source, bank, family, r, :, person, target, unit]
            result["program_effects"].append(
                {
                    **base,
                    "program": program,
                    **dict(
                        zip(
                            ("slope_c0", "slope_c1", "condition_level", "interaction"),
                            map(float, values),
                            strict=True,
                        )
                    ),
                }
            )
        for mask in range(8):
            result["program_characters"].append(
                {
                    **base,
                    "mask": mask,
                    "coefficient": float(
                        chars[source, bank, family, mask, person, target, unit]
                    ),
                }
            )
    return result


def analyze(out):
    """Save complete finite results separately from psychological interpretation."""
    cube, checks = observations(out)
    effects, chars, fields = derive(cube)
    folder = out / "02"
    folder.mkdir()
    with (folder / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, cube=cube, effects=effects, characters=chars)
    for name, records in tables(effects, chars).items():
        H.table(folder / (name + ".csv"), records)
    H.write(
        folder / "summary.json",
        {
            "status": "FINITE_OBSERVATION_PROGRAM_COMPARISON_NOT_HUMAN_VALIDITY",
            "checks": checks,
            "fields": fields,
            "batches": 108,
            "endpoints": 1512,
            "model_seconds": H.load(out / "status.json")["model_seconds"],
            "new_fits": 0,
            "original_scope_complete": False,
        },
    )
    H.write(folder / "output_seal.json", H.seals(out, folder))
    print("ANALYZED HM136", checks, flush=True)


def verify(out):
    """Read-only alternate primitive and character arithmetic plus table replay."""
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    cube, checks = observations(out)
    effects, chars, fields = derive(cube)
    with np.load(out / "02/observations.npz", allow_pickle=False) as z:
        for n, value in (("cube", cube), ("effects", effects), ("characters", chars)):
            if not np.array_equal(z[n], value):
                raise ValueError("Saved complete field differs")
    error = 0.0
    for source, bank, family in itertools.product(range(2), range(2), range(6)):
        x = np.moveaxis(cube[source, bank, family], 2, 0)
        for r in range(2):
            alternate = x[r, 1, 1] - x[r, 0, 1] - x[r, 1, 0] + x[r, 0, 0]
            error = max(
                error,
                float(np.max(abs(alternate - effects[source, bank, family, r, 3]))),
            )
        error = max(
            error, float(np.max(abs(M.reconstruct(chars[source, bank, family]) - x)))
        )
        error = max(
            error,
            float(
                np.max(
                    abs(
                        effects[source, bank, family, 1, 3]
                        - effects[source, bank, family, 0, 3]
                        - 8 * chars[source, bank, family, 7]
                    )
                )
            ),
        )
    for d in H.load(out / "00/design.json"):
        raw = raw_unit(out, d)
        logits = raw["logits"]
        w = np.exp(logits - logits.max(axis=1)[:, None])
        p = w / w.sum(axis=1)[:, None]
        alternate = np.stack(
            (
                p @ np.arange(10) / 9,
                np.log(w[:, 5:].sum(axis=1) / w[:, :5].sum(axis=1)),
                np.exp(logits.max(axis=1) + np.log(w.sum(axis=1)) - raw["normalizer"]),
                -np.sum(p * np.log(p), axis=1),
            ),
            axis=1,
        )
        error = max(error, float(np.max(abs(alternate - raw["readings"]))))
    if error > 1e-10:
        raise ValueError("Alternate finite-field or primitive arithmetic failed")
    saved = H.load(out / "02/summary.json")
    if saved["checks"] != checks or saved["fields"] != fields:
        raise ValueError("Summary changed")
    counts = {}
    for name, rows in tables(effects, chars).items():
        with (out / f"02/{name}.csv").open() as stream:
            actual = list(csv.DictReader(stream))
        if actual != [{n: str(v) for n, v in r.items()} for r in rows]:
            raise ValueError("Full signed table changed")
        counts[name] = len(rows)
    print(
        "VERIFIED HM136",
        {"checks": checks, "independent_error": error, "table_rows": counts},
        flush=True,
    )


def main():
    """Separate explicit phases; no automatic continuation or scientific retry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("prepare", "collect", "analyze", "verify")
    )
    args = parser.parse_args()
    out = ROOT / H.load(SPEC)["out"]
    {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
        args.phase
    ](out)


if __name__ == "__main__":
    main()
