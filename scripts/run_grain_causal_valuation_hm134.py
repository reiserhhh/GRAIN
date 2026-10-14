#!/usr/bin/env python3
"""Finite cause-qualified native judgment experiment, not accuracy optimization."""

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
from grain_research.construction import math_causal_valuation as M
from scripts import run_grain_native_key_sectors_hm133 as R

H, A, N = R.H, R.A, R.R.M
SPEC = ROOT / "configs/grain_causal_valuation_hm134.json"
CODE = (
    *R.CODE,
    "configs/grain_causal_valuation_hm134.json",
    "docs/GRAIN_CAUSAL_VALUATION_PLAN_HM134.md",
    "grain_research/construction/math_causal_valuation.py",
    "scripts/run_grain_causal_valuation_hm134.py",
    "tests/test_grain_causal_valuation.py",
)
MODES = ("capture", "k", "v", "kv")
UNITS = ("score", "logodds", "digit_mass", "entropy")


def validate_rows(rows):
    """Require complete b/c support and identical accessible prefixes per pair."""
    by_key = {(r["bank"], r["family"], r["b"], r["c"]): r for r in rows}
    want = set(itertools.product(range(2), range(6), (-1, 1), (-1, 1)))
    if len(rows) != 48 or set(by_key) != want:
        raise ValueError("All48 registered histories required")
    fixed = None
    for r in rows:
        groups = A.partition(r)
        actual = [
            (groups[n], [r["input_ids"][i] for i in groups[n]]) for n in ("S", "B", "C")
        ]
        if fixed is not None and actual != fixed:
            raise ValueError("Non-history body changed")
        fixed = actual
        if len(r["input_ids"]) != 187 or r["cause"] != list(range(149, 168)):
            raise ValueError("Body or causal statement address changed")
    for bank, family, b in itertools.product(range(2), range(6), (-1, 1)):
        lo, hi = (by_key[bank, family, b, c] for c in (-1, 1))
        if (
            lo["input_ids"][:149] != hi["input_ids"][:149]
            or lo["padding"] != hi["padding"]
        ):
            raise ValueError("Cause pair changed earlier behavior or support")
    return by_key


def fixed_design(rows):
    """Capture all endpoints before paired K/V interchanges with four controls."""
    by_key = validate_rows(rows)
    design = []

    def append(r, mode, donor):
        design.append(
            {"batch": len(design), "row": r["id"], "mode": mode, "donor": donor["id"]}
        )

    for r in rows:
        append(r, "capture", r)
    for b, c in itertools.product((-1, 1), repeat=2):
        append(by_key[0, 0, b, c], "noop", by_key[0, 0, b, c])
    for r in rows:
        for mode in MODES[1:]:
            append(r, mode, by_key[r["bank"], r["family"], r["b"], -r["c"]])
    if len(design) != 196:
        raise ValueError("Fixed196-batch plan changed")
    return design


def check(out):
    """Authenticate the complete frozen plan, source and input records."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Register new public evidence and equal-size cause slots before inference."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if out.exists():
        raise FileExistsError("Do not overwrite a previous scientific packet")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    old = R.check(source)
    H.verify_hashes(source, H.load(source / "02/output_seal.json"))
    base = H.load(source / "00/rows.json")[0]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows = []
    for bank, family, b, c in itertools.product(range(2), range(6), (-1, 1), (-1, 1)):
        definition = spec["banks"][bank]
        text = (
            definition["prefix"]
            + " "
            + definition["behaviours"][family][(b + 1) // 2]
            + "."
        )
        action = tokenizer.encode(text, add_special_tokens=False)
        cause_text = "\n" + definition["causes"][(c + 1) // 2]
        cause = tokenizer.encode(cause_text, add_special_tokens=False)
        if len(cause) != 19 or len(action) > 103:
            raise ValueError("Frozen cause width or available action slot exceeded")
        ids = list(base["input_ids"])
        for i in base["history"]:
            ids[i] = tokenizer.pad_token_id
        action_rows = base["history"][: len(action)]
        for i, token in zip(action_rows, action, strict=True):
            ids[i] = token
        cause_rows = list(range(spec["cause_start"], spec["cause_start"] + len(cause)))
        for i, token in zip(cause_rows, cause, strict=True):
            ids[i] = token
        rows.append(
            {
                "id": len(rows),
                "bank": bank,
                "family": family,
                "b": b,
                "c": c,
                "history": base["history"],
                "current": base["current"],
                "input_ids": ids,
                "padding": sorted(set(base["history"]) - set(action_rows + cause_rows)),
                "action": action_rows,
                "cause": cause_rows,
                "history_text": text + cause_text,
            }
        )
    query = H.load(source / "00/queries.json")
    query = {k: list(v) if isinstance(v, list) else v for k, v in query.items()}
    for person in ("A", "B"):
        q = (
            f"Does the account explicitly report that Person {person} carried out the described action, "
            "regardless of why it was done? Use 0 for no and 9 for yes. Return only one digit from 0 to 9."
        )
        suffix = tokenizer.encode(
            "\n\nQuestion about the named person:\n"
            + q
            + "\nAnswer:"
            + "<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        query["questions"].append(q)
        query["suffixes"].append(suffix)
    if len(query["suffixes"]) != 14:
        raise ValueError("All12 trait plus2 factual endpoints required")
    design = fixed_design(rows)
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, data in (("rows", rows), ("design", design), ("queries", query)):
        H.write(out / f"00/{name}.json", data)
    source_paths = [
        source / n
        for n in (
            "00/manifest.json",
            "00/rows.json",
            "00/queries.json",
            "02/output_seal.json",
        )
    ]
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "model": old["model"],
            "model_blocks": old["model_blocks"],
            "code": {p: H.sha(ROOT / p) for p in dict.fromkeys(CODE)},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in source_paths},
            "inputs": H.seals(out, out / "00"),
            "source_outputs_fit": False,
            "scope": "new fictional behavior/causal-context;native H-only program,not human traits",
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
    print("PREPARED HM134", len(rows), len(design), len(query["suffixes"]), flush=True)


def committed(out, d):
    """Require the exact address receipt as well as the primitive file hash."""
    meta = H.load(out / f"01/units/b{d['batch']:03d}.json")
    if any(meta[n] != value for n, value in d.items()) or meta["endpoints"] != 14:
        raise ValueError("Scientific batch receipt differs from frozen design")
    return R.R.committed(out, d["batch"])


def collect(out):
    """Run once under the original cumulative cap, preserving uncertain work."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No automatic retry of terminal or uncertain science")
    H.verify_hashes(MODEL, manifest["model"])
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    design, query = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    captures = {}
    tol = manifest["spec"]["tolerance"]
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
                    raise TimeoutError("Original cumulative model budget reached")
                row, donors = rows[d["row"]], None
                donor_hashes = {}
                if d["mode"] != "capture":
                    base, donor = (
                        committed(out, captures[d[key]]) for key in ("row", "donor")
                    )
                    indices = [row["history"].index(i) for i in row["cause"]]
                    donors = M.install_cause(
                        base["history_key"],
                        base["history_value"],
                        donor["history_key"],
                        donor["history_value"],
                        indices,
                        d["mode"],
                    )
                    donor_hashes = {
                        key: H.sha(
                            out / f"01/units/b{captures[d[key]]['batch']:03d}.npz"
                        )
                        for key in ("row", "donor")
                    }
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=d
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    arrays = N.native_batch(observer, row, query, donors=donors)
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    write_json(out / "status.json", status)
                if float(arrays["cache_spread"]) > tol:
                    raise ValueError("Causal prefix depends on future question")
                if d["mode"] == "capture":
                    captures[row["id"]] = d
                else:
                    for name in ("body_final", "k_raw", "v"):
                        if np.max(abs(arrays[name] - base[name])) > tol:
                            raise ValueError("Suffix operation altered body: " + name)
                    expected = base if d["mode"] == "noop" else donor
                    if (
                        d["mode"] in ("noop", "kv")
                        and np.max(abs(arrays["readings"] - expected["readings"])) > tol
                    ):
                        raise ValueError("Own/full-cause transfer control failed")
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
                        "donor_hashes": donor_hashes,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 12 == 0 or d["mode"] == "noop":
                    print("HM134", status, flush=True)
            if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Original cumulative cap crossed")
            status["state"] = "COLLECTED"
            write_json(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM134", status, flush=True)


def observations(out):
    """Reconstruct authenticated support and audit causal/local native guards."""
    manifest, status = check(out), H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 196
        or status["pending"] is not None
    ):
        raise ValueError("Incomplete evidence cannot be promoted")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    pairs = validate_rows(list(rows.values()))
    design, query = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    cube = np.full((2, 6, 2, 2, 4, 2, 7, 4), np.nan)
    captures, filled = {}, set()
    checks = dict.fromkeys(
        (
            "cache_spread",
            "units",
            "body_final",
            "k_raw",
            "v",
            "noop",
            "kv_transfer",
            "pre_cause_key",
            "pre_cause_value",
        ),
        0.0,
    )
    for d in design:
        raw, row = committed(out, d), rows[d["row"]]
        for name in ("blocks", "interchange_blocks"):
            if not np.array_equal(raw[name], np.arange(manifest["model_blocks"])):
                raise ValueError("Missed a registered block")
        masks = [
            A.permissions(row, len(s), int(raw["execution_length"]), 1)
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
            raise ValueError("Actual native access or suffix endpoint changed")
        checks["cache_spread"] = max(checks["cache_spread"], float(raw["cache_spread"]))
        checks["units"] = max(
            checks["units"],
            float(
                np.max(abs(A.units(raw["logits"], raw["normalizer"]) - raw["readings"]))
            ),
        )
        if d["mode"] == "capture":
            captures[row["id"]] = (d, raw)
        else:
            bd, base = captures[row["id"]]
            dd, donor = captures[d["donor"]]
            receipt = H.load(out / f"01/units/b{d['batch']:03d}.json")
            if receipt["donor_hashes"] != {
                "row": H.sha(out / f"01/units/b{bd['batch']:03d}.npz"),
                "donor": H.sha(out / f"01/units/b{dd['batch']:03d}.npz"),
            }:
                raise ValueError("Causal interchange receipt changed")
            for name in ("body_final", "k_raw", "v"):
                checks[name] = max(
                    checks[name], float(np.max(abs(raw[name] - base[name])))
                )
            if d["mode"] in ("noop", "kv"):
                name = "noop" if d["mode"] == "noop" else "kv_transfer"
                expected = base if name == "noop" else donor
                checks[name] = max(
                    checks[name],
                    float(np.max(abs(raw["readings"] - expected["readings"]))),
                )
            if d["mode"] == "noop":
                continue
        index = (
            row["bank"],
            row["family"],
            (row["b"] + 1) // 2,
            (row["c"] + 1) // 2,
            MODES.index(d["mode"]),
        )
        if index in filled:
            raise ValueError("Duplicate operation substituted for missing evidence")
        filled.add(index)
        cube[index] = np.stack(
            (
                np.concatenate((raw["readings"][:6], raw["readings"][12:13])),
                np.concatenate((raw["readings"][6:12], raw["readings"][13:14])),
            )
        )
    for bank, family, b in itertools.product(range(2), range(6), (-1, 1)):
        lo, hi = (captures[pairs[bank, family, b, c]["id"]][1] for c in (-1, 1))
        prior = 149 - 46
        for name in ("key", "value"):
            checks["pre_cause_" + name] = max(
                checks["pre_cause_" + name],
                float(
                    np.max(
                        abs(
                            lo["history_" + name][:, :, :prior]
                            - hi["history_" + name][:, :, :prior]
                        )
                    )
                ),
            )
    if len(filled) != 192 or not np.isfinite(cube).all():
        raise ValueError("Missing complete causal field")
    if max(checks.values()) > manifest["spec"]["tolerance"]:
        raise ValueError("Causal scope/noop/transfer/units guard failed")
    return cube, checks


def derive(cube):
    """Retain natural slopes and every signed selective causal component."""
    natural = cube[..., 0, :, :, :]
    # Axes are bank,family,b,c,mode,person,target,unit; four natural quantities.
    slopes = natural[:, :, 1] - natural[:, :, 0]
    level = (natural[:, :, :, 1] - natural[:, :, :, 0]).mean(axis=2)
    interaction = slopes[:, :, 1] - slopes[:, :, 0]
    effects = np.stack((slopes[:, :, 0], slopes[:, :, 1], level, interaction), axis=2)
    parts = M.cause_parts(np.moveaxis(cube, 4, 0))
    parts = np.moveaxis(parts, 0, 4)
    summaries = []
    for bank, person, unit in itertools.product(range(2), range(2), range(4)):
        ref = (
            slopes[bank, :, 0, person, :6, unit] - slopes[bank, :, 1, person, :6, unit]
        )
        action_parts = (
            parts[bank, :, 1, 1, :, person, :6, unit]
            - parts[bank, :, 0, 1, :, person, :6, unit]
        )
        denominator = float(np.sum(ref**2))
        summaries.append(
            {
                "bank": bank,
                "person": "AB"[person],
                "unit": UNITS[unit],
                "assigned_behavior_rms": float(
                    np.sqrt(np.mean(slopes[bank, :, 0, person, :6, unit] ** 2))
                ),
                "chosen_behavior_rms": float(
                    np.sqrt(np.mean(slopes[bank, :, 1, person, :6, unit] ** 2))
                ),
                "cause_level_rms": float(
                    np.sqrt(np.mean(level[bank, :, person, :6, unit] ** 2))
                ),
                "behavior_cause_interaction_rms": float(np.sqrt(np.mean(ref**2))),
                "chosen_diagonal": np.diag(
                    slopes[bank, :, 1, person, :6, unit]
                ).tolist(),
                "assigned_diagonal": np.diag(
                    slopes[bank, :, 0, person, :6, unit]
                ).tolist(),
                "factual_cause_max": float(
                    np.max(
                        abs(
                            natural[bank, :, :, 1, person, 6, unit]
                            - natural[bank, :, :, 0, person, 6, unit]
                        )
                    )
                ),
                "cause_parts_signed_projections": [
                    float(np.sum(action_parts[:, j] * ref) / denominator)
                    if denominator
                    else None
                    for j in range(3)
                ],
                "part_reconstruction_error": float(
                    np.max(abs(action_parts.sum(axis=1) - ref))
                ),
            }
        )
    return effects, parts, summaries


def tables(effects, parts):
    """Export all targets/persons/units, including negative and factual fields."""
    targets = H.load(SPEC)["targets"] + ["action_fact"]
    result = {"natural_effects": [], "selective_parts": []}
    for bank, family, person, target, unit in itertools.product(
        range(2), range(6), range(2), range(7), range(4)
    ):
        row = {
            "bank": bank,
            "family": targets[family],
            "person": "AB"[person],
            "target": targets[target],
            "unit": UNITS[unit],
        }
        result["natural_effects"].append(
            {
                **row,
                **dict(
                    zip(
                        (
                            "assigned_behavior",
                            "chosen_behavior",
                            "cause_level",
                            "behavior_cause_interaction",
                        ),
                        map(float, effects[bank, family, :, person, target, unit]),
                        strict=True,
                    )
                ),
            }
        )
        for b, c, part in itertools.product(range(2), range(2), range(3)):
            result["selective_parts"].append(
                {
                    **row,
                    "b": 2 * b - 1,
                    "c": 2 * c - 1,
                    "part": ("k", "v", "kv_interaction")[part],
                    "value": float(
                        parts[bank, family, b, c, part, person, target, unit]
                    ),
                }
            )
    return result


def analyze(out):
    """Persist finite effects without tuning or sign repair."""
    cube, checks = observations(out)
    effects, parts, summaries = derive(cube)
    folder = out / "02"
    folder.mkdir()
    with (folder / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, cube=cube, effects=effects, parts=parts)
    for name, records in tables(effects, parts).items():
        H.table(folder / (name + ".csv"), records)
    H.write(
        folder / "summary.json",
        {
            "status": "FINITE_CAUSAL_VALUATION_NOT_GLOBAL_OR_HUMAN_VALIDITY",
            "checks": checks,
            "fields": summaries,
            "batches": 196,
            "endpoints": 2744,
            "model_seconds": H.load(out / "status.json")["model_seconds"],
            "original_scope_complete": False,
        },
    )
    H.write(folder / "output_seal.json", H.seals(out, folder))
    print("ANALYZED HM134", {"checks": checks, "fields": summaries}, flush=True)


def verify(out):
    """Read-only independent corner/logit arithmetic and full-table replay."""
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    cube, checks = observations(out)
    effects, parts, summaries = derive(cube)
    with np.load(out / "02/observations.npz", allow_pickle=False) as z:
        for name, value in (("cube", cube), ("effects", effects), ("parts", parts)):
            if not np.array_equal(z[name], value):
                raise ValueError("Saved field changed: " + name)
    error = 0.0
    for bank, family, person in itertools.product(range(2), range(6), range(2)):
        n = cube[bank, family, :, :, 0, person]
        bc = (
            sum(
                b * c * n[(b + 1) // 2, (c + 1) // 2]
                for b, c in itertools.product((-1, 1), repeat=2)
            )
            / 4
        )
        error = max(
            error, float(np.max(abs(4 * bc - effects[bank, family, 3, person])))
        )
    error = max(
        error,
        float(
            np.max(abs(parts.sum(axis=4) - (cube[:, :, :, :, 3] - cube[:, :, :, :, 0])))
        ),
    )
    for d in H.load(out / "00/design.json"):
        raw = committed(out, d)
        z = raw["logits"]
        w = np.exp(z - z.max(axis=1)[:, None])
        p = w / w.sum(axis=1)[:, None]
        reconstructed = np.stack(
            (
                p @ np.arange(10) / 9,
                np.log(w[:, 5:].sum(axis=1) / w[:, :5].sum(axis=1)),
                np.exp(z.max(axis=1) + np.log(w.sum(axis=1)) - raw["normalizer"]),
                -np.sum(p * np.log(p), axis=1),
            ),
            axis=1,
        )
        error = max(error, float(np.max(abs(reconstructed - raw["readings"]))))
    if error > 1e-10:
        raise ValueError("Independent primitive or finite-corner check failed")
    saved = H.load(out / "02/summary.json")
    if saved["checks"] != checks or saved["fields"] != summaries:
        raise ValueError("Scientific summary changed")
    counts = {}
    for name, records in tables(effects, parts).items():
        with (out / f"02/{name}.csv").open() as stream:
            actual = list(csv.DictReader(stream))
        if actual != [{n: str(v) for n, v in row.items()} for row in records]:
            raise ValueError("Full result table changed")
        counts[name] = len(records)
    print(
        "VERIFIED HM134",
        {"checks": checks, "independent_error": error, "table_rows": counts},
        flush=True,
    )


def main():
    """Separated stages; no automatic budget renewal or uncertain science replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("prepare", "collect", "analyze", "verify")
    )
    parser.add_argument("--out", type=Path, default=ROOT / H.load(SPEC)["out"])
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyze": analyze, "verify": verify}[
        args.phase
    ](args.out)


if __name__ == "__main__":
    main()
