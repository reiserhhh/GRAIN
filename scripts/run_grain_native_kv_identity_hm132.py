#!/usr/bin/env python3
"""Execute a finite historical K/V ownership interchange, with no predictor fit."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_native_kv_identity as M
from scripts import run_grain_native_access_hm131 as R

H, A = R.H, R.M
SPEC = ROOT / "configs/grain_native_kv_identity_hm132.json"
CODE = tuple(
    dict.fromkeys(
        (
            *R.CODE,
            "configs/grain_native_kv_identity_hm132.json",
            "docs/GRAIN_NATIVE_KV_IDENTITY_PLAN_HM132.md",
            "grain_research/construction/math_native_kv_identity.py",
            "scripts/run_grain_native_kv_identity_hm132.py",
            "tests/test_grain_native_kv_identity.py",
        )
    )
)
BANKS = (0, 3)


def validate_rows(rows):
    """Require complete paired support and fixed non-history token addresses."""
    wanted = set(itertools.product(BANKS, range(6), (-1, 1), (-1, 1)))
    by_key = {
        (r["bank"], r["family"], r["polarity"], r["acting_person"]): r for r in rows
    }
    if len(rows) != 48 or set(by_key) != wanted:
        raise ValueError("All48 paired old/new histories are required")
    reference = None
    for r in rows:
        groups = A.partition(r)
        fixed = [
            (groups[n], [r["input_ids"][i] for i in groups[n]]) for n in ("S", "B", "C")
        ]
        if len(r["input_ids"]) != 187 or len(r["history"]) != 122:
            raise ValueError("Frozen body and history slot dimensions changed")
        if reference is not None and fixed != reference:
            raise ValueError("Non-history body tokens or positions changed")
        reference = fixed
    for bank, family, b in itertools.product(BANKS, range(6), (-1, 1)):
        left, right = (by_key[bank, family, b, o] for o in (-1, 1))
        if left["history"] != right["history"] or left["padding"] != right["padding"]:
            raise ValueError("Historical donors need identical positional support")
        if Counter(left["input_ids"]) != Counter(right["input_ids"]):
            raise ValueError("Owner pairs must retain the same token bag")
        if (
            left["history_text"].rsplit("\n", 1)[0]
            != right["history_text"].rsplit("\n", 1)[0]
        ):
            raise ValueError("Paired actions differ outside the ownership declaration")
    return by_key


def fixed_design(rows):
    """Freeze48 captures,4 early matched controls,48 off-diagonal interchanges."""
    by_key, design = validate_rows(rows), []

    def append(row, mode, k, v):
        key = (row["bank"], row["family"], row["polarity"])
        design.append(
            {
                "batch": len(design),
                "row": row["id"],
                "mode": mode,
                "k": k,
                "v": v,
                "donor_k": by_key[*key, k]["id"],
                "donor_v": by_key[*key, v]["id"],
            }
        )

    for row in rows:
        append(row, "capture", row["acting_person"], row["acting_person"])
    for b, owner in itertools.product((-1, 1), (-1, 1)):
        append(by_key[0, 0, b, 1], "noop", owner, owner)
    for bank, family, b in itertools.product(BANKS, range(6), (-1, 1)):
        for k, v in ((-1, 1), (1, -1)):
            append(by_key[bank, family, b, 1], "mixed", k, v)
    if len(design) != 100:
        raise ValueError("Fixed100-batch design is incomplete")
    return design


def check(out):
    """Authenticate the source, fixed program, and frozen experiment inputs."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Create paired new public materials before any new model forward."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL, write_json

    if out.exists():
        raise FileExistsError("Do not overwrite old or failed experiment roots")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    old = R.check(source)
    H.verify_hashes(source, H.load(source / "02/output_seal.json"))
    old_rows = H.load(source / "00/rows.json")
    old_design = H.load(source / "00/design.json")
    source_batches = {
        d["row"]: d["batch"] for d in old_design if d["route"] == 1 and not d["noop"]
    }
    rows = []
    selected = [r for r in old_rows if r["bank"] == 0]
    for r in selected:
        rows.append(
            {
                **r,
                "id": len(rows),
                "source_row": r["id"],
                "source_batch": source_batches[r["id"]],
            }
        )
    owners = H.load(R.P.SPEC)["owners"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    for r in selected:
        phrases = [
            spec["new_behaviours"][target][r["polarity"] + 1 if j == r["family"] else 1]
            for j, target in enumerate(spec["targets"])
        ]
        text = "\n".join(
            f"Earlier entry E{j + 1}: the person {p}." for j, p in enumerate(phrases)
        )
        text += "\n" + owners[r["acting_person"] + 1]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > len(r["history"]):
            raise ValueError(
                f"New history too long: family{r['family']},n={len(tokens)}"
            )
        ids = list(r["input_ids"])
        for offset, position in enumerate(r["history"]):
            ids[position] = (
                tokens[offset] if offset < len(tokens) else tokenizer.pad_token_id
            )
        rows.append(
            {
                **r,
                "id": len(rows),
                "bank": 3,
                "history_text": text,
                "input_ids": ids,
                "padding": r["history"][len(tokens) :],
                "source_row": None,
                "source_batch": None,
            }
        )
    query = H.load(source / "00/queries.json")
    if len(query["suffixes"]) != 12 or spec["targets"] != old["spec"]["targets"]:
        raise ValueError("All six targets and both query people required")
    design = fixed_design(rows)
    if len(design) != spec["max_batches"]:
        raise ValueError("Do not silently change the model-call budget")
    sources = [
        source / p
        for p in (
            "00/manifest.json",
            "00/rows.json",
            "00/design.json",
            "00/queries.json",
            "01/raw_seal.json",
            "02/summary.json",
            "02/output_seal.json",
        )
    ]
    for r in rows[:24]:
        R.raw_unit(source, r["source_batch"])
        sources.extend(
            source / f"01/units/b{r['source_batch']:03d}.{ext}"
            for ext in ("json", "npz")
        )
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, data in (("rows", rows), ("design", design), ("queries", query)):
        H.write(out / f"00/{name}.json", data)
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "model": old["model"],
            "model_blocks": old["model_blocks"],
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
            "inputs": H.seals(out, out / "00"),
            "natural_old_previously_observed": True,
            "new_predictor_fits": 0,
            "new_history_texts": 24,
            "new_model_batches": 100,
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
    print("PREPARED HM132", len(rows), len(design), flush=True)


def committed(out, index):
    """Read a committed raw cache with its exact design receipt."""
    stem = out / f"01/units/b{index:03d}"
    meta = H.load(stem.with_suffix(".json"))
    if H.sha(stem.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Raw historical primitive changed")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        return {name: z[name] for name in z.files}


def collect(out):
    """Execute once, preserving any uncertain attempt without budget renewal."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No automatic uncertain or terminal replay")
    H.verify_hashes(MODEL, manifest["model"])
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    design = H.load(out / "00/design.json")
    query = H.load(out / "00/queries.json")
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
                row = rows[d["row"]]
                donors, receipts = None, {}
                if d["mode"] != "capture":
                    key, value = (
                        committed(out, captures[d[n]]) for n in ("donor_k", "donor_v")
                    )
                    donors = (key["history_key"], value["history_value"])
                    receipts = {
                        n: H.sha(out / f"01/units/b{captures[d[n]]:03d}.npz")
                        for n in ("donor_k", "donor_v")
                    }
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=d
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    arrays = M.native_batch(observer, row, query, donors=donors)
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    write_json(out / "status.json", status)
                if float(arrays["cache_spread"]) > tol:
                    raise ValueError("Historical cache depends on future query")
                if d["mode"] == "capture":
                    captures[row["id"]] = d["batch"]
                    if row["source_batch"] is not None:
                        old = R.raw_unit(
                            ROOT / manifest["spec"]["source"], row["source_batch"]
                        )
                        if np.max(abs(arrays["readings"] - old["readings"])) > tol:
                            raise ValueError(
                                "Unchanged old H-only source reading changed"
                            )
                else:
                    base = committed(out, captures[row["id"]])
                    for name in ("body_final", "k_raw", "v"):
                        if np.max(abs(arrays[name] - base[name])) > tol:
                            raise ValueError("Suffix interchange changed body: " + name)
                    if d["mode"] == "noop":
                        expected = committed(out, captures[d["donor_k"]])
                        if np.max(abs(arrays["readings"] - expected["readings"])) > tol:
                            raise ValueError("Matched whole-cache control failed")
                stem = out / f"01/units/b{d['batch']:03d}"
                temporary = stem.with_suffix(".tmp")
                with temporary.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                temporary.replace(stem.with_suffix(".npz"))
                H.write(
                    stem.with_suffix(".json"),
                    {
                        **d,
                        "seconds": seconds,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "donor_hashes": receipts,
                        "endpoints": 12,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 12 == 0 or d["mode"] == "noop":
                    print("HM132", status, flush=True)
            if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Original finite cap crossed")
            status["state"] = "COLLECTED"
            write_json(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM132", status, flush=True)


def observations(out):
    """Reconstruct complete role support and audit actual donor/cache operations."""
    manifest, status = check(out), H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["pending"] is not None
        or status["completed"] != 100
    ):
        raise ValueError("Incomplete support cannot yield a full K/V attribution")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    validate_rows(list(rows.values()))
    query = H.load(out / "00/queries.json")
    cube = np.full((2, 6, 2, 2, 2, 2, 6, 4), np.nan)
    captures = {}
    checks = dict.fromkeys(
        ("cache_spread", "body_final", "k_raw", "v", "noop", "old_source", "units"), 0.0
    )
    filled = set()
    for d in H.load(out / "00/design.json"):
        row, raw = rows[d["row"]], committed(out, d["batch"])
        meta = H.load(out / f"01/units/b{d['batch']:03d}.json")
        if any(meta[n] != v for n, v in d.items()):
            raise ValueError("Raw unit receipt differs from frozen design")
        for name in ("blocks", "interchange_blocks"):
            if not np.array_equal(raw[name], np.arange(manifest["model_blocks"])):
                raise ValueError("Actual operation missed a decoder block")
        width = int(raw["execution_length"])
        permissions = [A.permissions(row, len(s), width, 1) for s in query["suffixes"]]
        hashes = np.stack(
            [
                np.frombuffer(hashlib.sha256(p.tobytes()).digest(), dtype=np.uint8)
                for p in permissions
            ]
        )
        if not np.array_equal(raw["support_hash"], hashes):
            raise ValueError("Historical query access differs from planned support")
        if not np.array_equal(
            raw["endpoints"], [187 + len(s) - 1 for s in query["suffixes"]]
        ):
            raise ValueError("Native suffix was truncated or moved")
        checks["cache_spread"] = max(checks["cache_spread"], float(raw["cache_spread"]))
        checks["units"] = max(
            checks["units"],
            float(
                np.max(abs(A.units(raw["logits"], raw["normalizer"]) - raw["readings"]))
            ),
        )
        if d["mode"] == "capture":
            captures[row["id"]] = d["batch"]
            if row["source_batch"] is not None:
                old = R.raw_unit(ROOT / manifest["spec"]["source"], row["source_batch"])
                checks["old_source"] = max(
                    checks["old_source"],
                    float(np.max(abs(raw["readings"] - old["readings"]))),
                )
            for name in ("history_key", "history_value"):
                if (
                    raw[name].shape[0] != 28
                    or raw[name].shape[2] != 122
                    or not np.isfinite(raw[name]).all()
                ):
                    raise ValueError("Incomplete or invalid actual historical cache")
        else:
            base = committed(out, captures[row["id"]])
            for name in ("body_final", "k_raw", "v"):
                checks[name] = max(
                    checks[name], float(np.max(abs(raw[name] - base[name])))
                )
            expected_hashes = {
                n: H.sha(out / f"01/units/b{captures[d[n]]:03d}.npz")
                for n in ("donor_k", "donor_v")
            }
            if meta["donor_hashes"] != expected_hashes:
                raise ValueError("Interchange used an unaddressed donor")
            if d["mode"] == "noop":
                expected = committed(out, captures[d["donor_k"]])
                checks["noop"] = max(
                    checks["noop"],
                    float(np.max(abs(raw["readings"] - expected["readings"]))),
                )
                continue
        index = (
            BANKS.index(row["bank"]),
            row["family"],
            (row["polarity"] + 1) // 2,
            (d["k"] + 1) // 2,
            (d["v"] + 1) // 2,
        )
        if index in filled:
            raise ValueError(
                "Duplicate role corner cannot stand in for missing support"
            )
        filled.add(index)
        cube[index] = raw["readings"].reshape(2, 6, 4)[::-1]
    if len(filled) != 96 or not np.isfinite(cube).all():
        raise ValueError("Full natural and off-diagonal support is missing")
    if max(checks.values()) > manifest["spec"]["tolerance"]:
        raise ValueError("Actual primitive/body/no-op validation failed")
    return cube, checks


def derive(cube):
    """Retain every character and the exact natural-diagonal alias relation."""
    gamma = np.stack([[M.characters(c) for c in bank] for bank in cube])
    beta = np.stack([[A.characters(M.diagonal(c)) for c in bank] for bank in cube])
    error = max(
        float(np.max(abs(M.restrict_coefficients(g) - b)))
        for gb, bb in zip(gamma, beta, strict=True)
        for g, b in zip(gb, bb, strict=True)
    )
    summary = []
    for bank, unit in itertools.product(range(2), range(4)):
        key, value, natural = (
            gamma[bank, :, 11, :, unit],
            gamma[bank, :, 13, :, unit],
            beta[bank, :, 7, :, unit],
        )
        norm = float(np.sum(natural**2))
        summary.append(
            {
                "bank": BANKS[bank],
                "unit": R.UNITS[unit],
                "natural_bop_rms": float(np.sqrt(np.mean(natural**2))),
                "key_bkp_rms": float(np.sqrt(np.mean(key**2))),
                "value_bvp_rms": float(np.sqrt(np.mean(value**2))),
                "key_signed_projection": float(np.sum(key * natural) / norm)
                if norm
                else None,
                "value_signed_projection": float(np.sum(value * natural) / norm)
                if norm
                else None,
                "key_value_inner_product": float(np.mean(key * value)),
                "bkvp_rms": float(np.sqrt(np.mean(gamma[bank, :, 15, :, unit] ** 2))),
                "key_matched_diagonal": np.diag(key).tolist(),
                "value_matched_diagonal": np.diag(value).tolist(),
                "natural_matched_diagonal": np.diag(natural).tolist(),
            }
        )
    return gamma, beta, error, summary


def tables(gamma, beta):
    """Export full6x6 fields in all units, not a best-target subset."""
    spec = H.load(SPEC)
    output = []
    for array, roles in ((gamma, 16), (beta, 8)):
        rows = []
        for bank, family, role, target, unit in itertools.product(
            range(2), range(6), range(roles), range(6), range(4)
        ):
            rows.append(
                {
                    "bank": BANKS[bank],
                    "family": spec["targets"][family],
                    "character": role,
                    "target": spec["targets"][target],
                    "unit": R.UNITS[unit],
                    "value": float(array[bank, family, role, target, unit]),
                }
            )
        output.append(rows)
    return output


def analyze(out):
    """Save exact finite mechanism results without fitting or selecting a winner."""
    cube, checks = observations(out)
    gamma, beta, error, summary = derive(cube)
    folder = out / "02"
    folder.mkdir()
    with (folder / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, cube=cube, gamma=gamma, beta=beta)
    for name, records in zip(
        ("characters", "natural_diagonal"), tables(gamma, beta), strict=True
    ):
        H.table(folder / f"{name}.csv", records)
    H.write(
        folder / "summary.json",
        {
            "status": "FINITE_NATIVE_KV_INTERCHANGE_NOT_GLOBAL_THEORY",
            "checks": checks,
            "restriction_error": error,
            "fields": summary,
            "actual_batches": 100,
            "actual_endpoints": 1200,
            "original_scope_complete": False,
            "model_seconds": H.load(out / "status.json")["model_seconds"],
        },
    )
    H.write(out / "02/output_seal.json", H.seals(out, folder))
    print("ANALYZED HM132", {"checks": checks, "restriction_error": error}, flush=True)


def verify(out):
    """Independently substitute signs and recompute primitive units read-only."""
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    cube, checks = observations(out)
    gamma, beta, error, summary = derive(cube)
    with np.load(out / "02/observations.npz", allow_pickle=False) as z:
        if any(
            not np.array_equal(z[n], v)
            for n, v in (("cube", cube), ("gamma", gamma), ("beta", beta))
        ):
            raise ValueError("Saved complete role fields changed")
    max_error = 0.0
    for bank, family in itertools.product(range(2), range(6)):
        rebuilt = np.zeros_like(cube[bank, family])
        for bits in itertools.product(range(2), repeat=4):
            signs = [2 * bit - 1 for bit in bits]
            for mask in range(16):
                weight = 1
                for j in range(4):
                    if mask & (1 << j):
                        weight *= signs[j]
                rebuilt[bits] += weight * gamma[bank, family, mask]
        max_error = max(max_error, float(np.max(abs(rebuilt - cube[bank, family]))))
        for b, o, p in itertools.product(range(2), repeat=3):
            natural = cube[bank, family, b, o, o, p]
            max_error = max(
                max_error, float(np.max(abs(natural - rebuilt[b, o, o, p])))
            )
    for d in H.load(out / "00/design.json"):
        raw = committed(out, d["batch"])
        z = raw["logits"]
        shifted = z - z.max(axis=1)[:, None]
        weights = np.exp(shifted)
        probabilities = weights / weights.sum(axis=1)[:, None]
        independent = np.stack(
            (
                probabilities @ np.arange(10) / 9,
                np.log(weights[:, 5:].sum(axis=1) / weights[:, :5].sum(axis=1)),
                np.exp(z.max(axis=1) + np.log(weights.sum(axis=1)) - raw["normalizer"]),
                -np.sum(probabilities * np.log(probabilities), axis=1),
            ),
            axis=1,
        )
        max_error = max(max_error, float(np.max(abs(independent - raw["readings"]))))
    if max_error > 1e-10:
        raise ValueError("Independent primitive/sign reconstruction failed")
    saved = H.load(out / "02/summary.json")
    if (
        saved["checks"] != checks
        or saved["fields"] != summary
        or saved["restriction_error"] != error
    ):
        raise ValueError("Saved summary differs from authenticated raw observations")
    for name, records in zip(
        ("characters", "natural_diagonal"), tables(gamma, beta), strict=True
    ):
        import csv

        with (out / f"02/{name}.csv").open() as stream:
            actual = list(csv.DictReader(stream))
        expected = [{n: str(v) for n, v in row.items()} for row in records]
        if actual != expected:
            raise ValueError("Full signed result table differs")
    print(
        "VERIFIED HM132",
        {
            "max_independent_error": max_error,
            "checks": checks,
            "character_rows": 4608,
            "diagonal_rows": 2304,
        },
        flush=True,
    )


def main():
    """Run explicit stages; never auto-resume uncertain model work."""
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
