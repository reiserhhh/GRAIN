#!/usr/bin/env python3
"""Run a bounded historical-key sector experiment, never an accuracy search."""

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
from grain_research.construction import math_native_key_sectors as M
from scripts import run_grain_native_kv_identity_hm132 as R

H, A = R.H, R.A
SPEC = ROOT / "configs/grain_native_key_sectors_hm133.json"
CODE = tuple(
    dict.fromkeys(
        (
            *R.CODE,
            "configs/grain_native_key_sectors_hm133.json",
            "docs/GRAIN_NATIVE_KEY_SECTORS_PLAN_HM133.md",
            "grain_research/construction/math_native_key_sectors.py",
            "scripts/run_grain_native_key_sectors_hm133.py",
            "tests/test_grain_native_key_sectors.py",
        )
    )
)
BANKS = (4, 5)
MODES = ("blind", "role", "joint", "full", "common")
FLAGS = {"blind": (0, 0), "role": (1, 0), "joint": (0, 1), "full": (1, 1)}


def array_sha(value):
    """Hash a declared dtype/shape and its exact bytes, not merely a filename."""
    x = np.ascontiguousarray(value)
    return hashlib.sha256(
        str((x.dtype.str, x.shape)).encode() + x.tobytes()
    ).hexdigest()


def validate_rows(rows):
    """Verify all48 paired histories and a globally registered declaration."""
    by_key = {
        (r["bank"], r["family"], r["polarity"], r["acting_person"]): r for r in rows
    }
    wanted = set(itertools.product(BANKS, range(6), (-1, 1), (-1, 1)))
    if len(rows) != 48 or set(by_key) != wanted:
        raise ValueError("Complete two-bank/family/behavior/owner support required")
    reference = None
    for row in rows:
        groups = A.partition(row)
        signature = [
            (groups[n], [row["input_ids"][i] for i in groups[n]])
            for n in ("S", "B", "C")
        ]
        signature.append(row["declaration"])
        if len(row["input_ids"]) != 187 or len(row["history"]) != 122:
            raise ValueError("Frozen native body support changed")
        if reference is not None and signature != reference:
            raise ValueError("Absolute declaration or non-history support changed")
        reference = signature
        if set(row["padding"]) & set(row["declaration"]):
            raise ValueError("The declaration must remain actually visible")
    for bank, f, b in itertools.product(BANKS, range(6), (-1, 1)):
        left, right = (by_key[bank, f, b, o] for o in (-1, 1))
        if left["padding"] != right["padding"] or Counter(left["input_ids"]) != Counter(
            right["input_ids"]
        ):
            raise ValueError("Owner edits changed the token bag or opportunities")
        changed = {
            i
            for i, (x, y) in enumerate(
                zip(left["input_ids"], right["input_ids"], strict=True)
            )
            if x != y
        }
        if not changed or not changed <= set(left["declaration"]):
            raise ValueError("Only declaration ownership tokens may change")
    return by_key


def fixed_design(rows):
    """Freeze captures, early controls and every selective intervention."""
    by_key, design = validate_rows(rows), []

    def add(row, mode, control=False):
        design.append(
            {"batch": len(design), "row": row["id"], "mode": mode, "control": control}
        )

    for row in rows:
        add(row, "full")
    for b, o in itertools.product((-1, 1), repeat=2):
        add(by_key[4, 0, b, o], "full", True)
    for mode in MODES[:3]:
        for row in rows:
            add(row, mode)
    for row in rows:
        if row["bank"] == 5:
            add(row, "common")
    if len(design) != 220:
        raise ValueError("Fixed model-call budget changed")
    return design


def check(out):
    """Authenticate frozen protocol, inputs and inherited code provenance."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Register new public evidence without reading new outputs or truncating it."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL, write_json

    if out.exists():
        raise FileExistsError("Preserve old and failed roots")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    old = R.check(source)
    H.verify_hashes(source, H.load(source / "02/output_seal.json"))
    blueprint = H.load(source / "00/rows.json")[0]
    query = H.load(source / "00/queries.json")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    owners = H.load(R.R.P.SPEC)["owners"]
    declarations = {
        o: tokenizer.encode("\n" + owners[o + 1], add_special_tokens=False)
        for o in (-1, 1)
    }
    if len(declarations[-1]) != len(declarations[1]) or Counter(
        declarations[-1]
    ) != Counter(declarations[1]):
        raise ValueError("Declaration tokens must be matched")
    width = len(declarations[1])
    declaration = blueprint["history"][-width:]
    actions = blueprint["history"][:-width]
    rows = []
    for bank, family, b, o in itertools.product(BANKS, range(6), (-1, 1), (-1, 1)):
        phrases = [
            spec["behaviours"][BANKS.index(bank)][target][b + 1 if j == family else 1]
            for j, target in enumerate(spec["targets"])
        ]
        text = "\n".join(
            f"Earlier entry E{j + 1}: the person {phrase}."
            for j, phrase in enumerate(phrases)
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > len(actions):
            raise ValueError(
                f"No evidence truncation permitted:bank{bank}/family{family}/b{b}/{len(tokens)}>{len(actions)}"
            )
        ids = list(blueprint["input_ids"])
        for j, position in enumerate(actions):
            ids[position] = tokens[j] if j < len(tokens) else tokenizer.pad_token_id
        for position, token in zip(declaration, declarations[o], strict=True):
            ids[position] = token
        rows.append(
            {
                **blueprint,
                "id": len(rows),
                "bank": bank,
                "family": family,
                "polarity": b,
                "acting_person": o,
                "input_ids": ids,
                "padding": actions[len(tokens) :],
                "declaration": declaration,
                "history_text": text + "\n" + owners[o + 1],
                "action_token_count": len(tokens),
                "source_row": None,
                "source_batch": None,
            }
        )
    design = fixed_design(rows)
    if len(design) != spec["max_batches"] or spec["targets"] != old["spec"]["targets"]:
        raise ValueError("Target support or budget differs from protocol")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, data in (("rows", rows), ("design", design), ("queries", query)):
        H.write(out / f"00/{name}.json", data)
    paths = [
        source / p
        for p in (
            "00/manifest.json",
            "00/rows.json",
            "00/queries.json",
            "02/summary.json",
            "02/output_seal.json",
        )
    ]
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "model": old["model"],
            "model_blocks": 28,
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in paths},
            "inputs": H.seals(out, out / "00"),
            "declaration_positions": declaration,
            "new_public_histories": 48,
            "new_predictor_fits": 0,
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
    print(
        "PREPARED HM133",
        {
            "rows": 48,
            "batches": 220,
            "declaration": declaration,
            "action_counts": sorted({r["action_token_count"] for r in rows}),
        },
        flush=True,
    )


def basis(out, rows):
    """Compile content/owner projections from48 captures, never their readouts."""
    lookup = validate_rows(rows)
    sectors = []
    error, earlier = 0.0, 0.0
    for bank in BANKS:
        group = []
        for f in range(6):
            keys = np.stack(
                [
                    np.stack(
                        [
                            R.committed(out, lookup[bank, f, b, o]["id"])["history_key"]
                            for o in (-1, 1)
                        ]
                    )
                    for b in (-1, 1)
                ]
            )
            s = M.key_characters(keys)
            for bi, oi in itertools.product(range(2), repeat=2):
                reconstructed = M.reconstruct_key(
                    s, 2 * bi - 1, 2 * oi - 1, 1, 1
                ).astype(np.float32)
                error = max(error, float(np.max(abs(reconstructed - keys[bi, oi]))))
            first = lookup[bank, f, 1, 1]["declaration"][0] - min(rows[0]["history"])
            earlier = max(earlier, float(np.max(abs(s[2:, :, :, :first]))))
            group.append(s)
        sectors.append(np.stack(group))
    if error != 0 or earlier != 0:
        raise ValueError(
            "Exact key reconstruction or causal pre-declaration support failed"
        )
    return np.stack(sectors), {
        "key_reconstruction_error": error,
        "owner_sectors_before_declaration": earlier,
    }


def donor(sectors, common, row, mode):
    """Manufacture only the predeclared finite key sector combination."""
    r, s = FLAGS[mode] if mode != "common" else (1, 0)
    return M.reconstruct_key(
        sectors[BANKS.index(row["bank"]), row["family"]],
        row["polarity"],
        row["acting_person"],
        r,
        s,
        common_role=common if mode == "common" else None,
    ).astype(np.float32)


def collect(out):
    """Collect exactly once; retain uncertain failures and the original budget."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No automatic uncertain or terminal replay")
    H.verify_hashes(MODEL, manifest["model"])
    rows, design = H.load(out / "00/rows.json"), H.load(out / "00/design.json")
    query, sectors, common = H.load(out / "00/queries.json"), None, None
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
            for d in design:
                if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Original finite model budget reached")
                row, actual_donors, receipt = rows[d["row"]], None, {}
                if d["batch"] >= 48:
                    if sectors is None:
                        sectors, checks = basis(out, rows)
                        common = sectors[0, :, 2].mean(axis=0)
                        with (out / "01/basis.npz").open("xb") as stream:
                            np.savez_compressed(stream, sectors=sectors, common=common)
                        H.write(
                            out / "01/basis.json",
                            {
                                **checks,
                                "sha256": H.sha(out / "01/basis.npz"),
                                "source_bank": 4,
                                "common_role_sha256": array_sha(common),
                                "source_readouts_used": False,
                                "frozen_before_interventions": True,
                            },
                        )
                    base = R.committed(out, row["id"])
                    key = donor(sectors, common, row, d["mode"])
                    actual_donors = key, base["history_value"]
                    receipt = {
                        "key_sha256": array_sha(key),
                        "value_sha256": array_sha(base["history_value"]),
                        "base_sha256": H.sha(out / f"01/units/b{row['id']:03d}.npz"),
                    }
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=d
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    arrays = R.M.native_batch(
                        observer, row, query, donors=actual_donors
                    )
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    write_json(out / "status.json", status)
                if float(arrays["cache_spread"]) > tol:
                    raise ValueError("Historical carriers depend on future suffix")
                if actual_donors is not None:
                    for name in ("body_final", "k_raw", "v"):
                        if np.max(abs(arrays[name] - base[name])) > tol:
                            raise ValueError(
                                "Key-only suffix edit changed body: " + name
                            )
                    if (
                        d["control"]
                        and np.max(abs(arrays["readings"] - base["readings"])) > tol
                    ):
                        raise ValueError("Reconstructed full-key no-op failed")
                stem = out / f"01/units/b{d['batch']:03d}"
                with stem.with_suffix(".tmp").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                stem.with_suffix(".tmp").replace(stem.with_suffix(".npz"))
                H.write(
                    stem.with_suffix(".json"),
                    {
                        **d,
                        **receipt,
                        "seconds": seconds,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "endpoints": 12,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 24 == 0 or d["control"]:
                    print("HM133", status, flush=True)
            if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Original finite cap crossed")
            status["state"] = "COLLECTED"
            write_json(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM133", status, flush=True)


def observations(out):
    """Audit actual support, donors and no-ops while rebuilding complete fields."""
    manifest, status = check(out), H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 220
        or status["pending"] is not None
    ):
        raise ValueError("Incomplete support is not a mechanism result")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows, query = H.load(out / "00/rows.json"), H.load(out / "00/queries.json")
    validate_rows(rows)
    sectors, basis_checks = basis(out, rows)
    common = sectors[0, :, 2].mean(axis=0)
    with np.load(out / "01/basis.npz", allow_pickle=False) as z:
        if not np.array_equal(z["sectors"], sectors) or not np.array_equal(
            z["common"], common
        ):
            raise ValueError("Source sectors were changed after intervention")
    fields = np.full((2, 4, 6, 2, 2, 2, 6, 4), np.nan)
    shared = np.full((6, 2, 2, 2, 6, 4), np.nan)
    checks = {
        **basis_checks,
        **dict.fromkeys(
            ("body_final", "k_raw", "v", "noop", "cache_spread", "units"), 0.0
        ),
    }
    filled = set()
    for d in H.load(out / "00/design.json"):
        row, raw = rows[d["row"]], R.committed(out, d["batch"])
        meta = H.load(out / f"01/units/b{d['batch']:03d}.json")
        if any(meta[n] != v for n, v in d.items()):
            raise ValueError("Receipt no longer matches the addressed design")
        for name in ("blocks", "interchange_blocks"):
            if not np.array_equal(raw[name], np.arange(28)):
                raise ValueError("Operation skipped a block")
        expected = np.stack(
            [
                np.frombuffer(
                    hashlib.sha256(
                        A.permissions(
                            row, len(s), int(raw["execution_length"]), 1
                        ).tobytes()
                    ).digest(),
                    dtype=np.uint8,
                )
                for s in query["suffixes"]
            ]
        )
        if not np.array_equal(raw["support_hash"], expected) or not np.array_equal(
            raw["endpoints"], [187 + len(s) - 1 for s in query["suffixes"]]
        ):
            raise ValueError("Suffix permissions or endpoint support changed")
        checks["cache_spread"] = max(checks["cache_spread"], float(raw["cache_spread"]))
        checks["units"] = max(
            checks["units"],
            float(
                np.max(abs(A.units(raw["logits"], raw["normalizer"]) - raw["readings"]))
            ),
        )
        if d["batch"] >= 48:
            base = R.committed(out, row["id"])
            key = donor(sectors, common, row, d["mode"])
            if (
                meta["key_sha256"] != array_sha(key)
                or meta["value_sha256"] != array_sha(base["history_value"])
                or meta["base_sha256"] != H.sha(out / f"01/units/b{row['id']:03d}.npz")
            ):
                raise ValueError("Actual key/value source receipt differs")
            for name in ("body_final", "k_raw", "v"):
                checks[name] = max(
                    checks[name], float(np.max(abs(raw[name] - base[name])))
                )
            if d["control"]:
                checks["noop"] = max(
                    checks["noop"],
                    float(np.max(abs(raw["readings"] - base["readings"]))),
                )
                continue
        value = raw["readings"].reshape(2, 6, 4)[::-1]
        index = (
            row["family"],
            (row["polarity"] + 1) // 2,
            (row["acting_person"] + 1) // 2,
        )
        address = (row["bank"], d["mode"], *index)
        if address in filled:
            raise ValueError("Duplicate corner cannot replace missing support")
        filled.add(address)
        if d["mode"] == "common":
            shared[index] = value
        else:
            fields[BANKS.index(row["bank"]), MODES.index(d["mode"]), *index] = value
    if (
        len(filled) != 216
        or not np.isfinite(fields).all()
        or not np.isfinite(shared).all()
    ):
        raise ValueError("Full sector/bank/target support is missing")
    if max(checks.values()) > manifest["spec"]["tolerance"]:
        raise ValueError("Actual invariance or primitive-unit guard failed")
    return fields, shared, sectors, checks


def derive(fields, shared):
    """Retain all role characters and finite output interactions, with signs."""
    beta = np.stack(
        [[np.stack([A.characters(c) for c in arm]) for arm in bank] for bank in fields]
    )
    common_beta = np.stack([A.characters(c) for c in shared])
    parts = np.stack([M.intervention_parts(b) for b in beta])
    summaries = []
    for bank, unit in itertools.product(range(2), range(4)):
        qualified = beta[bank, :, :, 7, :, unit]
        reference = qualified[3] - qualified[0]
        norm = float(np.sum(reference**2))
        record = {
            "bank": BANKS[bank],
            "unit": R.R.UNITS[unit],
            "natural_bop_rms": float(np.sqrt(np.mean(qualified[3] ** 2))),
            "blind_bop_rms": float(np.sqrt(np.mean(qualified[0] ** 2))),
            "full_minus_blind_rms": float(np.sqrt(np.mean(reference**2))),
        }
        for j, name in enumerate(("role", "joint", "interaction"), 1):
            x = parts[bank, j, :, 7, :, unit]
            record[name + "_rms"] = float(np.sqrt(np.mean(x**2)))
            record[name + "_signed_projection"] = (
                float(np.sum(x * reference) / norm) if norm else None
            )
            record[name + "_matched_diagonal"] = np.diag(x).tolist()
        record["natural_matched_diagonal"] = np.diag(qualified[3]).tolist()
        if bank == 1:
            x, local = (
                common_beta[:, 7, :, unit] - qualified[0],
                qualified[1] - qualified[0],
            )
            record["source_common_role_rms"] = float(np.sqrt(np.mean(x**2)))
            record["source_common_to_local_cosine"] = (
                float(np.sum(x * local) / np.sqrt(np.sum(x**2) * np.sum(local**2)))
                if np.sum(x**2) * np.sum(local**2)
                else None
            )
            record["source_common_minus_local_rms"] = float(
                np.sqrt(np.mean((x - local) ** 2))
            )
            record["source_common_signed_projection"] = (
                float(np.sum(x * reference) / norm) if norm else None
            )
        summaries.append(record)
    return beta, common_beta, parts, summaries


def tables(beta, common_beta, parts):
    """Export every target, character, intervention account and author law."""
    targets, characters, accounts, authors = H.load(SPEC)["targets"], [], [], []
    for bank, mode, family, char, target, unit in itertools.product(
        range(2), range(4), range(6), range(8), range(6), range(4)
    ):
        address = {
            "bank": BANKS[bank],
            "mode": MODES[mode],
            "family": targets[family],
            "character": char,
            "target": targets[target],
            "unit": R.R.UNITS[unit],
        }
        characters.append(
            {**address, "value": float(beta[bank, mode, family, char, target, unit])}
        )
        accounts.append(
            {
                **address,
                "part": ("blind", "role", "joint", "interaction")[mode],
                "value": float(parts[bank, mode, family, char, target, unit]),
            }
        )
    for bank, mode, family in itertools.product(range(2), range(4), range(6)):
        value = M.author_difference(beta[bank, mode, family])
        for p, target, unit in itertools.product(range(2), range(6), range(4)):
            authors.append(
                {
                    "bank": BANKS[bank],
                    "mode": MODES[mode],
                    "family": targets[family],
                    "queried_person": 2 * p - 1,
                    "target": targets[target],
                    "unit": R.R.UNITS[unit],
                    "value": float(value[p, target, unit]),
                }
            )
    for family, char, target, unit in itertools.product(
        range(6), range(8), range(6), range(4)
    ):
        characters.append(
            {
                "bank": 5,
                "mode": "common",
                "family": targets[family],
                "character": char,
                "target": targets[target],
                "unit": R.R.UNITS[unit],
                "value": float(common_beta[family, char, target, unit]),
            }
        )
    return {
        "characters": characters,
        "intervention_accounts": accounts,
        "author_accounts": authors,
    }


def analyze(out):
    """Save full finite results without a predictor, winner or polarity repair."""
    fields, shared, sectors, checks = observations(out)
    beta, common_beta, parts, summaries = derive(fields, shared)
    folder = out / "02"
    folder.mkdir()
    with (folder / "observations.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            fields=fields,
            shared=shared,
            beta=beta,
            common_beta=common_beta,
            parts=parts,
        )
    for name, records in tables(beta, common_beta, parts).items():
        H.table(folder / f"{name}.csv", records)
    norms = np.sqrt(np.mean(sectors**2, axis=(3, 4, 5, 6)))
    H.write(
        folder / "summary.json",
        {
            "status": "FINITE_KEY_SECTOR_MECHANISM_NOT_GLOBAL_THEORY",
            "checks": checks,
            "key_sector_rms_by_bank_family": norms.tolist(),
            "fields": summaries,
            "batches": 220,
            "endpoints": 2640,
            "model_seconds": H.load(out / "status.json")["model_seconds"],
            "original_scope_complete": False,
        },
    )
    H.write(out / "02/output_seal.json", H.seals(out, folder))
    print("ANALYZED HM133", {"checks": checks, "fields": summaries}, flush=True)


def verify(out):
    """Read-only independent sign, integral and primitive-logit reconstruction."""
    import csv

    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    fields, shared, _sectors, checks = observations(out)
    beta, common_beta, parts, summaries = derive(fields, shared)
    with np.load(out / "02/observations.npz", allow_pickle=False) as z:
        for name, value in (
            ("fields", fields),
            ("shared", shared),
            ("beta", beta),
            ("common_beta", common_beta),
            ("parts", parts),
        ):
            if not np.array_equal(z[name], value):
                raise ValueError("Saved finite field changed: " + name)
    error = float(np.max(abs(parts.sum(axis=1) - beta[:, 3])))
    for bank, mode, family in itertools.product(range(2), range(4), range(6)):
        cube = fields[bank, mode, family]
        error = max(
            error,
            float(
                np.max(abs(M.independent_characters(cube) - beta[bank, mode, family]))
            ),
        )
        direct = np.stack(
            [
                sum(
                    0.25 * b * o * cube[(b + 1) // 2, (o + 1) // 2, p]
                    for b, o in itertools.product((-1, 1), repeat=2)
                )
                for p in range(2)
            ]
        )
        error = max(
            error,
            float(np.max(abs(direct - M.author_difference(beta[bank, mode, family])))),
        )
    for d in H.load(out / "00/design.json"):
        raw = R.committed(out, d["batch"])
        logits = raw["logits"]
        shifted = logits - logits.max(axis=1)[:, None]
        weights = np.exp(shifted)
        probabilities = weights / weights.sum(axis=1)[:, None]
        units = np.stack(
            (
                probabilities @ np.arange(10) / 9,
                np.log(weights[:, 5:].sum(axis=1) / weights[:, :5].sum(axis=1)),
                np.exp(
                    logits.max(axis=1) + np.log(weights.sum(axis=1)) - raw["normalizer"]
                ),
                -np.sum(probabilities * np.log(probabilities), axis=1),
            ),
            axis=1,
        )
        error = max(error, float(np.max(abs(units - raw["readings"]))))
    if error > 1e-10:
        raise ValueError("Independent numerical reconstruction failed")
    saved = H.load(out / "02/summary.json")
    if saved["checks"] != checks or saved["fields"] != summaries:
        raise ValueError("Summary differs from authenticated observations")
    counts = {}
    for name, records in tables(beta, common_beta, parts).items():
        with (out / f"02/{name}.csv").open() as stream:
            actual = list(csv.DictReader(stream))
        if actual != [{n: str(v) for n, v in row.items()} for row in records]:
            raise ValueError("Complete signed result table changed: " + name)
        counts[name] = len(records)
    print(
        "VERIFIED HM133",
        {"independent_error": error, "checks": checks, "table_rows": counts},
        flush=True,
    )


def main():
    """Separate stages; never silently resume uncertain or terminal science."""
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
