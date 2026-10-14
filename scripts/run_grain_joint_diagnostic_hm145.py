#!/usr/bin/env python3
"""Probe joint diagnostic operation once, not improve personality accuracy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_diagnostic_law as N
from grain_research.construction import math_joint_diagnostic as M
from scripts import run_grain_diagnostic_law_hm141 as PREV

H, A = PREV.H, PREV.A
SPEC = ROOT / "configs/grain_joint_diagnostic_hm145.json"
CODE = tuple(
    dict.fromkeys(
        (
            *PREV.CODE,
            "configs/grain_joint_diagnostic_hm145.json",
            "docs/GRAIN_JOINT_DIAGNOSTIC_PLAN_HM145.md",
            "grain_research/construction/math_joint_diagnostic.py",
            "scripts/run_grain_joint_diagnostic_hm145.py",
            "tests/test_grain_joint_diagnostic.py",
        )
    )
)
MODES = ("capture", "k", "v", "kv")
QUERIES = ("A_named", "A_category", "B_category", "A_first_R", "B_any_choice")
UNITS = ("score", "logodds", "digit_mass", "entropy")


def status_write(path, value):
    """Atomically replace mutable runtime status, never scientific primitives."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def validate_rows(rows):
    """Require all32 new cells and reference-only token-bag-preserving edits."""
    by_key = {(row["bank"], *row["b"], row["c"]): row for row in rows}
    want = set(itertools.product(range(2), (-1, 1), (-1, 1), (-1, 1), (-1, 1)))
    if len(rows) != 32 or set(by_key) != want:
        raise ValueError("Complete, nonduplicated joint support required")
    fixed = None
    for row in rows:
        groups = A.partition(row)
        common = [
            (groups[name], [row["input_ids"][i] for i in groups[name]])
            for name in ("S", "B", "C")
        ] + [row["history"], row["reference_slot"]]
        if fixed is not None and common != fixed:
            raise ValueError("Fixed system/relay/current/reference coordinates changed")
        fixed = common
        if not set(row["reference"]) <= set(row["reference_slot"]) or set(
            row["reference"]
        ) & set(row["padding"]):
            raise ValueError("Reference scope must be visible within its slot")
    for bank, *b in itertools.product(range(2), (-1, 1), (-1, 1), (-1, 1)):
        lo, hi = (by_key[bank, *b, c] for c in (-1, 1))
        if any(lo[key] != hi[key] for key in ("action_text", "padding", "reference")):
            raise ValueError("Event or support changed across reference tables")
        changed = {
            i
            for i, (x, y) in enumerate(
                zip(lo["input_ids"], hi["input_ids"], strict=True)
            )
            if x != y
        }
        if not changed or not changed <= set(lo["reference"]):
            raise ValueError("Only reference count tokens may change")
        if Counter(lo["input_ids"]) != Counter(hi["input_ids"]):
            raise ValueError("Reference edit changed the complete token bag")
    return by_key


def fixed_design(rows):
    """Register complete captures, own-reference controls and K/V/KV swaps."""
    by_key = validate_rows(rows)
    design = []

    def add(row, mode, donor):
        design.append(
            {"batch": len(design), "row": row["id"], "mode": mode, "donor": donor["id"]}
        )

    for row in rows:
        add(row, "capture", row)
    for b, c in itertools.product(((-1, -1, -1), (1, 1, 1)), (-1, 1)):
        row = by_key[0, *b, c]
        add(row, "noop", row)
    for row in rows:
        for mode in MODES[1:]:
            add(row, mode, by_key[row["bank"], *row["b"], -row["c"]])
    if len(design) != 132:
        raise ValueError("Finite intervention support changed")
    return design


def questions():
    """Fix named, anonymous, nonactor and unchanged-fact channels in advance."""
    ending = " Use 0 for probability zero and 9 for probability one. Return only one digit from 0 to 9."
    return [
        "Using the population table and all three observed choices, what is the probability that Person A has high extraversion?"
        + ending,
        "Using the population table and all three observed choices, what is the probability that Person A belongs to H?"
        + ending,
        "Using the population table and the information about B, what is the probability that Person B belongs to H?"
        + ending,
        "Does the record say Person A chose R on the first occasion?" + ending,
        "Does the record contain any observed choice personally made by Person B?"
        + ending,
    ]


def check(out):
    """Authenticate the code ancestry, reference artifacts and prepared inputs."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Seal all new texts, counts and supports before model calls or outcomes."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if out.exists():
        raise FileExistsError("Never overwrite a previous packet")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    old = PREV.check(source)
    H.verify_hashes(source, H.load(source / "02/output_seal.json"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    prefix = tokenizer.encode(
        "<|im_start|>system\nRead the fictional record. Answer only the requested probability, without adding observations.<|im_end|>\n<|im_start|>user\n",
        add_special_tokens=False,
    )
    relay = tokenizer.encode("\nEnd of historical record.\n", add_special_tokens=False)
    current = tokenizer.encode(
        "No additional events or category observations are available.",
        add_special_tokens=False,
    )
    candidates = []
    for bank, b, c in itertools.product(range(2), list(M.POINTS), (-1, 1)):
        definition = spec["banks"][bank]
        counts = M.population_counts(c)
        table = "\n".join(
            "".join("R" if i == 1 else "S" for i in point)
            + f": H {int(counts[0, j])}, L {int(counts[1, j])}."
            for j, point in enumerate(M.POINTS)
        )
        letters = ["R" if i == 1 else "S" for i in b]
        action = definition["action"].format(x=letters[0], y=letters[1], z=letters[2])
        reference = "\n" + definition["reference"].format(table=table)
        candidates.append(
            {
                "id": len(candidates),
                "bank": bank,
                "b": b.tolist(),
                "c": c,
                "action_text": action,
                "reference_text": reference,
                "action_ids": tokenizer.encode(action, add_special_tokens=False),
                "reference_ids": tokenizer.encode(reference, add_special_tokens=False),
            }
        )
    widths = [
        max(len(row[name]) for row in candidates)
        for name in ("action_ids", "reference_ids")
    ]
    history = list(range(len(prefix), len(prefix) + sum(widths)))
    current_indices = list(
        range(history[-1] + 1 + len(relay), history[-1] + 1 + len(relay) + len(current))
    )
    rows = []
    for row in candidates:
        action, reference = row.pop("action_ids"), row.pop("reference_ids")
        h = [tokenizer.pad_token_id] * sum(widths)
        h[: len(action)] = action
        h[widths[0] : widths[0] + len(reference)] = reference
        rows.append(
            {
                **row,
                "history": history,
                "current": current_indices,
                "input_ids": prefix + h + relay + current,
                "action": history[: len(action)],
                "reference": history[widths[0] : widths[0] + len(reference)],
                "reference_slot": history[widths[0] :],
                "padding": history[len(action) : widths[0]]
                + history[widths[0] + len(reference) :],
            }
        )
    query_text = questions()
    queries = {
        "questions": query_text,
        "suffixes": [
            tokenizer.encode(
                "\n\nQuestion:\n" + q + "\nAnswer:<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )
            for q in query_text
        ],
        "alphabet": H.load(source / "00/queries.json")[0]["alphabet"],
    }
    maximum = len(rows[0]["input_ids"]) + max(map(len, queries["suffixes"]))
    if maximum > spec["max_tokens"]:
        raise ValueError(f"No truncation allowed: actual token width {maximum}")
    design = fixed_design(rows)
    if len(design) != spec["max_batches"]:
        raise ValueError("Frozen batch cap mismatch")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in (("rows", rows), ("queries", queries), ("design", design)):
        H.write(out / f"00/{name}.json", value)
    files = [
        source / p
        for p in ("00/manifest.json", "00/queries.json", "02/output_seal.json")
    ]
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "model": old["model"],
            "model_blocks": old["model_blocks"],
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in files},
            "inputs": H.seals(out, out / "00"),
            "slot_widths": widths,
            "maximum_tokens": maximum,
            "new_public_histories": 32,
            "native_route": 7,
            "new_predictor_fits": 0,
            "original_scope_complete": False,
        },
    )
    status_write(
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
        "PREPARED HM145",
        len(design),
        "batches; max tokens",
        maximum,
        "slots",
        widths,
        flush=True,
    )


def committed(out, entry):
    """Read a primitive only when its receipt binds the requested operation."""
    stem = out / f"01/units/b{entry['batch']:03d}"
    receipt = H.load(stem.with_suffix(".json"))
    if any(receipt[k] != v for k, v in entry.items()) or receipt["endpoints"] != 5:
        raise ValueError("Primitive receipt address mismatch")
    if receipt["sha256"] != H.sha(stem.with_suffix(".npz")):
        raise ValueError("Primitive hash changed")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def collect(out):
    """Perform the finite sealed design once, with cumulative load/call charges."""
    from grain_research.vector_mechanisms.navigation import MODEL
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No replay of terminal or uncertain model work")
    H.verify_hashes(MODEL, manifest["model"])
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    design, queries = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    captures, tolerance = {}, manifest["spec"]["tolerance"]
    try:
        with collector_guard(out):
            start = time.monotonic()
            try:
                observer = load_observer()
            finally:
                status["model_seconds"] += time.monotonic() - start
                status_write(out / "status.json", status)
            H.write(out / "model_manifest.json", observer.manifest())
            if len(observer.model.model.layers) != manifest["model_blocks"]:
                raise ValueError("Pinned model depth changed")
            for entry in design:
                if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Original cumulative model budget reached")
                row, donors, donor_hashes = rows[entry["row"]], None, {}
                if entry["mode"] != "capture":
                    base, donor = (
                        committed(out, captures[entry[k]]) for k in ("row", "donor")
                    )
                    indices = [row["history"].index(i) for i in row["reference"]]
                    donors = N.install_reference(
                        base["history_key"],
                        base["history_value"],
                        donor["history_key"],
                        donor["history_value"],
                        indices,
                        entry["mode"],
                    )
                    donor_hashes = {
                        k: H.sha(
                            out / f"01/units/b{captures[entry[k]]['batch']:03d}.npz"
                        )
                        for k in ("row", "donor")
                    }
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=entry
                )
                status_write(out / "status.json", status)
                start = time.monotonic()
                try:
                    arrays = N.native_batch(observer, row, queries, donors=donors)
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    status_write(out / "status.json", status)
                if float(arrays["cache_spread"]) > tolerance:
                    raise ValueError(
                        "A future query changed causal historical carriers"
                    )
                if entry["mode"] == "capture":
                    captures[row["id"]] = entry
                else:
                    for name in ("body_final", "k_raw", "v"):
                        if np.max(abs(arrays[name] - base[name])) > tolerance:
                            raise ValueError(
                                "Suffix intervention changed prefix " + name
                            )
                    if (
                        entry["mode"] == "noop"
                        and np.max(abs(arrays["readings"] - base["readings"]))
                        > tolerance
                    ):
                        raise ValueError("Own-reference swap changed output")
                stem = out / f"01/units/b{entry['batch']:03d}"
                with stem.with_suffix(".tmp").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                stem.with_suffix(".tmp").replace(stem.with_suffix(".npz"))
                H.write(
                    stem.with_suffix(".json"),
                    {
                        **entry,
                        "seconds": seconds,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "endpoints": 5,
                        "donor_hashes": donor_hashes,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                status_write(out / "status.json", status)
                if status["completed"] % 8 == 0 or entry["mode"] == "noop":
                    print("HM145", status, flush=True)
            if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Original model budget crossed")
            status["state"] = "COLLECTED"
            status_write(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status_write(out / "status.json", status)
        raise
    print("COLLECTED HM145", status, flush=True)


def observe(out):
    """Reconstruct complete fields from actual primitive logits and supports."""
    manifest, status = check(out), H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 132
        or status["pending"] is not None
    ):
        raise ValueError("Incomplete collection cannot support a completed result")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    pairs = validate_rows(list(rows.values()))
    design, queries = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    cube = np.full((2, 2, 2, 2, 2, 4, 5, 4), np.nan)
    captures, filled = {}, set()
    guards = dict.fromkeys(
        (
            "cache_spread",
            "body_final",
            "current_key",
            "current_value",
            "own_reference",
            "pre_reference_key",
            "pre_reference_value",
            "readout_reconstruction",
        ),
        0.0,
    )
    for entry in design:
        row, arrays = rows[entry["row"]], committed(out, entry)
        if not np.array_equal(arrays["blocks"], np.arange(28)) or not np.array_equal(
            arrays["interchange_blocks"], np.arange(28)
        ):
            raise ValueError("All28 blocks required")
        width = int(arrays["execution_length"])
        desired = np.stack(
            [A.permissions(row, len(s), width, 7) for s in queries["suffixes"]]
        )
        hashes = np.stack(
            [
                np.frombuffer(hashlib.sha256(s.tobytes()).digest(), dtype=np.uint8)
                for s in desired
            ]
        )
        if not np.array_equal(hashes, arrays["support_hash"]) or not np.array_equal(
            arrays["endpoints"],
            [len(row["input_ids"]) + len(s) - 1 for s in queries["suffixes"]],
        ):
            raise ValueError("Actual mask/endpoint differs from frozen support")
        probabilities = softmax(arrays["logits"], axis=1)
        alternate = np.stack(
            (
                probabilities @ np.linspace(0, 1, 10),
                logsumexp(arrays["logits"][:, 5:], axis=1)
                - logsumexp(arrays["logits"][:, :5], axis=1),
                np.exp(logsumexp(arrays["logits"], axis=1) - arrays["normalizer"]),
                -(probabilities * np.log(probabilities)).sum(axis=1),
            ),
            axis=1,
        )
        guards["readout_reconstruction"] = max(
            guards["readout_reconstruction"],
            float(np.max(abs(alternate - arrays["readings"]))),
        )
        guards["cache_spread"] = max(
            guards["cache_spread"], float(arrays["cache_spread"])
        )
        address = (row["bank"], *((i + 1) // 2 for i in row["b"]), (row["c"] + 1) // 2)
        if entry["mode"] == "capture":
            captures[row["id"]] = arrays
        else:
            base = captures[row["id"]]
            for name, key in (
                ("body_final", "body_final"),
                ("k_raw", "current_key"),
                ("v", "current_value"),
            ):
                guards[key] = max(
                    guards[key], float(np.max(abs(arrays[name] - base[name])))
                )
            receipt = H.load(out / f"01/units/b{entry['batch']:03d}.json")
            for key in ("row", "donor"):
                capture_entry = next(d for d in design[:32] if d["row"] == entry[key])
                if receipt["donor_hashes"][key] != H.sha(
                    out / f"01/units/b{capture_entry['batch']:03d}.npz"
                ):
                    raise ValueError("Unauthorized intervention donor")
            if entry["mode"] == "noop":
                guards["own_reference"] = max(
                    guards["own_reference"],
                    float(np.max(abs(arrays["readings"] - base["readings"]))),
                )
                continue
        cell = (*address, MODES.index(entry["mode"]))
        if cell in filled:
            raise ValueError("Duplicate scientific field")
        filled.add(cell)
        cube[cell] = arrays["readings"]
    if not np.isfinite(cube).all() or len(filled) != 128:
        raise ValueError("Missing joint conditions")
    for bank, *b in itertools.product(range(2), (-1, 1), (-1, 1), (-1, 1)):
        lo, hi = (pairs[bank, *b, c] for c in (-1, 1))
        before = lo["history"].index(min(lo["reference"]))
        for name, key in (
            ("history_key", "pre_reference_key"),
            ("history_value", "pre_reference_value"),
        ):
            guards[key] = max(
                guards[key],
                float(
                    np.max(
                        abs(
                            captures[lo["id"]][name][:, :, :before]
                            - captures[hi["id"]][name][:, :, :before]
                        )
                    )
                ),
            )
    if max(guards.values()) > manifest["spec"]["tolerance"]:
        raise ValueError("Scope guards failed: " + str(guards))
    return cube, guards, status, manifest


def outputs(cube):
    """Keep full character fields, parity contrasts, controls and signed swaps."""
    coefficient_rows, summary, audit_error = [], [], 0.0
    basis = M.character_matrix()
    for bank, mode, unit in itertools.product(range(2), range(4), range(4)):
        field = cube[bank, :, :, :, :, mode, :, unit]
        beta = M.coefficients(field)
        direct = np.stack(
            [
                sum(basis[i, mask] * field.reshape(16, 5)[i] for i in range(16)) / 16
                for mask in range(16)
            ]
        )
        audit_error = max(
            audit_error,
            float(np.max(abs(beta - direct))),
            float(np.max(abs(M.reconstruct(beta) - field))),
        )
        for mask, q in itertools.product(range(16), range(5)):
            label = (
                "*".join(
                    name
                    for i, name in enumerate(("b1", "b2", "b3", "reference"))
                    if mask & (1 << i)
                )
                or "constant"
            )
            coefficient_rows.append(
                {
                    "bank": bank,
                    "mode": MODES[mode],
                    "unit": UNITS[unit],
                    "query": QUERIES[q],
                    "mask": mask,
                    "term": label,
                    "coefficient": float(beta[mask, q]),
                }
            )
    for bank, q, unit in itertools.product(range(2), range(5), range(4)):
        field = cube[bank, :, :, :, :, :, q, unit]
        native = field[..., 0]
        beta = M.coefficients(native)
        contrasts = M.parity_contrasts(native)
        natural = native[..., ::-1] - native
        swaps = {
            MODES[m]: M.effect_comparison(field[..., m] - native, natural)
            for m in range(1, 4)
        }
        for m in range(1, 4):
            swaps[MODES[m]]["joint_coefficient"] = float(
                M.coefficients(field[..., m])[15]
            )
        summary.append(
            {
                "bank": bank,
                "query": QUERIES[q],
                "unit": UNITS[unit],
                "native_joint_coefficient": float(beta[15]),
                "reference_main_coefficient": float(beta[8]),
                "other_nonconstant_coefficient_rms": float(
                    np.sqrt(np.mean(beta[1:15] ** 2))
                ),
                "parity_contrasts_negative_positive_reference": contrasts.tolist(),
                "predicted_direction_reversal": bool(contrasts[0] < 0 < contrasts[1]),
                "largest_reference_effect": float(np.max(abs(natural))),
                "swaps": swaps,
            }
        )
    return coefficient_rows, summary, audit_error


def law_audit():
    """Check tables by direct count enumeration, independently of native outputs."""
    errors, checks = [], 0
    low, high = (M.lower_marginals(c) for c in (-1, 1))
    for indices in low:
        errors.append(float(np.max(abs(low[indices] - high[indices]))))
        checks += low[indices].size
    for b, c in itertools.product(M.POINTS, (-1, 1)):
        counts = M.population_counts(c)
        for degree in range(4):
            for observed in itertools.combinations(range(3), degree):
                compatible = [
                    j
                    for j, point in enumerate(M.POINTS)
                    if all(point[i] == b[i] for i in observed)
                ]
                p = sum(int(counts[0, j]) for j in compatible) / sum(
                    int(counts[:, j].sum()) for j in compatible
                )
                wanted = 0.5 if degree < 3 else 0.5 + 0.3 * c * int(np.prod(b))
                errors.extend((abs(p - M.posterior(b, c, observed)), abs(p - wanted)))
                checks += 2
    field = np.asarray(
        [M.posterior(row[:3], row[3]) for row in M.FIELD_POINTS]
    ).reshape(2, 2, 2, 2)
    wanted_beta = np.zeros(16)
    wanted_beta[[0, 15]] = [0.5, 0.3]
    errors.append(float(np.max(abs(M.coefficients(field) - wanted_beta))))
    return {
        "checks": checks + 16,
        "maximum_error": max(errors),
        "full_posterior_levels": [0.2, 0.8],
        "every_lower_posterior": 0.5,
        "normative_joint_coefficient": 0.3,
    }


def analyze(out):
    """Save mechanism findings without fitting or selecting any scoring channel."""
    cube, guards, status, manifest = observe(out)
    coefficients, summary, error = outputs(cube)
    audit = law_audit()
    result = {
        "packet": "HM145",
        "status": "COMPLETE_FINITE_MECHANISM_NOT_THEORY_CLOSURE",
        "original_scope_complete": False,
        "new_public_histories": 32,
        "native_batches": 132,
        "native_endpoints": 660,
        "model_seconds": status["model_seconds"],
        "maximum_tokens": manifest["maximum_tokens"],
        "guards": guards,
        "alternate_character_error": error,
        "population_audit": audit,
        "all_channels": summary,
        "decision": "Inspect all named/category/nonactor/fact joint terms; do not promote generic table response to personality-specific mechanism or tune reader accuracy",
    }
    target = out / "02"
    target.mkdir()
    with (target / "fields.npz").open("xb") as stream:
        np.savez_compressed(stream, cube=cube)
    with (target / "coefficients.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coefficients[0]))
        writer.writeheader()
        writer.writerows(coefficients)
    H.write(target / "result.json", result)
    H.write(target / "output_seal.json", H.seals(out, target))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "packet",
                    "native_batches",
                    "native_endpoints",
                    "model_seconds",
                    "guards",
                    "population_audit",
                )
            },
            indent=2,
        )
    )
    for row in summary:
        if row["unit"] == "score":
            print(json.dumps(row))


def verify(out):
    """Replay sealed logits, fields, all CSV cells and exact count mathematics."""
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    cube, guards, status, _ = observe(out)
    coefficients, summary, error = outputs(cube)
    result = H.load(out / "02/result.json")
    if (
        summary != result["all_channels"]
        or guards != result["guards"]
        or law_audit() != result["population_audit"]
        or status["model_seconds"] != result["model_seconds"]
        or error != result["alternate_character_error"]
    ):
        raise ValueError("Saved mechanism result replay differs")
    with np.load(out / "02/fields.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["cube"], cube):
            raise ValueError("Saved field differs from actual primitives")
    with (out / "02/coefficients.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(coefficients) or any(
        any(row[k] != str(value) for k, value in wanted.items())
        for row, wanted in zip(saved, coefficients, strict=True)
    ):
        raise ValueError("Full coefficient table replay differs")
    if error > 1e-12 or result["population_audit"]["maximum_error"] > 1e-12:
        raise ValueError("Alternate mathematical check failed")
    print(
        "VERIFIED HM145",
        len(coefficients),
        "coefficients; max alternate error",
        error,
        flush=True,
    )


def main():
    """Run explicit bounded phases; terminal packets cannot silently rerun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / H.load(SPEC)["out"]
    globals()[args.phase](out)


if __name__ == "__main__":
    main()
