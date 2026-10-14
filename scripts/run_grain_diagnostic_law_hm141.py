#!/usr/bin/env python3
"""Test fixed-event diagnostic-law reversal, never improve teacher agreement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_diagnostic_law as M
from scripts import run_grain_causal_valuation_hm134 as PREV

H, A = PREV.H, PREV.A
SPEC = ROOT / "configs/grain_diagnostic_law_hm141.json"
CODE = tuple(
    dict.fromkeys(
        (
            *PREV.CODE,
            "configs/grain_diagnostic_law_hm141.json",
            "docs/GRAIN_DIAGNOSTIC_LAW_REVERSAL_PLAN_HM141.md",
            "grain_research/construction/math_diagnostic_law.py",
            "scripts/run_grain_diagnostic_law_hm141.py",
            "tests/test_grain_diagnostic_law.py",
        )
    )
)
MODES = ("capture", "k", "v", "kv")
QUERIES = ("A_named", "A_category", "B_named", "A_chose_R", "B_chose_any")
UNITS = ("score", "logodds", "digit_mass", "entropy")


def validate_rows(rows):
    """Reject incomplete cells or reference edits that change the actual event."""
    by_key = {(r["bank"], r["family"], r["b"], r["c"]): r for r in rows}
    want = set(itertools.product(range(2), range(6), (-1, 1), (-1, 1)))
    if len(rows) != 48 or set(by_key) != want:
        raise ValueError("All48 new histories required, without duplicates")
    common = None
    for row in rows:
        groups = A.partition(row)
        fixed = [
            (groups[n], [row["input_ids"][i] for i in groups[n]])
            for n in ("S", "B", "C")
        ]
        fixed += [row["reference_slot"], row["history"]]
        if common is not None and fixed != common:
            raise ValueError("Reference address or non-history support changed")
        common = fixed
        if set(row["reference"]) & set(row["padding"]):
            raise ValueError("Every reference token must be visible")
        if not set(row["reference"]) <= set(row["reference_slot"]):
            raise ValueError("Reference tokens must occupy the registered slot")
    for bank, family, b in itertools.product(range(2), range(6), (-1, 1)):
        lo, hi = (by_key[bank, family, b, c] for c in (-1, 1))
        if (
            lo["action_text"] != hi["action_text"]
            or lo["padding"] != hi["padding"]
            or lo["reference"] != hi["reference"]
        ):
            raise ValueError("Actual action or support changed across reference laws")
        changed = {
            i
            for i, (x, y) in enumerate(
                zip(lo["input_ids"], hi["input_ids"], strict=True)
            )
            if x != y
        }
        if not changed or not changed <= set(lo["reference"]):
            raise ValueError("Only registered reference tokens may change")
        if Counter(lo["input_ids"]) != Counter(hi["input_ids"]):
            raise ValueError("Reference reversal must preserve the input token bag")
    return by_key


def fixed_design(rows):
    """Capture the complete grid, then own controls and all three reference swaps."""
    by_key, design = validate_rows(rows), []

    def add(row, mode, donor):
        design.append(
            {"batch": len(design), "row": row["id"], "mode": mode, "donor": donor["id"]}
        )

    for row in rows:
        add(row, "capture", row)
    for b, c in itertools.product((-1, 1), repeat=2):
        row = by_key[0, 0, b, c]
        add(row, "noop", row)
    for row in rows:
        for mode in MODES[1:]:
            add(row, mode, by_key[row["bank"], row["family"], row["b"], -row["c"]])
    if len(design) != 196:
        raise ValueError("Registered batch support changed")
    return design


def questions(trait):
    """Specify fixed named/category/fact questions, not an optimized prompt grid."""
    prefix = "Using only the stated reference population and observed choice, "
    ending = " Use 0 for probability zero and 9 for probability one. Return only one digit from 0 to 9."
    return [
        prefix + f"what is the probability that Person A has high {trait}?" + ending,
        prefix
        + "what is the probability that Person A belongs to category H?"
        + ending,
        prefix + f"what is the probability that Person B has high {trait}?" + ending,
        "Does the record say that Person A chose option R rather than S?" + ending,
        "Does the record say that Person B personally chose either option?" + ending,
    ]


def check(out):
    """Verify frozen inputs and all inherited/current source code hashes."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Register all new public texts and token supports before any model execution."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if out.exists():
        raise FileExistsError("Do not overwrite previous or failed experiments")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    old = PREV.check(source)
    H.verify_hashes(source, H.load(source / "02/output_seal.json"))
    base = H.load(source / "00/rows.json")[0]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    candidates = []
    for bank, family, b, c in itertools.product(range(2), range(6), (-1, 1), (-1, 1)):
        definition = spec["banks"][bank]
        low, high = definition["behaviors"][family]
        text = definition["menu"].format(low=low, high=high) + " "
        text += definition["choice"].format(option="R" if b == 1 else "S")
        counts = spec["counts"]["positive" if c == 1 else "negative"]
        reference = "\n" + definition["reference"].format(
            trait=spec["targets"][family],
            hr=counts[0][0],
            hs=counts[0][1],
            lr=counts[1][0],
            ls=counts[1][1],
        )
        candidates.append(
            {
                "bank": bank,
                "family": family,
                "b": b,
                "c": c,
                "action_text": text,
                "reference_text": reference,
                "action_ids": tokenizer.encode(text, add_special_tokens=False),
                "reference_ids": tokenizer.encode(reference, add_special_tokens=False),
            }
        )
    action_width = max(len(row["action_ids"]) for row in candidates)
    reference_width = max(len(row["reference_ids"]) for row in candidates)
    width = action_width + reference_width
    start, end = min(base["history"]), max(base["history"]) + 1
    prefix, tail = base["input_ids"][:start], base["input_ids"][end:]
    shift = width - len(base["history"])
    history = list(range(start, start + width))
    rows = []
    for row in candidates:
        action, reference = row.pop("action_ids"), row.pop("reference_ids")
        h = [tokenizer.pad_token_id] * width
        h[: len(action)] = action
        h[action_width : action_width + len(reference)] = reference
        rows.append(
            {
                **row,
                "id": len(rows),
                "history": history,
                "current": [i + shift for i in base["current"]],
                "input_ids": prefix + h + tail,
                "action": history[: len(action)],
                "reference": history[action_width : action_width + len(reference)],
                "reference_slot": history[action_width:],
                "padding": history[len(action) : action_width]
                + history[action_width + len(reference) :],
            }
        )
    query_sets = []
    alphabet = H.load(source / "00/queries.json")["alphabet"]
    for trait in spec["targets"]:
        q = questions(trait)
        query_sets.append(
            {
                "questions": q,
                "suffixes": [
                    tokenizer.encode(
                        "\n\nQuestion about the named person:\n"
                        + question
                        + "\nAnswer:<|im_end|>\n<|im_start|>assistant\n",
                        add_special_tokens=False,
                    )
                    for question in q
                ],
                "alphabet": alphabet,
            }
        )
    for row in rows:
        if (
            len(row["input_ids"]) + max(map(len, query_sets[row["family"]]["suffixes"]))
            > 512
        ):
            raise ValueError("No truncation of a native evidence/question record")
    design = fixed_design(rows)
    if len(design) != spec["max_batches"]:
        raise ValueError("Frozen cap changed")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in (("rows", rows), ("queries", query_sets), ("design", design)):
        H.write(out / f"00/{name}.json", value)
    files = [
        source / p
        for p in (
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
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in files},
            "inputs": H.seals(out, out / "00"),
            "slot_widths": [action_width, reference_width],
            "native_route": 7,
            "new_public_histories": 48,
            "new_predictor_fits": 0,
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
    print(
        "PREPARED HM141",
        len(design),
        "batches; slots",
        action_width,
        reference_width,
        flush=True,
    )


def committed(out, entry):
    """Bind every primitive to its exact design address and receipt."""
    stem = out / f"01/units/b{entry['batch']:03d}"
    receipt = H.load(stem.with_suffix(".json"))
    if (
        any(receipt[key] != value for key, value in entry.items())
        or receipt["endpoints"] != 5
    ):
        raise ValueError("Wrong experimental batch receipt")
    if H.sha(stem.with_suffix(".npz")) != receipt["sha256"]:
        raise ValueError("Primitive hash differs from receipt")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def collect(out):
    """Run once under the original cumulative budget; retain uncertain forwards."""
    from grain_research.vector_mechanisms.navigation import MODEL
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No automatic replay of terminal or uncertain model work")
    H.verify_hashes(MODEL, manifest["model"])
    rows = {row["id"]: row for row in H.load(out / "00/rows.json")}
    design, queries = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    captures, tolerance = {}, manifest["spec"]["tolerance"]
    try:
        with collector_guard(out):
            start = time.monotonic()
            try:
                observer = load_observer()
            finally:
                status["model_seconds"] += time.monotonic() - start
                H.write(out / "status.json", status)
            H.write(out / "model_manifest.json", observer.manifest())
            if len(observer.model.model.layers) != manifest["model_blocks"]:
                raise ValueError("Pinned model depth changed")
            for entry in design:
                if status["model_seconds"] >= manifest["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Original cumulative native budget reached")
                row, donors = rows[entry["row"]], None
                donor_hashes = {}
                if entry["mode"] != "capture":
                    base, donor = (
                        committed(out, captures[entry[key]]) for key in ("row", "donor")
                    )
                    indices = [row["history"].index(i) for i in row["reference"]]
                    donors = M.install_reference(
                        base["history_key"],
                        base["history_value"],
                        donor["history_key"],
                        donor["history_value"],
                        indices,
                        entry["mode"],
                    )
                    donor_hashes = {
                        key: H.sha(
                            out / f"01/units/b{captures[entry[key]]['batch']:03d}.npz"
                        )
                        for key in ("row", "donor")
                    }
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=entry
                )
                H.write(out / "status.json", status)
                start = time.monotonic()
                try:
                    arrays = M.native_batch(
                        observer, row, queries[row["family"]], donors=donors
                    )
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    H.write(out / "status.json", status)
                if float(arrays["cache_spread"]) > tolerance:
                    raise ValueError("Causal body depends on the future question")
                if entry["mode"] == "capture":
                    captures[row["id"]] = entry
                else:
                    for name in ("body_final", "k_raw", "v"):
                        if np.max(abs(arrays[name] - base[name])) > tolerance:
                            raise ValueError("Suffix operation changed prefix: " + name)
                    if (
                        entry["mode"] == "noop"
                        and np.max(abs(arrays["readings"] - base["readings"]))
                        > tolerance
                    ):
                        raise ValueError("Own-reference interchange changed readout")
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
                H.write(out / "status.json", status)
                if status["completed"] % 12 == 0 or entry["mode"] == "noop":
                    print("HM141", status, flush=True)
            if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Original budget crossed")
            status["state"] = "COLLECTED"
            H.write(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        H.write(out / "status.json", status)
        raise
    print("COLLECTED HM141", status, flush=True)


def observe(out):
    """Reconstruct every cell and check primitive scope without changing results."""
    manifest, status = check(out), H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 196
        or status["pending"] is not None
    ):
        raise ValueError("An incomplete packet has no completed scientific field")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = {row["id"]: row for row in H.load(out / "00/rows.json")}
    pairs = validate_rows(list(rows.values()))
    design, queries = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    cube = np.full((2, 6, 2, 2, 4, 5, 4), np.nan)
    captures, filled = {}, set()
    checks = dict.fromkeys(
        (
            "cache_spread",
            "stock_support",
            "body_final",
            "current_key",
            "current_value",
            "own_reference",
            "pre_reference_keys",
            "pre_reference_values",
            "readout_reconstruction",
        ),
        0.0,
    )
    for entry in design:
        row, arrays = rows[entry["row"]], committed(out, entry)
        q = queries[row["family"]]
        checks["cache_spread"] = max(
            checks["cache_spread"], float(arrays["cache_spread"])
        )
        if not np.array_equal(arrays["blocks"], np.arange(28)) or not np.array_equal(
            arrays["interchange_blocks"], np.arange(28)
        ):
            raise ValueError("All28 native blocks required")
        width = int(arrays["execution_length"])
        # These masks are the actual supports recorded by the native runner.
        desired = np.stack(
            [A.permissions(row, len(suffix), width, 7) for suffix in q["suffixes"]]
        )
        hashes = np.stack(
            [
                np.frombuffer(hashlib.sha256(s.tobytes()).digest(), dtype=np.uint8)
                for s in desired
            ]
        )
        if not np.array_equal(hashes, arrays["support_hash"]) or not np.array_equal(
            arrays["endpoints"],
            [len(row["input_ids"]) + len(s) - 1 for s in q["suffixes"]],
        ):
            raise ValueError(
                "Actual native attention/endpoints differ from frozen support"
            )
        z, norm = arrays["logits"], arrays["normalizer"]
        probabilities = softmax(z, axis=1)
        alternate = np.stack(
            (
                probabilities @ np.linspace(0, 1, 10),
                logsumexp(z[:, 5:], axis=1) - logsumexp(z[:, :5], axis=1),
                np.exp(logsumexp(z, axis=1) - norm),
                -(probabilities * np.log(probabilities)).sum(axis=1),
            ),
            axis=1,
        )
        checks["readout_reconstruction"] = max(
            checks["readout_reconstruction"],
            float(np.max(abs(alternate - arrays["readings"]))),
        )
        address = (row["bank"], row["family"], (row["b"] + 1) // 2, (row["c"] + 1) // 2)
        if entry["mode"] == "capture":
            captures[row["id"]] = arrays
        else:
            base = captures[row["id"]]
            for name, key in (
                ("body_final", "body_final"),
                ("k_raw", "current_key"),
                ("v", "current_value"),
            ):
                checks[key] = max(
                    checks[key], float(np.max(abs(arrays[name] - base[name])))
                )
            for key in ("row", "donor"):
                source_entry = next(d for d in design[:48] if d["row"] == entry[key])
                receipt = H.load(out / f"01/units/b{entry['batch']:03d}.json")
                if receipt["donor_hashes"][key] != H.sha(
                    out / f"01/units/b{source_entry['batch']:03d}.npz"
                ):
                    raise ValueError("Intervention used an unauthorized donor")
            if entry["mode"] == "noop":
                checks["own_reference"] = max(
                    checks["own_reference"],
                    float(np.max(abs(arrays["readings"] - base["readings"]))),
                )
                continue
        cell = (*address, MODES.index(entry["mode"]))
        if cell in filled:
            raise ValueError("Duplicate scientific field")
        filled.add(cell)
        cube[cell] = arrays["readings"]
    if not np.isfinite(cube).all() or len(filled) != 192:
        raise ValueError("Missing fixed experimental conditions")
    for bank, family, b in itertools.product(range(2), range(6), (-1, 1)):
        lo, hi = (pairs[bank, family, b, c] for c in (-1, 1))
        count = lo["history"].index(min(lo["reference"]))
        for name, key in (
            ("history_key", "pre_reference_keys"),
            ("history_value", "pre_reference_values"),
        ):
            diff = (
                captures[lo["id"]][name][:, :, :count]
                - captures[hi["id"]][name][:, :, :count]
            )
            checks[key] = max(checks[key], float(np.max(abs(diff))))
    if max(checks.values()) > manifest["spec"]["tolerance"]:
        raise ValueError("Numerical scope checks failed: " + str(checks))
    return cube, checks, status, manifest


def outputs(cube, manifest):
    """Compute all mechanism contrasts, without choosing a favorable metric."""
    coefficients, accounts, summary = [], [], {}
    for bank, family, mode, unit in itertools.product(
        range(2), range(6), range(4), range(4)
    ):
        corners = cube[bank, family, :, :, mode, :, unit]
        beta = M.characters(corners)
        if not np.allclose(
            beta,
            np.stack(
                [
                    (
                        sum(
                            (2 * b - 1) ** bool(mask & 1)
                            * (2 * c - 1) ** bool(mask & 2)
                            * corners[b, c]
                            for b, c in itertools.product(range(2), repeat=2)
                        )
                        / 4
                    )
                    for mask in range(4)
                ]
            ),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Independent character enumeration differs")
        difference = M.author_contrasts(corners)
        for query in range(5):
            coefficients.append(
                {
                    "bank": bank,
                    "target": manifest["spec"]["targets"][family],
                    "mode": MODES[mode],
                    "unit": UNITS[unit],
                    "query": QUERIES[query],
                    **{
                        name: float(beta[i, query])
                        for i, name in enumerate(
                            ("constant", "behavior", "reference", "behavior_reference")
                        )
                    },
                    "negative_law_slope": float(
                        corners[1, 0, query] - corners[0, 0, query]
                    ),
                    "positive_law_slope": float(
                        corners[1, 1, query] - corners[0, 1, query]
                    ),
                }
            )
            for c in range(2):
                accounts.append(
                    {
                        "bank": bank,
                        "target": manifest["spec"]["targets"][family],
                        "mode": MODES[mode],
                        "unit": UNITS[unit],
                        "query": QUERIES[query],
                        "reference_sign": 2 * c - 1,
                        "author_difference": float(difference[c, query]),
                    }
                )
    for bank, unit in itertools.product(range(2), range(4)):
        natural = cube[bank, :, :, :, 0, :, unit]
        slopes = natural[:, 1] - natural[:, 0]
        law_difference = natural[:, :, 1] - natural[:, :, 0]
        key = f"bank{bank}/{UNITS[unit]}"
        summary[key] = {
            "queries": {
                name: {
                    "negative_law_slope_mean": float(slopes[:, 0, q].mean()),
                    "positive_law_slope_mean": float(slopes[:, 1, q].mean()),
                    "reversals_in_predicted_direction": int(
                        ((slopes[:, 0, q] < 0) & (slopes[:, 1, q] > 0)).sum()
                    ),
                    "behavior_reference_RMS": float(
                        np.sqrt(np.mean((slopes[:, 1, q] - slopes[:, 0, q]) ** 2))
                    ),
                    "reference_effect_max": float(np.max(abs(law_difference[:, :, q]))),
                }
                for q, name in enumerate(QUERIES)
            },
            "named_category_difference_RMS": float(
                np.sqrt(np.mean((natural[..., 0] - natural[..., 1]) ** 2))
            ),
            "named_category_interaction_difference_RMS": float(
                np.sqrt(
                    np.mean(
                        (
                            (slopes[:, 1, 0] - slopes[:, 0, 0])
                            - (slopes[:, 1, 1] - slopes[:, 0, 1])
                        )
                        ** 2
                    )
                )
            ),
        }
    effects = np.stack(
        [
            cube[:, :, :, ::-1, 0] - cube[:, :, :, :, 0],
            cube[:, :, :, :, 1] - cube[:, :, :, :, 0],
            cube[:, :, :, :, 2] - cube[:, :, :, :, 0],
            cube[:, :, :, :, 3] - cube[:, :, :, :, 0],
        ],
        axis=4,
    )
    parts = M.intervention_parts(np.moveaxis(cube, 4, 0))
    if not np.allclose(parts.sum(axis=0), effects[:, :, :, :, 3], atol=1e-12, rtol=0):
        raise ValueError("K/V effect reconstruction differs")
    action_table = []
    for bank, query, unit in itertools.product(range(2), range(5), range(4)):
        full = effects[bank, :, :, :, 0, query, unit].reshape(-1)
        denominator = float(full @ full)
        for mode in range(1, 4):
            intervention = effects[bank, :, :, :, mode, query, unit].reshape(-1)
            norm = float(np.linalg.norm(intervention) * np.linalg.norm(full))
            action_table.append(
                {
                    "bank": bank,
                    "query": QUERIES[query],
                    "unit": UNITS[unit],
                    "mode": MODES[mode],
                    "effect_RMS": float(np.sqrt(np.mean(intervention**2))),
                    "natural_reference_RMS": float(np.sqrt(np.mean(full**2))),
                    "signed_projection": None
                    if denominator == 0
                    else float(intervention @ full / denominator),
                    "cosine": None if norm == 0 else float(intervention @ full / norm),
                    "error_over_natural_RMS": None
                    if denominator == 0
                    else float(
                        np.linalg.norm(intervention - full) / np.linalg.norm(full)
                    ),
                }
            )
    return coefficients, accounts, action_table, summary, effects, parts


def table(path, rows):
    """Write all declared rows once, never only a winning subset."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    """Save the complete fields and normative mechanism comparison once."""
    cube, checks, status, manifest = observe(out)
    coefficients, accounts, actions, summary, effects, parts = outputs(cube, manifest)
    destination = out / "02"
    destination.mkdir()
    table(destination / "characters.csv", coefficients)
    table(destination / "author_contrasts.csv", accounts)
    table(destination / "interventions.csv", actions)
    counts = manifest["spec"]["counts"]
    normative = M.law_cube(counts["positive"], counts["negative"])
    with (destination / "fields.npz").open("xb") as stream:
        np.savez_compressed(
            stream, cube=cube, effects=effects, parts=parts, normative=normative
        )
    result = {
        "status": "FINITE_DIAGNOSTIC_LAW_EXPERIMENT_COMPLETE_NOT_GLOBAL_CLOSURE",
        "checks": checks,
        "fields": summary,
        "normative_posterior": normative.tolist(),
        "completed_batches": status["completed"],
        "actual_model_seconds": status["model_seconds"],
        "new_public_histories": 48,
        "endpoints": 980,
        "new_predictor_fits": 0,
        "human_participants": 0,
        "original_scope_complete": False,
    }
    H.write(destination / "summary.json", result)
    H.write(destination / "output_seal.json", H.seals(out, destination))
    print("ANALYZED HM141", summary["bank0/score"], summary["bank1/score"], flush=True)


def verify(out):
    """Read-only replay of every primitive, field and complete exported table."""
    import io

    cube, checks, _status, manifest = observe(out)
    data = outputs(cube, manifest)
    saved = H.load(out / "02/summary.json")
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    if saved["checks"] != checks or saved["fields"] != data[3]:
        raise ValueError("Saved numerical mechanism summary differs")
    for name, rows in zip(
        ("characters", "author_contrasts", "interventions"), data[:3], strict=True
    ):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        with (out / f"02/{name}.csv").open(newline="") as saved_table:
            if stream.getvalue() != saved_table.read():
                raise ValueError("Complete output table differs: " + name)
    with np.load(out / "02/fields.npz", allow_pickle=False) as archive:
        for name, value in (("cube", cube), ("effects", data[4]), ("parts", data[5])):
            if not np.array_equal(archive[name], value):
                raise ValueError("Saved full field differs: " + name)
    print("VERIFIED HM141", checks, "980 endpoints; all tables", flush=True)


def main():
    """Expose explicit phases without automatically replaying an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--out", type=Path, default=ROOT / H.load(SPEC)["out"])
    args = parser.parse_args()
    globals()[args.phase](args.out.resolve())


if __name__ == "__main__":
    main()
