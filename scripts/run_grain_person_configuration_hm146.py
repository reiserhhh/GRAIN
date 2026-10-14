#!/usr/bin/env python3
"""Separate fixed-bag person facts from conditional joint target valuation."""

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
from grain_research.construction import math_person_configuration as M
from scripts import run_grain_joint_diagnostic_hm145 as PREV

H, A, N, J = PREV.H, PREV.A, PREV.N, PREV.M
SPEC = ROOT / "configs/grain_person_configuration_hm146.json"
CODE = tuple(
    dict.fromkeys(
        (
            *PREV.CODE,
            "configs/grain_person_configuration_hm146.json",
            "docs/GRAIN_PERSON_CONFIGURATION_PLAN_HM146.md",
            "grain_research/construction/math_person_configuration.py",
            "scripts/run_grain_person_configuration_hm146.py",
            "tests/test_grain_person_configuration.py",
        )
    )
)
MODES = ("capture", "action_k", "action_v", "action_kv", "reference_kv")
FAMILIES = ("named", "category", "first_R")
QUERIES = tuple(f"{person}_{family}" for family in FAMILIES for person in ("B", "A"))
UNITS = PREV.UNITS
status_write = PREV.status_write


def validate_rows(rows):
    """Certify the entire token multiset is fixed for every assignment/reference."""
    by_key = {(r["bank"], *r["b"], r["c"]): r for r in rows}
    want = set(itertools.product(range(2), (-1, 1), (-1, 1), (-1, 1), (-1, 1)))
    if len(rows) != 32 or set(by_key) != want:
        raise ValueError("Complete nonduplicated person/reference support required")
    common = None
    for row in rows:
        groups = A.partition(row)
        fixed = [
            (groups[n], [row["input_ids"][i] for i in groups[n]])
            for n in ("S", "B", "C")
        ] + [row["history"], row["reference_slot"]]
        if common is not None and fixed != common:
            raise ValueError("System/relay/current/history addresses changed")
        common = fixed
        if set(row["action"]) & set(row["reference"]) or not (
            set(row["action"]) | set(row["reference"]) | set(row["padding"])
        ) == set(row["history"]):
            raise ValueError("Historical scopes overlap or omit tokens")
    for bank in range(2):
        group = [r for r in rows if r["bank"] == bank]
        base = group[0]
        for row in group:
            for name in ("padding", "action", "reference", "event_spans"):
                if row[name] != base[name]:
                    raise ValueError("Person/reference edit changed positions: " + name)
            if Counter(row["input_ids"]) != Counter(base["input_ids"]):
                raise ValueError(
                    "Complete bag changed across assignments or references"
                )
            # Pairwise edits at fixed c change only the registered event rows.
            paired = by_key[bank, *row["b"], base["c"]]
            changed = {
                i
                for i, (x, y) in enumerate(
                    zip(base["input_ids"], paired["input_ids"], strict=True)
                )
                if x != y
            }
            if not changed <= set(itertools.chain.from_iterable(base["event_spans"])):
                raise ValueError("Assignment edit escaped the actual event sentences")
            donor = by_key[bank, *row["b"], -row["c"]]
            changed = {
                i
                for i, (x, y) in enumerate(
                    zip(row["input_ids"], donor["input_ids"], strict=True)
                )
                if x != y
            }
            if not changed or not changed <= set(row["reference"]):
                raise ValueError("Reference edit is not reference-only")
    return by_key


def fixed_design(rows):
    """Register all captures, own controls, three action swaps and reference KV."""
    indexed = validate_rows(rows)
    design = []

    def add(row, mode, donor):
        design.append(
            {"batch": len(design), "row": row["id"], "mode": mode, "donor": donor["id"]}
        )

    for row in rows:
        add(row, "capture", row)
    for b, c in itertools.product(((-1, -1, -1), (1, 1, 1)), (-1, 1)):
        row = indexed[0, *b, c]
        add(row, "noop", row)
    for row in rows:
        for mode in MODES[1:4]:
            add(row, mode, indexed[row["bank"], *(-x for x in row["b"]), row["c"]])
        add(row, "reference_kv", indexed[row["bank"], *row["b"], -row["c"]])
    if len(design) != 164:
        raise ValueError("Finite batch support changed")
    return design


def questions():
    """Fix both-person named/category/fact questions without output selection."""
    ending = " Use 0 for probability zero and 9 for probability one. Return only one digit from 0 to 9."
    result = []
    for family in FAMILIES:
        for person in ("B", "A"):
            if family == "named":
                text = f"Using the population table and the complete observed choice paths, what is the probability that Person {person} has high extraversion?"
            elif family == "category":
                text = f"Using the population table and the complete observed choice paths, what is the probability that Person {person} belongs to H?"
            else:
                text = f"Does the record say Person {person} chose R on the first occasion?"
            result.append(text + ending)
    return result


def check(out):
    """Authenticate the sealed code ancestry, source artifacts and new inputs."""
    manifest = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, manifest["code"] | manifest["source"])
    H.verify_hashes(out, manifest["inputs"])
    return manifest


def prepare(out):
    """Freeze all whole-bag-matched records and exact supports before forwards."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL

    if out.exists():
        raise FileExistsError("Never overwrite an earlier packet")
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
        action = definition["intro"]
        event_chars = []
        for i, sign in enumerate(b):
            start = len(action)
            action += definition["event"].format(
                n=i + 1, r="A" if sign == 1 else "B", s="B" if sign == 1 else "A"
            )
            event_chars.append((start, len(action)))
        action += definition["sampling"]
        encoded = tokenizer(
            action, add_special_tokens=False, return_offsets_mapping=True
        )
        spans = [
            [
                j
                for j, (a, z) in enumerate(encoded["offset_mapping"])
                if a < end and z > start
            ]
            for start, end in event_chars
        ]
        counts = J.population_counts(c)
        table = "\n".join(
            "".join("R" if i == 1 else "S" for i in point)
            + f": H {int(counts[0, j])}, L {int(counts[1, j])}."
            for j, point in enumerate(M.POINTS)
        )
        reference = "\n" + definition["reference"].format(table=table)
        candidates.append(
            {
                "id": len(candidates),
                "bank": bank,
                "b": b.tolist(),
                "c": c,
                "action_text": action,
                "reference_text": reference,
                "action_ids": encoded["input_ids"],
                "relative_event_spans": spans,
                "reference_ids": tokenizer.encode(reference, add_special_tokens=False),
            }
        )
    widths = [
        max(len(r[n]) for r in candidates) for n in ("action_ids", "reference_ids")
    ]
    history = list(range(len(prefix), len(prefix) + sum(widths)))
    current_indices = list(
        range(history[-1] + 1 + len(relay), history[-1] + 1 + len(relay) + len(current))
    )
    rows = []
    for row in candidates:
        action, reference = row.pop("action_ids"), row.pop("reference_ids")
        spans = row.pop("relative_event_spans")
        h = [tokenizer.pad_token_id] * sum(widths)
        h[: len(action)], h[widths[0] : widths[0] + len(reference)] = action, reference
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
                "event_spans": [[history[i] for i in span] for span in spans],
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
        "alphabet": H.load(source / "00/queries.json")["alphabet"],
    }
    maximum = len(rows[0]["input_ids"]) + max(map(len, queries["suffixes"]))
    if maximum > spec["max_tokens"]:
        raise ValueError(f"No truncation allowed: actual width {maximum}")
    design = fixed_design(rows)
    if len(design) != spec["max_batches"]:
        raise ValueError("Frozen batch cap mismatch")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in (("rows", rows), ("queries", queries), ("design", design)):
        H.write(out / f"00/{name}.json", value)
    sources = [
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
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
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
        "PREPARED HM146",
        len(design),
        "batches; tokens",
        maximum,
        "slots",
        widths,
        flush=True,
    )


def committed(out, entry):
    """Read only a primitive whose exclusive receipt authenticates this operation."""
    stem = out / f"01/units/b{entry['batch']:03d}"
    receipt = H.load(stem.with_suffix(".json"))
    if (
        any(receipt[k] != v for k, v in entry.items())
        or receipt["endpoints"] != 6
        or receipt["sha256"] != H.sha(stem.with_suffix(".npz"))
    ):
        raise ValueError("Primitive receipt mismatch")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        return {n: archive[n].copy() for n in archive.files}


def collect(out):
    """Run this finite frozen design once, without accuracy rescue or new fits."""
    from grain_research.vector_mechanisms.navigation import MODEL
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    manifest, status = check(out), H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"] is not None:
        raise RuntimeError("No replay of terminal or uncertain native work")
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
                    raise TimeoutError("Cumulative model cap reached")
                row, donors, hashes = rows[entry["row"]], None, {}
                if entry["mode"] != "capture":
                    base, donor = (
                        committed(out, captures[entry[k]]) for k in ("row", "donor")
                    )
                    scope = "reference" if entry["mode"] == "reference_kv" else "action"
                    mode = (
                        "kv"
                        if entry["mode"] == "reference_kv"
                        else entry["mode"].removeprefix("action_")
                    )
                    indices = [row["history"].index(i) for i in row[scope]]
                    donors = N.install_reference(
                        base["history_key"],
                        base["history_value"],
                        donor["history_key"],
                        donor["history_value"],
                        indices,
                        mode,
                    )
                    hashes = {
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
                    raise ValueError("Future queries altered causal historical states")
                if entry["mode"] == "capture":
                    captures[row["id"]] = entry
                else:
                    for name in ("body_final", "k_raw", "v"):
                        if np.max(abs(arrays[name] - base[name])) > tolerance:
                            raise ValueError("Suffix swap altered prefix " + name)
                    if (
                        entry["mode"] == "noop"
                        and np.max(abs(arrays["readings"] - base["readings"]))
                        > tolerance
                    ):
                        raise ValueError("Own-donor swap changed readings")
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
                        "endpoints": 6,
                        "donor_hashes": hashes,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                status_write(out / "status.json", status)
                if status["completed"] % 8 == 0 or entry["mode"] == "noop":
                    print("HM146", status, flush=True)
            if status["model_seconds"] > manifest["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Cumulative model cap crossed")
            status["state"] = "COLLECTED"
            status_write(out / "status.json", status)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status_write(out / "status.json", status)
        raise
    print("COLLECTED HM146", status, flush=True)


def observe(out):
    """Reconstruct native fields and scopes from sealed primitive operations."""
    manifest, status = check(out), H.load(out / "status.json")
    if (
        status["state"] != "COLLECTED"
        or status["completed"] != 164
        or status["pending"] is not None
    ):
        raise ValueError("Incomplete work is not a completed finite experiment")
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    rows = {r["id"]: r for r in H.load(out / "00/rows.json")}
    validate_rows(list(rows.values()))
    design, queries = H.load(out / "00/design.json"), H.load(out / "00/queries.json")
    cube = np.full((2, 2, 2, 2, 2, 5, 3, 2, 4), np.nan)
    captures, filled = {}, set()
    guards = dict.fromkeys(
        (
            "cache_spread",
            "body_final",
            "current_key",
            "current_value",
            "own_swap",
            "readout_reconstruction",
        ),
        0.0,
    )
    for entry in design:
        row, arrays = rows[entry["row"]], committed(out, entry)
        if not np.array_equal(arrays["blocks"], np.arange(28)) or not np.array_equal(
            arrays["interchange_blocks"], np.arange(28)
        ):
            raise ValueError("All28 block scope changed")
        width = int(arrays["execution_length"])
        wanted = np.stack(
            [A.permissions(row, len(s), width, 7) for s in queries["suffixes"]]
        )
        hashes = np.stack(
            [
                np.frombuffer(hashlib.sha256(s.tobytes()).digest(), dtype=np.uint8)
                for s in wanted
            ]
        )
        if not np.array_equal(hashes, arrays["support_hash"]) or not np.array_equal(
            arrays["endpoints"],
            [len(row["input_ids"]) + len(s) - 1 for s in queries["suffixes"]],
        ):
            raise ValueError("Actual mask or endpoint differs from frozen support")
        p = softmax(arrays["logits"], axis=1)
        read = np.stack(
            (
                p @ np.linspace(0, 1, 10),
                logsumexp(arrays["logits"][:, 5:], axis=1)
                - logsumexp(arrays["logits"][:, :5], axis=1),
                np.exp(logsumexp(arrays["logits"], axis=1) - arrays["normalizer"]),
                -(p * np.log(p)).sum(axis=1),
            ),
            axis=1,
        )
        guards["readout_reconstruction"] = max(
            guards["readout_reconstruction"],
            float(np.max(abs(read - arrays["readings"]))),
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
                    raise ValueError("Wrong intervention donor")
            if entry["mode"] == "noop":
                guards["own_swap"] = max(
                    guards["own_swap"],
                    float(np.max(abs(arrays["readings"] - base["readings"]))),
                )
                continue
        cell = (*address, MODES.index(entry["mode"]))
        if cell in filled:
            raise ValueError("Duplicated finite field")
        filled.add(cell)
        cube[cell] = arrays["readings"].reshape(3, 2, 4)
    if (
        not np.isfinite(cube).all()
        or len(filled) != 160
        or max(guards.values()) > manifest["spec"]["tolerance"]
    ):
        raise ValueError("Incomplete field or failed scope guard")
    return cube, guards, status, manifest


def outputs(cube):
    """Preserve all group components, person sectors and signed swap responses."""
    coefficient_rows, summary, error = [], [], 0.0
    basis = M.character_matrix()
    for bank, mode, family, unit in itertools.product(
        range(2), range(5), range(3), range(4)
    ):
        field = cube[bank, :, :, :, :, mode, family, :, unit]
        beta = M.coefficients(field)
        flat = field.reshape(32)
        direct = np.array(
            [sum(basis[i, m] * flat[i] for i in range(32)) / 32 for m in range(32)]
        )
        error = max(
            error,
            float(np.max(abs(beta - direct))),
            float(np.max(abs(M.reconstruct(beta) - field))),
        )
        for mask in range(32):
            label = (
                "*".join(
                    n
                    for i, n in enumerate(("b1", "b2", "b3", "reference", "person"))
                    if mask & (1 << i)
                )
                or "constant"
            )
            coefficient_rows.append(
                {
                    "bank": bank,
                    "mode": MODES[mode],
                    "family": FAMILIES[family],
                    "unit": UNITS[unit],
                    "mask": mask,
                    "term": label,
                    "coefficient": float(beta[mask]),
                }
            )
    for bank, family, unit in itertools.product(range(2), range(3), range(4)):
        fields = cube[bank, :, :, :, :, :, family, :, unit]
        native = fields[..., 0, :]
        beta = M.coefficients(native)
        common, personal = M.person_sectors(native)
        reference_effect, person_effect = (
            native[:, :, :, ::-1] - native,
            np.flip(native, axis=(0, 1, 2)) - native,
        )
        interventions = {}
        for mode in range(1, 5):
            actual = fields[..., mode, :] - native
            _, actual_personal = M.person_sectors(actual)
            _, wanted_personal = M.person_sectors(
                reference_effect if mode == 4 else person_effect
            )
            interventions[MODES[mode]] = {
                "full": J.effect_comparison(
                    actual, reference_effect if mode == 4 else person_effect
                ),
                "person_difference": J.effect_comparison(
                    actual_personal, wanted_personal
                ),
                "joint_person_coefficient": float(
                    M.coefficients(fields[..., mode, :])[31]
                ),
                "first_person_coefficient": float(
                    M.coefficients(fields[..., mode, :])[17]
                ),
            }
        parity = J.parity_contrasts(personal)
        summary.append(
            {
                "bank": bank,
                "family": FAMILIES[family],
                "unit": UNITS[unit],
                "joint_person_coefficient": float(beta[31]),
                "first_person_coefficient": float(beta[17]),
                "common_reference_coefficient": float(beta[8]),
                "person_name_bias": float(beta[16]),
                "personal_parity_contrasts_negative_positive_reference": parity.tolist(),
                "predicted_personal_reversal": bool(parity[0] < 0 < parity[1]),
                "assignment_rms_over_bag_observer": float(
                    np.sqrt(np.mean((native - M.bag_observer(native)) ** 2))
                ),
                "common_nonconstant_rms": float(np.std(common)),
                "personal_rms": float(np.sqrt(np.mean(personal**2))),
                "exchange_defect": M.exchange_defect(native),
                "interventions": interventions,
            }
        )
    return coefficient_rows, summary, error


def law_audit():
    """Enumerate the unselected independent law and condition on actual selection."""
    errors, checks, summaries = [], 0, []
    for c in (-1, 1):
        selected = np.zeros((2, 2, 8))
        total = 0.0
        for a, d, i, j in itertools.product(range(2), range(2), range(8), range(8)):
            ta, tb = 1 - 2 * a, 1 - 2 * d
            pa = (1 + 0.6 * ta * c * int(np.prod(M.POINTS[i]))) / 8
            pb = (1 + 0.6 * tb * c * int(np.prod(M.POINTS[j]))) / 8
            mass = 0.25 * pa * pb
            total += mass
            if np.array_equal(M.POINTS[j], -M.POINTS[i]):
                selected[a, d, i] += mass
            checks += 1
        probability = float(selected.sum())
        selected /= probability
        actual, actual_probability = M.selected_pair_law(c)
        errors.extend(
            (
                abs(total - 1),
                abs(probability - 0.125),
                abs(actual_probability - probability),
                float(np.max(abs(selected - actual))),
            )
        )
        t = np.array([1, -1])
        class_table = selected.sum(2)
        errors.append(
            float(np.max(abs(class_table - (1 - 0.36 * t[:, None] * t[None, :]) / 4)))
        )
        for i, person in itertools.product(range(8), (-1, 1)):
            part = selected[:, :, i]
            direct = (part[0].sum() if person == 1 else part[:, 0].sum()) / part.sum()
            wanted = 0.5 + 0.3 * person * c * int(np.prod(M.POINTS[i]))
            errors.extend(
                (
                    abs(direct - wanted),
                    abs(direct - M.pair_posterior(M.POINTS[i], c, person)),
                )
            )
            checks += 2
        # Marginalizing any of the three observed occasions removes diagnosis.
        for degree in range(3):
            for indices in itertools.combinations(range(3), degree):
                for values in itertools.product((-1, 1), repeat=degree):
                    mask = np.ones(8, dtype=bool)
                    for index, value in zip(indices, values, strict=True):
                        mask &= M.POINTS[:, index] == value
                    part = selected[:, :, mask].sum(2)
                    for direct in (
                        part[0].sum() / part.sum(),
                        part[:, 0].sum() / part.sum(),
                    ):
                        errors.append(abs(direct - 0.5))
                        checks += 1
        summaries.append(
            {
                "reference": c,
                "selection_probability": probability,
                "selected_class_joint": class_table.tolist(),
                "class_correlation": float(
                    np.sum(class_table * t[:, None] * t[None, :])
                ),
            }
        )
    truth = np.array(
        [M.pair_posterior(x[:3], x[3], x[4]) for x in M.FIELD_POINTS]
    ).reshape(2, 2, 2, 2, 2)
    beta = np.zeros(32)
    beta[[0, 31]] = [0.5, 0.3]
    errors.extend(
        (
            float(np.max(abs(M.coefficients(truth) - beta))),
            M.exchange_defect(truth)["maximum"],
        )
    )
    return {
        "checks": checks + 32,
        "maximum_error": max(errors),
        "selected_laws": summaries,
        "normative_target_coefficient": 0.3,
        "every_lower_posterior": 0.5,
        "normative_first_fact_coefficient": 0.5,
    }


def analyze(out):
    """Save all finite results without optimizing trait-prediction accuracy."""
    cube, guards, status, manifest = observe(out)
    coefficient_rows, summary, error = outputs(cube)
    result = {
        "packet": "HM146",
        "status": "COMPLETE_FINITE_MECHANISM_NOT_THEORY_CLOSURE",
        "original_scope_complete": False,
        "new_public_histories": 32,
        "native_batches": 164,
        "native_endpoints": 984,
        "model_seconds": status["model_seconds"],
        "maximum_tokens": manifest["maximum_tokens"],
        "guards": guards,
        "alternate_character_error": error,
        "population_audit": law_audit(),
        "all_channels": summary,
        "decision": "Distinguish qualified fact response, named/anonymous joint target valuation and common context bias at fixed whole bag; do not rescue a failed diagnostic with precision tuning",
    }
    target = out / "02"
    target.mkdir()
    with (target / "fields.npz").open("xb") as stream:
        np.savez_compressed(stream, cube=cube)
    with (target / "coefficients.csv").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coefficient_rows[0]))
        writer.writeheader()
        writer.writerows(coefficient_rows)
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
    """Replay all raw logits, fields, coefficients and exact selected-law audits."""
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    cube, guards, status, _ = observe(out)
    coefficients, summary, error = outputs(cube)
    result = H.load(out / "02/result.json")
    if (
        summary != result["all_channels"]
        or guards != result["guards"]
        or law_audit() != result["population_audit"]
        or error != result["alternate_character_error"]
        or status["model_seconds"] != result["model_seconds"]
    ):
        raise ValueError("Saved mechanism result differs from primitive replay")
    with np.load(out / "02/fields.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["cube"], cube):
            raise ValueError("Saved scientific fields differ")
    with (out / "02/coefficients.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    if len(saved) != len(coefficients) or any(
        any(row[k] != str(v) for k, v in wanted.items())
        for row, wanted in zip(saved, coefficients, strict=True)
    ):
        raise ValueError("Complete signed coefficient table differs")
    if max(error, result["population_audit"]["maximum_error"]) > 1e-12:
        raise ValueError("Alternate mathematical checks fail")
    print("VERIFIED HM146", len(coefficients), "coefficients; error", error, flush=True)


def main():
    """Require explicit phases; a sealed terminal packet is never silently rerun."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    globals()[args.phase](args.output or ROOT / H.load(SPEC)["out"])


if __name__ == "__main__":
    main()
