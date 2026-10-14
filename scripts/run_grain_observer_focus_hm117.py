#!/usr/bin/env python3
"""Test observation focus as a separate measurement variable, without fitting."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observer_focus as M
from scripts import run_grain_actor_action_hm116 as R

SPEC = ROOT / "configs/grain_observer_focus_hm117.json"
CODE = (
    "configs/grain_observer_focus_hm117.json",
    "docs/GRAIN_OBSERVER_FOCUS_PLAN_HM117.md",
    "grain_research/construction/math_observer_focus.py",
    "scripts/run_grain_observer_focus_hm117.py",
    "tests/test_grain_math_observer_focus.py",
    *R.CODE,
)


def check(out):
    """Verify frozen code, complete inherited arms and new prepared tokens."""
    m = R.H.load(out / "00/manifest.json")
    R.H.verify_hashes(ROOT, m["code"] | m["source"])
    R.H.verify_hashes(out, m["inputs"])
    return m


def prepare(out):
    """Change only one current focus token and seal all source actions/support."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.navigation import MODEL, write_json

    if out.exists():
        raise FileExistsError("Preserve prior or failed results")
    spec = R.H.load(SPEC)
    source = ROOT / spec["source"]
    old = R.check(source)
    parent_status = R.H.load(source / "status.json")
    if parent_status["pending"] or parent_status["state"] != "BUDGET_STOP":
        raise ValueError(
            "The retained terminal parent, not a revived collector, is required"
        )
    R.H.verify_hashes(source, R.H.load(source / "partial_readback/output_seal.json"))
    if spec["arms"] != list(R.M.ARMS) or spec["targets"] != old["spec"]["targets"]:
        raise ValueError("No action or target selection")
    originals = {
        r["id"]: r
        for r in R.H.load(source / "00/rows.json")
        if r["wording"] == spec["source_wording"]
        and r["variant"] == spec["source_variant"]
    }
    inherited = [
        d
        for d in R.H.load(source / "00/design.json")
        if d["row"] in originals and d["arm"] in R.M.ARMS
    ]
    if len(originals) != 48 or len(inherited) != 240:
        raise ValueError("All five complete A-focus arms required")
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    current = tok.encode(spec["current_B"], add_special_tokens=False)
    rows = []
    for original in originals.values():
        ids = original["input_ids"].copy()
        positions = original["current"]
        if (
            len(current) != len(positions)
            or sum(ids[p] != t for p, t in zip(positions, current, strict=True)) != 1
        ):
            raise ValueError("Only one equal-length current focus token may change")
        for p, t in zip(positions, current, strict=True):
            ids[p] = t
        rows.append(
            {
                **original,
                "id": len(rows),
                "source_row": original["id"],
                "input_ids": ids,
                "current_text": spec["current_B"],
                "focus": "B",
            }
        )
    design = []
    for r in rows:
        design.append({"batch": len(design), "row": r["id"], "arm": "baseline"})
    for r in rows:
        if (r["family"], r["condition"]) == tuple(spec["noop_group"]):
            design.append({"batch": len(design), "row": r["id"], "arm": "noop"})
    for arm in R.M.ARMS[1:]:
        for r in rows:
            design.append({"batch": len(design), "row": r["id"], "arm": arm})
    if (
        len(design) != spec["max_batches"]
        or sum(d["arm"] == "noop" for d in design) != 4
    ):
        raise ValueError("Complete support and early no-op required")
    sources = [
        source / p
        for p in (
            "00/manifest.json",
            "00/actions.npz",
            "00/queries.json",
            "00/rows.json",
            "status.json",
            "partial_readback/summary.json",
            "partial_readback/output_seal.json",
        )
    ]
    for d in inherited:
        R.raw_unit(source, d["batch"])
        stem = source / f"01/units/b{d['batch']:03d}"
        sources.extend([stem.with_suffix(".npz"), stem.with_suffix(".json")])
    folder = out / "00"
    folder.mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    R.H.write(folder / "rows.json", rows)
    R.H.write(folder / "design.json", design)
    R.H.write(folder / "inherited.json", inherited)
    R.H.write(
        folder / "manifest.json",
        {
            "spec": spec,
            "code": {p: R.H.sha(ROOT / p) for p in CODE},
            "source": {str(p.relative_to(ROOT)): R.H.sha(p) for p in sources},
            "inputs": R.H.seals(out, folder),
            "model": old["model"],
            "model_blocks": old["model_blocks"],
            "physical_shape": old["physical_shape"],
            "one_changed_token_per_row": True,
            "complete_inherited_first_wording": True,
            "parent_full_protocol_passed": False,
            "new_action_fits": 0,
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
        "PREPARED HM117",
        len(rows),
        "B-focus rows",
        len(design),
        "new batches",
        flush=True,
    )


def raw_unit(out, index):
    """Read a committed batch and authenticate its paired numeric file."""
    stem = out / f"01/units/b{index:03d}"
    meta = R.H.load(stem.with_suffix(".json"))
    if R.H.sha(stem.with_suffix(".npz")) != meta["sha256"]:
        raise ValueError("Capture changed")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def collect(out):
    """Execute new focus support once, under a load/failure-inclusive cap."""
    from grain_research.vector_mechanisms.navigation import MODEL, write_json
    from scripts.run_grain_math_span_binding import load_observer
    from scripts.run_grain_upstream_routes import collector_guard

    m, status = check(out), R.H.load(out / "status.json")
    if status["state"] != "PREPARED" or status["pending"]:
        raise RuntimeError("No terminal/uncertain replay or renewed budget")
    source = ROOT / m["spec"]["source"]
    R.H.verify_hashes(MODEL, m["model"])
    queries = R.H.load(source / "00/queries.json")
    rows = {r["id"]: r for r in R.H.load(out / "00/rows.json")}
    design = R.H.load(out / "00/design.json")
    bases = {d["row"]: d["batch"] for d in design if d["arm"] == "baseline"}
    with np.load(source / "00/actions.npz", allow_pickle=False) as z:
        actions = {k: z[k] for k in z.files}
    try:
        with collector_guard(out):
            start = time.monotonic()
            try:
                observer = load_observer()
            finally:
                status["model_seconds"] += time.monotonic() - start
                write_json(out / "status.json", status)
            R.H.write(out / "model_manifest.json", observer.manifest())
            for d in design:
                if status["model_seconds"] >= m["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    raise TimeoutError("Finite model cap, no renewal")
                row = rows[d["row"]]
                delta = (
                    None if d["arm"] == "baseline" else np.zeros(m["physical_shape"])
                )
                if d["arm"] in actions:
                    delta = (
                        -row["polarity"]
                        * row["owner"]
                        * actions[d["arm"]][row["family"], (row["condition"] + 1) // 2]
                    )
                status.update(
                    state="RUNNING", attempts=status["attempts"] + 1, pending=d
                )
                write_json(out / "status.json", status)
                start = time.monotonic()
                try:
                    readings, arrays = R.B.native_batch(
                        observer,
                        row,
                        queries["suffixes"],
                        queries["alphabet"],
                        delta,
                        m["spec"]["layer"],
                    )
                finally:
                    seconds = time.monotonic() - start
                    status["model_seconds"] += seconds
                    write_json(out / "status.json", status)
                logits = np.array([r["candidate_logits"] for r in readings])
                norm = np.array([r["full_vocab_log_normalizer"] for r in readings])
                arrays.update(
                    logits=logits,
                    normalizer=norm,
                    readings=np.array(
                        [
                            R.V.native_units(z, n)
                            for z, n in zip(logits, norm, strict=True)
                        ]
                    ),
                    requested_delta=np.zeros(m["physical_shape"])
                    if delta is None
                    else delta,
                )
                if d["arm"] == "noop":
                    base = raw_unit(out, bases[d["row"]])
                    if (
                        np.max(abs(arrays["readings"] - base["readings"]))
                        > m["spec"]["isolation_tolerance"]
                    ):
                        raise ValueError("Early no-op gate failed; do not collect more")
                stem = out / f"01/units/b{d['batch']:03d}"
                with stem.with_suffix(".npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                R.H.write(
                    stem.with_suffix(".json"),
                    {
                        **d,
                        "sha256": R.H.sha(stem.with_suffix(".npz")),
                        "seconds": seconds,
                        "endpoints": 12,
                    },
                )
                status.update(completed=status["completed"] + 1, pending=None)
                write_json(out / "status.json", status)
                if status["completed"] % 24 == 0 or status["completed"] == 52:
                    print("HM117", status, flush=True)
            if status["model_seconds"] > m["spec"]["max_model_seconds"]:
                status["state"] = "BUDGET_STOP"
                raise TimeoutError("Finite cap crossed")
            status["state"] = "COLLECTED"
            write_json(out / "status.json", status)
            R.H.write(out / "01/raw_seal.json", R.H.seals(out, out / "01/units"))
    except BaseException as exc:
        if status["state"] != "BUDGET_STOP":
            status["state"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "status.json", status)
        raise
    print("COLLECTED HM117", status, flush=True)


def assemble(out):
    """Retain the full person/focus law and check actual edits and permissions."""
    m = check(out)
    source = ROOT / m["spec"]["source"]
    rows = {r["id"]: r for r in R.H.load(out / "00/rows.json")}
    original_to_new = {r["source_row"]: r["id"] for r in rows.values()}
    design = R.H.load(out / "00/design.json")
    bases = {d["row"]: d["batch"] for d in design if d["arm"] == "baseline"}
    native = np.full((5, 2, 6, 2, 2, 2, 2, 6, 4), np.nan)
    errors = {k: 0.0 for k in ("key", "before", "edit", "requested", "readout", "noop")}
    with np.load(source / "00/actions.npz", allow_pickle=False) as z:
        actions = {k: z[k] for k in z.files}

    def validate_readings(raw):
        if raw["readings"].shape != (12, 4) or not np.array_equal(
            raw["masked_blocks"], np.tile(np.arange(m["model_blocks"]), (12, 1))
        ):
            raise ValueError("All12 endpoints/all causal blocks required")
        replay = np.array(
            [
                R.V.native_units(z, n)
                for z, n in zip(raw["logits"], raw["normalizer"], strict=True)
            ]
        )
        errors["readout"] = max(
            errors["readout"], float(np.max(abs(replay - raw["readings"])))
        )

    def insert(raw, row, arm, focus):
        validate_readings(raw)
        idx = (
            R.M.ARMS.index(arm),
            focus,
            row["family"],
            (row["condition"] + 1) // 2,
            (row["polarity"] + 1) // 2,
            (row["owner"] + 1) // 2,
        )
        native[idx] = raw["readings"].reshape(2, 6, 4)[::-1]

    for d in R.H.load(out / "00/inherited.json"):
        raw = R.raw_unit(source, d["batch"])
        row = rows[original_to_new[d["row"]]]
        insert(raw, row, d["arm"], 1)
        fresh_base = raw_unit(out, bases[row["id"]])
        for field in (
            "support_hash",
            "execution_support_hash",
            "masked_blocks",
            "execution_length",
            "endpoint",
        ):
            if not np.array_equal(raw[field], fresh_base[field]):
                raise ValueError("Cross-focus execution shape/permissions changed")
    for d in design:
        raw, row = raw_unit(out, d["batch"]), rows[d["row"]]
        base = raw_unit(out, bases[d["row"]])
        for field in (
            "support_hash",
            "execution_support_hash",
            "masked_blocks",
            "execution_length",
            "endpoint",
        ):
            if not np.array_equal(raw[field], base[field]):
                raise ValueError("Edit changed support")
        for key, field in (("key", "k"), ("before", "before")):
            errors[key] = max(errors[key], float(np.max(abs(raw[field] - base[field]))))
        request = np.zeros(m["physical_shape"])
        if d["arm"] in actions:
            request = (
                -row["polarity"]
                * row["owner"]
                * actions[d["arm"]][row["family"], (row["condition"] + 1) // 2]
            )
        errors["requested"] = max(
            errors["requested"], float(np.max(abs(request - raw["requested_delta"])))
        )
        errors["edit"] = max(
            errors["edit"], float(np.max(abs(raw["after"] - raw["before"] - request)))
        )
        if d["arm"] == "noop":
            validate_readings(raw)
            errors["noop"] = max(
                errors["noop"], float(np.max(abs(raw["readings"] - base["readings"])))
            )
            continue
        insert(raw, row, d["arm"], 0)
    if not np.isfinite(native).all() or errors["edit"] > m["spec"]["edit_tolerance"]:
        raise ValueError("Incomplete support or unrealized edit")
    if (
        max(errors[k] for k in ("key", "before", "noop"))
        > m["spec"]["isolation_tolerance"]
        or max(errors[k] for k in ("requested", "readout"))
        > m["spec"]["arithmetic_tolerance"]
    ):
        raise ValueError("Actual isolation/no-op/arithmetic gate failed")
    return native, errors


def accounts(native):
    """Separate focus, queried person and their interaction without dropping any."""
    beta = np.array(
        [
            [
                M.decompose(native[a, :, f].transpose(1, 2, 3, 4, 0, 5, 6))
                for f in range(6)
            ]
            for a in range(5)
        ]
    )
    return beta, beta[0:1] - beta


def tables(native, beta, effects, spec):
    """Produce allcondition/target/author tables and separately named sectors."""
    target, author, summary, sectors = [], [], [], []
    for a, arm in enumerate(R.M.ARMS):
        for f, family in enumerate(spec["targets"]):
            law = M.positive_author_integral(
                native[a, :, f].transpose(1, 2, 3, 4, 0, 5, 6), spec["author_amplitude"]
            )
            for ci, c in enumerate((-1, 1)):
                for ri, r in enumerate((-1, 1)):
                    values = M.qualified_effect(effects[a, f], c, r)
                    for q, query in enumerate(spec["targets"]):
                        for u, unit in enumerate(R.UNITS):
                            target.append(
                                {
                                    "arm": arm,
                                    "family": family,
                                    "condition": c,
                                    "focus": "A" if r == 1 else "B",
                                    "query": query,
                                    "unit": unit,
                                    "actor_difference": float(
                                        values["actor_difference"][q, u]
                                    ),
                                    "actor_common": float(values["actor_common"][q, u]),
                                }
                            )
            for pi, p in enumerate(("B", "A")):
                for ri, r in enumerate(("B", "A")):
                    for q, query in enumerate(spec["targets"]):
                        for u, unit in enumerate(R.UNITS):
                            author.append(
                                {
                                    "arm": arm,
                                    "family": family,
                                    "person": p,
                                    "focus": r,
                                    "query": query,
                                    "unit": unit,
                                    "positive_law_difference": float(law[pi, ri, q, u]),
                                }
                            )
        for ri, r in enumerate((-1, 1)):
            for u, unit in enumerate(R.UNITS):
                x = effects[a, :, 14, :, u] + r * effects[a, :, 30, :, u]
                g = effects[a, :, 6, :, u] + r * effects[a, :, 22, :, u]
                summary.append(
                    {
                        "arm": arm,
                        "focus": "A" if r == 1 else "B",
                        "unit": unit,
                        "actor_difference_RMS": float(np.sqrt(np.mean(x * x))),
                        "actor_common_RMS": float(np.sqrt(np.mean(g * g))),
                        "diagonal_mean": float(np.trace(x) / 6),
                        "off_target_mean": float(x[~np.eye(6, dtype=bool)].mean()),
                        "diagonal_minus_off": float(
                            np.trace(x) / 6 - x[~np.eye(6, dtype=bool)].mean()
                        ),
                    }
                )
        for u, unit in enumerate(R.UNITS):
            sectors.append(
                {
                    "arm": arm,
                    "unit": unit,
                    **{
                        name: float(np.sqrt(np.mean(effects[a, :, mask, :, u] ** 2)))
                        for name, mask in (
                            ("bo_common", 6),
                            ("bop_query_person", 14),
                            ("bor_focus_common", 22),
                            ("bopr_focus_person", 30),
                        )
                    },
                }
            )
    return target, author, summary, sectors


def analyze(out):
    """Construct the observation law; no encoder, classifier or metric fitting."""
    m = check(out)
    if R.H.load(out / "status.json")["state"] != "COLLECTED":
        raise ValueError("Complete HM117 support required")
    R.H.verify_hashes(out, R.H.load(out / "01/raw_seal.json"))
    native, checks = assemble(out)
    beta, effects = accounts(native)
    folder = out / "02"
    folder.mkdir()
    np.savez_compressed(
        folder / "accounts.npz", raw=native, native=beta, effects=effects
    )
    all_tables = tables(native, beta, effects, m["spec"])
    for name, values in zip(
        ("target_effects", "author_accounts", "focus_summaries", "routing_sectors"),
        all_tables,
        strict=True,
    ):
        R.H.table(folder / f"{name}.csv", values)
    R.H.write(
        folder / "summary.json",
        {
            "packet": "HM117",
            "status": R.H.load(out / "status.json"),
            "checks": checks,
            "new_calls": 244,
            "new_endpoints": 2928,
            "inherited_batches": 240,
            "character_count": 32,
            "table_rows": [len(t) for t in all_tables],
            "new_fits": 0,
            "parent_HM116_passed": False,
            "neural_runtime_loaded": any(
                n in sys.modules for n in ("torch", "transformers")
            ),
            "original_scope_complete": False,
        },
    )
    R.H.write(folder / "output_seal.json", R.H.seals(out, folder))
    print("ANALYZED HM117", checks, [len(t) for t in all_tables], flush=True)


def verify(out):
    """Independently sum all32 characters, author integrals and saved CSV rows."""
    m = check(out)
    R.H.verify_hashes(out, R.H.load(out / "01/raw_seal.json"))
    R.H.verify_hashes(out, R.H.load(out / "02/output_seal.json"))
    native, checks = assemble(out)
    with np.load(out / "02/accounts.npz", allow_pickle=False) as z:
        saved = {k: z[k] for k in z.files}
    errors = {
        "raw": float(np.max(abs(native - saved["raw"]))),
        "effects": float(
            np.max(abs(saved["effects"] - (saved["native"][0:1] - saved["native"])))
        ),
        "character": 0.0,
        "reconstruction": 0.0,
        "author": 0.0,
        "CSV": 0.0,
    }
    for a, f in itertools.product(range(5), range(6)):
        x = native[a, :, f].transpose(1, 2, 3, 4, 0, 5, 6)
        for mask in range(32):
            value = np.zeros((6, 4))
            for idx in itertools.product(range(2), repeat=5):
                sign = np.prod(
                    [(2 * idx[j] - 1) if mask & (1 << j) else 1 for j in range(5)]
                )
                value += sign * x[idx] / 32
            errors["character"] = max(
                errors["character"],
                float(np.max(abs(value - saved["native"][a, f, mask]))),
            )
        errors["reconstruction"] = max(
            errors["reconstruction"],
            float(np.max(abs(M.reconstruct(saved["native"][a, f]) - x))),
        )
        direct = M.positive_author_integral(x, m["spec"]["author_amplitude"])
        for pi, p in enumerate((-1, 1)):
            for ri, r in enumerate((-1, 1)):
                expected = (2 * m["spec"]["author_amplitude"] / 6) * (
                    saved["native"][a, f, 6]
                    + p * saved["native"][a, f, 14]
                    + r * saved["native"][a, f, 22]
                    + p * r * saved["native"][a, f, 30]
                )
                errors["author"] = max(
                    errors["author"], float(np.max(abs(direct[pi, ri] - expected)))
                )
    beta, effects = accounts(native)
    for name, expected in zip(
        ("target_effects", "author_accounts", "focus_summaries", "routing_sectors"),
        tables(native, beta, effects, m["spec"]),
        strict=True,
    ):
        with (out / f"02/{name}.csv").open(newline="") as stream:
            actual = list(csv.DictReader(stream))
        if len(actual) != len(expected):
            raise ValueError("CSV support changed")
        for actual_row, expected_row in zip(actual, expected, strict=True):
            for key, value in expected_row.items():
                if isinstance(value, (float, int)):
                    errors["CSV"] = max(
                        errors["CSV"], abs(float(actual_row[key]) - value)
                    )
                elif actual_row[key] != value:
                    raise ValueError("CSV scientific address changed")
    if max(errors.values()) > m["spec"]["arithmetic_tolerance"]:
        raise ValueError(f"Independent replay failed:{errors}")
    R.H.write(
        out / "verification.json",
        {
            "status": "VERIFIED_FINITE_HM117_NOT_GLOBAL",
            "errors": errors,
            "raw_checks": checks,
            "no_op_scope": "New B-focus four-cell group only;does not fill parent's missing A no-op",
            "same_agent_not_third_party": True,
            "original_scope_complete": False,
        },
    )
    print("VERIFIED HM117", errors, flush=True)


def main():
    """Run one explicitly chosen stage, never automatically recover failed work."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / R.H.load(SPEC)["output"])
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())


if __name__ == "__main__":
    main()
