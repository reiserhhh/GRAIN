#!/usr/bin/env python3
"""Finite source-only event-state composition and qualified author consequences."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_author_evidence_merge as A
from grain_research.construction import math_history_routing as R
from grain_research.construction import math_qualified_composition as M
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_history_routing_hm101 import check as reader_check
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_qualified_composition_hm110.json"
CODE = (
    "configs/grain_qualified_composition_hm110.json",
    "docs/GRAIN_QUALIFIED_COMPOSITION_PLAN_HM110.md",
    "grain_research/construction/math_qualified_composition.py",
    "scripts/run_grain_qualified_composition_hm110.py",
    "scripts/score_grain_qualified_composition_cold_hm110.py",
    "tests/test_grain_math_qualified_composition.py",
    "grain_research/construction/math_author_evidence_merge.py",
    "grain_research/construction/math_history_routing.py",
    "grain_research/vector_mechanisms/prefix.py",
    "scripts/run_grain_author_evidence_merge_hm102.py",
    "scripts/run_grain_history_routing_hm101.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
)
FIELDS = ("actual", "additive_read", "additive_state", "source_read", "source_state")
PAIRS = list(itertools.combinations(range(6), 2))


def save(path, **arrays):
    """Retain every existing output, including failed or incomplete artifacts."""
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def check(out):
    """Check source calibration and frozen input/code identities for each phase."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(MODEL, m["model"])
    H.verify_hashes(out, m["inputs"])
    return m


def prepare(out):
    """Fix all text streams, observations and author laws before model calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve earlier packet")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    reader_check(source)
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM101_NOT_GLOBAL"
    ):
        raise ValueError("Verified source reader required")
    bundle = H.load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"] or spec["author_law_strength"] != 0.4:
        raise ValueError("Inherited target order and positive laws fixed")
    original = H.load(ROOT / spec["source_material"])
    application = H.load(ROOT / spec["application_material"])
    banks = [
        {t: original["behaviours"][t][0] for t in spec["targets"]},
        application["new_behaviours"],
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    system = H.load(ROOT / spec["system_material"])["system"]
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "__GRAIN_BODY__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    def tokens(text):
        """Segmented inherited tokenization, without answer or property queries."""
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = tokens(rendered.split("__GRAIN_BODY__")[0])
    designs = list(M.design())
    histories, bags = {}, 0
    for bank, row in itertools.product(range(2), designs):
        phrases = [
            banks[bank][t][level]
            for t, level in zip(spec["targets"], row["levels"], strict=True)
        ]
        text = "\n".join(
            f"Earlier entry E{i + 1}: the person {p}." for i, p in enumerate(phrases)
        )
        text += "\n" + application["owners"][2 if row["owner"] == 1 else 0]
        histories[bank, M.key(row)] = {"text": text, "tokens": tokens(text)}
    for bank, row in itertools.product(range(2), designs):
        if row["owner"] == -1:
            other = dict(row, owner=1)
            if Counter(histories[bank, M.key(row)]["tokens"]) != Counter(
                histories[bank, M.key(other)]["tokens"]
            ):
                raise ValueError("A/B owner contrast changes token bag")
            bags += 1
    width = max(len(h["tokens"]) for h in histories.values())
    bridge = tokens(
        "\n"
        + application["bridges"][spec["bridge_index"]]
        + "\nCurrent entry: Person A"
    )
    current_tokens = tokens(application["currents"][spec["current_index"]])
    current = list(
        range(
            len(prefix) + width + len(bridge),
            len(prefix) + width + len(bridge) + len(current_tokens),
        )
    )
    rows = []
    for bank, row in itertools.product(range(2), designs):
        history = histories[bank, M.key(row)]
        ids = (
            prefix
            + history["tokens"]
            + [tokenizer.pad_token_id] * (width - len(history["tokens"]))
            + bridge
            + current_tokens
        )
        hpos = list(range(len(prefix), len(prefix) + width))
        excluded = set(hpos) | set(current)
        rows.append(
            {
                **row,
                "unit": len(rows),
                "bank": bank,
                "history_text": history["text"],
                "input_ids": ids,
                "history": hpos,
                "current": current,
                "relay": [p for p in range(len(ids)) if p not in excluded],
                "padding": list(
                    range(len(prefix) + len(history["tokens"]), len(prefix) + width)
                ),
            }
        )
    if len(rows) != spec["max_forwards"] or bags != 146:
        raise ValueError("Registered battery cardinality changed")
    (out / "00").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    H.write(out / "00/rows.json", rows)
    H.write(
        out / "00/reader.json",
        {
            "targets": bundle["targets"],
            "directions": bundle["directions"],
            "queries": A.registered_queries(bundle, max(current)).tolist(),
            "current_tokens": current_tokens,
            "positions": current,
        },
    )
    sources = [
        source / name
        for name in ("00/manifest.json", "00/bundle.json", "verification.json")
    ]
    sources += [
        ROOT / spec[k]
        for k in ("source_material", "application_material", "system_material")
    ]
    H.write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {p: H.sha(ROOT / p) for p in CODE},
            "source": {str(p.relative_to(ROOT)): H.sha(p) for p in sources},
            "model": {p: H.sha(MODEL / p) for p in ("config.json", "tokenizer.json")},
            "inputs": H.seals(out, out / "00"),
            "history_width": width,
            "owner_token_bag_checks": bags,
            "actual_forwards_planned": len(rows),
        },
    )
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "model_seconds": 0.0,
            "pending": None,
        },
    )
    print(
        "PREPARED HM110",
        {"forwards": len(rows), "bags": bags, "width": width},
        flush=True,
    )


def raw_unit(out, row):
    """Require a committed actual capture with the declared current-token shape."""
    stem = out / f"01/units/unit_{row['unit']:04d}"
    metadata = H.load(stem.with_suffix(".json"))
    if (
        metadata["unit"] != row["unit"]
        or H.sha(stem.with_suffix(".npz")) != metadata["sha256"]
    ):
        raise ValueError("Capture commit mismatch")
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
        raw = {k: z[k].astype(float) for k in ("k", "v")}
    if (
        raw["k"].shape != raw["v"].shape
        or len(raw["k"]) != len(row["current"])
        or not np.isfinite(raw["k"] + raw["v"]).all()
    ):
        raise ValueError("Missing or incompatible current capture")
    return raw


def anchors(out, bank, rows):
    """Read only actual baseline/single units, never a measured pair interaction."""
    return {
        M.key(r): raw_unit(out, r)
        for r in rows
        if r["bank"] == bank and r["kind"] != "pair"
    }


def predictions(states, q, d, read=M.attention):
    """Construct all pair states from single evidence, with no pair-effect inputs."""
    read_sum, state_sum = [], []
    for i, j in PAIRS:
        a, b = [], []
        for x, y, owner in M.SIGNS:
            mid = [1] * 6
            first, second = mid.copy(), mid.copy()
            first[i], second[j] = int(x + 1), int(y + 1)
            base, ri, rj = (
                states[tuple(levels), int(owner)] for levels in (mid, first, second)
            )
            constructed = {k: M.compose(base[k], ri[k], rj[k]) for k in ("k", "v")}
            a.append(
                read(ri["k"], ri["v"], q, d)
                + read(rj["k"], rj["v"], q, d)
                - read(base["k"], base["v"], q, d)
            )
            b.append(read(constructed["k"], constructed["v"], q, d))
        read_sum.append(a)
        state_sum.append(b)
    return np.array(read_sum), np.array(state_sum)


def freeze_source(out, rows, reader):
    """Seal source-only compositions before collecting any application prefix."""
    if (out / "source_seal.json").exists():
        H.verify_hashes(out, H.load(out / "source_seal.json"))
        return
    inputs = [r for r in rows if r["bank"] == 0 and r["kind"] != "pair"]
    application_files = [
        p
        for r in rows
        if r["bank"] == 1
        for p in (
            out / f"01/units/unit_{r['unit']:04d}.npz",
            out / f"01/units/unit_{r['unit']:04d}.json",
        )
        if p.exists()
    ]
    if application_files:
        raise ValueError("Source constructor was not frozen before application capture")
    states = anchors(out, 0, rows)
    a, b = predictions(
        states, np.array(reader["queries"]), np.array(reader["directions"])
    )
    dest = out / "02_source"
    dest.mkdir()
    save(dest / "predictions.npz", additive_read=a, additive_state=b)
    H.write(
        dest / "manifest.json",
        {
            "input_units": [r["unit"] for r in inputs],
            "input_count": len(inputs),
            "pair_captures_used": 0,
            "application_captured_before_freeze": False,
            "observed_application_file_count": len(application_files),
            "hashes": {
                f"01/units/unit_{r['unit']:04d}.npz": H.sha(
                    out / f"01/units/unit_{r['unit']:04d}.npz"
                )
                for r in inputs
            },
        },
    )
    H.write(out / "source_seal.json", H.seals(out, dest))
    print("HM110 SOURCE COMPOSITION FROZEN:26 single/base captures only", flush=True)


def collect(out):
    """Capture one finite unit per committed attempt, without uncertain retries."""
    m = check(out)
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    status_path = out / "01/status.json"
    status = H.load(status_path)
    if status["pending"] is not None or status["state"] not in (
        "PREPARED",
        "COLLECTING",
    ):
        raise ValueError("Uncertain or terminal attempt must not be retried")
    with collector_guard(out):
        start = time.monotonic()
        observer = load_observer()
        status["load_seconds"] = (
            status.get("load_seconds", 0.0) + time.monotonic() - start
        )
        try:
            for row in rows:
                if row["bank"] == 1:
                    freeze_source(out, rows, reader)
                stem = out / f"01/units/unit_{row['unit']:04d}"
                if stem.with_suffix(".json").exists():
                    raw_unit(out, row)
                    continue
                if stem.with_suffix(".npz").exists():
                    raise RuntimeError("Uncommitted capture retained; no retry")
                if status["model_seconds"] >= m["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    write_json(status_path, status)
                    raise TimeoutError("Finite budget exhausted, no rescue")
                status.update(
                    state="COLLECTING",
                    pending=row["unit"],
                    attempts=status["attempts"] + 1,
                )
                write_json(status_path, status)
                start = time.monotonic()
                raw, trace = R.capture_routed_prefix(observer, row, 1, 1)
                seconds = time.monotonic() - start
                save(stem.with_suffix(".npz"), **raw)
                H.write(
                    stem.with_suffix(".json"),
                    {
                        "unit": row["unit"],
                        "trace": trace,
                        "seconds": seconds,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "source_seal_before_forward": H.sha(out / "source_seal.json")
                        if row["bank"] == 1
                        else None,
                    },
                )
                status.update(
                    completed=row["unit"] + 1,
                    pending=None,
                    model_seconds=status["model_seconds"] + seconds,
                )
                write_json(status_path, status)
                if status["completed"] % 26 == 0:
                    print("HM110", status, flush=True)
            H.write(out / "01/raw_seal.json", H.seals(out, out / "01/units"))
            status.update(state="COLLECTED", actual_forwards=len(rows))
            write_json(status_path, status)
        except BaseException as exc:
            if status["state"] != "BUDGET_STOP":
                status.update(state="FAILED", error=repr(exc))
                write_json(status_path, status)
            raise
    print("COLLECTED HM110", status, flush=True)


def fields(out, rows, reader, read=M.attention):
    """Calculate actual and anchored fields for every pair, owner and target."""
    q, d = np.array(reader["queries"]), np.array(reader["directions"])
    bank_fields = []
    for bank in range(2):
        single = anchors(out, bank, rows)
        a, b = predictions(single, q, d, read)
        selected = {
            M.key(r): r for r in rows if r["bank"] == bank and r["kind"] == "pair"
        }
        actual = []
        for i, j in PAIRS:
            values = []
            for x, y, owner in M.SIGNS:
                levels = [1] * 6
                levels[i], levels[j] = int(x + 1), int(y + 1)
                raw = raw_unit(out, selected[tuple(levels), int(owner)])
                values.append(read(raw["k"], raw["v"], q, d))
            actual.append(values)
        bank_fields.append(np.stack((np.array(actual), a, b), axis=2))
    return np.array(bank_fields)


def tables(data, source, cold):
    """Return all signed observations/coefficients/author contrasts without selection."""
    values, characters, authors = [], [], []
    for bank, pair in itertools.product(range(2), range(15)):
        complete = np.concatenate((data[bank, pair], source[pair]), axis=1)
        coeff = M.coefficients(complete)
        for corner, target in itertools.product(range(8), range(6)):
            values.append(
                {
                    "bank": bank,
                    "pair": pair,
                    "i": PAIRS[pair][0],
                    "j": PAIRS[pair][1],
                    "corner": corner,
                    "x": M.SIGNS[corner, 0],
                    "y": M.SIGNS[corner, 1],
                    "owner": M.SIGNS[corner, 2],
                    "target": target,
                    **{
                        name: complete[corner, f, target]
                        for f, name in enumerate(FIELDS)
                    },
                }
            )
        for character, target in itertools.product(range(8), range(6)):
            characters.append(
                {
                    "bank": bank,
                    "pair": pair,
                    "power": "".join(map(str, M.POWERS[character])),
                    "target": target,
                    **{
                        name: coeff[character, f, target]
                        for f, name in enumerate(FIELDS)
                    },
                }
            )
        for field, target in itertools.product(range(5), range(6)):
            authors.append(
                {
                    "bank": bank,
                    "pair": pair,
                    "i": PAIRS[pair][0],
                    "j": PAIRS[pair][1],
                    "field": FIELDS[field],
                    "target": target,
                    "plus": cold[bank][pair, 0, field, target],
                    "minus": cold[bank][pair, 1, field, target],
                    "contrast": cold[bank][pair, 0, field, target]
                    - cold[bank][pair, 1, field, target],
                }
            )
    return {"corners": values, "characters": characters, "authors": authors}


def summarize(data, cold):
    """Keep reading/encoding signed accounts separate from prediction accuracy."""
    records = []
    for bank in range(2):
        split = M.account(data[bank, :, :, 0], data[bank, :, :, 1], data[bank, :, :, 2])
        total, read, encode = (split[k] for k in ("defect", "reading", "encoding"))
        power = float(np.sum(total * total))
        coeff = np.array([M.coefficients(c) for c in data[bank]])
        author = cold[bank][:, 0] - cold[bank][:, 1]
        records.append(
            {
                "bank": bank,
                "defect_RMS": M.rms(total),
                "reading_RMS": M.rms(read),
                "encoding_RMS": M.rms(encode),
                "reading_signed_alignment": float(np.sum(read * total) / power)
                if power
                else None,
                "encoding_signed_alignment": float(np.sum(encode * total) / power)
                if power
                else None,
                "reading_encoding_mean_product": float(np.mean(read * encode)),
                "xy_actual_RMS": M.rms(coeff[:, 6, 0]),
                "xyo_actual_RMS": M.rms(coeff[:, 7, 0]),
                "xy_additive_read_RMS": M.rms(coeff[:, 6, 1]),
                "xyo_additive_read_RMS": M.rms(coeff[:, 7, 1]),
                "author_contrast_RMS": {
                    f: M.rms(author[:, i]) for i, f in enumerate(FIELDS)
                },
                "author_encoding_remainder_RMS": M.rms(author[:, 0] - author[:, 2]),
            }
        )
    return {
        "banks": records,
        "new_forwards": 292,
        "new_fits": 0,
        "cold_processes": 2,
        "designed_author_laws": 60,
        "events_per_law": 20,
        "original_scope_complete": False,
    }


def analyze(out):
    """Execute positive cold author aggregation and a signed construction account."""
    start = time.monotonic()
    check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    H.verify_hashes(out, H.load(out / "source_seal.json"))
    if H.load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Only complete actual battery may be analyzed")
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    data = fields(out, rows, reader)
    with np.load(out / "02_source/predictions.npz", allow_pickle=False) as z:
        source = np.stack((z["additive_read"], z["additive_state"]), axis=2)
    dest = out / "03"
    dest.mkdir()
    save(dest / "fields.npz", data=data, source=source)
    cold = []
    for bank in range(2):
        obs = np.concatenate(
            (data[bank][:, M.AUTHOR_CORNERS], source[:, M.AUTHOR_CORNERS]), axis=2
        )
        if obs.shape != (15, 4, 5, 6):
            raise ValueError(
                "Cold evidence layout differs from declared pair/corner/field"
            )
        path, result = dest / f"cold_input_{bank}.npz", dest / f"cold_output_{bank}.npz"
        save(path, observations=obs, counts=M.LAW_COUNTS, owners=np.ones((15, 4), int))
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_qualified_composition_cold_hm110.py"),
                "--input",
                str(path),
                "--output",
                str(result),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        H.write(
            dest / f"cold_process_{bank}.json",
            {
                "command": completed.args,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
            },
        )
        with np.load(result, allow_pickle=False) as z:
            cold.append(z["scores"])
    for name, table in tables(data, source, cold).items():
        H.table(dest / f"{name}.csv", table)
    summary = summarize(data, cold)
    summary["analysis_seconds"] = time.monotonic() - start
    H.write(dest / "summary.json", summary)
    H.write(out / "output_seal.json", H.seals(out, dest))
    print("ANALYZED HM110", summary, flush=True)


def verify(out):
    """Recompute actual readouts, all coefficients and explicit author event sums."""
    start = time.monotonic()
    m = check(out)
    for seal in ("01/raw_seal.json", "source_seal.json", "output_seal.json"):
        H.verify_hashes(out, H.load(out / seal))
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    status = H.load(out / "01/status.json")
    if (
        status["state"] != "COLLECTED"
        or status["pending"] is not None
        or status["attempts"] != len(rows)
        or status["completed"] != len(rows)
    ):
        raise ValueError("Attempt/commit coverage does not match actual finite support")
    source_manifest = H.load(out / "02_source/manifest.json")
    if len(source_manifest["input_units"]) != 26 or any(
        rows[i]["bank"] or rows[i]["kind"] == "pair"
        for i in source_manifest["input_units"]
    ):
        raise ValueError("Pair/application values entered the source constructor")
    H.verify_hashes(out, source_manifest["hashes"])
    for row in rows:
        capture = H.load(out / f"01/units/unit_{row['unit']:04d}.json")
        if capture["trace"]["masked_blocks"] != list(range(17)):
            raise ValueError("Incomplete actual prefix capture")
        if row["bank"] == 1 and capture["source_seal_before_forward"] != H.sha(
            out / "source_seal.json"
        ):
            raise ValueError("Application forward did not record the frozen source")
    reconstructed = fields(out, rows, reader, M.attention_loop)
    with np.load(out / "03/fields.npz", allow_pickle=False) as z:
        data, source = z["data"], z["source"]
    errors = {"primitive_read": float(np.max(abs(reconstructed - data)))}
    q, d = np.array(reader["queries"]), np.array(reader["directions"])
    sr, ss = predictions(anchors(out, 0, rows), q, d, M.attention_loop)
    errors["source_construction"] = float(
        np.max(abs(np.stack((sr, ss), axis=2) - source))
    )
    identity = (reader["current_tokens"], reader["positions"])
    if any(
        ([r["input_ids"][p] for p in r["current"]], r["current"]) != identity
        for r in rows
    ):
        raise ValueError("Carrier coordinates not common")
    errors["signed_account"] = 0.0
    errors["character_reconstruction"] = 0.0
    errors["coefficient_sum"] = 0.0
    errors["author_event_sum"] = 0.0
    errors["author_xy_identity"] = 0.0
    cold = []
    for bank in range(2):
        with np.load(out / f"03/cold_output_{bank}.npz", allow_pickle=False) as z:
            cold.append(z["scores"])
        complete = np.concatenate((data[bank], source), axis=2)
        split = M.account(*[data[bank, :, :, f] for f in (0, 1, 2)])
        errors["signed_account"] = max(
            errors["signed_account"],
            float(np.max(abs(split["defect"] - split["reading"] - split["encoding"]))),
        )
        for pair in range(15):
            c = M.coefficients(complete[pair])
            errors["character_reconstruction"] = max(
                errors["character_reconstruction"],
                float(
                    np.max(
                        abs(np.einsum("cs,sft->cft", M.CHARACTERS, c) - complete[pair])
                    )
                ),
            )
            for power, exponents in enumerate(M.POWERS):
                explicit = (
                    sum(
                        np.prod(signs**exponents) * complete[pair, k]
                        for k, signs in enumerate(M.SIGNS)
                    )
                    / 8
                )
                errors["coefficient_sum"] = max(
                    errors["coefficient_sum"], float(np.max(abs(explicit - c[power])))
                )
            owned = complete[pair, M.AUTHOR_CORNERS]
            for law, counts in enumerate(M.LAW_COUNTS):
                repeated = np.concatenate(
                    [np.repeat(owned[k][None], n, axis=0) for k, n in enumerate(counts)]
                )
                errors["author_event_sum"] = max(
                    errors["author_event_sum"],
                    float(np.max(abs(repeated.mean(0) - cold[bank][pair, law]))),
                )
            axy = (c[6] + c[7]) * 0.8
            errors["author_xy_identity"] = max(
                errors["author_xy_identity"],
                float(np.max(abs(axy - (cold[bank][pair, 0] - cold[bank][pair, 1])))),
            )
    for name, records in tables(data, source, cold).items():
        expected = out / f"verify_{name}.csv"
        H.table(expected, records)
        if H.sha(expected) != H.sha(out / f"03/{name}.csv"):
            raise ValueError("Full table replay mismatch:" + name)
    if max(errors.values()) > m["spec"]["arithmetic_atol"]:
        H.write(out / "verification_failure.json", errors)
        raise ValueError("Independent numerical account failed")
    result = {
        "status": "VERIFIED_FINITE_HM110_NOT_GLOBAL",
        "errors": errors,
        "actual_forwards": len(rows),
        "model_seconds": status["model_seconds"],
        "verify_seconds": time.monotonic() - start,
        "original_scope_complete": False,
    }
    H.write(out / "verification.json", result)
    print("VERIFIED HM110", result, flush=True)


def main():
    """Run one explicitly selected phase; never automatically restart a packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / H.load(SPEC)["output"]
    globals()[args.phase](out)


if __name__ == "__main__":
    main()
