#!/usr/bin/env python3
"""Execute a finite event-context versus common-collector mechanism experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
from grain_research.construction import math_event_context_routes as R
from grain_research.construction import math_qualified_composition as M
from grain_research.vector_mechanisms.navigation import MODEL, write_json
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts import run_grain_qualified_composition_hm110 as Q
from scripts.run_grain_history_routing_hm101 import check as source_check
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_event_context_routes_hm111.json"
CODE = tuple(
    dict.fromkeys(
        (
            *Q.CODE,
            "configs/grain_event_context_routes_hm111.json",
            "docs/GRAIN_EVENT_CONTEXT_ROUTES_PLAN_HM111.md",
            "grain_research/construction/math_event_context_routes.py",
            "scripts/run_grain_event_context_routes_hm111.py",
            "tests/test_grain_math_event_context_routes.py",
        )
    )
)


def check(out):
    """Authenticate fixed rules, source observer, model configuration and inputs."""
    m = H.load(out / "00/manifest.json")
    H.verify_hashes(ROOT, m["code"] | m["source"])
    H.verify_hashes(MODEL, m["model"])
    H.verify_hashes(out, m["inputs"])
    return m


def prepare(out):
    """Fix per-entry slots and four actual permissions before any model call."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve earlier and partial results")
    spec = H.load(SPEC)
    source = ROOT / spec["source"]
    source_check(source)
    if (
        H.load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM101_NOT_GLOBAL"
    ):
        raise ValueError("Verified frozen source reader required")
    bundle = H.load(source / "00/bundle.json")
    if bundle["targets"] != spec["targets"] or spec["arms"] != [
        list(a) for a in R.ARMS
    ]:
        raise ValueError("Fixed target order and four actual arms required")
    original, application = (
        H.load(ROOT / spec[k]) for k in ("source_material", "application_material")
    )
    banks = [
        {t: original["behaviours"][t][0] for t in spec["targets"]},
        application["new_behaviours"],
    ]
    tok = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )

    def encode(text):
        """Encode an explicit segment without downloading or adding suffix tokens."""
        return tok.encode(text, add_special_tokens=False)

    system = H.load(ROOT / spec["system_material"])["system"]
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "__BODY__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix = encode(rendered.split("__BODY__")[0])
    entries = {
        (b, j, level): encode(
            ("\n" if j else "")
            + f"Earlier entry E{j + 1}: the person {banks[b][t][level]}."
        )
        for b, (j, t), level in itertools.product(
            range(2), enumerate(spec["targets"]), range(3)
        )
    }
    widths = [
        max(
            len(entries[b, j, level])
            for b, level in itertools.product(range(2), range(3))
        )
        for j in range(6)
    ]
    owners = {
        o: encode("\n" + application["owners"][2 if o == 1 else 0]) for o in (-1, 1)
    }
    if Counter(owners[-1]) != Counter(owners[1]):
        raise ValueError("Owner token bags differ")
    bridge = encode("\n" + application["bridges"][0] + "\nCurrent entry: Person A")
    current_ids = encode(application["currents"][0])
    rows = []
    for bank, design in itertools.product(range(2), list(M.design())):
        ids, event_pos, pad = prefix.copy(), [], []
        for j, level in enumerate(design["levels"]):
            tokens = entries[bank, j, level]
            event_pos.append(list(range(len(ids), len(ids) + len(tokens))))
            pad.extend(range(len(ids) + len(tokens), len(ids) + widths[j]))
            ids += tokens + [tok.pad_token_id] * (widths[j] - len(tokens))
        relay = list(
            range(len(ids), len(ids) + len(owners[design["owner"]]) + len(bridge))
        )
        ids += owners[design["owner"]] + bridge
        current = list(range(len(ids), len(ids) + len(current_ids)))
        ids += current_ids
        for arm in R.ARMS:
            row = {
                **design,
                "unit": len(rows),
                "bank": bank,
                "arm": list(arm),
                "input_ids": ids,
                "prefix": list(range(len(prefix))),
                "events": event_pos,
                "relay": relay,
                "current": current,
                "padding": pad,
            }
            R.permission_mask(row, *arm)
            rows.append(row)
    if (
        len(rows) != spec["max_forwards"]
        or len({tuple(r["current"]) for r in rows}) != 1
    ):
        raise ValueError("Finite battery or common positions changed")
    lookup = {(r["bank"], M.key(r), tuple(r["arm"])): r for r in rows}
    bags = 0
    for row in rows:
        if row["owner"] == -1:
            match = lookup[row["bank"], (tuple(row["levels"]), 1), tuple(row["arm"])]
            if Counter(row["input_ids"]) != Counter(match["input_ids"]):
                raise ValueError("A/B contrast changes full token bag")
            bags += 1
    for folder in ("00", "01/units", "01/references"):
        (out / folder).mkdir(parents=True)
    H.write(out / "00/rows.json", rows)
    H.write(
        out / "00/reader.json",
        {
            "targets": bundle["targets"],
            "directions": bundle["directions"],
            "queries": A.registered_queries(bundle, max(current)).tolist(),
            "current_ids": current_ids,
            "current": current,
        },
    )
    sources = [
        source / n for n in ("00/manifest.json", "00/bundle.json", "verification.json")
    ] + [
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
            "slot_widths": widths,
            "prefix_length": len(prefix),
            "total_token_length": len(ids),
            "owner_bag_checks": bags,
            "source_coefficients_fitted": 0,
        },
    )
    write_json(
        out / "01/status.json",
        {
            "state": "PREPARED",
            "completed": 0,
            "attempts": 0,
            "pending": None,
            "model_seconds": 0.0,
        },
    )
    print(
        "HM111 PREPARED",
        {
            "forwards": len(rows),
            "tokens": len(ids),
            "widths": widths,
            "owner_bags": bags,
        },
        flush=True,
    )


def probes(out, row, raw):
    """Check actual contextual independence against frozen same-token references."""
    candidates = [("prefix", row["prefix"])]
    if row["arm"][0] == 0:
        candidates += [
            (f"event_b{row['bank']}_e{j}_l{row['levels'][j]}", positions)
            for j, positions in enumerate(row["events"])
        ]
    if row["arm"][1] == 0:
        candidates.append((f"relay_o{row['owner']}", row["relay"]))
    checks = []
    for name, positions in candidates:
        path = out / f"01/references/{name}.npz"
        value = R.referenced_probe(raw, positions)
        ids = np.array(row["input_ids"])[positions]
        if not path.exists():
            Q.save(path, value=value, positions=np.array(positions), ids=ids)
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(ids, z["ids"]) or not np.array_equal(
                positions, z["positions"]
            ):
                raise ValueError("Isolation reference has different token support")
            error = float(np.max(abs(value - z["value"])))
        checks.append(
            {
                "name": name,
                "max_error": error,
                "value_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        )
    return checks


def collect(out):
    """Commit finite actual interventions without uncertain retries or new fitting."""
    m = check(out)
    rows = H.load(out / "00/rows.json")
    status_path = out / "01/status.json"
    status = H.load(status_path)
    if status["pending"] is not None or status["state"] not in (
        "PREPARED",
        "COLLECTING",
    ):
        raise ValueError("Uncertain or terminal attempt retained; no retry")
    with collector_guard(out):
        start = time.monotonic()
        observer = load_observer()
        status["load_seconds"] = (
            status.get("load_seconds", 0) + time.monotonic() - start
        )
        try:
            for row in rows:
                stem = out / f"01/units/unit_{row['unit']:04d}"
                if stem.with_suffix(".json").exists():
                    Q.raw_unit(out, row)
                    continue
                if stem.with_suffix(".npz").exists():
                    raise RuntimeError("Uncommitted capture cannot be retried")
                if status["model_seconds"] >= m["spec"]["max_model_seconds"]:
                    status["state"] = "BUDGET_STOP"
                    write_json(status_path, status)
                    raise TimeoutError("Cumulative model cap; no rescue")
                status.update(
                    state="COLLECTING",
                    pending=row["unit"],
                    attempts=status["attempts"] + 1,
                )
                write_json(status_path, status)
                start = time.monotonic()
                full, trace = R.capture(observer, row)
                seconds = time.monotonic() - start
                isolation = probes(out, row, full)
                trace["isolation"] = isolation
                if max(p["max_error"] for p in isolation) > m["spec"]["isolation_atol"]:
                    raise ValueError(
                        "Actual states violate declared event/relay independence"
                    )
                current = np.array(row["current"])
                Q.save(
                    stem.with_suffix(".npz"), k=full["k"][current], v=full["v"][current]
                )
                H.write(
                    stem.with_suffix(".json"),
                    {
                        "unit": row["unit"],
                        "trace": trace,
                        "seconds": seconds,
                        "sha256": H.sha(stem.with_suffix(".npz")),
                        "rule_seal_before_forward": H.sha(out / "00/manifest.json"),
                    },
                )
                status.update(
                    completed=row["unit"] + 1,
                    pending=None,
                    model_seconds=status["model_seconds"] + seconds,
                )
                write_json(status_path, status)
                if status["completed"] % 104 == 0:
                    print("HM111", status, flush=True)
            H.write(
                out / "01/raw_seal.json",
                H.seals(out, out / "01/units") | H.seals(out, out / "01/references"),
            )
            status.update(state="COLLECTED", actual_forwards=len(rows))
            write_json(status_path, status)
        except BaseException as exc:
            if status["state"] != "BUDGET_STOP":
                status.update(state="FAILED", error=repr(exc))
                write_json(status_path, status)
            raise
    print("HM111 COLLECTED", status, flush=True)


def fields(out, rows, reader, read=M.attention):
    """Read all routes and their own baseline/single-state constructions."""
    q, d = np.array(reader["queries"]), np.array(reader["directions"])
    result = []
    for bank in range(2):
        route_values = []
        for arm in R.ARMS:
            selected = {
                M.key(r): r
                for r in rows
                if r["bank"] == bank and tuple(r["arm"]) == arm
            }
            states = {
                key: Q.raw_unit(out, row)
                for key, row in selected.items()
                if row["kind"] != "pair"
            }
            a, b = Q.predictions(states, q, d, read)
            actual = []
            for i, j in Q.PAIRS:
                corners = []
                for x, y, owner in M.SIGNS:
                    levels = [1] * 6
                    levels[i], levels[j] = int(x + 1), int(y + 1)
                    raw = Q.raw_unit(out, selected[tuple(levels), int(owner)])
                    corners.append(read(raw["k"], raw["v"], q, d))
                actual.append(corners)
            route_values.append(np.stack((np.array(actual), a, b), axis=2))
        result.append(route_values)
    return np.array(result)


def output_tables(data, cold):
    """Keep every pair/target and signed route component without winner selection."""
    corners, authors, routes = [], [], []
    for bank, arm, pair, corner, target in itertools.product(
        range(2), range(4), range(15), range(8), range(6)
    ):
        corners.append(
            {
                "bank": bank,
                "arm": "".join(map(str, R.ARMS[arm])),
                "pair": pair,
                "corner": corner,
                "target": target,
                **{
                    name: data[bank, arm, pair, corner, f, target]
                    for f, name in enumerate(
                        ("actual", "additive_read", "additive_state")
                    )
                },
            }
        )
    for bank in range(2):
        contrast = (cold[bank][:, 0] - cold[bank][:, 1]).transpose(1, 0, 2)
        components = R.route_account(contrast)
        for arm, pair, target in itertools.product(range(4), range(15), range(6)):
            authors.append(
                {
                    "bank": bank,
                    "arm": "".join(map(str, R.ARMS[arm])),
                    "pair": pair,
                    "target": target,
                    "plus": cold[bank][pair, 0, arm, target],
                    "minus": cold[bank][pair, 1, arm, target],
                    "contrast": contrast[arm, pair, target],
                }
            )
            routes.append(
                {
                    "bank": bank,
                    "component": R.COMPONENTS[arm],
                    "pair": pair,
                    "target": target,
                    "effect": components[arm, pair, target],
                }
            )
    return {"corners": corners, "authors": authors, "route_components": routes}


def summarize(data, cold):
    """Describe construction consequences, not prediction accuracy or information shares."""
    banks = []
    for bank in range(2):
        contrast = (cold[bank][:, 0] - cold[bank][:, 1]).transpose(1, 0, 2)
        components = R.route_account(contrast)
        arm_records = []
        for arm in range(4):
            actual, additive, summed = (data[bank, arm, :, :, f] for f in range(3))
            coefficients = np.array([M.coefficients(x) for x in actual])
            arm_records.append(
                {
                    "arm": list(R.ARMS[arm]),
                    "actual_author_contrast_RMS": M.rms(contrast[arm]),
                    "author_contrast_by_target_RMS": np.sqrt(
                        np.mean(contrast[arm] ** 2, axis=0)
                    ).tolist(),
                    "xy_RMS": M.rms(coefficients[:, 6]),
                    "xyo_RMS": M.rms(coefficients[:, 7]),
                    "single_read_defect_RMS": M.rms(actual - additive),
                    "single_state_defect_RMS": M.rms(actual - summed),
                }
            )
        full = contrast[3]
        denominator = float(np.sum(full**2))
        banks.append(
            {
                "bank": bank,
                "arms": arm_records,
                "route_component_RMS": {
                    name: M.rms(components[i]) for i, name in enumerate(R.COMPONENTS)
                },
                "route_component_signed_alignment": {
                    name: float(np.sum(components[i] * full) / denominator)
                    if denominator
                    else None
                    for i, name in enumerate(R.COMPONENTS)
                },
                "components_are_not_independent_variance_or_personality_information": True,
            }
        )
    return {
        "banks": banks,
        "actual_forwards": 1168,
        "new_fits": 0,
        "cold_processes": 2,
        "designed_author_laws": 60,
        "original_scope_complete": False,
    }


def analyze(out):
    """Create cold qualified-author observations and signed route accounts."""
    start = time.monotonic()
    check(out)
    H.verify_hashes(out, H.load(out / "01/raw_seal.json"))
    if H.load(out / "01/status.json")["state"] != "COLLECTED":
        raise ValueError("Incomplete actual support cannot become a full result")
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    data = fields(out, rows, reader)
    dest = out / "02"
    dest.mkdir()
    Q.save(dest / "fields.npz", data=data)
    cold = []
    for bank in range(2):
        obs = np.take(data[bank, :, :, :, 0], M.AUTHOR_CORNERS, axis=2).transpose(
            1, 2, 0, 3
        )
        source, result = (
            dest / f"cold_input_{bank}.npz",
            dest / f"cold_output_{bank}.npz",
        )
        Q.save(
            source, observations=obs, counts=M.LAW_COUNTS, owners=np.ones((15, 4), int)
        )
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/score_grain_qualified_composition_cold_hm110.py"),
                "--input",
                str(source),
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
                "command": p.args,
                "returncode": p.returncode,
                "stdout": p.stdout,
                "interface": "Reused frozen HM110 qualified numeric integrator; no text/model/truth",
            },
        )
        with np.load(result, allow_pickle=False) as z:
            cold.append(z["scores"])
    for name, table in output_tables(data, cold).items():
        H.table(dest / f"{name}.csv", table)
    summary = summarize(data, cold)
    summary["analysis_seconds"] = time.monotonic() - start
    H.write(dest / "summary.json", summary)
    H.write(out / "output_seal.json", H.seals(out, dest))
    print("HM111 ANALYZED", summary, flush=True)


def verify(out):
    """Replay raw reads, graph isolation, signed identities and positive author sums."""
    start = time.monotonic()
    manifest = check(out)
    for seal in ("01/raw_seal.json", "output_seal.json"):
        H.verify_hashes(out, H.load(out / seal))
    rows, reader = H.load(out / "00/rows.json"), H.load(out / "00/reader.json")
    status = H.load(out / "01/status.json")
    if (
        status["state"] != "COLLECTED"
        or status["pending"] is not None
        or status["completed"] != len(rows)
        or status["attempts"] != len(rows)
    ):
        raise ValueError("Attempt/commit support incomplete")
    isolation, probe_count = 0.0, 0
    for row in rows:
        meta = H.load(out / f"01/units/unit_{row['unit']:04d}.json")
        trace, mask = meta["trace"], R.permission_mask(row, *row["arm"])
        if (
            trace["masked_blocks"] != list(range(17))
            or trace["permission_sha256"] != hashlib.sha256(mask.tobytes()).hexdigest()
            or meta["rule_seal_before_forward"] != H.sha(out / "00/manifest.json")
        ):
            raise ValueError("Actual invocation does not match the frozen rules")
        if [row["input_ids"][p] for p in row["current"]] != reader[
            "current_ids"
        ] or row["current"] != reader["current"]:
            raise ValueError("Different current carrier coordinates")
        isolation = max(isolation, max(p["max_error"] for p in trace["isolation"]))
        probe_count += len(trace["isolation"])
    reconstructed = fields(out, rows, reader, M.attention_loop)
    with np.load(out / "02/fields.npz", allow_pickle=False) as z:
        data = z["data"]
    errors = {
        "primitive_read": float(np.max(abs(reconstructed - data))),
        "route_identity": 0.0,
        "author_event_sum": 0.0,
        "walsh_reconstruction": 0.0,
    }
    cold = []
    for bank in range(2):
        with np.load(out / f"02/cold_output_{bank}.npz", allow_pickle=False) as z:
            cold.append(z["scores"])
        contrast = (cold[bank][:, 0] - cold[bank][:, 1]).transpose(1, 0, 2)
        errors["route_identity"] = max(
            errors["route_identity"],
            float(np.max(abs(R.route_account(contrast).sum(0) - contrast[3]))),
        )
        for arm, pair in itertools.product(range(4), range(15)):
            field = data[bank, arm, pair, :, 0]
            coeff = M.coefficients(field)
            errors["walsh_reconstruction"] = max(
                errors["walsh_reconstruction"],
                float(np.max(abs(M.CHARACTERS @ coeff - field))),
            )
            owned = field[M.AUTHOR_CORNERS]
            for law, counts in enumerate(M.LAW_COUNTS):
                explicit = np.concatenate(
                    [np.repeat(owned[k][None], n, axis=0) for k, n in enumerate(counts)]
                ).mean(0)
                errors["author_event_sum"] = max(
                    errors["author_event_sum"],
                    float(np.max(abs(explicit - cold[bank][pair, law, arm]))),
                )
    for name, table in output_tables(data, cold).items():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        if stream.getvalue().encode() != (out / f"02/{name}.csv").read_bytes():
            raise ValueError("Table does not replay: " + name)
    expected, recorded = summarize(data, cold), H.load(out / "02/summary.json")
    if expected != {k: v for k, v in recorded.items() if k != "analysis_seconds"}:
        raise ValueError("Summary does not replay")
    if (
        max(errors.values()) > manifest["spec"]["arithmetic_atol"]
        or isolation > manifest["spec"]["isolation_atol"]
    ):
        raise ValueError("Numerical or isolation failure retained")
    result = {
        "status": "VERIFIED_FINITE_HM111_NOT_GLOBAL",
        "actual_forwards": len(rows),
        "errors": errors,
        "isolation_max_error": isolation,
        "actual_isolation_probe_checks": probe_count,
        "cold_processes": 2,
        "verification_seconds": time.monotonic() - start,
        "same_agent_not_third_party": True,
        "original_scope_complete": False,
    }
    H.write(out / "verification.json", result)
    print("HM111 VERIFIED", result, flush=True)


def main():
    """Run one explicit finite stage; never overwrite prior packets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / H.load(SPEC)["output"]
    globals()[args.phase](out)


if __name__ == "__main__":
    main()
