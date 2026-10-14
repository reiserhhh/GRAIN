#!/usr/bin/env python3
"""HM59 finite passive experiment: declared event links, constant action/token bags."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_explicit_event_links as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_construct_routing import token_positions
from grain_research.construction.math_event_probability_pooling import probabilities
from grain_research.construction.math_evidence_response_operation import readout
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_event_relation_renaming import CODE as SOURCE_CODE
from scripts.run_grain_math_event_relation_renaming import check as source_check
from scripts.run_grain_math_event_relation_renaming import (
    observation as source_observation,
)
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_explicit_event_links_v1.json"
OUT = ROOT / "results/grain_math_response_v1/explicit_event_links_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_explicit_event_links_v1.json",
            "docs/GRAIN_EXPLICIT_EVENT_LINK_MECHANISM_PLAN_V1.md",
            "grain_research/construction/math_explicit_event_links.py",
            "scripts/run_grain_math_explicit_event_links.py",
            "scripts/verify_grain_math_explicit_event_links.py",
            "tests/test_grain_math_explicit_event_links.py",
            *SOURCE_CODE,
        )
    )
)
FIELDS = (
    "unit",
    "content",
    "query",
    "family",
    "polarity",
    "order",
    "syntax",
    "declaration_actor",
    "scheme",
)


def source_atoms(spec):
    """Bind each isolated law to its true content/family/polarity/surface code."""
    source = ROOT / spec["source"]
    m = source_check(source)
    if (
        read_json(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM58_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM58 required")
    atoms, lineage = {}, {}
    for name in (
        "00/manifest.json",
        "00/rows.json",
        "verification.json",
        "completion_manifest.json",
    ):
        lineage[str((source / name).relative_to(ROOT))] = digest(source / name)
    for r in read_json(source / "00/rows.json"):
        if r["role"] != "atom":
            continue
        path = (
            ROOT / m["reused"][str(r["unit"])]["meta"]
            if r["reused"]
            else source / f"01/units/unit_{r['unit']:04d}.json"
        )
        raw = source_observation(source, r, m)
        key = (r["content"], r["family"], r["polarity"], r["history"][0]["event_id"])
        if key in atoms:
            raise ValueError("Duplicate isolated atom")
        atoms[key] = probabilities(raw["candidate_logits"])
        for p in (path, path.with_suffix(".npz")):
            lineage[str(p.relative_to(ROOT))] = digest(p)
    if len(atoms) != spec["expected_atom_readouts"]:
        raise ValueError("Complete 96 actual-code atom laws required")
    return atoms, m, lineage


def render_document(row, system, tokenizer):
    """Bind exact local chat inputs, action prefix and declared relation token slots."""
    text, ids, part, cues, actor = render(
        row["body"], row["content"], "A", 1, system, tokenizer
    )
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start = text.index(row["body"])
    action_end = start + len(row["action_block"])
    declaration_start = start + len(row["action_block"]) + 2
    action = token_positions(encoded["offset_mapping"], start, action_end)
    declaration = token_positions(
        encoded["offset_mapping"], declaration_start, start + len(row["body"])
    )
    if not action or not declaration or set(action) & set(declaration):
        raise ValueError("Disjoint action/declaration slots required")
    return {
        **row,
        "rendered": text,
        "input_ids": ids,
        "partition": part,
        "cue_slots": cues,
        "actor_slot": actor,
        "action_positions": action,
        "declaration_positions": declaration,
    }


def token_proof(a, b):
    """Prove equal full token bags and unchanged action prefix/query identities."""
    if a["action_block"] != b["action_block"] or M.word_bag(a["body"]) != M.word_bag(
        b["body"]
    ):
        raise ValueError("Changed action text/raw word bag")
    x, y = a["input_ids"], b["input_ids"]
    if len(x) != len(y) or Counter(x) != Counter(y):
        raise ValueError("Full actual token bags must agree")
    if a["action_positions"] != b["action_positions"]:
        raise ValueError("Action token positions changed")
    prefix_end = max(a["action_positions"]) + 1
    if x[:prefix_end] != y[:prefix_end]:
        raise ValueError("Action prefix identity changed")
    s, t = a["partition"]["suffix"], b["partition"]["suffix"]
    if s != t or [x[i] for i in s] != [y[i] for i in t]:
        raise ValueError("Ordered query suffix must be unchanged")
    changed = [i for i, (u, v) in enumerate(zip(x, y, strict=True)) if u != v]
    allowed = set(a["declaration_positions"]) | set(b["declaration_positions"])
    if not changed or not set(changed) <= allowed:
        raise ValueError("Only declaration tokens may change")
    return {
        "lo": a["unit"],
        "hi": b["unit"],
        "full_length": len(x),
        "changed_positions": changed,
        "word_bag_equal": True,
        "full_token_bag_equal": True,
        "action_prefix_equal": True,
        "query_suffix_equal": True,
        "not_digit_only": True,
    }


def prepare(out):
    """Freeze all materials and proofs before any bounded passive readout."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve previous scientific packets")
    spec = read_json(SPEC)
    atoms, old, lineage = source_atoms(spec)
    reference = ROOT / spec["behaviour_reference"]
    parent = read_json(reference)
    lineage[str(reference.relative_to(ROOT))] = digest(reference)
    if (
        spec["layer"] != old["layer"]
        or spec["schemes"] != list(M.SCHEMES)
        or spec["action_orders"] != [list(x) for x in M.ORDERS]
        or spec["fit_patch_generation_API_private_training"]
    ):
        raise ValueError("Unchanged finite passive design required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows = [
        render_document(r, old["system"], tokenizer)
        for r in M.documents(parent["spec"]["behaviours"], old["targets"])
    ]
    pairs = M.paired_roster(rows)
    proofs = [token_proof(rows[p["lo"]], rows[p["hi"]]) for p in pairs]
    if (len(rows), len(pairs), len(proofs)) != (384, 192, 192):
        raise ValueError("Complete declared design required")
    for folder in ("00/code", "01/units", "03"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    write(out / "00/rows.json", rows)
    write(out / "00/pairs.json", pairs)
    write(out / "00/token_proofs.json", proofs)
    write(
        out / "00/atoms.json",
        [
            {"key": list(k), "probabilities": p.tolist()}
            for k, p in sorted(atoms.items())
        ],
    )
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM59",
            "spec": spec,
            "code": code,
            "source_hashes": lineage,
            "targets": old["targets"],
            "alphabet": old["alphabet"],
            "system": old["system"],
            "observer_reference": old["observer_reference"],
            "observer_reference_sha256": old["observer_reference_sha256"],
            "model_config_sha256": digest(MODEL / "config.json"),
            "assets": {
                n: digest(out / n)
                for n in (
                    "00/rows.json",
                    "00/pairs.json",
                    "00/token_proofs.json",
                    "00/atoms.json",
                )
            },
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "independent_human_units": 0,
            "new_neural_training": False,
            "model_weights_not_separately_hashed": True,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "model_seconds": 0.0,
            "native_forward_seconds": 0.0,
            "in_flight": None,
        },
    )
    print("PREPARED HM59:384 passive readouts/192 exact token-bag pairs", flush=True)


def check(out):
    """Reject drift of live/frozen code, local assets, atom lineage or observer."""
    m = read_json(out / "00/manifest.json")
    for name, expected in m["code"].items():
        if (
            digest(ROOT / name) != expected
            or digest(out / "00/code" / name) != expected
        ):
            raise ValueError("Frozen code drift:" + name)
    for mapping, base in ((m["assets"], out), (m["source_hashes"], ROOT)):
        for name, expected in mapping.items():
            if digest(base / name) != expected:
                raise ValueError("Frozen asset/source drift:" + name)
    if (
        digest(MODEL / "config.json") != m["model_config_sha256"]
        or digest(ROOT / m["observer_reference"]) != m["observer_reference_sha256"]
    ):
        raise ValueError("Observer reference drift")
    return m


def observation(out, row, manifest):
    """Read a bound raw capture, without invoking the observer."""
    path = out / f"01/units/unit_{row['unit']:04d}.json"
    meta = read_json(path)
    if meta["row"] != row or digest(path.with_suffix(".npz")) != meta["array_sha256"]:
        raise ValueError("Exact raw-material binding mismatch")
    raw = meta["native"]
    p = probabilities(raw["candidate_logits"])
    v = np.asarray(manifest["alphabet"]["values"])
    tol = manifest["spec"]["arithmetic_tolerance"]
    if (
        abs(float(p @ v) - raw["conditional_score"]) > tol
        or np.max(np.abs(p - raw["conditional_probabilities"])) > tol
    ):
        raise ValueError("Raw candidate readout mismatch")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        if set(arrays.files) != {"declaration", "suffix", "endpoint"}:
            raise ValueError("Complete declared captures required")
        width = len(arrays["endpoint"])
        for name, shape in (
            ("declaration", (len(row["declaration_positions"]), width)),
            ("suffix", (len(row["partition"]["suffix"]), width)),
            ("endpoint", (width,)),
        ):
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError("Capture shape/finite-state mismatch")
        if not np.array_equal(arrays["endpoint"], arrays["suffix"][-1]):
            raise ValueError("Endpoint must be the final suffix state")
    return raw


def collect(out):
    """Run each passive input once; stop instead of retrying uncertain forwards."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("Terminal/uncertain attempts require explicit inspection")
    if (
        len(list((out / "01/units").glob("unit_*.json"))) != status["completed_units"]
        or status["completed_units"] != status["forward_calls"]
    ):
        raise ValueError("Raw capture/status mismatch")
    start, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        current, prior = observer.manifest(), read_json(ROOT / m["observer_reference"])
        for k in (
            "model",
            "torch",
            "transformers",
            "dtype",
            "device",
            "hidden_size",
            "layers",
            "max_tokens",
            "attention",
            "training",
            "template_sha256",
        ):
            if current[k] != prior[k]:
                raise ValueError("Actual observer drift:" + k)
        write(out / "01/observer.json", current)
        for row in read_json(out / "00/rows.json"):
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                observation(out, row, m)
                continue
            elapsed = previous + time.monotonic() - start
            if (
                status["attempted_forward_calls"] >= m["spec"]["max_forwards"]
                or elapsed >= m["spec"]["max_model_seconds"]
            ):
                raise TimeoutError("384-forward/900-second bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            forward_start = time.monotonic()
            raw, arrays = observe_full(observer, row, m["alphabet"], m["spec"]["layer"])
            forward_seconds = time.monotonic() - forward_start
            status["native_forward_seconds"] += forward_seconds
            status["forward_calls"] += 1
            h = arrays["full"]
            save(
                path,
                {
                    "row": row,
                    "native": raw,
                    "finished_epoch": time.time(),
                    "forward_seconds": forward_seconds,
                },
                {
                    "declaration": h[row["declaration_positions"]],
                    "suffix": h[row["partition"]["suffix"]],
                    "endpoint": h[-1],
                },
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - start,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 64 == 0:
                print(f"ACTUAL:{status['completed_units']}/384", flush=True)
        if (
            any(
                status[k] != 384
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > 900
        ):
            raise ValueError("Incomplete finite acquisition")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - start,
        )
        write(out / "run_status.json", status)
        raise


def summaries(pairs, interactions):
    """Enumerate fixed descriptive scopes without choosing a successful syntax."""
    rows = []
    for name, data in (("scheme", pairs), ("actor_interaction", interactions)):
        rows.append({"kind": name, "group": "all", **M.summarize(data)})
        for key in ("content", "syntax", "order"):
            for value in sorted({r[key] for r in data}):
                rows.append(
                    {
                        "kind": name,
                        "group": key + "=" + str(value),
                        **M.summarize([r for r in data if r[key] == value]),
                    }
                )
        if name == "scheme":
            for actor in ("A", "B"):
                selected = [r for r in data if r["declaration_actor"] == actor]
                for content in ("all", *sorted({r["content"] for r in data})):
                    group = (
                        selected
                        if content == "all"
                        else [r for r in selected if r["content"] == content]
                    )
                    rows.append(
                        {
                            "kind": name,
                            "group": f"actor={actor}/content={content}",
                            **M.summarize(group),
                        }
                    )
    return rows


def analyze(out):
    """Read all laws and incidence references; no fitting or head selection."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] != "COLLECTED" or status["in_flight"] is not None:
        raise ValueError("Complete terminal acquisition required")
    if (out / "03/summary.json").exists():
        raise FileExistsError("Do not overwrite scientific analysis")
    rows, v = read_json(out / "00/rows.json"), m["alphabet"]["values"]
    atoms = {
        tuple(r["key"]): np.asarray(r["probabilities"])
        for r in read_json(out / "00/atoms.json")
    }
    native, laws, references, weights, records, lookup = [], [], [], [], {}, {}
    for row in rows:
        raw = observation(out, row, m)
        p = probabilities(raw["candidate_logits"])
        ref = M.references(row, atoms, v)
        keys = {k: row[k] for k in FIELDS}
        records[row["unit"]] = {**readout(raw), "probabilities": p, "references": ref}
        native.append(
            {
                **keys,
                **readout(raw),
                "events": ref["events"],
                "mentions": ref["mentions"],
                **{k + "_score": s for k, s in ref["scores"].items()},
            }
        )
        for j, probability in enumerate(p):
            laws.append(
                {"unit": row["unit"], "candidate": j, "probability": float(probability)}
            )
        for basis, law in ref["laws"].items():
            for j, probability in enumerate(law):
                references.append(
                    {
                        "unit": row["unit"],
                        "basis": basis,
                        "candidate": j,
                        "probability": float(probability),
                    }
                )
        for j, record in enumerate(row["history"]):
            weights.append(
                {
                    "unit": row["unit"],
                    "position": j,
                    "record_id": record["record_id"],
                    "occasion_code": record["occasion_code"],
                    "event_id": record["event_id"],
                    "event_weight": float(ref["event_weights"][j]),
                    "mention_weight": float(ref["mention_weights"][j]),
                }
            )
    pairs = []
    for pair in read_json(out / "00/pairs.json"):
        a, b = records[pair["lo"]], records[pair["hi"]]
        result = {**pair, **M.contrast(a, b, pair["orientation"])}
        for basis in a["references"]["laws"]:
            low, high = a["references"]["laws"][basis], b["references"]["laws"][basis]
            change = b["references"]["scores"][basis] - a["references"]["scores"][basis]
            result[basis + "_score"] = float(change)
            result[basis + "_tv"] = float(np.abs(high - low).sum() / 2)
        pairs.append(result)
        key = tuple(
            pair[k]
            for k in (
                "content",
                "family",
                "polarity",
                "order",
                "syntax",
                "declaration_actor",
            )
        )
        lookup[key] = result
    interactions = []
    for content, family, polarity, order, syntax in itertools.product(
        m["targets"], (0, 1), (0, 1), (0, 1), (0, 1)
    ):
        key = (content, family, polarity, order, syntax)
        a, b = lookup[(*key, "A")], lookup[(*key, "B")]
        delta = a["score"] - b["score"]
        interactions.append(
            {
                "interaction": len(interactions),
                "content": content,
                "family": family,
                "polarity": polarity,
                "order": order,
                "syntax": syntax,
                "score": float(delta),
                "oriented_score": float(delta * (2 * polarity - 1)),
                "log_odds": float(a["log_odds"] - b["log_odds"]),
                "probability_L1_half": float(
                    np.abs(
                        records[a["hi"]]["probabilities"]
                        - records[a["lo"]]["probabilities"]
                        - records[b["hi"]]["probabilities"]
                        + records[b["lo"]]["probabilities"]
                    ).sum()
                    / 2
                ),
                "A_pair": a["pair"],
                "B_pair": b["pair"],
                "event_code_neutral_score": a["event_code_neutral_score"]
                - b["event_code_neutral_score"],
            }
        )
    chars = []
    for content, family, polarity in itertools.product(m["targets"], (0, 1), (0, 1)):
        cells = {
            (scope, syntax, order): lookup[
                (content, family, polarity, order, syntax, "A" if scope == 0 else "B")
            ]["score"]
            for scope, syntax, order in itertools.product((0, 1), repeat=3)
        }
        for mask, coefficient in M.character_components(cells).items():
            chars.append(
                {
                    "content": content,
                    "family": family,
                    "polarity": polarity,
                    "character_scope_syntax_order": mask,
                    "coefficient": coefficient,
                }
            )
    tables = {
        "native": native,
        "probabilities": laws,
        "reference_laws": references,
        "event_weights": weights,
        "paired_changes": pairs,
        "actor_interactions": interactions,
        "character_components": chars,
        "summary_by_group": summaries(pairs, interactions),
    }
    for name, data in tables.items():
        write_csv(out / f"03/{name}.csv", data)
    write(
        out / "03/summary.json",
        {
            "packet": "HM59",
            "status": "ANALYZED_FINITE_NOT_GLOBAL",
            "actual_calls": status["forward_calls"],
            "model_seconds": status["model_seconds"],
            "rows": {name: len(data) for name, data in tables.items()},
            "summaries": tables["summary_by_group"],
            "fit_patch_generation_API_private_training": [0] * 6,
            "independent_human_units": 0,
            "original_scope_complete": False,
            "reference_is_not_personality_ground_truth": True,
            "captures_not_causal_localization": True,
        },
    )
    write(
        out / "03/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted((out / "03").glob("*"))
            if p.name != "artifact_hashes.json"
        },
    )
    print("ANALYZED HM59:384 laws/192 pairs/96 actor interactions", flush=True)


def main():
    """Expose separate immutable preparation, acquisition and numerical analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "collect":
        with collector_guard(args.out):
            collect(args.out)
    else:
        {"prepare": prepare, "analyze": analyze}[args.stage](args.out)


if __name__ == "__main__":
    main()
