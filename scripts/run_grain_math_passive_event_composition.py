#!/usr/bin/env python3
"""HM55: acquire passive event histories, not intervention fidelity or accuracy."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_passive_event_composition as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_evidence_response_operation import readout
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_evidence_response_operation import CODE as PARENT_CODE
from scripts.run_grain_math_evidence_response_operation import check as parent_check
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_passive_event_composition_v1.json"
OUT = ROOT / "results/grain_math_response_v1/passive_event_composition_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_passive_event_composition_v1.json",
            "docs/GRAIN_PASSIVE_EVENT_COMPOSITION_PLAN_V1.md",
            "grain_research/construction/math_passive_event_composition.py",
            "grain_research/evidence_history.py",
            "scripts/run_grain_math_passive_event_composition.py",
            "scripts/verify_grain_math_passive_event_composition.py",
            "tests/test_grain_math_passive_event_composition.py",
            *PARENT_CODE,
        )
    )
)
FIELDS = ("unit", "role", "target", "family", "polarity", "order", "regime")


def token_proofs(rows, tokenizer):
    """Require identical action/token sequences except two occasion digits."""
    index = {
        (r["target"], r["family"], r["polarity"], r["order"], r["regime"]): r
        for r in rows
        if r["role"] == "account"
    }
    proofs = []
    for t, f, p, order in itertools.product(
        sorted({r["target"] for r in rows}), (0, 1), (0, 1), (0, 1)
    ):
        a, b = (index[t, f, p, order, k] for k in ("alias_focus", "distinct_focus"))
        if len(a["input_ids"]) != len(b["input_ids"]):
            raise ValueError("Alias/distinct token length differs")
        changed = [
            i
            for i, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"], strict=True))
            if x != y
        ]
        offsets_a = tokenizer(
            a["rendered"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        offsets_b = tokenizer(
            b["rendered"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        chars = [
            [r["rendered"][offsets[i][0] : offsets[i][1]] for i in changed]
            for r, offsets in ((a, offsets_a), (b, offsets_b))
        ]
        if (
            len(changed) != 2
            or chars != [["1", "1"], ["3", "4"]]
            or not set(changed) <= set(a["partition"]["body"])
        ):
            raise ValueError("Only K1->K3/K4 occasion digits may differ")
        proofs.append(
            {
                "alias": a["unit"],
                "distinct": b["unit"],
                "changed_positions": changed,
                "changed_characters": chars,
                "tokens": len(a["input_ids"]),
            }
        )
    return proofs


def prepare(out):
    """Freeze fresh materials and unchanged observer before any forward."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing run")
    spec = read_json(SPEC)
    prior = parent_check(ROOT / spec["parent"])
    if (
        spec["targets"] != prior["spec"]["targets"]
        or spec["layer"] != prior["spec"]["layer"]
        or spec["system"] != prior["spec"]["system"]
        or read_json(ROOT / spec["parent"] / "verification.json")["status"]
        != "VERIFIED_FINITE_HM54_NOT_GLOBAL"
    ):
        raise ValueError("Verified unchanged HM54 reader required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows = []
    for doc in M.documents(spec):
        text, ids, part, cues, actor = render(
            doc["body"], doc["target"], "A", 1, spec["system"], tokenizer
        )
        rows.append(
            {
                **doc,
                "rendered": text,
                "input_ids": ids,
                "partition": part,
                "cue_slots": cues,
                "actor_slot": actor,
            }
        )
    if (
        len(rows) != spec["max_forwards"]
        or sum(r["role"] == "atom" for r in rows) != 24
    ):
        raise ValueError("Full 24+384 design required")
    proofs = token_proofs(rows, tokenizer)
    for directory in ("00/code", "01/units", "03"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    write(out / "00/rows.json", rows)
    write(out / "00/token_proofs.json", proofs)
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM55",
            "spec": spec,
            "code": code,
            "alphabet": prior["alphabet"],
            "assets": {
                n: digest(out / n) for n in ("00/rows.json", "00/token_proofs.json")
            },
            "parent_files": {
                n: digest(ROOT / spec["parent"] / n)
                for n in ("00/manifest.json", "verification.json", "01/observer.json")
            },
            "model_config_sha256": digest(MODEL / "config.json"),
            "model_weights_not_separately_hashed": True,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "independent_human_units": 0,
            "fit_or_patch": False,
            "unique_body_count": len({r["body"] for r in rows}),
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
            "in_flight": None,
        },
    )
    print(
        "PREPARED:24 atoms+384 accounts,48 exact occasion-only token pairs", flush=True
    )


def check(out):
    """Verify frozen material/code and the retained parent reader."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen material drift " + name)
    for name, sha in m["parent_files"].items():
        if digest(ROOT / m["spec"]["parent"] / name) != sha:
            raise ValueError("Parent drift " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Observer configuration drift")
    return m


def collect(out):
    """Acquire once, persisting attempted calls before each actual native forward."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("No automatic retry of stopped or uncertain forwards")
    rows = read_json(out / "00/rows.json")
    existing = list((out / "01/units").glob("unit_*.json"))
    if len(existing) != status["completed_units"]:
        raise RuntimeError("Saved unit/status mismatch requires inspection")
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        old = read_json(ROOT / m["spec"]["parent"] / "01/observer.json")
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
            if actual[k] != old[k]:
                raise ValueError("Observer drift " + k)
        write(out / "01/observer.json", actual)
        for row in rows:
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                meta, _ = load_arrays(path)
                if any(meta[k] != row[k] for k in FIELDS):
                    raise ValueError("Saved identity mismatch")
                continue
            elapsed = previous + time.monotonic() - started
            if not call_allowed(status, m["spec"], elapsed):
                raise TimeoutError("408-forward/600-second acquisition bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(
                observer, row, m["alphabet"], m["spec"]["layer"]
            )
            status["forward_calls"] += 1
            save(
                path,
                {k: row[k] for k in FIELDS}
                | {
                    "native": native,
                    "finished_epoch": time.time(),
                },
                arrays,
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 48 == 0:
                print(
                    f"ACTUAL:{status['completed_units']}/408 passive forwards",
                    flush=True,
                )
        if (
            any(
                status[k] != m["spec"]["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > m["spec"]["max_model_seconds"]
        ):
            raise ValueError("Incomplete or over-budget acquisition")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Measure event/mention effects; never force native reading to obey a profile."""
    m = check(out)
    status = read_json(out / "run_status.json")
    if status["status"] != "COLLECTED":
        raise ValueError("Complete actual acquisition required")
    rows = read_json(out / "00/rows.json")
    values, states, atoms = {}, {}, {}
    for r in rows:
        meta, a = load_arrays(out / f"01/units/unit_{r['unit']:04d}.json")
        if any(meta[k] != r[k] for k in FIELDS):
            raise ValueError("Capture identity mismatch")
        values[r["unit"]] = readout(meta["native"])
        states[r["unit"]] = a["full"][-1].astype(float)
        if r["role"] == "atom":
            atoms[r["target"], r["family"], r["polarity"]] = values[r["unit"]]["score"]
    native, channels = [], {}
    for r in rows:
        atom = {
            (f, p): atoms[r["target"], f, p]
            for f, p in itertools.product((0, 1), repeat=2)
        }
        channel = M.channels(r["history"], atom)
        channels[r["unit"]] = channel
        native.append(
            {k: r[k] for k in FIELDS}
            | values[r["unit"]]
            | {
                k: channel[k]
                for k in (
                    "event_profile",
                    "mention_profile",
                    "own_events",
                    "own_mentions",
                )
            }
            | {"tokens": len(r["input_ids"])}
        )
    index = {
        (r["target"], r["family"], r["polarity"], r["order"], r["regime"]): r
        for r in rows
        if r["role"] == "account"
    }
    pairs, vectors, identities, orders = [], [], [], []
    for t, f, p, order in itertools.product(
        m["spec"]["targets"], (0, 1), (0, 1), (0, 1)
    ):
        orientation = np.sign(atoms[t, f, p] - atoms[t, 1 - f, 1 - p])
        for name, hi, lo in M.CONTRASTS:
            a, b = index[t, f, p, order, hi], index[t, f, p, order, lo]
            effect = {
                k: values[a["unit"]][k] - values[b["unit"]][k]
                for k in ("score", "log_odds")
            }
            for key in ("event_profile", "mention_profile"):
                x, y = channels[a["unit"]][key], channels[b["unit"]][key]
                effect[key] = None if x is None or y is None else x - y
            pairs.append(
                {
                    "target": t,
                    "family": f,
                    "polarity": p,
                    "order": order,
                    "contrast": name,
                    "hi": a["unit"],
                    "lo": b["unit"],
                    **effect,
                    "oriented_score": float(effect["score"] * orientation)
                    if orientation
                    else None,
                }
            )
            expected = 0.0
            if name in ("distinct_minus_balanced", "distinct_minus_alias"):
                expected = 0.25 * (atoms[t, f, p] - atoms[t, 1 - f, 1 - p])
            if name in (
                "alias_minus_balanced",
                "other_minus_balanced",
                "revision_alias_minus_revision",
                "distinct_minus_balanced",
                "distinct_minus_alias",
            ):
                identities.append(
                    {
                        "target": t,
                        "family": f,
                        "polarity": p,
                        "order": order,
                        "contrast": name,
                        "event_error": effect["event_profile"] - expected,
                    }
                )
        a, b, c = (
            index[t, f, p, order, k]
            for k in ("balanced", "alias_focus", "distinct_focus")
        )
        vectors.append(
            {
                "target": t,
                "family": f,
                "polarity": p,
                "order": order,
                **M.displacement(
                    states[a["unit"]], states[b["unit"]], states[c["unit"]]
                ),
            }
        )
    for t, f, p, regime in itertools.product(
        m["spec"]["targets"], (0, 1), (0, 1), M.REGIMES
    ):
        a, b = (index[t, f, p, o, regime] for o in (0, 1))
        orders.append(
            {
                "target": t,
                "family": f,
                "polarity": p,
                "regime": regime,
                **{
                    k: values[b["unit"]][k] - values[a["unit"]][k]
                    for k in ("score", "log_odds")
                },
            }
        )
    groups = defaultdict(list)
    targets = defaultdict(list)
    for r in pairs:
        groups[r["contrast"]].append(r)
        targets[r["target"], r["contrast"]].append(r)
    summary_groups = [{"contrast": k, **M.summarize(v)} for k, v in groups.items()]
    tables = {
        "native.csv": native,
        "paired_effects.csv": pairs,
        "contrast_summary.csv": summary_groups,
        "target_contrast_summary.csv": [
            {"target": k[0], "contrast": k[1], **M.summarize(v)}
            for k, v in targets.items()
        ],
        "endpoint_displacements.csv": vectors,
        "interface_identities.csv": identities,
        "order_effects.csv": orders,
    }
    for name, table in tables.items():
        write_csv(out / "03" / name, table)
    write(
        out / "03/summary.json",
        {
            "packet": "HM55",
            "actual_forwards": status["forward_calls"],
            "model_seconds": status["model_seconds"],
            "single_event_references": 24,
            "whole_accounts": 384,
            "matched_occasion_only_pairs": 48,
            "groups": summary_groups,
            "interface_max_error": max(abs(r["event_error"]) for r in identities),
            "other_only_unobserved": sum(
                r["regime"] == "other_only" and r["event_profile"] is None
                for r in native
            ),
            "order_score_rms": float(
                np.sqrt(np.mean([r["score"] ** 2 for r in orders]))
            ),
            "minimum_candidate_mass": min(r["candidate_mass"] for r in native),
            "isolated_high_minus_low": [
                {"target": t, "family": f, "score": atoms[t, f, 1] - atoms[t, f, 0]}
                for t, f in itertools.product(m["spec"]["targets"], (0, 1))
            ],
            "independent_human_units": 0,
            "fit_patch_generation_calls": 0,
            "original_scope_complete": False,
        },
    )
    paths = list((out / "01").rglob("[!._]*.json")) + list(
        (out / "01").rglob("[!._]*.npz")
    )
    paths += list((out / "03").glob("[!._]*.csv")) + [out / "03/summary.json"]
    write(
        out / "03/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    print(
        "ANALYZED:384 paired effects,192 order effects,240 interface identities",
        flush=True,
    )


def main():
    """Run one explicit finite stage without restarting any historical queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "collect":
        with collector_guard(args.output):
            collect(args.output)
    else:
        {"prepare": prepare, "analyze": analyze}[args.stage](args.output)


if __name__ == "__main__":
    main()
