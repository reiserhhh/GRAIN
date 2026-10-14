#!/usr/bin/env python3
"""HM52: fixed source operations on fresh evidence classes, without accuracy search."""

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

from grain_research.construction import math_relational_operation_specificity as M
from grain_research.construction.math_relational_appraisal import render
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.construction.math_source_relational_transport import prove_transport
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_source_relational_transport import CODE as PARENT_CODE
from scripts.run_grain_math_source_relational_transport import check as parent_check
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_relational_operation_specificity_v1.json"
OUT = ROOT / "results/grain_math_response_v1/relational_operation_specificity_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_relational_operation_specificity_v1.json",
            "docs/GRAIN_RELATIONAL_OPERATION_SPECIFICITY_PLAN_V1.md",
            "grain_research/construction/math_relational_operation_specificity.py",
            "scripts/run_grain_math_relational_operation_specificity.py",
            "scripts/verify_grain_math_relational_operation_specificity.py",
            "tests/test_grain_math_relational_operation_specificity.py",
            *PARENT_CODE,
        )
    )
)


def prepare(out):
    """Seal new materials and fixed HM51 source assets before96+576 actual calls."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing result versions")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    prior = parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM51_NOT_GLOBAL"
        or spec["layer"] != prior["spec"]["layer"]
        or spec["targets"] != prior["spec"]["targets"]
        or spec["classes"] != list(M.CLASSES)
        or spec["arms"] != list(M.ARMS)
    ):
        raise ValueError("Verified unchanged source and full fixed scope required")
    hm50 = ROOT / prior["spec"]["parent"]
    source = {
        (e["content"], e["person"]): e
        for e in read_json(hm50 / "00/design.json")
        if e["kind"] == "capture" and e["role"] == "source" and e["direction"] == 0
    }
    old_rows = {r["unit"]: r for r in read_json(hm50 / "00/receivers.json")}
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    docs = M.documents(spec)
    doc_index = {(r["content"], r["class"], r["direction"]): r for r in docs}
    rows = []
    for target, kind, person, direction in itertools.product(
        spec["targets"], spec["classes"], ("A", "B"), (0, 1)
    ):
        doc = doc_index[target, kind, direction]
        rendered, encoded, _ = render(
            doc["body"], target, person, spec["system"], tokenizer
        )
        row = {
            **doc,
            "unit": len(rows),
            "query": target,
            "person": person,
            "rendered": rendered,
            "input_ids": list(encoded["input_ids"]),
        }
        row["partition"] = M.partition(row, encoded)
        src = source[target, person]
        row["source_mapping"] = prove_transport(
            old_rows[src["receiver"]], src, row, row
        )
        row["source_capture"] = src["unit"]
        rows.append(row)
    if len(rows) != spec["expected_receivers"] or len(source) != 12:
        raise ValueError("All96 receivers and12 source suffix templates required")
    index = {(r["content"], r["class"], r["person"], r["direction"]): r for r in rows}
    captures = []
    for r in rows:
        donor = index[r["content"], r["class"], r["person"], 1 - r["direction"]]
        captures.append(
            {
                **{
                    k: r[k]
                    for k in ("content", "class", "person", "direction", "partition")
                },
                "unit": r["unit"],
                "pair": r["unit"],
                "receiver": r["unit"],
                "donor": donor["unit"],
                "kind": "capture",
                "arm": "native",
                "mapping": M.prove_pair(r, donor),
            }
        )
    design = captures.copy()
    for e in captures:
        for arm in spec["arms"]:
            design.append({**e, "unit": len(design), "kind": "patch", "arm": arm})
    if len(design) != spec["max_forwards"] or len({r["rendered"] for r in rows}) != 96:
        raise ValueError("Incomplete or duplicated672 actual input/call design")
    for name in ("00/code", "01/units", "03"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write(out / "00/documents.json", docs)
    write(out / "00/receivers.json", rows)
    write(out / "00/design.json", design)
    (out / "00/source_vectors.npz").write_bytes(
        (parent / "00/source_vectors.npz").read_bytes()
    )
    write(
        out / "00/source_seal.json",
        {
            "before_all_new_calls": True,
            "new_source_forwards": 0,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_result": spec["parent"],
            "source_operator_unchanged": True,
        },
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
            "packet": "HM52",
            "spec": spec,
            "alphabet": prior["alphabet"],
            "code": code,
            "assets": {
                n: digest(out / n)
                for n in (
                    "00/documents.json",
                    "00/receivers.json",
                    "00/design.json",
                    "00/source_vectors.npz",
                    "00/source_seal.json",
                )
            },
            "parent_files": {
                n: digest(parent / n)
                for n in (
                    "00/manifest.json",
                    "00/source_vectors.npz",
                    "verification.json",
                    "01/observer.json",
                )
            },
            "model_config_sha256": digest(MODEL / "config.json"),
            "model_weights_not_separately_hashed": True,
            "new_controlled_family": True,
            "independent_human_units": 0,
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
    print("PREPARED:96 new receivers; frozen source;672 calls maximum", flush=True)


def check(out):
    """Check complete frozen material/program/source provenance, not just output names."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Changed frozen code " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed frozen asset " + name)
    parent = ROOT / m["spec"]["parent"]
    parent_check(parent)
    for name, sha in m["parent_files"].items():
        if digest(parent / name) != sha:
            raise ValueError("Changed source " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Observer config drift")
    return m


def vectors(out):
    """Read only the frozen source operator arrays."""
    with np.load(out / "00/source_vectors.npz", allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def collect(out):
    """Capture all96 states then576 patches; commit each attempt before model execution."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["in_flight"] is not None or status["status"] == "STOPPED":
        raise RuntimeError("Uncertain/stopped acquisition; no automatic retry")
    spec, source = m["spec"], vectors(out)
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    design = read_json(out / "00/design.json")
    states = {}
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        old = read_json(ROOT / spec["parent"] / "01/observer.json")
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
        for e in design:
            path = out / f"01/units/unit_{e['unit']:04d}.json"
            if path.exists():
                _, a = load_arrays(path)
                if e["kind"] == "capture":
                    states[e["receiver"]] = a["full"]
                continue
            elapsed = previous + time.monotonic() - started
            if not call_allowed(status, spec, elapsed):
                raise TimeoutError("Finite672call/600second budget reached")
            patch = None
            if e["kind"] == "patch":
                sectors = {
                    k: source[f"{e['content']}__{k}"]
                    for k in ("A", "B", "shared", "bound")
                }
                patch = M.patch_values(
                    e,
                    states[e["receiver"]],
                    states[e["donor"]],
                    sectors,
                    spec["random_seed"],
                )
            status.update(
                status="COLLECTING", in_flight=e["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(
                observer, rows[e["receiver"]], m["alphabet"], spec["layer"], patch
            )
            status["forward_calls"] += 1
            if patch is None:
                states[e["receiver"]] = arrays["full"]
            else:
                arrays.update(
                    patch_positions=np.asarray(patch[0], int), patch_values=patch[1]
                )
            save(
                path,
                {
                    k: e[k]
                    for k in (
                        "unit",
                        "pair",
                        "receiver",
                        "donor",
                        "kind",
                        "arm",
                        "content",
                        "class",
                        "person",
                        "direction",
                    )
                }
                | {"native": native},
                arrays,
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 96 == 0:
                print(f"ACTUAL:{status['completed_units']}/672 forwards", flush=True)
        if (
            any(
                status[k] != spec["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > spec["max_model_seconds"]
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
    """Keep all new evidence-class actions, log-odds and paired context contrasts."""
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Actual complete acquisition required")
    design = read_json(out / "00/design.json")
    captured = {}
    for e in design[:96]:
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        captured[e["receiver"]] = (meta["native"], a["full"])
    effects, native_rows, max_pre, max_noop = [], [], 0.0, 0.0
    for e in design:
        meta, a = load_arrays(out / f"01/units/unit_{e['unit']:04d}.json")
        current, state = captured[e["receiver"]]
        donor, _ = captured[e["donor"]]
        fields = {
            k: e[k]
            for k in ("unit", "pair", "content", "class", "person", "direction", "arm")
        }
        if e["kind"] == "capture":
            native_rows.append(
                fields
                | {
                    "native": current["conditional_score"],
                    "candidate_mass": current["candidate_mass"],
                    "native_log_odds": M.high_low_log_odds(current["candidate_logits"]),
                    "tokens": state.shape[0],
                }
            )
            continue
        y = meta["native"]["conditional_score"]
        native, full = current["conditional_score"], donor["conditional_score"]
        max_pre = max(
            max_pre, float(np.max(abs(a["before"].astype(float) - state.astype(float))))
        )
        if e["arm"] == "noop":
            max_noop = max(max_noop, abs(y - native))
        effects.append(
            fields
            | {
                "native": native,
                "donor": full,
                "patched": y,
                "candidate_mass": meta["native"]["candidate_mass"],
                "native_log_odds": M.high_low_log_odds(current["candidate_logits"]),
                "donor_log_odds": M.high_low_log_odds(donor["candidate_logits"]),
                "patch_log_odds": M.high_low_log_odds(
                    meta["native"]["candidate_logits"]
                ),
                "full_change": full - native,
                "patch_change": y - native,
                "remaining": full - y,
            }
        )
    groups, targets, people = defaultdict(list), defaultdict(list), defaultdict(list)
    index = {}
    for r in effects:
        groups[r["class"], r["arm"]].append(r)
        targets[r["content"], r["class"], r["arm"]].append(r)
        people[r["person"], r["class"], r["arm"]].append(r)
        index[r["content"], r["class"], r["person"], r["direction"], r["arm"]] = r
    contrasts = []
    for target, person, direction, arm, control in itertools.product(
        m["spec"]["targets"],
        ("A", "B"),
        (0, 1),
        m["spec"]["arms"],
        m["spec"]["classes"][1:],
    ):
        own = index[target, "own", person, direction, arm]
        other = index[target, control, person, direction, arm]
        score = own["patch_change"] - other["patch_change"]
        log = (own["patch_log_odds"] - own["native_log_odds"]) - (
            other["patch_log_odds"] - other["native_log_odds"]
        )
        eta = M.orientation(person, direction)
        contrasts.append(
            {
                "content": target,
                "person": person,
                "direction": direction,
                "arm": arm,
                "control": control,
                "score_action_difference": score,
                "log_odds_action_difference": log,
                "oriented_score_difference": eta * score,
                "oriented_log_odds_difference": eta * log,
            }
        )
    mixed = defaultdict(list)
    for r in contrasts:
        mixed[r["control"], r["arm"]].append(r)
    contrast_summary = [
        {
            "control": k[0],
            "arm": k[1],
            "rows": len(v),
            "oriented_score_difference": float(
                np.mean([r["oriented_score_difference"] for r in v])
            ),
            "oriented_log_odds_difference": float(
                np.mean([r["oriented_log_odds_difference"] for r in v])
            ),
            "score_difference_rms": float(
                np.sqrt(np.mean([r["score_action_difference"] ** 2 for r in v]))
            ),
            "log_odds_difference_rms": float(
                np.sqrt(np.mean([r["log_odds_action_difference"] ** 2 for r in v]))
            ),
        }
        for k, v in mixed.items()
    ]
    for name, data in (
        ("effects.csv", effects),
        ("native.csv", native_rows),
        (
            "class_summary.csv",
            [{"class": k[0], "arm": k[1], **M.summary(v)} for k, v in groups.items()],
        ),
        (
            "target_summary.csv",
            [
                {"content": k[0], "class": k[1], "arm": k[2], **M.summary(v)}
                for k, v in targets.items()
            ],
        ),
        (
            "person_summary.csv",
            [
                {"person": k[0], "class": k[1], "arm": k[2], **M.summary(v)}
                for k, v in people.items()
            ],
        ),
        ("class_contrasts.csv", contrasts),
        ("class_contrast_summary.csv", contrast_summary),
    ):
        write_csv(out / "03" / name, data)
    write(
        out / "03/summary.json",
        {
            "packet": "HM52",
            "native_captures": 96,
            "effects": len(effects),
            "class_contrasts": len(contrasts),
            "groups": [
                {"class": k[0], "arm": k[1], **M.summary(v)} for k, v in groups.items()
            ],
            "maximum_prepatch_difference": max_pre,
            "maximum_noop_error": max_noop,
            "minimum_candidate_mass": min(
                r["candidate_mass"] for r in effects + native_rows
            ),
            "new_source_calls": 0,
            "human_personality_accuracy": False,
            "original_scope_complete": False,
        },
    )
    paths = (
        [out / "01/observer.json"]
        + [
            out / f"01/units/unit_{e['unit']:04d}.{suffix}"
            for e in design
            for suffix in ("json", "npz")
        ]
        + list((out / "03").glob("[!._]*.csv"))
        + [out / "03/summary.json"]
    )
    write(
        out / "03/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    print("ANALYZED:576 actions/432 context contrasts; no model selected", flush=True)


def main():
    """Expose explicit bounded stages, without implicit acquisition or old queue recovery."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.output):
        {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](
            args.output
        )


if __name__ == "__main__":
    main()
