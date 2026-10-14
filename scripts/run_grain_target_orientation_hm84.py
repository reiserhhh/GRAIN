#!/usr/bin/env python3
"""HM84: finite orientation certificates and frozen new-wording consequences."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_cold_joint_author as J
from grain_research.construction import math_condition_order as C
from grain_research.construction import math_target_orientation as M
from scripts.run_grain_author_contrast_hm82 import load, seal, sha, table, write
from scripts.run_grain_condition_order_hm83 import OUT as HM83
from scripts.run_grain_condition_order_hm83 import SOURCE
from scripts.run_grain_condition_order_hm83 import check as check_hm83
from scripts.run_grain_math_construct_routing import write as checkpoint

OUT = ROOT / "results/grain_math_response_v1/target_orientation_r1"
CODE = (
    "docs/GRAIN_TARGET_ORIENTATION_PLAN_HM84.md",
    "grain_research/construction/math_target_orientation.py",
    "grain_research/construction/math_condition_order.py",
    "grain_research/construction/math_cold_joint_author.py",
    "grain_research/construction/math_author_contrast.py",
    "scripts/run_grain_target_orientation_hm84.py",
    "scripts/verify_grain_target_orientation_hm84.py",
    "tests/test_grain_math_target_orientation.py",
)


def source():
    """Bind480 source bodies to their frozen numeric design and observations."""
    rows = load(
        ROOT
        / "results/grain_math_response_v1/role_assignment_dealiasing_r1/00/rows.json"
    )
    docs = {}
    for r in rows:
        if r["family"] == 0:
            identity = {
                k: r[k] for k in ("doc_id", "focal", "companion", "coordinates")
            }
            if r["doc_id"] in docs and docs[r["doc_id"]] != identity:
                raise ValueError("source document design differs between queries")
            docs[r["doc_id"]] = identity
    metadata = [docs[i] for i in sorted(docs)]
    with np.load(SOURCE / "source.npz", allow_pickle=False) as a:
        design, p = a["design"], a["reference"]
        if len(metadata) != 480 or not np.array_equal(
            design, [r["coordinates"] for r in metadata]
        ):
            raise ValueError("source design ordering mismatch")
        marginal, joint, c = (
            a["marginal10_features"],
            a["joint49_features"],
            a["conditions"],
        )
        arms = C.matrices(marginal[:, :4], marginal[:, 4:], joint, c)
    return metadata, arms, p


def prepare():
    """Freeze one finite roster and source coefficients, never using new outcomes."""
    start = time.monotonic()
    if OUT.exists():
        raise FileExistsError("preserve existing HM84")
    if load(HM83 / "verification.json")["status"] != "VERIFIED_FINITE_HM83_NOT_GLOBAL":
        raise ValueError("verified HM83 required")
    check_hm83(HM83)
    OUT.mkdir()
    (OUT / "00/code").mkdir(parents=True)
    for name in CODE:
        path = OUT / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    checkpoint(
        OUT / "checkpoint.json",
        {"status": "CALIBRATING_SOURCE", "finished_cases": 0, "new_calls": 0},
    )
    metadata, arms, p = source()
    inherited = load(HM83 / "00/bundle.json")
    observation, functions, certificates, summaries, witnesses = (
        inherited["observation"],
        {},
        {},
        [],
        [],
    )
    arrays, addresses = {}, None
    for arm, b in arms.items():
        d, addressed = M.contrast_rows(b, metadata)
        if addresses is not None and addresses != addressed:
            raise ValueError("different source contrast identities")
        addresses = addressed
        arrays[arm] = b
        arrays[arm + "_D"] = d
        for scope in M.SCOPES:
            for target in ("common", *J.TARGETS):
                if time.monotonic() - start > 300:
                    raise TimeoutError("finite source analysis budget exceeded")
                key = f"{arm}__{scope}__{target}"
                ix = M.scope_indices(addresses, scope, target)
                certificate = M.orientation(d[ix])
                certificates[key] = certificate | {"indices": ix.tolist()}
                w = certificate["w"]
                functions[key] = {
                    "arm": arm,
                    "scope": scope,
                    "target": target,
                    "status": certificate["status"],
                    "w": w,
                    "offset": None if w is None else float(-(p @ b) @ w),
                }
                summaries.append(
                    {
                        "arm": arm,
                        "scope": scope,
                        "target": target,
                        "rows": len(ix),
                        "status": certificate["status"],
                        "L1": certificate["L1"],
                        "source_min_margin": certificate["minimum_source_margin"],
                        "certificate_error": certificate["certificate_error"],
                    }
                )
                if w is None:
                    for j, mass in enumerate(certificate["dual"]):
                        if mass > 0:
                            witnesses.append(
                                {
                                    "function": key,
                                    "source_contrast": int(ix[j]),
                                    "mass": mass,
                                    **addresses[ix[j]],
                                }
                            )
                checkpoint(
                    OUT / "checkpoint.json",
                    {
                        "status": "CALIBRATING_SOURCE",
                        "finished_cases": len(certificates),
                        "new_calls": 0,
                    },
                )
            pooled = inherited["functions"][arm]
            key = f"{arm}__{scope}__inherited_pooled_x"
            functions[key] = {
                "arm": arm,
                "scope": scope,
                "target": "inherited_pooled_x",
                "status": "INHERITED_POOLED_CALIBRATION",
                "w": np.asarray(pooled["beta"])[:, 1].tolist(),
                "offset": pooled["offset"][1],
                "not_orientation_certificate": True,
            }
    np.savez_compressed(OUT / "00/source_arrays.npz", **arrays)
    write(OUT / "00/addresses.json", addresses)
    write(OUT / "00/certificates.json", certificates)
    write(
        OUT / "00/bundle.json",
        {
            "packet": "HM84",
            "observation": observation,
            "functions": functions,
            "not_psychological_calibration": True,
        },
    )
    table(OUT / "00/cone_summary.csv", summaries)
    if witnesses:
        table(OUT / "00/cancellation_witnesses.csv", witnesses)
    inputs = [
        SOURCE / "source.npz",
        SOURCE / "source_freeze.json",
        HM83 / "verification.json",
        HM83 / "00/bundle.json",
        HM83 / "00/application.json",
        HM83 / "00/manifest.json",
        HM83 / "01/seal.json",
        HM83 / "03/seal.json",
        HM83 / "03/terminal_arrays.npz",
        ROOT
        / "results/grain_math_response_v1/role_assignment_dealiasing_r1/00/rows.json",
    ]
    write(
        OUT / "00/manifest.json",
        {
            "packet": "HM84",
            "retrospective_known_application": True,
            "code": {name: sha(ROOT / name) for name in CODE},
            "inputs": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
            "assets": seal(
                OUT,
                [
                    p
                    for p in (OUT / "00").iterdir()
                    if p.is_file() and not p.name.startswith("._")
                ],
            ),
            "source_LP_cases": len(certificates),
            "source_seconds": time.monotonic() - start,
            "new_model_calls": 0,
            "new_neural_or_lexical_or_teacher_fits": 0,
            "prior_actual_readings_reused": [2880, 720],
            "original_scope_complete": False,
        },
    )
    checkpoint(
        OUT / "checkpoint.json",
        {"status": "PREPARED", "finished_cases": len(certificates), "new_calls": 0},
    )
    print(
        "HM84 prepared:56 source LPs; known720 new-wording readings; no new calls",
        flush=True,
    )


def check():
    """Check new source code and sealed numeric inputs, without new model reads."""
    m = load(OUT / "00/manifest.json")
    for name, digest in (m["code"] | m["inputs"]).items():
        if sha(ROOT / name) != digest:
            raise ValueError("source/code drift:" + name)
    for name, digest in m["code"].items():
        if sha(OUT / "00/code" / name) != digest:
            raise ValueError("snapshot drift:" + name)
    for name, digest in m["assets"].items():
        if sha(OUT / name) != digest:
            raise ValueError("source result drift:" + name)
    return m


def cold(args):
    """Run a four-file score process; no target identity or application truth."""
    allowed = [
        str(p.resolve())
        for p in (args.bundle, args.observations, args.scores, args.audit)
    ]
    opened = []

    def guard(event, payload):
        if event == "open" and isinstance(payload[0], (str, bytes)):
            path = str(Path(payload[0]).resolve())
            if path not in allowed:
                raise PermissionError("undeclared cold read:" + path)
            if path not in opened:
                opened.append(path)

    sys.addaudithook(guard)
    bundle = load(args.bundle)
    records = [json.loads(s) for s in args.observations.read_text().splitlines()]
    with args.scores.open("x") as stream:
        for record in records:
            stream.write(
                json.dumps(M.cold_score(bundle, record), allow_nan=False) + "\n"
            )
    with args.audit.open("x") as stream:
        json.dump({"allowed": allowed, "opened": opened}, stream, indent=2)
    if set(opened) != set(allowed):
        raise ValueError("cold file roster mismatch")


def analyze():
    """Evaluate all frozen functions on known held-wording observations and laws."""
    manifest = check()
    for name, digest in load(HM83 / "03/seal.json")["files"].items():
        if sha(HM83 / name) != digest:
            raise ValueError("HM83 application output drift")
    new = OUT / "01"
    new.mkdir()
    metadata = load(HM83 / "00/application.json")
    with np.load(HM83 / "03/terminal_arrays.npz") as arrays:
        grain, query = arrays["grain"], arrays["query"]
    bundle = load(OUT / "00/bundle.json")
    records = [
        {
            "event_id": r["event_id"],
            "observer_id": bundle["observation"]["observer_id"],
            "grain": g.tolist(),
            "query": q.tolist(),
            "condition": r["c"],
        }
        for r, g, q in zip(metadata, grain, query, strict=True)
    ]
    records.append(
        records[0] | {"event_id": "unknown_condition_control", "condition": None}
    )
    with (new / "observations.jsonl").open("x") as stream:
        for r in records:
            stream.write(json.dumps(r) + "\n")
    proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "cold",
            "--bundle",
            str(OUT / "00/bundle.json"),
            "--observations",
            str(new / "observations.jsonl"),
            "--scores",
            str(new / "scores.jsonl"),
            "--audit",
            str(new / "cold_audit.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    write(
        new / "cold_process.json",
        {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
    )
    if proc.returncode:
        raise RuntimeError("cold process failed; preserve partial outputs")
    outputs = [json.loads(s) for s in (new / "scores.jsonl").read_text().splitlines()]
    cells = {
        (r["focal"], r["companion"], r["x"], r["c"]): i for i, r in enumerate(metadata)
    }
    if len(cells) != 120 or len(outputs) != 121:
        raise ValueError("application120 cells plus unknown-condition control required")
    actions, companion_actions, target_rows, author_rows = [], [], [], []
    for key, fn in bundle["functions"].items():
        values = [r["functions"][key]["value"] for r in outputs[:120]]
        for focal in J.TARGETS:
            differences = []
            for companion in J.TARGETS:
                if companion == focal:
                    continue
                for c in (-1, 1):
                    l, h = cells[focal, companion, -1, c], cells[focal, companion, 1, c]
                    action = None if values[l] is None else values[h] - values[l]
                    differences.append(action)
                    actions.append(
                        {
                            "function": key,
                            "focal": focal,
                            "companion": companion,
                            "c": c,
                            "low": l,
                            "high": h,
                            "action": action,
                        }
                    )
            valid = differences[0] is not None
            mean = float(np.mean(differences)) if valid else None
            target_rows.append(
                {
                    "function": key,
                    "focal": focal,
                    "mean_action": mean,
                    "positive": int(np.sum(np.array(differences) > 0))
                    if valid
                    else None,
                    "total": len(differences),
                    "status": fn["status"],
                }
            )
            p, q = M.author_laws(metadata, focal)
            observed = float((p - q) @ values) if valid else None
            expected = mean / 12 if valid else None
            if valid and abs(observed - expected) > 1e-9 * (1 + abs(expected)):
                raise ArithmeticError("author law/action identity failed")
            author_rows.append(
                {
                    "function": key,
                    "changed_evidence_target": focal,
                    "author_score_contrast": observed,
                    "from_mean_action": expected,
                    "identity_error": abs(observed - expected) if valid else None,
                }
            )
            for companion in J.TARGETS:
                if companion == focal:
                    continue
                for x in (-1, 1):
                    l, h = cells[focal, companion, x, -1], cells[focal, companion, x, 1]
                    action = None if values[l] is None else values[h] - values[l]
                    companion_actions.append(
                        {
                            "function": key,
                            "focal": focal,
                            "companion": companion,
                            "x": x,
                            "low": l,
                            "high": h,
                            "action": action,
                        }
                    )
    table(new / "new_focal_actions.csv", actions)
    table(new / "new_other_person_companion_actions.csv", companion_actions)
    table(new / "target_action_summary.csv", target_rows)
    table(new / "author_composition.csv", author_rows)
    write(
        new / "summary.json",
        {
            "status": "FINITE_HM84_EXECUTED_AWAITING_VERIFICATION",
            "source_LP_cases": manifest["source_LP_cases"],
            "new_calls": 0,
            "known_application_bodies_reads": [120, 720],
            "score_functions": len(bundle["functions"]),
            "actions": len(actions),
            "author_comparisons": len(author_rows),
            "not_psychological_accuracy": True,
            "original_scope_complete": False,
        },
    )
    write(
        new / "seal.json",
        {
            "files": seal(
                OUT,
                [
                    p
                    for p in new.iterdir()
                    if p.is_file() and not p.name.startswith("._")
                ],
            )
        },
    )
    checkpoint(
        OUT / "checkpoint.json",
        {
            "status": "ANALYZED_AWAITING_VERIFICATION",
            "new_calls": 0,
            "active_processes": [],
        },
    )
    print(
        "HM84 all frozen functions/actions/author laws evaluated; no new calls",
        flush=True,
    )


def main():
    """Dispatch finite stages; failures remain resumable and never auto-retry."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "analyze", "cold"))
    for name in ("bundle", "observations", "scores", "audit"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=1):
            if args.mode == "prepare":
                prepare()
            elif args.mode == "analyze":
                analyze()
            else:
                cold(args)
    except Exception as exc:
        if args.mode != "cold" and OUT.exists():
            checkpoint(
                OUT / "checkpoint.json",
                {
                    "status": "FAILED",
                    "stage": args.mode,
                    "error": repr(exc),
                    "new_calls": 0,
                    "active_processes": [],
                },
            )
        raise


if __name__ == "__main__":
    main()
