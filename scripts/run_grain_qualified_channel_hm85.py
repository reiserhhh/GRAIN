#!/usr/bin/env python3
"""HM85: fixed qualified-action capacity and retrospective full-family transport."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research import local_measure as L
from grain_research.construction import math_cold_joint_author as J
from grain_research.construction import math_qualified_channel as M
from scripts.run_grain_author_contrast_hm82 import load, seal, sha, table, write
from scripts.run_grain_math_construct_routing import write as checkpoint
from scripts.run_grain_target_orientation_hm84 import OUT as HM84
from scripts.run_grain_target_orientation_hm84 import check as check84
from scripts.run_grain_target_orientation_hm84 import source

OUT = ROOT / "results/grain_math_response_v1/qualified_channel_capacity_r1"
HM79 = ROOT / "results/grain_math_response_v1/role_assignment_dealiasing_r1"
CODE = (
    "docs/GRAIN_QUALIFIED_CHANNEL_CAPACITY_PLAN_HM85.md",
    "grain_research/construction/math_qualified_channel.py",
    "scripts/run_grain_qualified_channel_hm85.py",
    "scripts/verify_grain_qualified_channel_hm85.py",
    "tests/test_grain_math_qualified_channel.py",
)


def application_family():
    """Load every sealed family1 score/body by actual old/new capture identity."""
    rows = [r for r in load(HM79 / "00/rows.json") if r["family"] == 1]
    parent = ROOT / load(HM79 / "00/manifest.json")["spec"]["parent"]
    old_rows = {r["unit"]: r for r in load(parent / "00/rows.json")}
    seals = {
        str(HM79): load(HM79 / "01/raw_seal.json"),
        str(parent): load(parent / "01/raw_seal.json"),
    }
    docs, scores, seen, inputs = {}, {}, set(), {}
    for r in rows:
        old = r["old_unit"] is not None
        base, unit = (parent, r["old_unit"]) if old else (HM79, r["new_unit"])
        addressed = old_rows[unit] if old else r
        path = base / f"01/units/unit_{unit:04d}.json"
        files = seals[str(base)].get("files", seals[str(base)])
        digest = sha(path)
        if digest != files[str(path.relative_to(base))]:
            raise ValueError("application score seal drift")
        if old and any(
            r[k] != addressed[k]
            for k in ("family", "focal", "companion", "query", "body", "input_ids")
        ):
            raise ValueError("application reuse alias mismatch")
        captured = load(path)
        if captured["query"] != r["query"] or captured["family"] != 1:
            raise ValueError("application observation binding mismatch")
        identity = (r["doc_id"], J.TARGETS.index(r["query"]))
        if identity in seen:
            raise ValueError("duplicate query/document binding")
        seen.add(identity)
        metadata = {k: r[k] for k in ("doc_id", "focal", "companion", "coordinates")}
        doc = {"body": r["body"], **metadata}
        if r["doc_id"] in docs and docs[r["doc_id"]] != doc:
            raise ValueError("body/design differs across queries")
        docs[r["doc_id"]] = doc
        scores.setdefault(r["doc_id"], np.zeros(6))[identity[1]] = captured["native"][
            "conditional_score"
        ]
        inputs[str(path.relative_to(ROOT))] = digest
    if len(docs) != 480 or len(seen) != 2880:
        raise ValueError("complete480/2880 application family required")
    ordered = sorted(docs)
    return [docs[d] for d in ordered], np.array([scores[d] for d in ordered]), inputs


def run():
    """Calibrate24 source bounds then freeze before any application calculation."""
    start = time.monotonic()
    if OUT.exists():
        raise FileExistsError("preserve existing HM85")
    if load(HM84 / "verification.json")["status"] != "VERIFIED_FINITE_HM84_NOT_GLOBAL":
        raise ValueError("verified HM84 required")
    check84()
    OUT.mkdir()
    (OUT / "00/code").mkdir(parents=True)
    for name in CODE:
        dest = OUT / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    checkpoint(OUT / "checkpoint.json", {"status": "CALIBRATING", "new_calls": 0})
    metadata, old_spaces, _ = source()
    actor = [r["coordinates"][2] for r in metadata]
    matrices = M.spaces(old_spaces["marginal10"], old_spaces["joint49"], actor)
    funcs, summary, source_addresses, arrays = {}, [], {}, dict(matrices)
    for name, b in matrices.items():
        for target in J.TARGETS:
            if time.monotonic() - start > 300:
                raise TimeoutError("finite packet CPU budget reached")
            a, n, address = M.contrasts(b, metadata, target)
            result = M.capacity(a, n)
            key = f"{name}__{target}"
            funcs[key] = {"space": name, "target": target, **result}
            source_addresses[key] = address
            arrays[key + "_own"], arrays[key + "_foreign"] = a, n
            summary.append(
                {
                    "function": key,
                    "status": result["status"],
                    "foreign_bound": result.get("foreign_action_bound"),
                    "minimum_own_action": result.get("minimum_own_action"),
                    "coefficient_L1": result.get("coefficient_L1"),
                    "certificate_error": result.get("certificate_error"),
                }
            )
    bundle = {
        "packet": "HM85",
        "functions": funcs,
        "observation": load(HM84 / "00/bundle.json")["observation"],
        "owner_identity_is_design_oracle": True,
        "query_person": "A",
        "notebook_h_is_not_query_anchor": True,
    }
    write(OUT / "00/bundle.json", bundle)
    write(OUT / "00/addresses.json", source_addresses)
    np.savez_compressed(OUT / "00/source.npz", **arrays)
    table(OUT / "00/capacity.csv", summary)
    write(
        OUT / "00/freeze.json",
        {
            "code": {name: sha(ROOT / name) for name in CODE},
            "inputs": {
                str(p.relative_to(ROOT)): sha(p)
                for p in (HM84 / "00/manifest.json", HM84 / "verification.json")
            },
            "assets": seal(
                OUT,
                [p for p in (OUT / "00").iterdir() if p.is_file()],
            ),
            "source_calibrations": 24,
            "known_application_not_blind": True,
            "new_calls": 0,
        },
    )
    checkpoint(OUT / "checkpoint.json", {"status": "SOURCE_FROZEN", "new_calls": 0})
    application, query, inputs = application_family()
    grain = L.score_documents(
        bundle["observation"]["lexical"], [r["body"] for r in application]
    )["z"]
    joint, g, q = J.observed_features(bundle["observation"], grain, query)
    matrices = M.spaces(
        np.column_stack((g, q)), joint, [r["coordinates"][2] for r in application]
    )
    outputs = M.evaluate(funcs, matrices)
    new = OUT / "01"
    new.mkdir()
    np.savez_compressed(new / "contextual_observations.npz", **matrices)
    write(new / "scores.json", outputs["direct_channels"])
    write(new / "metadata.json", application)
    write(new / "input_bindings.json", inputs)
    actions, summaries = [], []
    for key, fn in funcs.items():
        _, _, addressed = M.contrasts(matrices[fn["space"]], application, fn["target"])
        values = outputs["direct_channels"][key]
        effect = {"own": [], "foreign": []}
        for a in addressed:
            action = None if values is None else values[a["high"]] - values[a["low"]]
            effect[a["kind"]].append(action)
            actions.append({"function": key, **a, "action": action})
        own, foreign = effect["own"], effect["foreign"]
        valid = values is not None
        summaries.append(
            {
                "function": key,
                "own_positive": int(np.sum(np.array(own) > 0)) if valid else None,
                "own_total": 60,
                "own_min": min(own) if valid else None,
                "own_mean": np.mean(own) if valid else None,
                "foreign_signed_mean": np.mean(foreign) if valid else None,
                "foreign_absolute_mean": np.mean(np.abs(foreign)) if valid else None,
                "foreign_max_absolute": max(np.abs(foreign)) if valid else None,
                "foreign_total": 20,
            }
        )
    table(new / "actions.csv", actions)
    table(new / "transport.csv", summaries)
    write(
        new / "summary.json",
        {
            "status": "EXECUTED_AWAITING_FINITE_VERIFICATION",
            "source_cases": 24,
            "known_application_bodies_previous_actual_reads": [480, 2880],
            "new_calls": 0,
            "source_calibrations": 24,
            "application_actions": len(actions),
            "contextual_observations_retained": True,
            "seconds": time.monotonic() - start,
            "not_psychological_accuracy": True,
            "original_scope_complete": False,
        },
    )
    write(
        new / "seal.json",
        {"files": seal(OUT, [p for p in new.iterdir() if p.is_file()])},
    )
    checkpoint(
        OUT / "checkpoint.json",
        {"status": "ANALYZED", "new_calls": 0, "active_processes": []},
    )
    print(
        "HM85 source24 LPs and known family1 full actions complete;0newcalls",
        flush=True,
    )


def main():
    """Run once; never overwrite frozen results or retry failed calibration."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        with threadpool_limits(limits=1):
            run()
    except Exception as exc:
        if OUT.exists():
            checkpoint(
                OUT / "checkpoint.json",
                {"status": "FAILED", "error": repr(exc), "new_calls": 0},
            )
        raise


if __name__ == "__main__":
    main()
