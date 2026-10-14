#!/usr/bin/env python3
"""Nonpromoting HM42 partial-source audit, with zero new neural calls."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

STARTED = time.monotonic()

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_route_action_audit as A
from grain_research.construction import math_text_query_route_cut as R
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_csv, write_json
from scripts.run_grain_math_target_active_support import load, save
from scripts.verify_grain_math_text_query_route_cut import moments, ordered_routes

PARENT = ROOT / "results/grain_math_response_v1/text_query_route_cut_r1"
OUT = ROOT / "results/grain_math_response_v1/text_query_route_partial_audit_r1"


def main():
    """Audit the actual prefix only; never relabel it as all96 or open application."""
    if OUT.exists():
        raise FileExistsError("preserve previous partial diagnostic")
    status = read_json(PARENT / "run_status.json")
    if (
        status["status"] != "STOPPED_FAILURE_OR_BUDGET"
        or status["committed_nodes"] != 32
        or status["in_flight"]["path"] != 32
    ):
        raise ValueError("this audit is fixed to the actual stopped33-source prefix")
    if (PARENT / "01/source_barrier.json").exists():
        raise ValueError("no full source acquisition occurred")
    for kind, count in (("forward", 33), ("first", 1320), ("second", 0)):
        if status[f"{kind}_attempted"] != count or status[f"{kind}_completed"] != count:
            raise ValueError("actual completed finite calls required")
    m = check(PARENT)
    design = read_json(PARENT / "00/design.json")
    paths = design[:33]
    if any(p["role"] != "source" for p in paths):
        raise ValueError("source-only prefix required")
    files = dict(m["files"])
    for p in PARENT.rglob("*"):
        if p.is_file() and not p.name.startswith("._"):
            files[str(p.relative_to(ROOT))] = digest(p)
    for name in (
        "grain_research/construction/math_route_action_audit.py",
        "scripts/audit_grain_math_partial_route_cut.py",
        "tests/test_grain_math_route_action_audit.py",
    ):
        files[name] = digest(ROOT / name)
    OUT.mkdir(parents=True)
    write_json(
        OUT / "manifest.json",
        {
            "files": files,
            "frozen_at": now(),
            "source_paths": list(range(33)),
            "missing_paths": list(range(33, 96)),
            "new_model_calls": 0,
            "new_fits": 0,
            "scope": "result-after prefix diagnostic; original HM42 remains STOPPED, no full source/application acceptance",
        },
    )
    observations, errors, descriptors = [], [], []
    maximum_alternate, maximum_physical = 0.0, 0.0
    bound_violation, output_replays, values_checked = 0.0, 0, 0
    try:
        for path in paths:
            file = f"path_{path['path']:03d}_node_1.json"
            meta, a = load(PARENT / "01/nodes" / file)
            parent, old = load(ROOT / m["spec"]["parent"] / "01/nodes" / file)
            if (
                meta["address"] != path
                or parent["address"] != path
                or meta["node"] != 1
                or meta["t"] != 0.5
                or meta["source_sha256"] is not None
                or max(meta["audit"].values()) != 0
            ):
                raise ValueError("changed source midpoint")
            masks = R.route_masks(path["left"])
            np.testing.assert_array_equal(masks, a["masks"])
            np.testing.assert_array_equal(a["logits"], old["logits"])
            if meta["score"] != parent["score"] or meta["native"] != parent["native"]:
                raise ValueError("actual forward output changed")
            output_replays += 1
            routes = R.link_routes(
                a["route_jacobian"], old["delta_h"], old["hessian"], old["p_jacobian"]
            )
            alternate = ordered_routes(
                a["route_jacobian"].astype(float),
                old["delta_h"].astype(float),
                old["hessian"].astype(float),
                old["p_jacobian"].astype(float),
            )
            jac_error = a["route_jacobian"].astype(float).sum(0) - old["h_jacobian"]
            witness = A.action_error_witness(
                jac_error,
                old["delta_h"],
                old["hessian"],
                old["p_jacobian"],
                old["delta_p"],
            )
            dual = A.dual_route_actions(
                a["route_jacobian"],
                old["delta_h"],
                old["hessian"],
                old["p_jacobian"],
                old["delta_p"],
            )
            primal = np.einsum("gapd,pd->ga", routes, old["delta_p"])
            covariance_error = routes.sum(0) - old["link"]
            physical_error = np.array(
                [np.sum(v * old["delta_p"]) for v in covariance_error]
            )
            alt_error = max(
                float(np.max(abs(routes - alternate))),
                float(np.max(abs(dual - primal))),
                float(np.max(abs(physical_error - witness["signed_error"]))),
            )
            if alt_error > 1e-8:
                raise ValueError("alternate physical contraction mismatch")
            maximum_alternate = max(maximum_alternate, alt_error)
            maximum_physical = max(maximum_physical, float(np.max(abs(physical_error))))
            values_checked += routes.size + dual.size + physical_error.size
            violation = float(
                np.max(abs(physical_error) - witness["directional_bound"])
            )
            bound_violation = max(bound_violation, violation)
            if violation > 1e-12:
                raise ValueError("error bound violated")
            rows, residual = R.action_rows(
                path, routes, old, m["spec"]["groups"], m["spec"]["arms"]
            )
            observations.extend(rows)
            save(
                OUT / f"path_{path['path']:03d}.json",
                {
                    "address": path,
                    "original_replay_valid": meta["replay_valid"],
                    "not_promoted": True,
                },
                {
                    "routes": routes,
                    "cut_roundoff": residual,
                    "jacobian_error": jac_error,
                    "dual_weights": witness["dual_weights"],
                    "physical_error": physical_error,
                    "directional_bound": witness["directional_bound"],
                    "frobenius_bound": witness["frobenius_bound"],
                },
            )
            for ai, arm in enumerate(m["spec"]["arms"]):
                errors.append(
                    {
                        "path": path["path"],
                        "query": path["query"],
                        "arm": arm,
                        "original_replay_valid": meta["replay_valid"],
                        "jacobian_max_error": float(np.max(abs(jac_error))),
                        "covector_max_error": float(np.max(abs(covariance_error[ai]))),
                        "signed_physical_error": float(physical_error[ai]),
                        "directional_bound": float(witness["directional_bound"][ai]),
                        "frobenius_bound": float(witness["frobenius_bound"][ai]),
                        "alternate_error": alt_error,
                    }
                )
            for gi, g in enumerate(R.GROUPS):
                descriptors.append(
                    {
                        "path": path["path"],
                        "query": path["query"],
                        "group": g,
                        "positions": int(masks[gi].sum()),
                        "jacobian_norm": float(
                            np.linalg.norm(a["route_jacobian"][gi].astype(float))
                        ),
                        "cut_gradient_norm": float(
                            np.linalg.norm(
                                a["cut_gradients"][:, masks[gi]].astype(float)
                            )
                        ),
                    }
                )
        paired, metrics = [], []
        # Trust has only one point: preserve it, never manufacture a balanced mean/contrast.
        for query, arm in itertools.product(
            m["spec"]["queries"][:2], m["spec"]["arms"]
        ):
            words = [r for r in observations if r["query"] == query and r["arm"] == arm]
            if len(words) != 16:
                raise ValueError("complete original16-source query prefix required")
            pairs = []
            for cell in range(8):
                pairs.extend(
                    R.pair_rows(
                        sorted(
                            [r for r in words if r["cell"] == cell],
                            key=lambda r: r["wording"],
                        ),
                        [0, 2],
                    )
                )
            paired.extend(pairs)
            for measure in ("word", "mean", "contrast"):
                selected = (
                    words
                    if measure == "word"
                    else [r for r in pairs if r["measure"] == measure]
                )
                primary, alternate = R.moments(selected), moments(selected)
                error = max(
                    abs(float(primary[k]) - float(alternate[k])) for k in primary
                )
                if error > 1e-8:
                    raise ValueError("alternate signed moment mismatch")
                maximum_alternate = max(maximum_alternate, error)
                values_checked += len(primary)
                metrics.append(
                    {
                        "query": query,
                        "role": "source",
                        "arm": arm,
                        "measure": measure,
                        **primary,
                    }
                )
        write_csv(OUT / "observations.csv", observations)
        write_csv(OUT / "paired_actions.csv", paired)
        write_csv(OUT / "action_metrics.csv", metrics)
        write_csv(OUT / "error_witnesses.csv", errors)
        write_csv(OUT / "route_descriptors.csv", descriptors)
        with (PARENT / "01/source_calls.jsonl").open() as stream:
            calls = [json.loads(line) for line in stream]
        if len(calls) != 33 * 82:
            raise ValueError("incomplete actual call journal")
        counter = {"first": 0, "forward": 0}
        for i in range(33):
            block = calls[i * 82 : (i + 1) * 82]
            for j in range(41):
                before, after = block[j * 2 : j * 2 + 2]
                kind = "forward" if j == 0 else "first"
                counter[kind] += 1
                if (
                    before["path"],
                    after["path"],
                    before["kind"],
                    after["kind"],
                    before["phase"],
                    after["phase"],
                    before["number"],
                    after["number"],
                ) != (
                    i,
                    i,
                    kind,
                    kind,
                    "attempted",
                    "completed",
                    counter[kind],
                    counter[kind],
                ):
                    raise ValueError("changed actual derivative lifecycle")
        elapsed = time.monotonic() - STARTED
        # Include the exploratory numeric readback above with a conservative two-second reserve.
        combined = status["analysis_seconds"] + elapsed + 2.0
        if combined > 180:
            raise TimeoutError("original cumulative CPU budget exceeded")
        summary = {
            "status": "VERIFIED_NONPROMOTING_PARTIAL_SOURCE_HM42",
            "original_packet_status": status["status"],
            "source_captures": 33,
            "committed_source_nodes": 32,
            "failed_saved_node": 32,
            "missing_source_nodes": 15,
            "application_captures": 0,
            "missing_application_nodes": 48,
            "forward_calls": 33,
            "first_gradient_calls": 1320,
            "second_gradient_calls": 0,
            "model_seconds": status["model_seconds"],
            "cpu_seconds": elapsed,
            "readback_reserve_seconds": 2,
            "combined_cpu_seconds": combined,
            "maximum_analysis_seconds": 180,
            "maximum_coordinate_error": max(r["jacobian_max_error"] for r in errors),
            "maximum_covector_error": max(r["covector_max_error"] for r in errors),
            "maximum_physical_action_error": maximum_physical,
            "maximum_alternate_error": maximum_alternate,
            "maximum_bound_violation": bound_violation,
            "output_replays": output_replays,
            "numeric_values_checked": values_checked,
            "ancestry_and_prefix_hashes": len(files),
            "observations": len(observations),
            "paired_actions": len(paired),
            "metric_groups": len(metrics),
            "new_model_calls": 0,
            "new_fits": 0,
            "source_barrier_exists": False,
            "HM42_promoted": False,
            "scope": "same-agent result-after audit of32 accepted+1 failed source midpoints; no full-Jacobian or application acceptance",
        }
        write_json(OUT / "summary.json", summary)
        write_json(
            OUT / "artifact_hashes.json",
            {
                str(p.relative_to(OUT)): digest(p)
                for p in OUT.rglob("*")
                if p.is_file()
                and not p.name.startswith("._")
                and p.name != "artifact_hashes.json"
            },
        )
        print(summary, flush=True)
    except Exception as exc:
        write_json(
            OUT / "failure.json",
            {
                "error": repr(exc),
                "status": "PARTIAL_DIAGNOSTIC_FAILED",
                "primary_unchanged": True,
                "new_model_calls": 0,
            },
        )
        raise


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
