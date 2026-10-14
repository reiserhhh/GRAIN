#!/usr/bin/env python3
"""HM43 alternate scalar summation, frozen weight, lifecycle and table audit."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import signal
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.monotonic()

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction.math_dual_route_action import (
    COMPONENTS,
    FIELDS,
    GROUPS,
    LINK_COMPONENTS,
)
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_json
from scripts.run_grain_math_target_active_support import load
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/dual_route_action_r1"
DEST = ROOT / "results/grain_math_response_v1/dual_route_action_verification_r1"


def pairing(a, b):
    """Pair original units using compensated scalar products, not primary BLAS order."""
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite aligned physical pairing required")
    return math.fsum(float(v) * float(w) for v, w in zip(x, y, strict=True))


def errors(row):
    """Independently retain raw output and exact correction identities."""
    return {
        "full_sum_error": abs(math.fsum(row[k] for k in COMPONENTS) - row["full"]),
        "link_sum_error": abs(math.fsum(row[k] for k in LINK_COMPONENTS) - row["link"]),
    }


def moments(rows):
    """Compensated signed mean, square and all21 cross terms."""
    n, result = len(rows), {"count": len(rows)}
    for key in FIELDS:
        result[key + "_mean"] = math.fsum(r[key] for r in rows) / n
        result[key + "_ms"] = math.fsum(r[key] ** 2 for r in rows) / n
        result[key + "_rms"] = math.sqrt(result[key + "_ms"])
    for a, b in itertools.combinations(COMPONENTS, 2):
        result[a + "__" + b + "_cross"] = math.fsum(2 * r[a] * r[b] for r in rows) / n
    result.update({k: max(errors(r)[k] for r in rows) for k in errors(rows[0])})
    return result


def call_journal(out, design, barrier):
    """Require precisely one forward and four actual first calls per original node."""
    totals = {"forward": 0, "first": 0}
    for role in ("source", "application"):
        with (out / "01" / f"{role}_calls.jsonl").open() as stream:
            records = [json.loads(line) for line in stream]
        paths = [p["path"] for p in design if p["role"] == role]
        if len(records) != 48 * 10 or [r["path"] for r in records[::10]] != paths:
            raise ValueError("48 ordered actual nodes per role required")
        for i, path in enumerate(paths):
            block = records[i * 10 : (i + 1) * 10]
            if any(r["path"] != path for r in block):
                raise ValueError("mixed journal addresses")
            for j in range(5):
                a, b = block[2 * j : 2 * j + 2]
                kind = "forward" if j == 0 else "first"
                totals[kind] += 1
                if (
                    a["kind"],
                    b["kind"],
                    a["phase"],
                    b["phase"],
                    a["number"],
                    b["number"],
                ) != (kind, kind, "attempted", "completed", totals[kind], totals[kind]):
                    raise ValueError("actual call lifecycle mismatch")
                if (
                    a["at"] > b["at"]
                    or (role == "source" and b["at"] > barrier["frozen_at"])
                    or (role == "application" and a["at"] < barrier["frozen_at"])
                ):
                    raise ValueError("source-first chronology mismatch")
    if totals != {"forward": 96, "first": 384}:
        raise ValueError("expanded/incomplete calls")
    return sum(totals.values()) * 2


def verify(out, dest):
    """Audit all original96 acquisitions without any additional model invocation."""
    if dest.exists():
        raise FileExistsError("preserve verification")
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "EVALUATED"
        or status["in_flight"] is not None
        or status["committed_nodes"] != 96
        or status["new_fits"]
    ):
        raise ValueError("complete unfitted evaluated packet required")
    for kind, n in (("forward", 96), ("first", 384), ("second", 0)):
        if status[f"{kind}_attempted"] != n or status[f"{kind}_completed"] != n:
            raise ValueError("actual call count mismatch")
    if status["model_seconds"] > 600:
        raise TimeoutError("model cap exceeded")
    remaining = 180 - status["analysis_seconds"] - (time.monotonic() - PROCESS_STARTED)
    if remaining <= 0:
        raise TimeoutError("CPU cap exhausted")
    dest.mkdir(parents=True)
    handler = signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("fixed CPU cap"))
    )
    signal.setitimer(signal.ITIMER_REAL, remaining)
    maximum, checked = 0.0, 0

    def close(a, b):
        nonlocal maximum, checked
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("finite aligned replay required")
        e = float(np.max(abs(x - y))) if x.size else 0.0
        maximum, checked = max(maximum, e), checked + x.size
        if e > 1e-8:
            raise ValueError(f"original physical gate exceeded: {e}")

    def row_close(saved, calc):
        for key, value in calc.items():
            if isinstance(value, str):
                if saved[key] != value:
                    raise ValueError("changed identity " + key)
            else:
                close(float(saved[key]), value)

    try:
        m = check(out)
        spec, design = m["spec"], read_json(out / "00/design.json")
        if m["application_weights_prepared"] is not True:
            raise ValueError("must disclose inherited local application weights")
        ledger = read_json(out / "02/artifact_hashes.json")
        for name, sha in ledger.items():
            if digest(out / name) != sha:
                raise ValueError("changed output " + name)
        barrier = read_json(out / "01/source_barrier.json")
        if (
            barrier["source_nodes"],
            barrier["new_application_nodes_acquired"],
            barrier["application_weights_prepared"],
        ) != (48, 0, True):
            raise ValueError("source capture barrier/weight disclosure mismatch")
        for name, sha in barrier["files"].items():
            if digest(out / name) != sha:
                raise ValueError("changed source " + name)
        source_sha = digest(out / "01/source_barrier.json")
        records = call_journal(out, design, barrier)
        tables = {
            k: table(out / "02" / f"{k}.csv")
            for k in (
                "observations",
                "paired_actions",
                "action_metrics",
                "route_descriptors",
            )
        }
        if [len(v) for v in tables.values()] != [192, 192, 36, 96]:
            raise ValueError("all finite rows required")
        obs = {(int(r["path"]), r["arm"]): r for r in tables["observations"]}
        desc = {int(r["path"]): r for r in tables["route_descriptors"]}
        if len(obs) != 192 or len(desc) != 96:
            raise ValueError("duplicate addresses")
        regenerated, max_action, max_weight, max_arithmetic = [], 0.0, 0.0, 0.0
        for path in design:
            name = f"path_{path['path']:03d}_node_1.json"
            meta, a = load(out / "01/nodes" / name)
            parent, old = load(ROOT / spec["parent"] / "01/nodes" / name)
            wm, fixed = load(out / "00/weights" / f"path_{path['path']:03d}.json")
            if (
                meta["address"] != path
                or parent["address"] != path
                or wm["address"] != path
                or meta["node"] != 1
                or meta["t"] != 0.5
                or not meta["replay_valid"]
                or meta["native"] != parent["native"]
                or max(meta["audit"].values()) != 0
            ):
                raise ValueError("changed midpoint/failed actual replay")
            if meta["source_sha256"] != (
                source_sha if path["role"] == "application" else None
            ):
                raise ValueError("wrong source provenance")
            for k in ("weight64", "weight32"):
                if not np.array_equal(a[k], fixed[k]):
                    raise ValueError("changed fixed weight")
            paction = np.array([pairing(j, old["delta_p"]) for j in old["p_jacobian"]])
            w64 = np.array([pairing(old["hessian"][:, i], paction) for i in range(10)])
            close(a["weight64"], w64)
            if not np.array_equal(a["weight32"], a["weight64"].astype(np.float32)):
                raise ValueError("wrong float32 fixed weight")
            masks = np.zeros_like(a["masks"], dtype=bool)
            masks[0, : path["left"]["question_start"]] = True
            masks[1, path["left"]["early_query_positions"]] = True
            masks[2] = ~(masks[0] | masks[1])
            close(a["masks"], masks)
            close(a["logits"], old["logits"])
            close(meta["score"], parent["score"])
            for k in ("cut_state", "cut_gradient", "route_covectors"):
                if not np.isfinite(a[k]).all():
                    raise ValueError("nonfinite actual capture")
            dm, bank = load(out / "02" / f"path_{path['path']:03d}.json")
            if dm["path"] != path["path"] or dm["source_sha256"] != source_sha:
                raise ValueError("derived identity mismatch")
            action_errors = []
            for ai, arm in enumerate(spec["arms"]):
                d = old["delta_h"][ai]
                routes = [pairing(j, d) for j in a["route_covectors"]]
                h = np.array([pairing(j, d) for j in old["h_jacobian"]])
                wd = pairing(a["weight32"].astype(float) - a["weight64"], h)
                link = pairing(old["link"][ai], old["delta_p"])
                raw = math.fsum(routes)
                ad = raw - link - wd
                head = {k: path[k] for k in ("path", "query", "role", "wording")}
                head.update(cell=path["H"] * 4 + path["B"] * 2 + path["Q"], arm=arm)
                result = {
                    **head,
                    **dict(zip(GROUPS, routes, strict=True)),
                    "weight_correction": -wd,
                    "arithmetic_correction": -ad,
                    "weight_delta": wd,
                    "arithmetic_delta": ad,
                    "raw_link": raw,
                    "link": link,
                    "decoder": pairing(old["decoder"][ai], old["delta_p"]),
                    "old_roundoff": pairing(old["roundoff"][ai], old["delta_p"]),
                    "full": pairing(old["parent_total"][ai], old["delta_p"]),
                }
                for k in FIELDS:
                    close(bank[k][ai], result[k])
                result.update(errors(result))
                close(raw, link)
                row_close(obs[path["path"], arm], result)
                regenerated.append(result)
                action_errors.append(abs(raw - link))
                max_action, max_weight, max_arithmetic = (
                    max(max_action, abs(raw - link)),
                    max(max_weight, abs(wd)),
                    max(max_arithmetic, abs(ad)),
                )
            replay = {
                "logits_max_error": float(
                    np.max(abs(a["logits"].astype(float) - old["logits"]))
                ),
                "score_error": abs(meta["score"] - parent["score"]),
                "physical_action_max_error": max(action_errors),
            }
            row_close(meta["replay"], replay)
            descriptors = {k: path[k] for k in ("path", "query", "role", "wording")}
            descriptors.update(cell=path["H"] * 4 + path["B"] * 2 + path["Q"], **replay)
            for gi, g in enumerate(GROUPS):
                descriptors[g + "_positions"] = int(masks[gi].sum())
                descriptors[g + "_covector_norm"] = math.sqrt(
                    pairing(a["route_covectors"][gi], a["route_covectors"][gi])
                )
                v = a["cut_gradient"][masks[gi]]
                descriptors[g + "_cut_gradient_norm"] = math.sqrt(pairing(v, v))
            row_close(desc[path["path"]], descriptors)
        keys = ("query", "role", "cell", "arm")
        saved_pairs = {
            tuple(r[k] for k in (*keys, "measure")): r for r in tables["paired_actions"]
        }
        if len(saved_pairs) != 192:
            raise ValueError("duplicate wording pair")
        paired = []
        for key in sorted({tuple(r[k] for k in keys) for r in regenerated}):
            rr = sorted(
                [r for r in regenerated if tuple(r[k] for k in keys) == key],
                key=lambda r: r["wording"],
            )
            if [r["wording"] for r in rr] != [0, 2]:
                raise ValueError("missing wording")
            for measure, weights in (("mean", (0.5, 0.5)), ("contrast", (-0.5, 0.5))):
                calc = {
                    **dict(zip(keys, key, strict=True)),
                    "measure": measure,
                    **{
                        k: math.fsum(w * r[k] for w, r in zip(weights, rr, strict=True))
                        for k in FIELDS
                    },
                }
                calc.update(errors(calc))
                row_close(
                    saved_pairs[tuple(str(calc[k]) for k in (*keys, "measure"))], calc
                )
                paired.append(calc)
        for saved in tables["action_metrics"]:
            src = (
                regenerated
                if saved["measure"] == "word"
                else [r for r in paired if r["measure"] == saved["measure"]]
            )
            row_close(
                saved,
                moments(
                    [
                        r
                        for r in src
                        if all(r[k] == saved[k] for k in ("query", "role", "arm"))
                    ]
                ),
            )
        elapsed = time.monotonic() - PROCESS_STARTED
        if elapsed + status["analysis_seconds"] > 180:
            raise TimeoutError("combined CPU cap exceeded")
        summary = {
            "status": "VERIFIED_FINITE_HM43_NOT_GLOBAL",
            "verified_at": now(),
            "nodes": 96,
            "observations": 192,
            "paired_actions": 192,
            "metric_groups": 36,
            "call_journal_records": records,
            "output_hashes": len(ledger),
            "ancestry_hashes": len(m["files"]),
            "numeric_values_checked": int(checked),
            "maximum_alternate_error": maximum,
            "maximum_physical_action_replay_error": max_action,
            "maximum_weight_delta": max_weight,
            "maximum_arithmetic_delta": max_arithmetic,
            "verification_seconds": elapsed,
            "combined_analysis_seconds": elapsed + status["analysis_seconds"],
            "model_seconds": status["model_seconds"],
            "new_verification_model_calls": 0,
            "new_fits": 0,
            "application_weights_prepared": True,
            "HM42_full_jacobian_accepted": False,
            "scope": "same-agent alternate arithmetic on fixed local diagnostic actions, not independent neural replication or cold text prediction",
        }
        write_json(dest / "summary.json", summary)
        print(summary, flush=True)
    except Exception as exc:
        write_json(
            dest / "failure.json",
            {
                "status": "VERIFICATION_FAILED_NOT_ACCEPTED",
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - PROCESS_STARTED,
                "primary_unchanged": True,
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def main():
    """One finite arithmetic audit; never replay the neural observer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.out, args.dest)


if __name__ == "__main__":
    main()
