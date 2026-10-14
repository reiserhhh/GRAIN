#!/usr/bin/env python3
"""HM42 alternate contractions and complete raw/journal/table verification."""

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

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_json
from scripts.run_grain_math_target_active_support import load
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/text_query_route_cut_r1"
DEST = ROOT / "results/grain_math_response_v1/text_query_route_cut_verification_r1"
GROUPS = ("body", "early_query", "remainder")
COMPONENTS = (*GROUPS, "cut_roundoff", "decoder", "old_roundoff")
FIELDS = (*COMPONENTS, "link", "full")


def ordered_routes(jacobians, directions, c, p):
    """Apply C to each logit-direction column before applying the P transpose."""
    width = len(c)
    dd, pp = directions.reshape(len(directions), -1), p.reshape(width, -1)
    return np.stack(
        [
            ((c @ (j.reshape(width, -1) @ dd.T)).T @ pp).reshape(len(dd), *p.shape[1:])
            for j in jacobians
        ]
    )


def moments(rows):
    """Compensated signed scalar moments, independent of primary array means."""
    n, result = len(rows), {"count": len(rows)}
    for key in FIELDS:
        result[key + "_mean"] = math.fsum(r[key] for r in rows) / n
        result[key + "_ms"] = math.fsum(r[key] ** 2 for r in rows) / n
        result[key + "_rms"] = math.sqrt(result[key + "_ms"])
    for a, b in itertools.combinations(COMPONENTS, 2):
        result[a + "__" + b + "_cross"] = math.fsum(2 * r[a] * r[b] for r in rows) / n
    result["full_sum_error"] = max(
        abs(math.fsum(r[k] for k in COMPONENTS) - r["full"]) for r in rows
    )
    result["link_sum_error"] = max(
        abs(math.fsum(r[k] for k in (*GROUPS, "cut_roundoff")) - r["link"])
        for r in rows
    )
    return result


def call_journal(out, design, barrier):
    """Require paired actual call records, role separation and source chronology."""
    totals = {k: 0 for k in ("forward", "first")}
    for role in ("source", "application"):
        with (out / "01" / f"{role}_calls.jsonl").open() as stream:
            records = [json.loads(line) for line in stream]
        expected = [p["path"] for p in design if p["role"] == role]
        if [r["path"] for r in records[::82]] != expected or len(records) != 48 * 82:
            raise ValueError(
                "48 ordered paths with 82 attempted/completed records required"
            )
        for i, path in enumerate(expected):
            block = records[i * 82 : (i + 1) * 82]
            if any(r["path"] != path for r in block):
                raise ValueError("journal mixed midpoint addresses")
            for j in range(41):
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
    if totals != {"forward": 96, "first": 3840}:
        raise ValueError("expanded or incomplete model calls")
    return sum(totals.values()) * 2


def verify(out, dest):
    """Replay every finite output without calling the neural reader again."""
    if dest.exists():
        raise FileExistsError("preserve previous verification")
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "EVALUATED"
        or status["in_flight"] is not None
        or status["committed_nodes"] != 96
        or status["new_fits"]
    ):
        raise ValueError("fully evaluated unfitted cut required")
    for kind, count in (("forward", 96), ("first", 3840), ("second", 0)):
        if status[f"{kind}_attempted"] != count or status[f"{kind}_completed"] != count:
            raise ValueError("expanded or incomplete actual call counts")
    if status["model_seconds"] > 1200:
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
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("finite aligned numerical replay required")
        error = float(np.max(abs(a - b))) if a.size else 0.0
        maximum, checked = max(maximum, error), checked + a.size
        if error > 1e-8:
            raise ValueError(f"original physical gate exceeded: {error}")

    def row_close(saved, calc):
        for key, value in calc.items():
            if isinstance(value, str):
                if saved[key] != value:
                    raise ValueError("changed address " + key)
            else:
                close(float(saved[key]), value)

    def norm(a):
        x = np.asarray(a, float).ravel()
        return math.sqrt(float(x @ x))

    try:
        m = check(out)
        spec, design = m["spec"], read_json(out / "00/design.json")
        ledger = read_json(out / "02/artifact_hashes.json")
        for name, sha in ledger.items():
            if digest(out / name) != sha:
                raise ValueError("changed raw/derived output " + name)
        barrier = read_json(out / "01/source_barrier.json")
        if (barrier["source_nodes"], barrier["application_nodes_loaded"]) != (48, 0):
            raise ValueError("incomplete source checkpoint")
        for name, sha in barrier["files"].items():
            if digest(out / name) != sha:
                raise ValueError("changed source capture " + name)
        source_sha = digest(out / "01/source_barrier.json")
        journal_records = call_journal(out, design, barrier)
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
            raise ValueError("unique finite addresses required")
        regenerated = []
        for path in design:
            name = f"path_{path['path']:03d}_node_1.json"
            meta, a = load(out / "01/nodes" / name)
            parent, old = load(ROOT / spec["parent"] / "01/nodes" / name)
            if (
                meta["address"] != path
                or parent["address"] != path
                or meta["node"] != 1
                or meta["t"] != 0.5
                or not meta["replay_valid"]
                or meta["native"] != parent["native"]
                or max(meta["audit"].values()) != 0
            ):
                raise ValueError("changed input identity or failed actual replay")
            expected_sha = source_sha if path["role"] == "application" else None
            if meta["source_sha256"] != expected_sha:
                raise ValueError("source capture provenance mismatch")
            row = path["left"]
            masks = np.zeros_like(a["masks"], dtype=bool)
            masks[0, : row["question_start"]] = True
            masks[1, row["early_query_positions"]] = True
            masks[2] = np.logical_not(masks[0] | masks[1])
            close(a["masks"], masks)
            close(a["logits"], old["logits"])
            close(a["route_jacobian"].astype(float).sum(0), old["h_jacobian"])
            close(meta["score"], parent["score"])
            if (
                not np.isfinite(a["cut_state"]).all()
                or not np.isfinite(a["cut_gradients"]).all()
            ):
                raise ValueError("nonfinite cut capture")
            routes = ordered_routes(
                a["route_jacobian"].astype(float),
                old["delta_h"].astype(float),
                old["hessian"].astype(float),
                old["p_jacobian"].astype(float),
            )
            close(routes.sum(0), old["link"])
            derived, bank = load(out / "02" / f"path_{path['path']:03d}.json")
            if (
                derived["path"] != path["path"]
                or derived["source_sha256"] != source_sha
            ):
                raise ValueError("derived route identity changed")
            close(bank["routes"], routes)
            cut_residual = old["link"] - routes.sum(0)
            close(bank["cut_roundoff"], cut_residual)
            components = [
                *routes,
                cut_residual,
                old["decoder"],
                old["roundoff"],
                old["link"],
                old["parent_total"],
            ]
            for ai, arm in enumerate(spec["arms"]):
                # Scalar action via digit contractions independently of stored P covectors.
                h_actions = [
                    j.reshape(10, -1).astype(float) @ old["delta_h"][ai].ravel()
                    for j in a["route_jacobian"]
                ]
                p_action = old["p_jacobian"].reshape(10, -1) @ old["delta_p"].ravel()
                direct = [float(p_action @ old["hessian"] @ h) for h in h_actions]
                close(
                    direct,
                    [float(v[ai].ravel() @ old["delta_p"].ravel()) for v in routes],
                )
                head = {k: path[k] for k in ("path", "query", "role", "wording")}
                head.update(cell=path["H"] * 4 + path["B"] * 2 + path["Q"], arm=arm)
                result = {
                    **head,
                    **{
                        k: float(v[ai].ravel() @ old["delta_p"].ravel())
                        for k, v in zip(FIELDS, components, strict=True)
                    },
                }
                result["full_sum_error"] = abs(
                    math.fsum(result[k] for k in COMPONENTS) - result["full"]
                )
                result["link_sum_error"] = abs(
                    math.fsum(result[k] for k in (*GROUPS, "cut_roundoff"))
                    - result["link"]
                )
                close(result["full_sum_error"], 0)
                close(result["link_sum_error"], 0)
                row_close(obs[path["path"], arm], result)
                regenerated.append(result)
            descriptors = {k: path[k] for k in ("path", "query", "role", "wording")}
            descriptors.update(
                cell=path["H"] * 4 + path["B"] * 2 + path["Q"], **meta["replay"]
            )
            for gi, g in enumerate(GROUPS):
                descriptors[g + "_positions"] = int(masks[gi].sum())
                descriptors[g + "_jacobian_norm"] = norm(a["route_jacobian"][gi])
                descriptors[g + "_cut_gradient_norm"] = norm(
                    a["cut_gradients"][:, masks[gi]]
                )
            row_close(desc[path["path"]], descriptors)
        keys = ("query", "role", "cell", "arm")
        saved_pairs = {
            tuple(r[k] for k in (*keys, "measure")): r for r in tables["paired_actions"]
        }
        if len(saved_pairs) != 192:
            raise ValueError("duplicate pair addresses")
        paired = []
        for key in sorted({tuple(r[k] for k in keys) for r in regenerated}):
            rr = sorted(
                [r for r in regenerated if tuple(r[k] for k in keys) == key],
                key=lambda r: r["wording"],
            )
            if [r["wording"] for r in rr] != [0, 2]:
                raise ValueError("missing original wording")
            for measure in ("mean", "contrast"):
                weights = (0.5, 0.5) if measure == "mean" else (-0.5, 0.5)
                calc = {
                    **dict(zip(keys, key, strict=True)),
                    "measure": measure,
                    **{
                        k: math.fsum(w * r[k] for w, r in zip(weights, rr, strict=True))
                        for k in FIELDS
                    },
                }
                calc["full_sum_error"] = abs(
                    math.fsum(calc[k] for k in COMPONENTS) - calc["full"]
                )
                calc["link_sum_error"] = abs(
                    math.fsum(calc[k] for k in (*GROUPS, "cut_roundoff")) - calc["link"]
                )
                row_close(
                    saved_pairs[tuple(str(calc[k]) for k in (*keys, "measure"))], calc
                )
                paired.append(calc)
        for saved in tables["action_metrics"]:
            source = (
                regenerated
                if saved["measure"] == "word"
                else [r for r in paired if r["measure"] == saved["measure"]]
            )
            selected = [
                r
                for r in source
                if all(r[k] == saved[k] for k in ("query", "role", "arm"))
            ]
            row_close(saved, moments(selected))
        elapsed = time.monotonic() - PROCESS_STARTED
        if elapsed + status["analysis_seconds"] > 180:
            raise TimeoutError("combined CPU cap exceeded")
        summary = {
            "status": "VERIFIED_FINITE_HM42_NOT_GLOBAL",
            "verified_at": now(),
            "nodes": 96,
            "observations": 192,
            "paired_actions": 192,
            "metric_groups": 36,
            "call_journal_records": journal_records,
            "output_hashes": len(ledger),
            "ancestry_hashes": len(m["files"]),
            "numeric_values_checked": int(checked),
            "maximum_alternate_error": maximum,
            "verification_seconds": elapsed,
            "combined_analysis_seconds": elapsed + status["analysis_seconds"],
            "model_seconds": status["model_seconds"],
            "new_verification_model_calls": 0,
            "new_fits": 0,
            "scope": "same-agent alternate contractions, not independent neural acquisition or psychological validity",
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
    """One alternate verification; never acquire or retry model calls."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.out, args.dest)


if __name__ == "__main__":
    main()
