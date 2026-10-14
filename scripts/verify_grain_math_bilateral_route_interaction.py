#!/usr/bin/env python3
"""HM44 independently ordered digit contractions, lifecycles and all finite tables."""

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

from grain_research.construction.math_bilateral_route_interaction import (
    CELLS,
    COMPONENTS,
    FIELDS,
    GROUPS,
    PATHS,
)
from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_json
from scripts.run_grain_math_target_active_support import load
from scripts.verify_grain_math_crossed_field import table
from scripts.verify_grain_math_dual_route_action import pairing

OUT = ROOT / "results/grain_math_response_v1/bilateral_route_interaction_r1"
DEST = (
    ROOT / "results/grain_math_response_v1/bilateral_route_interaction_verification_r1"
)


def ordered_joint(h, p, d, e, c):
    """Pair each digit first, then apply C transpose to H and pair with P."""
    hs = np.array([[[pairing(j, di) for j in g] for g in h] for di in d])
    ps = np.array([[pairing(j, e) for j in g] for g in p])
    return np.array([[[pairing(c.T @ hi, pi) for pi in ps] for hi in ha] for ha in hs])


def errors(row):
    """Independently reconstruct signed identities without hiding raw discrepancy."""
    return {
        "full_sum_error": abs(math.fsum(row[k] for k in COMPONENTS) - row["full"]),
        "link_sum_error": abs(
            math.fsum(row[k] for k in (*CELLS, "cut_roundoff")) - row["link"]
        ),
        "H_margin_sum_error": abs(
            math.fsum(row["H_" + g] for g in GROUPS) - row["raw_link"]
        ),
        "P_margin_sum_error": abs(
            math.fsum(row["P_" + g] for g in GROUPS) - row["raw_link"]
        ),
    }


def moments(rows):
    """Compensated all-component signed moments, not primary array mean order."""
    n, result = len(rows), {"count": len(rows)}
    for k in FIELDS:
        result[k + "_mean"] = math.fsum(r[k] for r in rows) / n
        result[k + "_ms"] = math.fsum(r[k] ** 2 for r in rows) / n
        result[k + "_rms"] = math.sqrt(result[k + "_ms"])
    for a, b in itertools.combinations(COMPONENTS, 2):
        result[a + "___" + b + "_cross"] = math.fsum(2 * r[a] * r[b] for r in rows) / n
    result["link_decoder_cross"] = (
        math.fsum(2 * r["link"] * r["decoder"] for r in rows) / n
    )
    result.update({k: max(errors(r)[k] for r in rows) for k in errors(rows[0])})
    return result


def call_journal(out, design, barrier):
    """Require24 ordered actual forwards and960 first calls, source sealed first."""
    totals = {"forward": 0, "first": 0}
    for role in ("source", "application"):
        with (out / "01" / f"{role}_calls.jsonl").open() as stream:
            records = [json.loads(line) for line in stream]
        ids = [p["path"] for p in design if p["role"] == role]
        if len(records) != 12 * 82 or [r["path"] for r in records[::82]] != ids:
            raise ValueError("12 ordered complete path calls required")
        for i, path in enumerate(ids):
            block = records[i * 82 : (i + 1) * 82]
            if any(r["path"] != path for r in block):
                raise ValueError("mixed call addresses")
            for j in range(41):
                a, b = block[2 * j : 2 * j + 2]
                k = "forward" if j == 0 else "first"
                totals[k] += 1
                if (
                    a["kind"],
                    b["kind"],
                    a["phase"],
                    b["phase"],
                    a["number"],
                    b["number"],
                ) != (k, k, "attempted", "completed", totals[k], totals[k]):
                    raise ValueError("actual lifecycle mismatch")
                if (
                    a["at"] > b["at"]
                    or (role == "source" and b["at"] > barrier["frozen_at"])
                    or (role == "application" and a["at"] < barrier["frozen_at"])
                ):
                    raise ValueError("source-first chronology mismatch")
    if totals != {"forward": 24, "first": 960}:
        raise ValueError("incomplete/expanded calls")
    return 2 * sum(totals.values())


def verify(out, dest):
    """Verify all raw/derived results without any new model invocation."""
    if dest.exists():
        raise FileExistsError("preserve verification")
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "EVALUATED"
        or status["committed_nodes"] != 24
        or status["in_flight"] is not None
        or status["new_fits"]
    ):
        raise ValueError("complete unfitted packet required")
    for k, n in (("forward", 24), ("first", 960), ("second", 0)):
        if status[f"{k}_attempted"] != n or status[f"{k}_completed"] != n:
            raise ValueError("actual finite call count mismatch")
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
    maximum, checked, max_raw, max_margin, max_coords = 0.0, 0, 0.0, 0.0, 0.0

    def close(a, b):
        nonlocal maximum, checked
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("finite aligned replay required")
        err = float(np.max(abs(x - y))) if x.size else 0.0
        maximum, checked = max(maximum, err), checked + x.size
        if err > 1e-8:
            raise ValueError(f"original physical gate exceeded: {err}")

    def row_close(saved, calc):
        for k, v in calc.items():
            if isinstance(v, str):
                if saved[k] != v:
                    raise ValueError("changed identity " + k)
            else:
                close(float(saved[k]), v)

    try:
        m = check(out)
        spec, design = m["spec"], read_json(out / "00/design.json")
        if [p["path"] for p in design] != list(PATHS) or m[
            "application_factors_prepared"
        ] is not True:
            raise ValueError(
                "24 declared corners with diagnostic-factor disclosure required"
            )
        ledger = read_json(out / "02/artifact_hashes.json")
        for n, sha in ledger.items():
            if digest(out / n) != sha:
                raise ValueError("changed output " + n)
        barrier = read_json(out / "01/source_barrier.json")
        if (
            barrier["source_nodes"],
            barrier["new_application_nodes_acquired"],
            barrier["application_factors_prepared"],
        ) != (12, 0, True):
            raise ValueError("source barrier mismatch")
        for n, sha in barrier["files"].items():
            if digest(out / n) != sha:
                raise ValueError("changed source " + n)
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
        if [len(v) for v in tables.values()] != [48, 48, 36, 24]:
            raise ValueError("all finite rows required")
        obs = {(int(r["path"]), r["arm"]): r for r in tables["observations"]}
        desc = {int(r["path"]): r for r in tables["route_descriptors"]}
        if len(obs) != 48 or len(desc) != 24:
            raise ValueError("duplicate finite addresses")
        regenerated = []
        for path in design:
            name = f"path_{path['path']:03d}_node_1.json"
            meta, a = load(out / "01/nodes" / name)
            parent, old = load(ROOT / spec["parent"] / "01/nodes" / name)
            fm, factors = load(out / "00/factors" / f"path_{path['path']:03d}.json")
            if (
                meta["address"] != path
                or parent["address"] != path
                or fm["address"] != path
                or meta["node"] != 1
                or meta["t"] != 0.5
                or not meta["replay_valid"]
                or meta["native"] != parent["native"]
                or max(meta["audit"].values()) != 0
            ):
                raise ValueError("changed input or failed original replay")
            if meta["source_sha256"] != (
                source_sha if path["role"] == "application" else None
            ):
                raise ValueError("source provenance mismatch")
            for k in factors:
                if not np.array_equal(factors[k], old[k]):
                    raise ValueError("changed diagnostic factors")
            row = path["left"]
            masks = np.zeros_like(a["masks"], dtype=bool)
            (
                masks[0, : row["question_start"]],
                masks[1, row["early_query_positions"]],
            ) = True, True
            masks[2] = ~(masks[0] | masks[1])
            close(a["masks"], masks)
            close(a["logits"], old["logits"])
            close(meta["score"], parent["score"])
            if np.count_nonzero(a["p_routes"][0]):
                raise ValueError("causal query-to-past-body violation")
            for k in ("cut_state", "cut_gradients", "h_routes", "p_routes"):
                if not np.isfinite(a[k]).all():
                    raise ValueError("nonfinite captured arrays")
            joint = ordered_joint(
                a["h_routes"],
                a["p_routes"],
                old["delta_h"],
                old["delta_p"],
                old["hessian"],
            )
            dm, bank = load(out / "02" / f"path_{path['path']:03d}.json")
            if dm["path"] != path["path"] or dm["source_sha256"] != source_sha:
                raise ValueError("derived identity changed")
            close(bank["joint"], joint)
            _, hm43 = load(ROOT / spec["previous_packet"] / "01/nodes" / name)
            reference = np.array(
                [
                    [pairing(j, d) for j in hm43["route_covectors"]]
                    for d in old["delta_h"]
                ]
            )
            hmargin = np.array([[math.fsum(g) for g in mat] for mat in joint])
            close(hmargin, reference)
            per_raw = []
            for ai, arm in enumerate(spec["arms"]):
                mat = joint[ai]
                raw = math.fsum(mat.ravel())
                link = pairing(old["link"][ai], old["delta_p"])
                close(raw, link)
                per_raw.append(abs(raw - link))
                r = {k: path[k] for k in ("path", "query", "role", "wording")}
                r.update(cell=path["H"] * 4 + path["B"] * 2 + path["Q"], arm=arm)
                r.update(dict(zip(CELLS, mat.ravel(), strict=True)))
                r.update({"H_" + g: math.fsum(mat[i]) for i, g in enumerate(GROUPS)})
                r.update({"P_" + g: math.fsum(mat[:, i]) for i, g in enumerate(GROUPS)})
                r.update(
                    raw_link=raw,
                    link=link,
                    cut_roundoff=link - raw,
                    decoder=pairing(old["decoder"][ai], old["delta_p"]),
                    old_roundoff=pairing(old["roundoff"][ai], old["delta_p"]),
                    full=pairing(old["parent_total"][ai], old["delta_p"]),
                )
                for k in FIELDS:
                    close(bank[k][ai], r[k])
                r.update(errors(r))
                row_close(obs[path["path"], arm], r)
                regenerated.append(r)
            replay = {
                "logits_max_error": float(
                    np.max(abs(a["logits"].astype(float) - old["logits"]))
                ),
                "score_error": abs(meta["score"] - parent["score"]),
                "physical_action_max_error": max(per_raw),
                "H_margin_HM43_max_error": float(np.max(abs(hmargin - reference))),
                "causal_P_body_max": 0.0,
            }
            coords = {
                "H_max": float(
                    np.max(abs(a["h_routes"].astype(float).sum(0) - old["h_jacobian"]))
                ),
                "P_max": float(
                    np.max(abs(a["p_routes"].astype(float).sum(0) - old["p_jacobian"]))
                ),
            }
            row_close(meta["replay"], replay)
            row_close(meta["coordinate_errors_reported_not_acceptance"], coords)
            max_raw, max_margin, max_coords = (
                max(max_raw, replay["physical_action_max_error"]),
                max(max_margin, replay["H_margin_HM43_max_error"]),
                max(max_coords, max(coords.values())),
            )
            dd = {k: path[k] for k in ("path", "query", "role", "wording")}
            dd.update(
                cell=path["H"] * 4 + path["B"] * 2 + path["Q"], **replay, **coords
            )
            for gi, g in enumerate(GROUPS):
                dd[g + "_positions"] = int(masks[gi].sum())
                dd[g + "_H_norm"] = float(
                    np.sqrt(np.sum(a["h_routes"][gi].astype(float) ** 2))
                )
                dd[g + "_P_norm"] = float(
                    np.sqrt(np.sum(a["p_routes"][gi].astype(float) ** 2))
                )
            row_close(desc[path["path"]], dd)
        keys = ("query", "role", "cell", "arm")
        sp = {
            tuple(r[k] for k in (*keys, "measure")): r for r in tables["paired_actions"]
        }
        if len(sp) != 48:
            raise ValueError("duplicate paired address")
        pairs = []
        for key in sorted({tuple(r[k] for k in keys) for r in regenerated}):
            rr = sorted(
                [r for r in regenerated if tuple(r[k] for k in keys) == key],
                key=lambda r: r["wording"],
            )
            if [r["wording"] for r in rr] != [0, 2]:
                raise ValueError("missing original pair")
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
                row_close(sp[tuple(str(calc[k]) for k in (*keys, "measure"))], calc)
                pairs.append(calc)
        for saved in tables["action_metrics"]:
            src = (
                regenerated
                if saved["measure"] == "word"
                else [r for r in pairs if r["measure"] == saved["measure"]]
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
            "status": "VERIFIED_FINITE_HM44_NOT_GLOBAL",
            "verified_at": now(),
            "nodes": 24,
            "observations": 48,
            "paired_actions": 48,
            "metric_groups": 36,
            "maximum_alternate_error_including_original_replay": maximum,
            "maximum_physical_action_replay_error": max_raw,
            "maximum_H_margin_HM43_error": max_margin,
            "maximum_coordinate_error_reported_not_acceptance": max_coords,
            "causal_P_body_exact_zero": True,
            "numeric_values_checked": int(checked),
            "output_hashes": len(ledger),
            "ancestry_hashes": len(m["files"]),
            "journal_records": records,
            "verification_seconds": elapsed,
            "combined_analysis_seconds": elapsed + status["analysis_seconds"],
            "model_seconds": status["model_seconds"],
            "new_verification_model_calls": 0,
            "new_fits": 0,
            "application_factors_prepared": True,
            "HM42_coordinate_accepted": False,
            "scope": "same-agent alternate digit arithmetic on24 reused corner texts; not independent neural replication, all8cell transport or cold author prediction",
        }
        write_json(dest / "summary.json", summary)
        print(summary, flush=True)
    except Exception as exc:
        write_json(
            dest / "failure.json",
            {
                "status": "VERIFICATION_FAILED_NOT_ACCEPTED",
                "error": repr(exc),
                "seconds": time.monotonic() - PROCESS_STARTED,
                "primary_unchanged": True,
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)


def main():
    """One bounded no-model numerical audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.out, args.dest)


if __name__ == "__main__":
    main()
