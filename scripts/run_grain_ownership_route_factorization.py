#!/usr/bin/env python3
"""HM75: fixed body/suffix ownership routes, not observer accuracy optimization."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_ownership_route_factorization as M
from grain_research.construction.math_causal_ownership_sectors import (
    array_hash,
    coefficients,
    composition_contrasts,
    ownership_actions,
)
from grain_research.construction.math_heterogeneous_evidence_cube import decode
from grain_research.construction.math_target_bound_actions import CHARACTERS
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_causal_ownership_sectors import (
    CODE as PARENT_CODE,
)
from scripts.run_grain_causal_ownership_sectors import (
    check as parent_check,
)
from scripts.run_grain_causal_ownership_sectors import (
    payload_path as old_payload,
)
from scripts.run_grain_causal_ownership_sectors import (
    unit_path as old_unit,
)
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

OUT = ROOT / "results/grain_math_response_v1/ownership_route_factorization_r2"
SPEC = ROOT / "configs/grain_ownership_route_factorization_hm75.json"
STAGES = ("score", "log_odds", "centered_logits")
TABLE_ARMS = ("native", "query_remove_both", *M.ARMS)
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_ownership_route_factorization_hm75.json",
            "docs/GRAIN_OWNERSHIP_ROUTE_FACTORIZATION_PLAN_HM75.md",
            "grain_research/construction/math_ownership_route_factorization.py",
            "scripts/run_grain_ownership_route_factorization.py",
            "scripts/verify_grain_ownership_route_factorization.py",
            "tests/test_grain_ownership_route_factorization.py",
            *PARENT_CODE,
        )
    )
)


def unit_path(out, unit):
    """Address one actual attempt without conflating it with reused parent calls."""
    return out / f"02/units/unit_{unit:04d}.json"


def payload_path(out, cube, arm):
    """Address a complete pre-outcome route payload bundle."""
    return out / f"01/payloads/cube_{cube:03d}_{arm}.json"


def prepare(out):
    """Freeze all sources, geometry, noops and exact assignments before outcomes."""
    if out.exists():
        raise FileExistsError("Preserve previous results; no reset")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    prior = parent_check(parent)
    if (
        spec["arms"] != list(M.ARMS)
        or spec["layer"] != 16
        or read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM74_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM74 and fixed route design required")
    rows, cubes, old_design = (
        read_json(parent / n)
        for n in ("00/rows.json", "00/cubes.json", "00/design.json")
    )
    if len(rows) != 528 or len(cubes) != 66:
        raise ValueError("All original query cubes required")
    reused = {
        (d["receiver"], d["arm"]): d["unit"]
        for d in old_design
        if d["arm"] in ("native", "remove_both")
    }
    sources = {}
    for name in (
        "00/manifest.json",
        "verification.json",
        "raw_seal.json",
        "03/output_seal.json",
        "01/observer.json",
    ):
        p = parent / name
        sources[str(p.relative_to(ROOT))] = digest(p)
    raw_seal = read_json(parent / "raw_seal.json")
    for unit in reused.values():
        p = old_unit(parent, unit)
        for q in (p, p.with_suffix(".npz")):
            sha = digest(q)
            if sha != raw_seal[str(q.relative_to(parent))]:
                raise ValueError("Parent raw source drift")
            sources[str(q.relative_to(ROOT))] = sha
    for directory in ("00/code", "01/payloads", "02/units", "03"):
        (out / directory).mkdir(parents=True)
    design, proofs, payload_files = [], [], {}
    for cube in cubes:
        full = {
            i: load_arrays(old_unit(parent, i))[1]["full"] for i in cube["receivers"]
        }
        low_path = old_payload(parent, cube["cube"], "remove_both")
        _, low_arrays = load_arrays(low_path)
        for p in (low_path, low_path.with_suffix(".npz")):
            sources[str(p.relative_to(ROOT))] = digest(p)
        bundles = {arm: {} for arm in (*M.ARMS, "noop")}
        for corner, receiver in enumerate(cube["receivers"]):
            row, flip = rows[receiver], rows[cube["receivers"][corner ^ 1]]
            split = row["partition"]["suffix"][0]
            if row["partition"] != flip["partition"] or len(row["input_ids"]) != len(
                flip["input_ids"]
            ):
                raise ValueError("g partners must have exact same cut/positions")
            if (
                row["input_ids"][split:] != flip["input_ids"][split:]
                or len(
                    [
                        i
                        for i, (a, b) in enumerate(
                            zip(row["input_ids"], flip["input_ids"], strict=True)
                        )
                        if a != b
                    ]
                )
                != 2
            ):
                raise ValueError("Same query, two body-name differences required")
            a, b = full[receiver], full[flip["receiver"]]
            prefix = row["partition"]["prefix"]
            if not np.array_equal(a[prefix], b[prefix]):
                raise ValueError("System prefix unexpectedly depends on future g")
            for arm in (*M.ARMS, "noop"):
                if arm == "noop" and corner != 0:
                    continue
                p, v = M.assignment(
                    a,
                    b,
                    split,
                    arm,
                    low_arrays["values"][corner],
                    row["coordinates"][2],
                    spec["random_seed"] + 8 * cube["cube"] + corner // 2,
                )
                bundles[arm][f"positions_{corner}"] = p
                bundles[arm][f"values_{corner}"] = v
                design.append(
                    {
                        "unit": len(design),
                        "cube": cube["cube"],
                        "receiver": receiver,
                        "corner": corner,
                        "arm": arm,
                    }
                )
            proofs.append(
                {
                    "receiver": receiver,
                    "flipped": flip["receiver"],
                    "split": split,
                    "system_g_drift": 0.0,
                }
            )
        for arm, arrays in bundles.items():
            p = payload_path(out, cube["cube"], arm)
            save(p, {"cube": cube["cube"], "arm": arm}, arrays)
            for q in (p, p.with_suffix(".npz")):
                payload_files[str(q.relative_to(out))] = digest(q)
    if len(design) != spec["max_forwards"] or len(proofs) != 528:
        raise ValueError("Incomplete original-scope route experiment")
    for name, value in (
        ("rows", rows),
        ("cubes", cubes),
        ("design", design),
        ("token_proofs", proofs),
        (
            "reuse_map",
            [
                {"receiver": r, "arm": a, "unit": u}
                for (r, a), u in sorted(reused.items())
            ],
        ),
    ):
        write(out / f"00/{name}.json", value)
    for name in CODE:
        p = out / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
    assets = {
        str(p.relative_to(out)): digest(p)
        for p in (out / "00").glob("*.json")
        if not p.name.startswith("._")
    }
    write(
        out / "00/manifest.json",
        {
            "packet": "HM75",
            "spec": spec,
            "alphabet": prior["alphabet"],
            "code": {n: digest(ROOT / n) for n in CODE},
            "assets": assets,
            "sources": sources,
            "payloads": payload_files,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "calls_at_payload_seal": 0,
            "model_config_sha256": digest(MODEL / "config.json"),
            "known_material_followup": True,
            "original_scope_complete": False,
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "attempted_forward_calls": 0,
            "forward_calls": 0,
            "completed_units": 0,
            "in_flight": None,
            "model_seconds": 0.0,
        },
    )
    print(
        "HM75 PREPARED:66 full cubes/3234 fixed attempts;all payloads sealed before outputs",
        flush=True,
    )


def check(out):
    """Verify actual frozen code, reused captures, inputs and complete payloads."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift: " + name)
    for section in ("assets", "payloads"):
        for name, sha in m[section].items():
            if digest(out / name) != sha:
                raise ValueError("Frozen input drift: " + name)
    for name, sha in m["sources"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Parent source drift: " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Local model configuration drift")
    return m


def collect(out):
    """Execute bounded interventions once; uncertain attempts are never retried."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("Stopped/uncertain attempts require inspection")
    if len(list((out / "02/units").glob("unit_*.json"))) != status["completed_units"]:
        raise ValueError("Actual files and attempt ledger disagree")
    rows, cubes, design = (
        read_json(out / n) for n in ("00/rows.json", "00/cubes.json", "00/design.json")
    )
    parent = ROOT / m["spec"]["parent"]
    started, previous = time.monotonic(), status["model_seconds"]
    cached_cube, cache, natives = None, {}, {}
    try:
        observer = load_observer()
        actual, old = observer.manifest(), read_json(parent / "01/observer.json")
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
                raise ValueError("Frozen observer drift: " + k)
        write(out / "02/observer.json", actual)
        for d in design[status["completed_units"] :]:
            elapsed = previous + time.monotonic() - started
            if (
                elapsed >= m["spec"]["max_model_seconds"]
                or status["attempted_forward_calls"] >= m["spec"]["max_forwards"]
            ):
                raise TimeoutError("Finite cap reached; preserve partial results")
            if cached_cube != d["cube"]:
                cache = {
                    arm: load_arrays(payload_path(out, d["cube"], arm))[1]
                    for arm in (*M.ARMS, "noop")
                }
                natives = {
                    r: load_arrays(old_unit(parent, r))[1]["full"]
                    for r in cubes[d["cube"]]["receivers"]
                }
                cached_cube = d["cube"]
            row = rows[d["receiver"]]
            arrays = cache[d["arm"]]
            p, v = arrays[f"positions_{d['corner']}"], arrays[f"values_{d['corner']}"]
            status.update(
                status="COLLECTING", in_flight=d["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            score, captured = observe_full(
                observer, row, m["alphabet"], m["spec"]["layer"], (p, v)
            )
            before, after = captured["before"], captured["full"]
            native = natives[d["receiver"]]
            unpatched = np.setdiff1d(np.arange(len(native)), p)
            meta = {
                **d,
                "native": score,
                "before_sha256": array_hash(before),
                "parent_sha256": array_hash(native),
                "unpatched_before_sha256": array_hash(before[unpatched]),
                "unpatched_after_sha256": array_hash(after[unpatched]),
                "assigned_max_error": float(
                    np.max(abs(after[p].astype(float) - v.astype(float)))
                ),
                "native_cut_max_error": float(
                    np.max(abs(before.astype(float) - native.astype(float)))
                ),
            }
            if (
                meta["before_sha256"] != meta["parent_sha256"]
                or meta["unpatched_before_sha256"] != meta["unpatched_after_sha256"]
                or meta["assigned_max_error"] != 0
            ):
                raise ValueError("Actual cut/payload/undeclared positions disagree")
            save(unit_path(out, d["unit"]), meta, {"positions": p, "actual": after[p]})
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                forward_calls=status["forward_calls"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 64 == 0:
                print(
                    f"HM75 actual {status['completed_units']}/3234;elapsed={status['model_seconds']:.1f}s",
                    flush=True,
                )
        if status["completed_units"] != m["spec"]["max_forwards"]:
            raise ValueError("Incomplete finite collection")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
        write(
            out / "raw_seal.json",
            {
                str(p.relative_to(out)): digest(p)
                for p in (out / "02").rglob("*")
                if p.is_file() and not p.name.startswith("._")
            },
        )
        print("HM75 COLLECTED:3234 actual calls", flush=True)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def stage_values(meta):
    """Keep score, log odds and pre-answer logit vector in their own units."""
    z = np.asarray(meta["native"]["candidate_logits"], dtype=float)
    d = decode(z)
    return {
        "score": d["score"],
        "log_odds": d["log_odds"],
        "centered_logits": z - z.mean(),
    }


def analyze(out):
    """Report all routes/targets, signed interactions and author-profile consequences."""
    m = check(out)
    status = read_json(out / "run_status.json")
    if status["status"] != "COLLECTED" or status["in_flight"] is not None:
        raise ValueError("Actual full collection required")
    if (out / "03/output_seal.json").exists():
        raise FileExistsError("Preserve analyzed results")
    for n, s in read_json(out / "raw_seal.json").items():
        if digest(out / n) != s:
            raise ValueError("Capture drift")
    parent = ROOT / m["spec"]["parent"]
    cubes = read_json(out / "00/cubes.json")
    lookup = {
        (d["receiver"], d["arm"]): d["unit"]
        for d in read_json(out / "00/reuse_map.json")
    }
    readings = {}
    noops = []
    for d in read_json(out / "00/design.json"):
        v = stage_values(read_json(unit_path(out, d["unit"])))
        if d["arm"] == "noop":
            original = stage_values(read_json(old_unit(parent, d["receiver"])))
            noops.append(
                max(float(np.max(abs(np.asarray(v[k]) - original[k]))) for k in STAGES)
            )
        else:
            readings[d["receiver"], d["arm"]] = v
    for r in range(528):
        readings[r, "native"] = stage_values(
            read_json(old_unit(parent, lookup[r, "native"]))
        )
        readings[r, "query_remove_both"] = stage_values(
            read_json(old_unit(parent, lookup[r, "remove_both"]))
        )
    tables, coefs, actions, profiles, interactions, summaries = {}, {}, [], [], [], []
    coefficient_rows = []
    for cube in cubes:
        common = {k: cube[k] for k in ("cube", "focal", "companion", "query")}
        for stage in STAGES:
            t = np.asarray(
                [[readings[r, a][stage] for r in cube["receivers"]] for a in TABLE_ARMS]
            )
            tables[f"cube_{cube['cube']:03d}_{stage}"] = t
            c = np.asarray([coefficients(v) for v in t])
            coefs[cube["cube"], stage] = c
            for ai, arm in enumerate(TABLE_ARMS):
                for j, name in enumerate(CHARACTERS):
                    coefficient_rows.append(
                        {
                            **common,
                            "stage": stage,
                            "arm": arm,
                            "character": name,
                            "value": np.asarray(c[ai, j]).tolist(),
                        }
                    )
                for name, value in zip(
                    ("average", "x_low", "x_high"),
                    ownership_actions(t[ai]),
                    strict=True,
                ):
                    actions.append(
                        {
                            **common,
                            "stage": stage,
                            "arm": arm,
                            "context": name,
                            "value": np.asarray(value).tolist(),
                        }
                    )
                for name, profile in composition_contrasts(t[ai]).items():
                    profiles.append(
                        {
                            **common,
                            "stage": stage,
                            "arm": arm,
                            "character": name,
                            **{k: np.asarray(v).tolist() for k, v in profile.items()},
                        }
                    )
            joint = M.route_interaction(t[0], t[2], t[3], t[4])
            for i, v in enumerate(joint):
                interactions.append(
                    {
                        **common,
                        "stage": stage,
                        "corner": i,
                        "value": np.asarray(v).tolist(),
                    }
                )
    for kind in ("matched", "other", "neutral"):
        selected = [
            c
            for c in cubes
            if (
                "neutral"
                if c["companion"] == "neutral"
                else "matched"
                if c["query"] == c["companion"]
                else "other"
            )
            == kind
        ]
        for stage in STAGES:
            for ai, arm in enumerate(TABLE_ARMS):
                for j in (3, 7):
                    values = np.asarray(
                        [coefs[c["cube"], stage][ai, j] for c in selected]
                    )
                    summaries.append(
                        {
                            "query_role": kind,
                            "stage": stage,
                            "arm": arm,
                            "character": CHARACTERS[j],
                            "cubes": len(selected),
                            "coefficient_mean": float(values.mean()),
                            "coefficient_RMS": float(np.sqrt(np.mean(values**2))),
                            "components": values.size,
                        }
                    )
    save(
        out / "03/tables.json",
        {"arms": list(TABLE_ARMS), "stages": list(STAGES)},
        tables,
    )
    for name, rows in (
        ("coefficients", coefficient_rows),
        ("actions", actions),
        ("composition_consequences", profiles),
        ("route_interactions", interactions),
        ("route_summary", summaries),
    ):
        write_csv(out / f"03/{name}.csv", rows)
    both = max(
        float(np.max(abs(coefs[c["cube"], s][4, [1, 3, 5, 7]])))
        for c in cubes
        for s in STAGES
    )
    write(
        out / "03/summary.json",
        {
            "status": "ANALYZED_FINITE_HM75_NOT_GLOBAL",
            "actual_forwards": status["forward_calls"],
            "model_seconds": status["model_seconds"],
            "noop_output_max_error": max(noops),
            "both_even_odd_coefficients_max_error": both,
            "scope": "all66known HM74query cubes;two queries per nonneutral pair",
            "no_accuracy_search": True,
            "original_scope_complete": False,
        },
    )
    write(
        out / "03/output_seal.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in (out / "03").iterdir()
            if p.is_file() and not p.name.startswith("._")
        },
    )
    print("HM75 ANALYZED", flush=True)


def main():
    """Run explicit bounded phases without automatically resuming other queues."""
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("prepare", "collect", "analyze"))
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    if args.phase == "collect":
        with collector_guard(args.out):
            collect(args.out)
    else:
        globals()[args.phase](args.out)


if __name__ == "__main__":
    main()
