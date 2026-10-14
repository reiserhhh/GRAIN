#!/usr/bin/env python3
"""HM49: source-only mean ownership and exact donor interventions, no search."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_relational_carrier as M
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import observe, save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_appraisal import CODE as PARENT_CODE
from scripts.run_grain_math_relational_appraisal import check as parent_check
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_relational_carrier_v1.json"
OUT = ROOT / "results/grain_math_response_v1/relational_carrier_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_relational_carrier_v1.json",
            "docs/GRAIN_RELATIONAL_CARRIER_PLAN_V1.md",
            "grain_research/construction/math_relational_carrier.py",
            "scripts/run_grain_math_relational_carrier.py",
            "scripts/verify_grain_math_relational_carrier.py",
            "tests/test_grain_math_relational_carrier.py",
            *PARENT_CODE,
        )
    )
)


def prepare(out):
    """Check every selected old capture and freeze source vectors before any fresh call."""
    if out.exists():
        raise FileExistsError("Preserve existing versions")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    prior = parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM48_NOT_GLOBAL"
        or spec["targets"] != prior["spec"]["targets"]
        or spec["layer"] != prior["spec"]["layer"]
        or spec["arms"] != list(M.ARMS)
    ):
        raise ValueError("Require verified HM48 with unchanged observer and target set")
    all_rows = read_json(parent / "00/design.json")
    pair_rows = M.pairs(all_rows, spec["targets"])
    rows = {
        r["unit"]: r
        for r in all_rows
        if r["unit"] in {p["receiver"] for p in pair_rows}
    }
    captures, metas, hashes = {}, {}, {}
    for unit in sorted(rows):
        path = parent / f"01/units/unit_{unit:04d}.json"
        meta, arrays = load_arrays(path)
        if meta["unit"] != unit or meta["role"] != rows[unit]["role"]:
            raise ValueError("Wrong capture identity")
        captures[unit], metas[unit] = arrays, meta
        for p in (path, path.with_suffix(".npz")):
            hashes[str(p.relative_to(parent))] = digest(p)
    # Pass only source captures: a mechanical guard against application construction.
    source = {k: v for k, v in captures.items() if rows[k]["role"] == "source"}
    shared, per_content = M.source_vectors(pair_rows, rows, source, spec["targets"])
    random = M.random_vectors(shared, spec["random_seed"])
    design = []
    for pair in pair_rows:
        for arm in spec["arms"]:
            design.append({**pair, "arm": arm, "unit": len(design)})
    if (
        len(pair_rows) != spec["expected_receivers"]
        or len(design) != spec["max_forwards"]
        or sum(p["role"] == "source" for p in pair_rows)
        != spec["expected_source_receivers"]
    ):
        raise ValueError("Incomplete declared scope")
    for name in ("00/code", "01/units", "03"):
        (out / name).mkdir(parents=True, exist_ok=True)
    write(out / "00/design.json", design)
    write(out / "00/receivers.json", list(rows.values()))
    save(
        out / "00/source_vectors.json",
        {
            "source_units": sorted(source),
            "source_only": True,
            "targets": spec["targets"],
            "equal_token_person_content_means": True,
        },
        {"shared": shared, "per_content": per_content, "random": random},
    )
    code = {}
    for name in CODE:
        p = out / "00/code" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(p)
    for name in (
        "00/manifest.json",
        "00/design.json",
        "verification.json",
        "01/observer.json",
    ):
        hashes[name] = digest(parent / name)
    assets = {
        name: digest(out / name)
        for name in (
            "00/design.json",
            "00/receivers.json",
            "00/source_vectors.json",
            "00/source_vectors.npz",
        )
    }
    write(
        out / "00/manifest.json",
        {
            "packet": "HM49",
            "spec": spec,
            "alphabet": prior["alphabet"],
            "code": code,
            "assets": assets,
            "parent_files": hashes,
            "model_config_sha256": digest(MODEL / "config.json"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "known_material_followup": True,
            "model_weights_not_separately_hashed": True,
            "human_truth": False,
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
        f"PREPARED: {len(pair_rows)} receivers/{len(design)} interventions/source assets sealed",
        flush=True,
    )


def check(out):
    """Bind live code, frozen source assets and all used parent captures by hashes."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Changed frozen code " + name)
    for name, sha in manifest["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Changed frozen assets " + name)
    parent = ROOT / manifest["spec"]["parent"]
    for name, sha in manifest["parent_files"].items():
        if digest(parent / name) != sha:
            raise ValueError("Changed selected parent " + name)
    if digest(MODEL / "config.json") != manifest["model_config_sha256"]:
        raise ValueError("Changed model config")
    return manifest


def collect(out):
    """Run only the fixed384 arms, source then application; preserve uncertain failures."""
    manifest = check(out)
    spec, status = manifest["spec"], read_json(out / "run_status.json")
    if status["in_flight"] is not None:
        raise RuntimeError("Unknown call outcome: inspect, never automatically repeat")
    if status["status"] == "COLLECTED":
        return
    parent = ROOT / spec["parent"]
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    arrays, metas = {}, {}
    for unit in rows:
        metas[unit], arrays[unit] = load_arrays(
            parent / f"01/units/unit_{unit:04d}.json"
        )
    _, assets = load_arrays(out / "00/source_vectors.json")
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual, prior = observer.manifest(), read_json(parent / "01/observer.json")
        for key in (
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
            if actual[key] != prior[key]:
                raise ValueError("Observer drift " + key)
        write(out / "01/observer.json", actual)
        design = read_json(out / "00/design.json")
        for role in ("source", "application"):
            if role == "application":
                for name, sha in read_json(out / "01/source_barrier.json")[
                    "files"
                ].items():
                    if digest(out / name) != sha:
                        raise ValueError("Changed sealed source output")
            for entry in design:
                if entry["role"] != role:
                    continue
                path = out / f"01/units/unit_{entry['unit']:04d}.json"
                if path.exists():
                    load_arrays(path)
                    continue
                elapsed = previous + time.monotonic() - started
                if not M.call_allowed(status, spec, elapsed):
                    raise TimeoutError("Finite resource contract reached")
                receiver, donor = rows[entry["receiver"]], rows[entry["donor"]]
                positions, values = M.patch_values(
                    receiver,
                    donor,
                    arrays[receiver["unit"]],
                    arrays[donor["unit"]],
                    assets["shared"],
                    assets["random"],
                    entry["arm"],
                )
                status.update(
                    status="COLLECTING", in_flight=entry["unit"], model_seconds=elapsed
                )
                status["attempted_forward_calls"] += 1
                write(out / "run_status.json", status)
                native, captured = observe(
                    observer,
                    receiver,
                    manifest["alphabet"],
                    spec["layer"],
                    (positions, values),
                )
                status["forward_calls"] += 1
                save(
                    path,
                    {
                        "unit": entry["unit"],
                        "pair": entry["pair"],
                        "role": role,
                        "arm": entry["arm"],
                        "positions": positions,
                        "native": native,
                    },
                    {
                        **captured,
                        "patch_values": values,
                        "patch_positions": np.asarray(positions, int),
                    },
                )
                status.update(
                    in_flight=None,
                    completed_units=status["completed_units"] + 1,
                    model_seconds=previous + time.monotonic() - started,
                )
                write(out / "run_status.json", status)
                if status["completed_units"] % 48 == 0:
                    print(
                        f"{role}: {status['completed_units']}/384 actual patches",
                        flush=True,
                    )
            if role == "source":
                paths = [
                    out / f"01/units/unit_{e['unit']:04d}.json"
                    for e in design
                    if e["role"] == "source"
                ]
                files = {
                    str(p.relative_to(out)): digest(p)
                    for path in paths
                    for p in (path, path.with_suffix(".npz"))
                }
                write(
                    out / "01/source_barrier.json",
                    {
                        "source_calls": len(paths),
                        "files": files,
                        "created_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    },
                )
        if (
            any(
                status[k] != spec["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > spec["max_model_seconds"]
        ):
            raise ValueError("Incomplete or over-budget collection")
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


def summarize(rows):
    """Report all finite signed effects; no best-arm selection or population intervals."""
    change = np.asarray([r["patch_change"] for r in rows], float)
    full = np.asarray([r["full_change"] for r in rows], float)
    denom = float(full @ full)
    return {
        "rows": len(rows),
        "patch_rms": float(np.sqrt(np.mean(change**2))),
        "full_rms": float(np.sqrt(np.mean(full**2))),
        "signed_projection": float(change @ full / denom) if denom else None,
        "remaining_rmse": float(np.sqrt(np.mean((change - full) ** 2))),
        "direction_count": int(np.sum(change * full > 0)),
        "max_abs_change": float(np.max(abs(change))),
    }


def analyze(out):
    """Convert fixed interventions to all-row amplitude and transport ledgers."""
    manifest = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Actual collection required")
    parent, result, carrier = ROOT / manifest["spec"]["parent"], [], []
    rows = {r["unit"]: r for r in read_json(out / "00/receivers.json")}
    for entry in read_json(out / "00/design.json"):
        meta, arrays = load_arrays(out / f"01/units/unit_{entry['unit']:04d}.json")
        a, before = load_arrays(parent / f"01/units/unit_{entry['receiver']:04d}.json")
        b, _ = load_arrays(parent / f"01/units/unit_{entry['donor']:04d}.json")
        score, native, donor = (
            meta["native"]["conditional_score"],
            a["native"]["conditional_score"],
            b["native"]["conditional_score"],
        )
        result.append(
            {
                k: entry[k]
                for k in (
                    "unit",
                    "pair",
                    "role",
                    "content",
                    "person",
                    "direction",
                    "arm",
                )
            }
            | {
                "native": native,
                "donor": donor,
                "patched": score,
                "candidate_mass": meta["native"]["candidate_mass"],
                **M.effects(native, donor, score),
            }
        )
        receiver = rows[entry["receiver"]]
        for key in ("property", "endpoint", "query_end"):
            value = arrays[key].astype(float) - before[key].astype(float)
            carrier.append(
                {
                    "unit": entry["unit"],
                    "role": entry["role"],
                    "arm": entry["arm"],
                    "carrier": key,
                    "difference_norm": float(np.linalg.norm(value)),
                    "max_abs_difference": float(np.max(abs(value))),
                }
            )
        if any(i < 0 or i >= len(receiver["input_ids"]) for i in meta["positions"]):
            raise ValueError("Invalid patch address")
    groups, content = defaultdict(list), defaultdict(list)
    for row in result:
        groups[row["role"], row["arm"]].append(row)
        content[row["role"], row["content"], row["arm"]].append(row)
    write_csv(out / "03/effects.csv", result)
    write_csv(out / "03/carrier_checks.csv", carrier)
    write_csv(
        out / "03/arm_summary.csv",
        [{"role": k[0], "arm": k[1], **summarize(v)} for k, v in groups.items()],
    )
    write_csv(
        out / "03/content_summary.csv",
        [
            {"role": k[0], "content": k[1], "arm": k[2], **summarize(v)}
            for k, v in content.items()
        ],
    )
    write(
        out / "03/summary.json",
        {
            "packet": "HM49",
            "rows": len(result),
            "groups": {"/".join(k): summarize(v) for k, v in groups.items()},
            "no_op_max_error": max(
                abs(r["patch_change"]) for r in result if r["arm"] == "noop"
            ),
            "minimum_candidate_mass": min(r["candidate_mass"] for r in result),
            "known_HM48_followup_not_blind_replication": True,
            "human_personality_accuracy": False,
            "source_direction_not_unique_binding_ID": True,
            "original_scope_complete": False,
        },
    )
    paths = (
        list((out / "01").rglob("*.json"))
        + list((out / "01").rglob("*.npz"))
        + [
            out / "03" / name
            for name in (
                "effects.csv",
                "carrier_checks.csv",
                "arm_summary.csv",
                "content_summary.csv",
                "summary.json",
            )
        ]
    )
    write(
        out / "03/artifact_hashes.json",
        {str(p.relative_to(out)): digest(p) for p in paths},
    )
    print("ANALYZED:384 effects/1152 carrier checks/all16arm groups", flush=True)


def main():
    """Expose explicit finite stages without silently restarting failed acquisition."""
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
