#!/usr/bin/env python3
"""Acquire and analyze finite role-bound behaviour/appraisal on identical token bags."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_construct_routing as C
from grain_research.construction import math_relational_appraisal as M
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import observe, save, write
from scripts.run_grain_math_evaluative_routing import CODE as PARENT_CODE
from scripts.run_grain_math_evaluative_routing import check as parent_check
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_relational_appraisal_v1.json"
OUT = ROOT / "results/grain_math_response_v1/relational_appraisal_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_relational_appraisal_v1.json",
            "docs/GRAIN_RELATIONAL_APPRAISAL_PLAN_V1.md",
            "grain_research/construction/math_relational_appraisal.py",
            "scripts/run_grain_math_relational_appraisal.py",
            "scripts/verify_grain_math_relational_appraisal.py",
            "tests/test_grain_math_relational_appraisal.py",
            "scripts/run_grain_math_evaluative_routing.py",
            *PARENT_CODE,
        )
    )
)


def bag_key(row):
    """Keep query-person identity fixed while comparing identical whole-prompt bags."""
    return (
        row["role"],
        row["content"],
        row["query"],
        row["person"],
        row["appraisal"] == "none",
    )


def cube_key(row):
    """Define one complete role/content/query response cube."""
    return row["role"], row["content"], row["query"]


def prepare(out):
    """Freeze all material and prove actual token-bag and causal-prefix contracts."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve existing versions")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM47_NOT_GLOBAL"
    ):
        raise ValueError(
            "Require actually verified HM47 before following its mechanism"
        )
    materials = read_json(ROOT / spec["material_source"])
    if materials["targets"] != spec["targets"]:
        raise ValueError("Changed ordered content/query set")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = {"labels": list("0123456789"), "values": (np.arange(10) / 9).tolist()}
    encoded_digits = [
        tokenizer.encode(t, add_special_tokens=False) for t in alphabet["labels"]
    ]
    if (
        any(len(t) != 1 for t in encoded_digits)
        or len({t[0] for t in encoded_digits}) != 10
    ):
        raise ValueError("Distinct native single-token digits required")
    alphabet["token_ids"] = [t[0] for t in encoded_digits]
    docs, rows, bags, prefixes = M.documents(materials), [], {}, {}
    for doc in docs:
        for person, query in itertools.product(("A", "B"), spec["targets"]):
            rendered, enc, offset = M.render(
                doc["body"], query, person, spec["system"], tokenizer
            )
            positions = {
                actor: C.token_positions(
                    enc["offset_mapping"], offset + interval[0], offset + interval[1]
                )
                for actor, interval in doc["properties"].items()
            }
            body_positions = C.token_positions(
                enc["offset_mapping"], offset, offset + len(doc["body"])
            )
            row = {
                **doc,
                "unit": len(rows),
                "person": person,
                "query": query,
                "rendered": rendered,
                "input_ids": list(enc["input_ids"]),
                "property_positions": positions["A"] + positions["B"],
                "property_a_count": len(positions["A"]),
                "body_endpoint": body_positions[-1],
            }
            key, bag = bag_key(row), Counter(row["input_ids"])
            if key in bags and bags[key] != bag:
                raise ValueError("Actual whole-prompt token bags differ")
            bags[key] = bag
            first = positions[doc["first_actor"]]
            prefix = row["input_ids"][: first[-1] + 1]
            pk = cube_key(row) + (doc["behaviour"], doc["order"], person)
            if pk in prefixes and prefixes[pk] != prefix:
                raise ValueError(
                    "Future appraisal altered the first-action token prefix"
                )
            prefixes[pk] = prefix
            rows.append(row)
    if (
        len(rows) != spec["expected_scores"]
        or len(docs) != spec["expected_documents"]
        or len(bags) != 288
    ):
        raise ValueError("Incomplete planned scope")
    for folder in ("00/code", "01/units", "03"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    write(out / "00/design.json", rows)
    hashes = {}
    for name in CODE:
        target = out / "00/code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        hashes[name] = digest(target)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM48",
            "spec": spec,
            "alphabet": alphabet,
            "code": hashes,
            "design_sha256": digest(out / "00/design.json"),
            "parent_manifest_sha256": digest(parent / "00/manifest.json"),
            "parent_verification_sha256": digest(parent / "verification.json"),
            "material_source_sha256": digest(ROOT / spec["material_source"]),
            "model_config_sha256": digest(MODEL / "config.json"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "actual_token_bag_classes": len(bags),
            "all_forwards_fresh": True,
            "new_neutral_system_not_HM47_replication": True,
            "historical_model_weights_not_separately_hashed": True,
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
        f"PREPARED: {len(docs)} accounts/{len(rows)} fresh scores/{len(bags)} exact-token-bag classes",
        flush=True,
    )


def check(out):
    """Check declared material, code and direct parent identity, not the entire old queue."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Changed frozen code " + name)
    if (
        digest(out / "00/design.json") != manifest["design_sha256"]
        or digest(MODEL / "config.json") != manifest["model_config_sha256"]
    ):
        raise ValueError("Changed materials or observer config")
    parent = ROOT / manifest["spec"]["parent"]
    for name, key in (
        ("00/manifest.json", "parent_manifest_sha256"),
        ("verification.json", "parent_verification_sha256"),
    ):
        if digest(parent / name) != manifest[key]:
            raise ValueError("Changed parent identity")
    return manifest


def collect(out):
    """Acquire a bounded actual source/application battery with safe per-call checkpoints."""
    manifest = check(out)
    spec, status = manifest["spec"], read_json(out / "run_status.json")
    if status["in_flight"] is not None:
        raise RuntimeError("Uncertain call: inspect, never automatically repeat")
    if status["status"] == "COLLECTED":
        return
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        write(out / "01/observer.json", observer.manifest())
        rows = read_json(out / "00/design.json")
        for role in ("source", "application"):
            if role == "application":
                barrier = read_json(out / "01/source_barrier.json")
                for name, sha in barrier["files"].items():
                    if digest(out / name) != sha:
                        raise ValueError("Changed sealed source")
            for row in rows:
                if row["role"] != role:
                    continue
                path = out / f"01/units/unit_{row['unit']:04d}.json"
                if path.exists():
                    load_arrays(path)
                    continue
                if (
                    status["attempted_forward_calls"] >= spec["max_forwards"]
                    or previous + time.monotonic() - started
                    >= spec["max_model_seconds"]
                ):
                    raise TimeoutError("Finite resource contract reached")
                status.update(
                    status="COLLECTING",
                    in_flight=f"native/{row['unit']}",
                    model_seconds=previous + time.monotonic() - started,
                )
                status["attempted_forward_calls"] += 1
                write(out / "run_status.json", status)
                native, arrays = observe(
                    observer, row, manifest["alphabet"], spec["layer"]
                )
                status["forward_calls"] += 1
                save(
                    path, {"unit": row["unit"], "role": role, "native": native}, arrays
                )
                status.update(
                    in_flight=None,
                    completed_units=status["completed_units"] + 1,
                    model_seconds=previous + time.monotonic() - started,
                )
                write(out / "run_status.json", status)
                if status["completed_units"] % 144 == 0:
                    print(
                        f"{role}: {status['completed_units']} actual forwards",
                        flush=True,
                    )
            if role == "source" and not (out / "01/source_barrier.json").exists():
                paths = [
                    out / f"01/units/unit_{r['unit']:04d}.json"
                    for r in rows
                    if r["role"] == role
                ]
                if len(paths) != spec["source_scores"]:
                    raise ValueError("Full source required before application")
                write(
                    out / "01/source_barrier.json",
                    {
                        "source_scores": len(paths),
                        "sealed_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                        "files": {
                            str(p.relative_to(out)): digest(p)
                            for path in paths
                            for p in (path, path.with_suffix(".npz"))
                        },
                    },
                )
        if (
            status["forward_calls"] != spec["expected_scores"]
            or status["completed_units"] != spec["expected_scores"]
        ):
            raise ValueError("Actual finite scope incomplete")
        status.update(
            status="COLLECTED", model_seconds=previous + time.monotonic() - started
        )
        write(out / "run_status.json", status)
        print(json.dumps(status), flush=True)
    except BaseException as exc:
        status.update(
            status="STOPPED_WITH_CHECKPOINT",
            error=f"{type(exc).__name__}: {exc}",
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def rms(values):
    """Use explicit scalar-score RMS, not human variance or accuracy."""
    a = np.asarray(values, float)
    return float(np.sqrt(np.mean(a * a)))


def analyze(out):
    """Separate query-bound, shared-order and name-asymmetric channels without fits."""
    manifest = check(out)
    status = read_json(out / "run_status.json")
    if status["status"] != "COLLECTED" or status["in_flight"] is not None:
        raise ValueError("Complete actual acquisition required")
    design = read_json(out / "00/design.json")
    main, absent, bags = {}, {}, defaultdict(list)
    native_rows, coefficients, contrasts, captures = [], [], [], {}
    for row in design:
        meta, arrays = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        y = meta["native"]["conditional_score"]
        native_rows.append(
            {
                k: row[k]
                for k in (
                    "unit",
                    "doc_id",
                    "role",
                    "content",
                    "query",
                    "person",
                    "behaviour",
                    "appraisal",
                    "order",
                )
            }
            | {"score": y, "candidate_mass": meta["native"]["candidate_mass"]}
        )
        key = cube_key(row)
        b, o, r = row["behaviour"], row["order"], int(row["person"] == "B")
        if row["appraisal"] == "none":
            absent.setdefault(key, np.full((2,) * 3, np.nan))[b, o, r] = y
        else:
            e = int(row["appraisal"] == "B")
            main.setdefault(key, np.full((2,) * 4, np.nan))[b, e, o, r] = y
        bags[bag_key(row)].append(y)
        captures[key + (b, o, r, row["appraisal"])] = row, arrays
    group_diagnostics, matrices = [], {}
    for key in sorted(main):
        role, content, query = key
        for presence, table in (("present", main[key]), ("absent", absent[key])):
            decomposition = M.decompose(table)
            beta = dict(
                zip(decomposition["bits"], decomposition["coefficients"], strict=True)
            )
            for mask, value in beta.items():
                coefficients.append(
                    {
                        "role": role,
                        "content": content,
                        "query": query,
                        "presence": presence,
                        "mask": "".join(map(str, mask)),
                        "value": float(value),
                    }
                )
            fields = {
                k: v
                for k, v in decomposition.items()
                if k not in ("bits", "coefficients")
            }
            group_diagnostics.append(
                {
                    "role": role,
                    "content": content,
                    "query": query,
                    "presence": presence,
                    **fields,
                }
            )
            channels = {
                "behaviour_bound": (1, 0, 0, 1) if presence == "present" else (1, 0, 1),
                "behaviour_order_shared": (1, 0, 1, 0)
                if presence == "present"
                else (1, 1, 0),
                "query_order": (0, 0, 1, 1) if presence == "present" else (0, 1, 1),
            }
            if presence == "present":
                channels.update(
                    appraisal_bound=(0, 1, 0, 1), appraisal_order_shared=(0, 1, 1, 0)
                )
            for name, mask in channels.items():
                value = float(2 * beta[mask])
                contrasts.append(
                    {
                        "role": role,
                        "content": content,
                        "query": query,
                        "presence": presence,
                        "channel": name,
                        "value": value,
                    }
                )
                matrices.setdefault((role, presence, name), {})[content, query] = value
    bounds = [
        {
            "role": k[0],
            "content": k[1],
            "query": k[2],
            "person": k[3],
            "presence": "absent" if k[4] else "present",
            **M.bag_bound(y),
        }
        for k, y in sorted(bags.items())
    ]
    carrier_rows = []
    for key, (row, arrays) in captures.items():
        if row["appraisal"] == "none":
            continue
        _, baseline = captures[key[:-1] + ("none",)]
        split = row["property_a_count"]
        segment = slice(0, split) if row["first_actor"] == "A" else slice(split, None)
        for name, x, z in (
            (
                "first_property",
                baseline["property"][segment],
                arrays["property"][segment],
            ),
            ("endpoint", baseline["endpoint"], arrays["endpoint"]),
            ("query_end", baseline["query_end"], arrays["query_end"]),
        ):
            delta = np.asarray(z, float) - np.asarray(x, float)
            carrier_rows.append(
                {
                    "unit": row["unit"],
                    "role": row["role"],
                    "carrier": name,
                    "maximum_absolute_change": float(np.max(abs(delta))),
                    "change_norm": float(np.linalg.norm(delta)),
                }
            )
    targets = manifest["spec"]["targets"]
    matrix_rows, summary_groups = [], []
    for (role, presence, name), cells in sorted(matrices.items()):
        a = np.asarray([[cells[c, q] for q in targets] for c in targets], float)
        summary_groups.append(
            {
                "role": role,
                "presence": presence,
                "channel": name,
                "rms": rms(a),
                "diagonal": np.diag(a).tolist(),
                "matrix": a.tolist(),
            }
        )
        matrix_rows.extend(
            {
                "role": role,
                "presence": presence,
                "channel": name,
                "content": c,
                "query": q,
                "value": cells[c, q],
            }
            for c, q in itertools.product(targets, repeat=2)
        )
    for filename, values in (
        ("readouts.csv", native_rows),
        ("coefficients.csv", coefficients),
        ("contrasts.csv", contrasts),
        ("token_bag_bounds.csv", bounds),
        ("group_diagnostics.csv", group_diagnostics),
        ("carrier_changes.csv", carrier_rows),
        ("response_matrices.csv", matrix_rows),
    ):
        write_csv(out / "03" / filename, values)
    summary = {
        "status": "FINITE_HM48_ACQUIRED_AND_ANALYZED_NOT_GLOBAL",
        "run_status": status,
        "accounts": 144,
        "scores": len(native_rows),
        "coefficient_rows": len(coefficients),
        "bag_classes": len(bounds),
        "minimum_candidate_mass": min(r["candidate_mass"] for r in native_rows),
        "maximum_first_property_numerical_change_reported": max(
            r["maximum_absolute_change"]
            for r in carrier_rows
            if r["carrier"] == "first_property"
        ),
        "maximum_reconstruction_error": max(
            g["reconstruction_error"] for g in group_diagnostics
        ),
        "maximum_parseval_error": max(g["parseval_error"] for g in group_diagnostics),
        "group_summary": summary_groups,
        "new_fits": 0,
        "body_binding_not_causal_Binding_ID_localization": True,
        "new_system_not_direct_HM47_score_comparison": True,
        "human_personality_accuracy": False,
    }
    write(out / "03/summary.json", summary)
    write(
        out / "03/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for folder in ("00", "01", "03")
            for p in (out / folder).rglob("*")
            if p.is_file()
            and not p.name.startswith("._")
            and p.name != "artifact_hashes.json"
        },
    )
    print(
        json.dumps({k: v for k, v in summary.items() if k != "group_summary"}),
        flush=True,
    )


def main():
    """Run one explicit bounded stage, never dispatch a new model-search loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        {"prepare": prepare, "collect": collect, "analyze": analyze}[args.stage](
            args.out
        )


if __name__ == "__main__":
    main()
