#!/usr/bin/env python3
"""HM80: complete a role-count mechanism design without a precision search."""

from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_role_count_sectors as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_heterogeneous_evidence_cube import decode
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_role_assignment_hm79 import (
    CODE as PARENT_CODE,
)
from scripts.run_grain_role_assignment_hm79 import (
    OBSERVER_KEYS,
    seal_check,
)
from scripts.run_grain_role_assignment_hm79 import (
    check as parent_check,
)
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_role_count_sectors_hm80.json"
OUT = ROOT / "results/grain_math_response_v1/role_count_sectors_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_role_count_sectors_hm80.json",
            "docs/GRAIN_ROLE_COUNT_SECTORS_PLAN_HM80.md",
            "grain_research/construction/math_role_count_sectors.py",
            "scripts/run_grain_role_count_sectors_hm80.py",
            "scripts/verify_grain_role_count_sectors_hm80.py",
            "tests/test_grain_math_role_count_sectors.py",
            *PARENT_CODE,
        )
    )
)
IDENTITY = (
    "unit",
    "doc_id",
    "family",
    "focal",
    "companion",
    "corner",
    "coordinates",
    "query",
    "query_role",
    "person_a_events",
)


def prepare(out):
    """Freeze complete inputs, token-count orbits, code and finite resources."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve all old preparations/results")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    parent_manifest = parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM79_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM79 required")
    seal_check(parent, parent / "01/raw_seal.json")
    seal_check(parent, parent / "03/seal.json")
    material = read_json(ROOT / spec["material"])
    if spec["targets"] != list(M.TARGETS) or spec["layer"] != 16:
        raise ValueError("Fixed observer/target scope required")
    alphabet = parent_manifest["alphabet"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    if [tokenizer.encode(s, add_special_tokens=False) for s in alphabet["labels"]] != [
        [t] for t in alphabet["token_ids"]
    ]:
        raise ValueError("Candidate alphabet changed")
    rows, classes = [], {}
    docs = M.documents(material)
    for doc in docs:
        prefix = None
        for role, query in M.queries(doc["focal"], doc["companion"]).items():
            text, ids, part, _, _ = render(
                doc["body"], query, "A", 1, material["system"], tokenizer
            )
            before_query = ids[: part["body"][-1] + 1]
            if prefix is not None and prefix != before_query:
                raise ValueError("Query changed preceding evidence prefix")
            prefix = before_query
            row = {
                **doc,
                "query": query,
                "query_role": role,
                "unit": len(rows),
                "input_ids": ids,
                "rendered": text,
                "partition": part,
            }
            rows.append(row)
            key = (
                doc["family"],
                doc["focal"],
                doc["companion"],
                *doc["coordinates"][:2],
                doc["person_a_events"],
                role,
            )
            classes.setdefault(key, []).append(row["unit"])
    count_proofs = []
    for key, units in sorted(classes.items()):
        first = Counter(rows[units[0]]["input_ids"])
        if any(Counter(rows[u]["input_ids"]) != first for u in units):
            raise ValueError("Equal counts changed the actual token multiset")
        count_proofs.append(
            {"key": list(key), "units": units, "token_bag": sorted(first.items())}
        )
    lookup = {
        (r["family"], r["focal"], r["companion"], *r["coordinates"], r["query_role"]): r
        for r in rows
    }
    edits = []
    for row in rows:
        for bit in (2, 3, 4):
            if row["coordinates"][bit] != -1:
                continue
            point = row["coordinates"].copy()
            point[bit] = 1
            other = lookup[
                (
                    row["family"],
                    row["focal"],
                    row["companion"],
                    *point,
                    row["query_role"],
                )
            ]
            a, b = row["input_ids"], other["input_ids"]
            changed = [k for k, (u, v) in enumerate(zip(a, b, strict=True)) if u != v]
            if len(changed) != 1 or changed[0] not in row["partition"]["body"]:
                raise ValueError("Role change is not one body token")
            edits.append(
                {
                    "units": [row["unit"], other["unit"]],
                    "bit": bit,
                    "position": changed[0],
                }
            )
    if (
        len(rows) != spec["expected_readings"]
        or len(docs) != spec["expected_bodies"]
        or len({d["body"] for d in docs}) != len(docs)
        or len(classes) != spec["expected_count_bag_classes"]
        or len(edits) != 8640
    ):
        raise ValueError("Incomplete declared finite design")
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    for name, value in (
        ("rows", rows),
        ("count_proofs", count_proofs),
        ("edit_proofs", edits),
    ):
        write(out / f"00/{name}.json", value)
    code = {}
    for name in CODE:
        dest = out / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        code[name] = digest(dest)
    sources = [ROOT / spec["material"]] + [
        parent / n
        for n in (
            "00/manifest.json",
            "01/observer.json",
            "01/raw_seal.json",
            "03/seal.json",
            "verification.json",
            "run_status.json",
        )
    ]
    write(
        out / "00/manifest.json",
        {
            "packet": "HM80",
            "spec": spec,
            "alphabet": alphabet,
            "code": code,
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
            "assets": {
                f"00/{n}.json": digest(out / f"00/{n}.json")
                for n in ("rows", "count_proofs", "edit_proofs")
            },
            "tokenizer_files": {
                p.name: digest(p)
                for p in MODEL.glob("*token*")
                if p.is_file() and not p.name.startswith("._")
            },
            "model_config_sha256": digest(MODEL / "config.json"),
            "model_weights_separately_hashed": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "original_scope_complete": False,
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
        f"HM80 PREPARED: {len(rows)} new reads, {len(classes)} equal-count bags, {len(edits)} edits",
        flush=True,
    )


def check(out):
    """Require sealed inputs/code before reading or transforming observations."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift: " + name)
    for field, base in (("sources", ROOT), ("assets", out), ("tokenizer_files", MODEL)):
        for name, sha in manifest[field].items():
            if digest(base / name) != sha:
                raise ValueError("Frozen asset drift: " + name)
    if digest(MODEL / "config.json") != manifest["model_config_sha256"]:
        raise ValueError("Model config drift")
    return manifest


def collect(out):
    """Capture new full-model reads; keep source family seal and safe attempts."""
    manifest, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("Uncertain/stopped call needs inspection, not retry")
    rows, spec = read_json(out / "00/rows.json"), manifest["spec"]
    paths = [
        p for p in (out / "01/units").glob("unit_*.json") if not p.name.startswith("._")
    ]
    if len(paths) != status["completed_units"]:
        raise ValueError("Status/capture disagreement")
    start, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        old = read_json(ROOT / spec["parent"] / "01/observer.json")
        for key in OBSERVER_KEYS:
            if actual[key] != old[key]:
                raise ValueError("Observer changed: " + key)
        write(out / "01/observer.json", actual)
        for row in rows:
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                meta, _ = load_arrays(path)
                if any(meta[k] != row[k] for k in IDENTITY):
                    raise ValueError("Resume identity mismatch")
                continue
            if row["family"] == 1 and not (out / "01/family0_seal.json").exists():
                prior = sorted(
                    p
                    for p in (out / "01/units").glob("unit_*.*")
                    if not p.name.startswith("._")
                )
                if status["completed_units"] != 2880 or len(prior) != 5760:
                    raise ValueError("Family barrier incomplete")
                write(
                    out / "01/family0_seal.json",
                    {str(p.relative_to(out)): digest(p) for p in prior},
                )
            elapsed = previous + time.monotonic() - start
            if (
                elapsed >= spec["max_model_seconds"]
                or status["attempted_forward_calls"] >= spec["max_forwards"]
            ):
                raise TimeoutError("Declared finite resource limit")
            status.update(
                status="COLLECTING", in_flight=row["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(
                observer, row, manifest["alphabet"], spec["layer"]
            )
            save(
                path,
                {k: row[k] for k in IDENTITY} | {"native": native},
                {
                    "body_end": arrays["full"][row["partition"]["body"][-1]],
                    "query_end": arrays["full"][-1],
                },
            )
            status["forward_calls"] += 1
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - start,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 192 == 0:
                print(
                    f"HM80 ACTUAL {status['completed_units']}/5760; {status['model_seconds']:.1f}s",
                    flush=True,
                )
        if any(
            status[k] != 5760
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        ):
            raise ValueError("Incomplete actual collector")
        if observer.calls != 5760 - len(paths):
            raise ValueError("Actual observer count differs")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
        files = [
            out / "run_status.json",
            out / "01/observer.json",
            out / "01/family0_seal.json",
        ]
        files += sorted(
            p
            for p in (out / "01/units").glob("unit_*.*")
            if not p.name.startswith("._")
        )
        write(
            out / "01/raw_seal.json",
            {str(p.relative_to(out)): digest(p) for p in files},
        )
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - start,
        )
        write(out / "run_status.json", status)
        raise


def raw_tables(out):
    """Rebuild complete finite stage tables from actual identity-bound captures."""
    manifest = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Full finite collection required")
    seal_check(out, out / "01/raw_seal.json")
    seal_check(out, out / "01/family0_seal.json")
    tables, seen, prefix_error, decoder_error = {}, set(), 0.0, 0.0
    for row in read_json(out / "00/rows.json"):
        meta, arrays = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        if any(meta[k] != row[k] for k in IDENTITY):
            raise ValueError("Actual capture identity differs")
        key = (row["family"], row["focal"], row["companion"])
        if key not in tables:
            tables[key] = {
                "logits": np.full((32, 3, 10), np.nan),
                "body_endpoint": np.full((32, 1, 1536), np.nan),
                "query_endpoint": np.full((32, 3, 1536), np.nan),
            }
        table = tables[key]
        q = manifest["spec"]["query_roles"].index(row["query_role"])
        c = row["corner"]
        if (key, c, q) in seen:
            raise ValueError("Duplicate actual cell")
        seen.add((key, c, q))
        z = np.asarray(meta["native"]["candidate_logits"], dtype=float)
        decoder_error = max(
            decoder_error,
            abs(float(decode(z)["score"]) - meta["native"]["conditional_score"]),
        )
        table["logits"][c, q] = z
        body = arrays["body_end"].astype(float)
        if np.isfinite(table["body_endpoint"][c]).all():
            prefix_error = max(
                prefix_error, float(np.max(np.abs(table["body_endpoint"][c, 0] - body)))
            )
        else:
            table["body_endpoint"][c, 0] = body
        table["query_endpoint"][c, q] = arrays["query_end"].astype(float)
    if len(tables) != 60 or len(seen) != 5760:
        raise ValueError("Incomplete full table scope")
    for table in tables.values():
        if not all(np.isfinite(a).all() for a in table.values()):
            raise ValueError("Nonfinite actual table")
        decoded = decode(table["logits"])
        table["score"] = decoded["score"][..., None]
        table["log_odds"] = decoded["log_odds"][..., None]
        table["centered_logits"] = table.pop("logits")
        table["centered_logits"] -= table["centered_logits"].mean(-1, keepdims=True)
    if decoder_error > manifest["spec"]["arithmetic_tolerance"]:
        raise ValueError("Native decoder mismatch")
    return tables, {
        "body_prefix_max_error": prefix_error,
        "native_decoder_max_error": decoder_error,
    }


def analyze(out):
    """Report retained count and role sectors without fitting/choosing a winner."""
    if (out / "03").exists():
        raise FileExistsError("Preserve analyses")
    tables, raw_checks = raw_tables(out)
    manifest = read_json(out / "00/manifest.json")
    (out / "03").mkdir()
    archive, coefs, accounts, pairs, authors, groups = {}, [], [], [], [], {}
    terms = [
        "".join(v for v, b in zip(M.VARIABLES, bits, strict=True) if b) or "1"
        for bits in itertools.product((0, 1), repeat=5)
    ]
    epsilon = manifest["spec"]["profile_epsilon"]
    profile_error, group_error = 0.0, 0.0
    for context, (key, table) in enumerate(sorted(tables.items())):
        family, focal, companion = key
        query_map = M.queries(focal, companion)
        for stage in M.STAGES:
            values = table[stage]
            e = M.sectors(values)
            group_error = max(
                group_error,
                float(np.max(np.abs(e["count"] - M.permutation_projection(values)))),
            )
            for name in (
                "coefficients",
                "count",
                "role",
                "pair_coefficients",
                "pair_count",
                "pair_role",
            ):
                archive[f"{name}_{context:02d}_{stage}"] = e[name]
            archive[f"values_{context:02d}_{stage}"] = values
            roles = (
                ("query_independent",) if stage == "body_endpoint" else tuple(query_map)
            )
            for q, role in enumerate(roles):
                common = {
                    "context": context,
                    "family": family,
                    "focal": focal,
                    "companion": companion,
                    "query_role": role,
                    "query": query_map.get(role, role),
                    "stage": stage,
                }
                for k, term in enumerate(terms):
                    c = e["coefficients"][k, q]
                    coefs.append(
                        common
                        | {
                            "term": term,
                            "rms": float(np.sqrt(np.mean(c * c))),
                            "scalar": float(c[0]) if c.size == 1 else None,
                        }
                    )
                for k, edge in enumerate(M.EDGES):
                    c, r = e["pair_coefficients"][k, q], e["pair_role"][k, q]
                    pairs.append(
                        common
                        | {
                            "edge": edge,
                            "coefficient_rms": float(np.sqrt(np.mean(c * c))),
                            "coefficient_scalar": float(c[0]) if c.size == 1 else None,
                            "count_mean_rms": float(
                                np.sqrt(np.mean(e["pair_count"][k, q] ** 2))
                            ),
                            "role_rms": float(np.sqrt(np.mean(r * r))),
                            "role_scalar": float(r[0]) if r.size == 1 else None,
                        }
                    )
                for first, second in itertools.combinations(M.EDGES, 2):
                    plus, minus = M.profiles(first, second, epsilon)
                    difference = plus @ values[:, q] - minus @ values[:, q]
                    k, l = M.EDGES.index(first), M.EDGES.index(second)
                    formula = epsilon * (
                        e["pair_coefficients"][k, q] - e["pair_coefficients"][l, q]
                    )
                    err = float(np.max(np.abs(difference - formula)))
                    profile_error = max(profile_error, err)
                    authors.append(
                        common
                        | {
                            "first": first,
                            "second": second,
                            "difference_rms": float(np.sqrt(np.mean(difference**2))),
                            "difference_scalar": float(difference[0])
                            if difference.size == 1
                            else None,
                            "formula_max_error": err,
                        }
                    )
                v, c, r = values[:, q], e["count"][:, q], e["role"][:, q]
                ac = common | {
                    "total_variance": float(np.mean((v - v.mean(0)) ** 2)),
                    "count_variance": float(np.mean((c - c.mean(0)) ** 2)),
                    "role_variance": float(np.mean(r * r)),
                    "cross_inner_mean": float(np.mean((c - c.mean(0)) * r)),
                    "pair_count_scaled_mean_square": float(
                        np.mean((8 * e["pair_count"][:, q]) ** 2)
                    ),
                    "pair_role_scaled_mean_square": float(
                        np.mean((8 * e["pair_role"][:, q]) ** 2)
                    ),
                }
                accounts.append(ac)
                keys = [(family, stage, role)]
                if role != "query_independent":
                    keys.append((family, stage, "all_queries"))
                for group in keys:
                    groups.setdefault(group, []).append(ac)
    summaries = []
    fields = (
        "total_variance",
        "count_variance",
        "role_variance",
        "cross_inner_mean",
        "pair_count_scaled_mean_square",
        "pair_role_scaled_mean_square",
    )
    for (family, stage, role), rows in sorted(groups.items()):
        row = {
            "family": family,
            "stage": stage,
            "query_role": role,
            "contexts_queries": len(rows),
        }
        row.update(
            {field: float(np.mean([r[field] for r in rows])) for field in fields}
        )
        row["role_variation_fraction"] = (
            row["role_variance"] / row["total_variance"]
            if row["total_variance"]
            else None
        )
        row["pair_count_scaled_rms"] = float(
            np.sqrt(row["pair_count_scaled_mean_square"])
        )
        row["pair_role_scaled_rms"] = float(
            np.sqrt(row["pair_role_scaled_mean_square"])
        )
        summaries.append(row)
    np.savez_compressed(out / "03/tables_and_sectors.npz", **archive)
    for name, rows in (
        ("coefficients", coefs),
        ("count_accounts", accounts),
        ("pair_roles", pairs),
        ("author_compositions", authors),
        ("summary_by_group", summaries),
    ):
        write_csv(out / f"03/{name}.csv", rows)
    write(
        out / "03/summary.json",
        dict(
            packet="HM80",
            contexts=60,
            context_keys=[list(k) for k in sorted(tables)],
            actual_new_forwards=5760,
            old_neural_reuses=0,
            model_seconds=read_json(out / "run_status.json")["model_seconds"],
            group_average_max_error=group_error,
            profile_max_error=profile_error,
            stages=list(M.STAGES),
            summary_by_group=summaries,
            **raw_checks,
            new_fits_training_API_private=0,
            full_six_query_cube_per_context=False,
            count_compatible_not_discarded=True,
            slot_specific_attention_not_eliminated=True,
            profiles_are_hypothetical_not_human=True,
            original_scope_complete=False,
        ),
    )
    files = sorted(
        p for p in (out / "03").iterdir() if p.is_file() and not p.name.startswith("._")
    )
    write(out / "03/seal.json", {str(p.relative_to(out)): digest(p) for p in files})
    print(
        f"HM80 ANALYZED:60 full five-variable contexts; S3 error{group_error:.3g}",
        flush=True,
    )


def main():
    """Expose finite acquisition and analysis without automatic continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == "collect":
        with collector_guard(args.out):
            collect(args.out)
    else:
        {"prepare": prepare, "analyze": analyze}[args.action](args.out)


if __name__ == "__main__":
    main()
