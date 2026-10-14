#!/usr/bin/env python3
"""HM79: complete the aliased actor design, with no observer-accuracy search."""

from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_role_assignment as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_heterogeneous_evidence_cube import decode
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_role_assignment_hm79.json"
OUT = ROOT / "results/grain_math_response_v1/role_assignment_dealiasing_r1"
CODE = (
    "configs/grain_role_assignment_hm79.json",
    "docs/GRAIN_ROLE_ASSIGNMENT_DEALIASING_PLAN_HM79.md",
    "grain_research/construction/math_role_assignment.py",
    "scripts/run_grain_role_assignment_hm79.py",
    "scripts/verify_grain_role_assignment_hm79.py",
    "tests/test_grain_math_role_assignment.py",
    "grain_research/construction/math_ownership_composition.py",
    "grain_research/construction/math_heterogeneous_evidence_cube.py",
    "grain_research/construction/math_cross_target_operator.py",
    "grain_research/construction/math_answer_recoding_mechanism.py",
    "grain_research/construction/math_passive_event_composition.py",
    "grain_research/vector_mechanisms/navigation.py",
    "grain_research/vector_mechanisms/direct_traits.py",
    "grain_research/vector_mechanisms/trace.py",
    "scripts/run_grain_math_construct_routing.py",
    "scripts/run_grain_math_evaluative_routing.py",
    "scripts/run_grain_math_relational_cut.py",
    "scripts/run_grain_math_response.py",
    "scripts/run_grain_math_span_binding.py",
    "scripts/run_grain_upstream_routes.py",
)
IDENTITY = (
    "unit",
    "doc_id",
    "family",
    "focal",
    "companion",
    "corner",
    "query",
    "coordinates",
)
OBSERVER_KEYS = (
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
)


def seal_check(base, path):
    """Check named hashes rather than treating the existence of a seal as evidence."""
    for name, sha in read_json(path).items():
        if digest(base / name) != sha:
            raise ValueError("Sealed file drift: " + name)


def prepare(out):
    """Bind all old/new readings and independent actor edits before acquisition."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Keep existing preparations and failures")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM68_NOT_GLOBAL"
    ):
        raise ValueError("Verified frozen HM68 required")
    seal_check(parent, parent / "01/raw_seal.json")
    material = read_json(ROOT / spec["material"])
    pm = read_json(parent / "00/manifest.json")
    if digest(ROOT / spec["material"]) != pm["source_files"][spec["material"]]:
        raise ValueError("Original behaviour bank drift")
    if spec["targets"] != list(M.TARGETS) or spec["layer"] != pm["layer"]:
        raise ValueError("Fixed target/observer scope required")
    alphabet = pm["alphabet"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    if [tokenizer.encode(s, add_special_tokens=False) for s in alphabet["labels"]] != [
        [t] for t in alphabet["token_ids"]
    ]:
        raise ValueError("Candidate alphabet drift")
    old_rows = read_json(parent / "00/rows.json")
    old = {
        (r["family"], r["focal"], r["companion"], *r["coordinates"], r["query"]): r
        for r in old_rows
        if r["companion"] != "neutral"
    }
    rows, proofs, new_count, reused = [], [], 0, 0
    for doc in M.documents(material):
        prefix = None
        for query in M.TARGETS:
            text, ids, part, _, _ = render(
                doc["body"], query, "A", 1, material["system"], tokenizer
            )
            current = ids[: part["body"][-1] + 1]
            if prefix is not None and prefix != current:
                raise ValueError("Future query changed the evidence prefix")
            prefix = current
            x, y, g, h = doc["coordinates"]
            historical = (
                old[(doc["family"], doc["focal"], doc["companion"], x, y, g, query)]
                if h == -g
                else None
            )
            if historical is not None:
                if any(
                    historical[k] != v
                    for k, v in {
                        "body": doc["body"],
                        "rendered": text,
                        "input_ids": ids,
                        "partition": part,
                    }.items()
                ):
                    raise ValueError("Old fraction is not byte/token-identical")
                reused += 1
            row = {
                **doc,
                "unit": len(rows),
                "query": query,
                "input_ids": ids,
                "rendered": text,
                "partition": part,
                "old_unit": historical["unit"] if historical is not None else None,
                "new_unit": new_count if historical is None else None,
            }
            new_count += int(historical is None)
            rows.append(row)
    lookup = {
        (r["family"], r["focal"], r["companion"], *r["coordinates"], r["query"]): r
        for r in rows
    }
    for row in rows:
        for bit, event in ((2, "K2"), (3, "K3")):
            if row["coordinates"][bit] != -1:
                continue
            coordinates = row["coordinates"].copy()
            coordinates[bit] = 1
            partner = lookup[
                (
                    row["family"],
                    row["focal"],
                    row["companion"],
                    *coordinates,
                    row["query"],
                )
            ]
            ia, ib = row["input_ids"], partner["input_ids"]
            if len(ia) != len(ib):
                raise ValueError("Actor flip changed token opportunity")
            positions = [
                i for i, (a, b) in enumerate(zip(ia, ib, strict=True)) if a != b
            ]
            if len(positions) != 1 or positions[0] not in row["partition"]["body"]:
                raise ValueError("Actor flip must edit one body token")
            a_lines, b_lines = row["body"].splitlines(), partner["body"].splitlines()
            changed = [i for i in range(3) if a_lines[i] != b_lines[i]]
            if changed != [bit - 1]:
                raise ValueError("Wrong event changed by independent actor flip")
            proofs.append(
                {
                    "units": [row["unit"], partner["unit"]],
                    "bit": bit,
                    "event": event,
                    "position": positions[0],
                }
            )
    if (
        len(rows) != spec["expected_readings"]
        or new_count != spec["max_forwards"]
        or reused != spec["expected_reused_readings"]
    ):
        raise ValueError("Incomplete full factorial scope")
    if (
        len({r["body"] for r in rows if r["old_unit"] is None})
        != spec["expected_new_bodies"]
    ):
        raise ValueError("Incomplete fresh material roster")
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/edit_proofs.json", proofs)
    code = {}
    for name in CODE:
        dest = out / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        code[name] = digest(dest)
    source_paths = [ROOT / spec["material"]] + [
        parent / name
        for name in (
            "00/rows.json",
            "00/manifest.json",
            "01/raw_seal.json",
            "01/observer.json",
            "run_status.json",
            "verification.json",
        )
    ]
    write(
        out / "00/manifest.json",
        {
            "packet": "HM79",
            "spec": spec,
            "alphabet": alphabet,
            "code": code,
            "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
            "assets": {
                n: digest(out / n) for n in ("00/rows.json", "00/edit_proofs.json")
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
        f"HM79 PREPARED: {len(rows)} readings, {reused} reused, {new_count} new; {len(proofs)} independent one-token edits",
        flush=True,
    )


def check(out):
    """Verify frozen inputs and executable code before collection or analysis."""
    manifest = read_json(out / "00/manifest.json")
    for name, sha in manifest["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift: " + name)
    for key, base in (("assets", out), ("sources", ROOT), ("tokenizer_files", MODEL)):
        for name, sha in manifest[key].items():
            if digest(base / name) != sha:
                raise ValueError("Frozen asset drift: " + name)
    if digest(MODEL / "config.json") != manifest["model_config_sha256"]:
        raise ValueError("Model config drift")
    parent = ROOT / manifest["spec"]["parent"]
    seal_check(parent, parent / "01/raw_seal.json")
    return manifest


def collect(out):
    """Acquire only the missing fraction, preserving attempts and safe checkpoints."""
    manifest, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("Stopped/uncertain attempts require inspection, not retry")
    spec = manifest["spec"]
    fresh = [r for r in read_json(out / "00/rows.json") if r["new_unit"] is not None]
    paths = [
        p for p in (out / "01/units").glob("unit_*.json") if not p.name.startswith("._")
    ]
    if len(paths) != status["completed_units"]:
        raise ValueError("Checkpoint/capture disagreement")
    start, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        old = read_json(ROOT / spec["parent"] / "01/observer.json")
        for key in OBSERVER_KEYS:
            if actual[key] != old[key]:
                raise ValueError("Observer drift: " + key)
        write(out / "01/observer.json", actual)
        for row in fresh:
            path = out / f"01/units/unit_{row['new_unit']:04d}.json"
            if path.exists():
                meta, _ = load_arrays(path)
                if any(meta[k] != row[k] for k in IDENTITY):
                    raise ValueError("Resume identity mismatch")
                continue
            elapsed = previous + time.monotonic() - start
            if (
                status["attempted_forward_calls"] >= spec["max_forwards"]
                or elapsed >= spec["max_model_seconds"]
            ):
                raise TimeoutError("Declared finite collector bound")
            status.update(
                status="COLLECTING", in_flight=row["new_unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(
                observer, row, manifest["alphabet"], spec["layer"]
            )
            save(
                path,
                {k: row[k] for k in IDENTITY}
                | {"new_unit": row["new_unit"], "native": native},
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
            if status["completed_units"] % 144 == 0:
                print(
                    f"HM79 ACTUAL {status['completed_units']}/{spec['max_forwards']}; {status['model_seconds']:.1f}s",
                    flush=True,
                )
        if any(
            status[k] != spec["max_forwards"]
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        ):
            raise ValueError("Incomplete fresh collection")
        if observer.calls != spec["max_forwards"] - len(paths):
            raise ValueError("Actual observer call count disagreement")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
        files = [out / "run_status.json", out / "01/observer.json"] + sorted(
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
    """Rebuild every stage from identity-bound old and fresh actual captures."""
    manifest = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Complete acquisition required")
    seal_check(out, out / "01/raw_seal.json")
    parent = ROOT / manifest["spec"]["parent"]
    old_rows = {r["unit"]: r for r in read_json(parent / "00/rows.json")}
    tables, seen, prefix_error, decoder_error = {}, set(), 0.0, 0.0
    for row in read_json(out / "00/rows.json"):
        reused = row["old_unit"] is not None
        path = (
            parent / f"01/units/unit_{row['old_unit']:04d}.json"
            if reused
            else out / f"01/units/unit_{row['new_unit']:04d}.json"
        )
        meta, arrays = load_arrays(path)
        if reused:
            old = old_rows[row["old_unit"]]
            for k in (
                "family",
                "focal",
                "companion",
                "query",
                "body",
                "input_ids",
                "partition",
                "rendered",
            ):
                if row[k] != old[k]:
                    raise ValueError("Old reading binding drift")
            for k in (
                "unit",
                "doc_id",
                "family",
                "focal",
                "companion",
                "corner",
                "query",
            ):
                if meta[k] != old[k]:
                    raise ValueError("Old capture identity drift")
        elif any(meta[k] != row[k] for k in IDENTITY):
            raise ValueError("Fresh capture identity drift")
        key = (row["family"], row["focal"], row["companion"])
        table = tables.setdefault(
            key,
            {
                "logits": np.full((16, 6, 10), np.nan),
                "body_endpoint": np.full((16, 1, 1536), np.nan),
                "query_endpoint": np.full((16, 6, 1536), np.nan),
            },
        )
        q, c = M.TARGETS.index(row["query"]), row["corner"]
        if (key, c, q) in seen:
            raise ValueError("Duplicate design reading")
        seen.add((key, c, q))
        z = np.asarray(meta["native"]["candidate_logits"], dtype=float)
        decoder_error = max(
            decoder_error,
            float(abs(decode(z)["score"] - meta["native"]["conditional_score"])),
        )
        table["logits"][c, q] = z
        body = arrays["body_end"].astype(float)
        if np.isfinite(table["body_endpoint"][c]).all():
            prefix_error = max(
                prefix_error, float(np.max(np.abs(body - table["body_endpoint"][c, 0])))
            )
        else:
            table["body_endpoint"][c, 0] = body
        table["query_endpoint"][c, q] = arrays["query_end"].astype(float)
    if (
        len(tables) != manifest["spec"]["expected_contexts"]
        or len(seen) != manifest["spec"]["expected_readings"]
    ):
        raise ValueError("Incomplete context roster")
    for table in tables.values():
        if not all(np.isfinite(v).all() for v in table.values()):
            raise ValueError("Missing finite table cell")
        decoded = decode(table["logits"])
        table["score"] = decoded["score"][..., None]
        table["log_odds"] = decoded["log_odds"][..., None]
        table["centered_logits"] = table.pop("logits")
        table["centered_logits"] -= table["centered_logits"].mean(-1, keepdims=True)
    if decoder_error > manifest["spec"]["arithmetic_tolerance"]:
        raise ValueError("Native decoder disagreement")
    return tables, {
        "body_prefix_max_error": prefix_error,
        "native_decoder_max_error": decoder_error,
    }


def analyze(out):
    """Save all coefficients, conditional effects and fixed author consequences."""
    if (out / "03").exists():
        raise FileExistsError("Preserve completed analyses")
    tables, checks = raw_tables(out)
    manifest = read_json(out / "00/manifest.json")
    (out / "03").mkdir()
    archive, coef_rows, gate_rows, accounts, profiles, group_values = (
        {},
        [],
        [],
        [],
        [],
        {},
    )
    epsilon = manifest["spec"]["profile_epsilon"]
    pplus, pminus = M.profile_pair(epsilon)
    terms = [
        "".join(v for v, b in zip(M.VARIABLES, bits, strict=True) if b) or "1"
        for bits in itertools.product((0, 1), repeat=4)
    ]
    alias_error = 0.0
    for context, (key, table) in enumerate(sorted(tables.items())):
        family, focal, companion = key
        for stage in M.STAGES:
            values = table[stage]
            e = M.effects(values)
            c = e["coefficients"]
            archive[f"values_{context:02d}_{stage}"] = values
            archive[f"coefficients_{context:02d}_{stage}"] = c
            alias_error = max(
                alias_error,
                float(np.max(np.abs(e["old_coefficients"] - e["alias_from_full"]))),
            )
            labels = ("query_independent",) if stage == "body_endpoint" else M.TARGETS
            for q, query in enumerate(labels):
                common = {
                    "context": context,
                    "family": family,
                    "focal": focal,
                    "companion": companion,
                    "query": query,
                    "stage": stage,
                }
                for i, term in enumerate(terms):
                    coef_rows.append(
                        {
                            **common,
                            "term": term,
                            "rms": float(np.sqrt(np.mean(c[i, q] ** 2))),
                            "scalar": float(c[i, q, 0])
                            if values.shape[-1] == 1
                            else None,
                        }
                    )
                for g in (-1, 1):
                    for h in (-1, 1):
                        for role, gate in (
                            ("companion", e["companion_gate_at_h"][int(h > 0), q]),
                            ("anchor", e["anchor_gate_at_g"][int(g > 0), q]),
                        ):
                            gate_rows.append(
                                {
                                    **common,
                                    "g": g,
                                    "h": h,
                                    "edited_role": role,
                                    "gate_rms": float(np.sqrt(np.mean(gate**2))),
                                    "gate_scalar": float(gate[0])
                                    if gate.size == 1
                                    else None,
                                }
                            )
                a, b, old = c[14, q], c[13, q], e["old_coefficients"][7, q]
                accounts.append(
                    {
                        **common,
                        "old_energy": float(np.mean(old**2)),
                        "companion_energy": float(np.mean(a**2)),
                        "anchor_energy": float(np.mean(b**2)),
                        "signed_cross_term": float(-2 * np.mean(a * b)),
                        "four_way_energy": float(np.mean(c[15, q] ** 2)),
                    }
                )
                difference = pplus @ values[:, q] - pminus @ values[:, q]
                profiles.append(
                    {
                        **common,
                        "difference_rms": float(np.sqrt(np.mean(difference**2))),
                        "difference_scalar": float(difference[0])
                        if difference.size == 1
                        else None,
                        "formula_max_error": float(
                            np.max(np.abs(difference - 2 * epsilon * c[15, q]))
                        ),
                    }
                )
                groups = ["all_queries"]
                if query == "query_independent":
                    groups = ["query_independent"]
                elif query == focal:
                    groups.append("focal_query")
                elif query == companion:
                    groups.append("companion_query")
                else:
                    groups.append("other_queries")
                for group in groups:
                    group_values.setdefault((family, stage, group), []).append(
                        {
                            "old_gate": e["old_gate"][q],
                            "companion_only": 8 * c[14, q],
                            "anchor_only": 8 * c[13, q],
                            "joint_modulation": 8 * c[15, q],
                            "companion_h_low": e["companion_gate_at_h"][0, q],
                            "companion_h_high": e["companion_gate_at_h"][1, q],
                            "anchor_g_low": e["anchor_gate_at_g"][0, q],
                            "anchor_g_high": e["anchor_gate_at_g"][1, q],
                            "companion_main_role": 4 * c[6, q],
                            "anchor_main_role": 4 * c[5, q],
                        }
                    )
    summary_rows = []
    for (family, stage, group), entries in sorted(group_values.items()):
        row = {
            "family": family,
            "stage": stage,
            "group": group,
            "contexts_queries": len(entries),
        }
        for field in entries[0]:
            values = np.asarray([entry[field] for entry in entries])
            row[field + "_rms"] = float(np.sqrt(np.mean(values**2)))
            row[field + "_mean"] = (
                float(values.mean()) if values.shape[-1] == 1 else None
            )
        a = np.asarray([entry["companion_only"] for entry in entries])
        b = np.asarray([entry["anchor_only"] for entry in entries])
        row["signed_cross_term"] = float(-2 * np.mean(a * b))
        summary_rows.append(row)
    np.savez_compressed(out / "03/tables_and_coefficients.npz", **archive)
    for name, rows in (
        ("coefficients", coef_rows),
        ("conditional_effects", gate_rows),
        ("alias_energy_accounts", accounts),
        ("author_composition_witnesses", profiles),
        ("summary_by_group", summary_rows),
    ):
        write_csv(out / f"03/{name}.csv", rows)
    summary = {
        "packet": "HM79",
        "contexts": len(tables),
        "targets": list(M.TARGETS),
        "context_keys": [list(k) for k in sorted(tables)],
        "stages": list(M.STAGES),
        "actual_new_forwards": read_json(out / "run_status.json")["forward_calls"],
        "reused_actual_readings": manifest["spec"]["expected_reused_readings"],
        "new_fit_training_API_private_calls": 0,
        "alias_max_error": alias_error,
        **checks,
        "summary_by_group": summary_rows,
        "profile_max_error": max(r["formula_max_error"] for r in profiles),
        "profiles_are_hypothetical_compositions_not_human_authors": True,
        "old_compound_effect_retained_not_pure_ownership": True,
        "single_actor_flip_changes_name_exposure": True,
        "original_scope_complete": False,
    }
    write(out / "03/summary.json", summary)
    files = sorted(
        p for p in (out / "03").iterdir() if p.is_file() and not p.name.startswith("._")
    )
    write(out / "03/seal.json", {str(p.relative_to(out)): digest(p) for p in files})
    print(
        f"HM79 ANALYZED: {len(tables)} complete contexts, alias error {alias_error:.3g}",
        flush=True,
    )


def main():
    """Expose finite prepare/collect/analyze steps without automatic queues."""
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
