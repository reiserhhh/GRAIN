#!/usr/bin/env python3
"""HM61: bounded passive six-behavior composition, with fixed source transfer."""

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

from grain_research.construction import math_heterogeneous_evidence_cube as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_cross_target_operator import matrices
from grain_research.construction.math_heterogeneous_evidence_cube import cosine
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_cross_target_operator import CODE as OPERATOR_CODE
from scripts.run_grain_math_cross_target_operator import check as operator_check
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_passive_event_composition import CODE as MATERIAL_CODE
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_heterogeneous_evidence_cube_v1.json"
OUT = ROOT / "results/grain_math_response_v1/heterogeneous_evidence_cube_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_heterogeneous_evidence_cube_v1.json",
            "docs/GRAIN_HETEROGENEOUS_EVIDENCE_CUBE_PLAN_V1.md",
            "grain_research/construction/math_heterogeneous_evidence_cube.py",
            "scripts/run_grain_math_heterogeneous_evidence_cube.py",
            "scripts/verify_grain_math_heterogeneous_evidence_cube.py",
            "tests/test_grain_math_heterogeneous_evidence_cube.py",
            *OPERATOR_CODE,
            *MATERIAL_CODE,
        )
    )
)
IDENTITY = ("unit", "doc_id", "family", "role", "corner", "query", "x")


def prepare(out):
    """Freeze complete independent materials, old operator and actual token prefixes."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve prior packets")
    spec = read_json(SPEC)
    previous = operator_check(ROOT / spec["operator_source"])
    material = read_json(ROOT / spec["material_spec"])
    if (
        spec["targets"] != list(M.TARGETS)
        or spec["max_forwards"] != 768
        or spec["families"] != [0, 1]
        or spec["transfer_orders"] != [1, 2, 6]
        or spec["source_family"] != 0
        or spec["application_family"] != 1
        or spec["layer"] != material["layer"]
        or read_json(ROOT / spec["operator_source"] / "verification.json")["status"]
        != "VERIFIED_FINITE_HM60_NOT_GLOBAL"
    ):
        raise ValueError("Unchanged verified source and complete fixed design required")
    source = ROOT / spec["operator_source"]
    observations = read_json(source / "00/observations.json")
    atoms = {
        unit: {
            f"family{f}": matrices(observations, unit)[f"atom/f{f}"].tolist()
            for f in (0, 1)
        }
        for unit in ("score", "log_odds")
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    alphabet = read_json(
        ROOT
        / spec["observer_reference"].replace("01/observer.json", "00/manifest.json")
    )["alphabet"]
    if [tokenizer.encode(x, add_special_tokens=False) for x in alphabet["labels"]] != [
        [i] for i in alphabet["token_ids"]
    ]:
        raise ValueError("Candidate alphabet drift")
    rows, proofs = [], []
    for doc in M.documents(material):
        prefix = None
        for target in M.TARGETS:
            text, ids, part, cues, actor = render(
                doc["body"], target, "A", 1, material["system"], tokenizer
            )
            current = ids[: part["body"][-1] + 1]
            if prefix is not None and current != prefix:
                raise ValueError("Question-dependent evidence prefix")
            prefix = current
            rows.append(
                {
                    **doc,
                    "unit": len(rows),
                    "query": target,
                    "rendered": text,
                    "input_ids": ids,
                    "partition": part,
                    "cue_slots": cues,
                    "actor_slot": actor,
                    "body_token_count": len(part["body"]),
                }
            )
        proofs.append(
            {"doc_id": doc["doc_id"], "prefix_ids": prefix, "queries": list(M.TARGETS)}
        )
    if len(rows) != 768 or len({r["body"] for r in rows}) != 128:
        raise ValueError("128 complete distinct bodies required")
    (out / "00/code").mkdir(parents=True)
    (out / "01/units").mkdir(parents=True)
    write(out / "00/rows.json", rows)
    write(out / "00/prefix_proofs.json", proofs)
    write(out / "00/isolated_operators.json", atoms)
    code = {}
    for name in CODE:
        dest = out / "00/code" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        code[name] = digest(dest)
    assets = {
        n: digest(out / n)
        for n in ("00/rows.json", "00/prefix_proofs.json", "00/isolated_operators.json")
    }
    source_files = {
        str((source / n).relative_to(ROOT)): digest(source / n)
        for n in ("00/manifest.json", "00/observations.json", "verification.json")
    }
    source_files[spec["observer_reference"]] = digest(ROOT / spec["observer_reference"])
    write(
        out / "00/manifest.json",
        {
            "packet": "HM61",
            "spec": spec,
            "code": code,
            "assets": assets,
            "source_files": source_files,
            "alphabet": alphabet,
            "system": material["system"],
            "model_config_sha256": digest(MODEL / "config.json"),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_manifest_packet": previous["packet"],
            "human_units": 0,
            "no_neural_training_generation_API_private": True,
            "model_weights_not_separately_hashed": True,
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
        "HM61 PREPARED:128 six-event bodies,768 readings;source family0 first",
        flush=True,
    )


def check(out):
    """Reject frozen input/code/parent drift, without importing a model."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen material drift " + name)
    for name, sha in m["source_files"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Frozen source drift " + name)
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Observer configuration drift")
    return m


def read_tables(out, rows):
    """Bind captures to full design identities and assemble exact factorial tensors."""
    grouped, native_rows, max_prefix = {}, [], 0.0
    for family in sorted({r["family"] for r in rows}):
        selected = [r for r in rows if r["family"] == family]
        z = np.zeros((64, 6, 10))
        body, endpoint, lengths = {}, {}, np.zeros((64, 6), dtype=int)
        seen = set()
        for row in selected:
            meta, a = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
            if any(meta[k] != row[k] for k in IDENTITY):
                raise ValueError("Raw capture identity mismatch")
            q, k = M.TARGETS.index(row["query"]), row["corner"]
            if (k, q) in seen or row["x"] != M.grid()[k].tolist():
                raise ValueError("Wrong or duplicate cube cell")
            seen.add((k, q))
            raw = meta["native"]
            z[k, q] = raw["candidate_logits"]
            decoded = M.decode(z[k, q])
            if abs(decoded["score"] - raw["conditional_score"]) > 1e-10:
                raise ValueError("Candidate conversion mismatch")
            h = a["body_end"].astype(float)
            if k in body:
                max_prefix = max(max_prefix, float(np.max(np.abs(body[k] - h))))
            else:
                body[k] = h
            endpoint[k, q] = a["query_end"].astype(float)
            lengths[k, q] = len(row["input_ids"])
            native_rows.append(
                {k: row[k] for k in IDENTITY}
                | {
                    "score": float(decoded["score"]),
                    "log_odds": float(decoded["log_odds"]),
                    "candidate_mass": raw["candidate_mass"],
                    "greedy_is_candidate": raw["greedy_first_token_is_candidate"],
                    "input_tokens": len(row["input_ids"]),
                    "body_tokens": row["body_token_count"],
                }
            )
        if seen != set(itertools.product(range(64), range(6))):
            raise ValueError("Incomplete six-query cube")
        grouped[family] = {
            "logits": z,
            "body": np.asarray([body[k] for k in range(64)]),
            "endpoint": np.asarray(
                [[endpoint[k, q] for q in range(6)] for k in range(64)]
            ),
            "lengths": lengths,
        }
    return grouped, native_rows, max_prefix


def seal_source(out, rows):
    """Freeze source coefficients and raw hashes before the first application call."""
    path = out / "01/source_barrier.json"
    if path.exists():
        for name, sha in read_json(path)["files"].items():
            if digest(out / name) != sha:
                raise ValueError("Sealed source changed")
        return
    source = [r for r in rows if r["family"] == 0]
    tables, _, _ = read_tables(out, source)
    z = tables[0]["logits"]
    decoded = M.decode(z)
    centered = z - z.mean(axis=-1, keepdims=True)
    arrays = {u: M.coefficients(v) for u, v in decoded.items()}
    arrays["centered_logits"] = M.coefficients(centered)
    dest = out / "01/source_coefficients.npz"
    np.savez_compressed(dest, **arrays)
    files = {str(dest.relative_to(out)): digest(dest)}
    for row in source:
        meta = out / f"01/units/unit_{row['unit']:04d}.json"
        for p in (meta, meta.with_suffix(".npz")):
            files[str(p.relative_to(out))] = digest(p)
    write(
        path,
        {
            "source_family": 0,
            "actual_source_calls": 384,
            "files": files,
            "created_epoch": time.time(),
            "application_scores_used": False,
        },
    )
    print(
        "HM61 SOURCE SEALED:384 actual calls;fixed coefficients before application",
        flush=True,
    )


def collect(out):
    """Acquire each frozen reading once, with an actual source-to-application barrier."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("Inspect terminal/uncertain work;no automatic repeat")
    rows = read_json(out / "00/rows.json")
    if len(list((out / "01/units").glob("unit_*.json"))) != status["completed_units"]:
        raise RuntimeError("Raw/status disagreement")
    start, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        reference = read_json(ROOT / m["spec"]["observer_reference"])
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
            if actual[k] != reference[k]:
                raise ValueError("Observer drift " + k)
        write(out / "01/observer.json", actual)
        application_started = False
        for row in rows:
            if row["family"] == 1 and not application_started:
                seal_source(out, rows)
                application_started = True
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                meta, _ = load_arrays(path)
                if any(meta[k] != row[k] for k in IDENTITY):
                    raise ValueError("Resume identity mismatch")
                continue
            elapsed = previous + time.monotonic() - start
            if not call_allowed(status, m["spec"], elapsed):
                raise TimeoutError("768 forwards / 900 seconds bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, a = observe_full(observer, row, m["alphabet"], m["spec"]["layer"])
            status["forward_calls"] += 1
            kept = {
                "body_end": a["full"][row["partition"]["body"][-1]],
                "query_end": a["full"][-1],
            }
            save(
                path,
                {k: row[k] for k in IDENTITY}
                | {"native": native, "finished_epoch": time.time()},
                kept,
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - start,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 64 == 0:
                print(
                    f"HM61 ACTUAL:{status['completed_units']}/768;{status['model_seconds']:.1f}s",
                    flush=True,
                )
        if any(
            status[k] != 768
            for k in ("forward_calls", "attempted_forward_calls", "completed_units")
        ):
            raise ValueError("Incomplete acquisition")
        if status["model_seconds"] > m["spec"]["max_model_seconds"]:
            raise TimeoutError("Collector exceeded its fixed bound")
        status["status"] = "COLLECTED"
        write(out / "run_status.json", status)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - start,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Describe all terms and frozen transfers, not choose a best-scoring model."""
    started = time.monotonic()
    m = check(out)
    if read_json(out / "run_status.json")["status"] != "COLLECTED":
        raise ValueError("Complete actual collection required")
    if (out / "03").exists():
        raise FileExistsError("Preserve previous scientific analysis")
    seal_source(out, read_json(out / "00/rows.json"))
    tables, native, prefix_error = read_tables(out, read_json(out / "00/rows.json"))
    if prefix_error > m["spec"]["body_prefix_tolerance"]:
        raise ValueError("Body state unexpectedly depends on later question")
    atoms = read_json(out / "00/isolated_operators.json")
    with np.load(out / "01/source_coefficients.npz", allow_pickle=False) as source:
        source_coef = {k: source[k].copy() for k in source.files}
    term_rows, energy_rows, edge_rows, operator_rows, link_rows = [], [], [], [], []
    geometry, transfers, stats, arrays = [], [], {}, {}
    orders, mm = M.masks().sum(1), M.masks()
    for f, table in tables.items():
        z = table["logits"]
        centered = z - z.mean(axis=-1, keepdims=True)
        cz = M.coefficients(centered)
        additive_logits = M.evaluate(cz, order=1)
        raw, additive = M.decode(z), M.decode(additive_logits)
        stages = {
            "centered_logits": centered,
            "body_endpoint": table["body"],
            "query_endpoint": table["endpoint"],
            **raw,
        }
        stats[str(f)] = {}
        for stage, v in stages.items():
            c = M.coefficients(v)
            arrays[f"family{f}_{stage}_coefficients"] = c
            e = M.order_energy(c)
            stats[str(f)][stage] = e
            energy_rows.extend(
                {"family": f, "stage": stage, "order": k, "energy": value}
                for k, value in enumerate(e["energy"])
            )
        for unit, v in raw.items():
            c = M.coefficients(v)
            ac = M.coefficients(additive[unit])
            arrays[f"family{f}_{unit}_link_control_coefficients"] = ac
            er = M.order_energy(ac)
            stats[str(f)][unit + "_additive_logit_control"] = er
            stats[str(f)][unit + "_link_control_difference_rms"] = float(
                np.sqrt(np.mean((v - additive[unit]) ** 2))
            )
            for k, q in itertools.product(range(64), range(6)):
                term_rows.append(
                    {
                        "family": f,
                        "unit": unit,
                        "query": M.TARGETS[q],
                        "mask": "".join(map(str, mm[k])),
                        "order": int(orders[k]),
                        "coefficient": float(c[k, q]),
                        "additive_logit_control": float(ac[k, q]),
                    }
                )
                link_rows.append(
                    {
                        "family": f,
                        "unit": unit,
                        "corner": k,
                        "query": M.TARGETS[q],
                        "actual": float(v[k, q]),
                        "additive_logit_control": float(additive[unit][k, q]),
                    }
                )
            d = M.linear_operator(c)
            atom = np.asarray(atoms[unit][f"family{f}"])
            stats[str(f)][unit + "_mixed_atom_cosine"] = cosine(d, atom)
            stats[str(f)][unit + "_mixed_diagonal_excess"] = float(
                np.diag(d).mean() - (d.sum() - np.trace(d)) / 30
            )
            singular, u, vt = None, None, None
            u, singular, vt = np.linalg.svd(d, full_matrices=False)
            leading = singular[0] * np.outer(u[:, 0], vt[0])
            tail = d - leading
            stats[str(f)][unit + "_rank1_energy_fraction"] = float(
                singular[0] ** 2 / np.square(singular).sum()
            )
            stats[str(f)][unit + "_own_edges_nonpositive"] = int(
                sum(np.sum(M.edges(v)[j, :, j] <= 0) for j in range(6))
            )
            for q, j in itertools.product(range(6), repeat=2):
                operator_rows.append(
                    {
                        "family": f,
                        "unit": unit,
                        "query": M.TARGETS[q],
                        "content": M.TARGETS[j],
                        "mixed": float(d[q, j]),
                        "isolated": float(atom[q, j]),
                        "shared_rank1": float(leading[q, j]),
                        "remaining": float(tail[q, j]),
                    }
                )
            ee = M.edges(v)
            for j, k, q in itertools.product(range(6), range(32), range(6)):
                edge_rows.append(
                    {
                        "family": f,
                        "unit": unit,
                        "content": M.TARGETS[j],
                        "condition_index": k,
                        "query": M.TARGETS[q],
                        "contrast": float(ee[j, k, q]),
                    }
                )
            geometry.append(
                {"family": f, "unit": unit}
                | M.injectivity_diagnostic(c, np.asarray(atoms[unit]["family0"]) / 2)
            )
            if f == 1:
                for order in m["spec"]["transfer_orders"]:
                    predicted = M.edges(M.evaluate(source_coef[unit], order=order))
                    for q in range(6):
                        actual, pred = ee[:, :, q].ravel(), predicted[:, :, q].ravel()
                        transfers.append(
                            {
                                "unit": unit,
                                "query": M.TARGETS[q],
                                "order": order,
                                "edge_rms_error": float(
                                    np.sqrt(np.mean((actual - pred) ** 2))
                                ),
                                "actual_edge_rms": float(np.sqrt(np.mean(actual**2))),
                                "edge_cosine": cosine(actual, pred),
                                "same_nonzero_sign": int(np.sum(actual * pred > 0)),
                                "edge_count": 192,
                            }
                        )
    for unit in ("score", "log_odds"):
        c0 = arrays[f"family0_{unit}_coefficients"]
        c1 = arrays[f"family1_{unit}_coefficients"]
        for k in range(1, 7):
            stats[f"{unit}_order{k}_family_cosine"] = cosine(
                c0[orders == k], c1[orders == k]
            )
    (out / "03").mkdir()
    np.savez_compressed(out / "03/coefficients.npz", **arrays)
    for name, rows in (
        ("native", native),
        ("terms", term_rows),
        ("order_energy", energy_rows),
        ("edges", edge_rows),
        ("operators", operator_rows),
        ("link_control", link_rows),
        ("source_transfer", transfers),
        ("geometry", geometry),
    ):
        write_csv(out / f"03/{name}.csv", rows)
    write(
        out / "03/summary.json",
        {
            "packet": "HM61",
            "statistics": stats,
            "geometry": geometry,
            "actual_new_forwards": 768,
            "new_training_or_accuracy_search": False,
            "prefix_hidden_max_error": prefix_error,
            "source_before_application": True,
            "original_scope_complete": False,
            "new_human_personality_accuracy": False,
            "CPU_analysis_seconds": time.monotonic() - started,
            "table_counts": {
                "native": len(native),
                "terms": len(term_rows),
                "edges": len(edge_rows),
                "transfers": len(transfers),
            },
        },
    )
    write(
        out / "03/manifest.json",
        {
            "files": {
                str(p.relative_to(out)): digest(p)
                for p in sorted((out / "03").iterdir())
                if p.is_file()
            }
        },
    )
    print(
        "HM61 ANALYZED:complete composition terms,link control,source-only transfers",
        flush=True,
    )


def main():
    """Run only the requested finite stage; never overwrite or auto-retry science."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "collect":
        with collector_guard(args.out):
            collect(args.out)
    else:
        globals()[args.stage](args.out)


if __name__ == "__main__":
    main()
