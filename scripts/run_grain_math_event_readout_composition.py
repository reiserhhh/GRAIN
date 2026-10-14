#!/usr/bin/env python3
"""HM56: common event weights versus context-sensitive six-query interpretation."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.construction import math_event_readout_composition as M
from grain_research.construction.math_answer_recoding_mechanism import render
from grain_research.construction.math_evidence_response_operation import readout
from grain_research.construction.math_passive_event_composition import channels
from grain_research.construction.math_relational_cut import call_allowed
from grain_research.vector_mechanisms.navigation import MODEL, digest, read_json
from scripts.run_grain_math_construct_routing import save, write
from scripts.run_grain_math_evaluative_routing import load_arrays
from scripts.run_grain_math_passive_event_composition import CODE as PARENT_CODE
from scripts.run_grain_math_passive_event_composition import check as parent_check
from scripts.run_grain_math_passive_event_composition import token_proofs
from scripts.run_grain_math_relational_cut import observe_full
from scripts.run_grain_math_response import write_csv
from scripts.run_grain_math_span_binding import load_observer
from scripts.run_grain_upstream_routes import collector_guard

SPEC = ROOT / "configs/grain_math_event_readout_composition_v1.json"
OUT = ROOT / "results/grain_math_response_v1/event_readout_composition_r1"
CODE = tuple(
    dict.fromkeys(
        (
            "configs/grain_math_event_readout_composition_v1.json",
            "docs/GRAIN_EVENT_READOUT_COMPOSITION_PLAN_V1.md",
            "grain_research/construction/math_event_readout_composition.py",
            "scripts/run_grain_math_event_readout_composition.py",
            "scripts/verify_grain_math_event_readout_composition.py",
            "tests/test_grain_math_event_readout_composition.py",
            *PARENT_CODE,
        )
    )
)
FIELDS = (
    "unit",
    "parent_unit",
    "role",
    "content",
    "query",
    "family",
    "polarity",
    "order",
    "regime",
)


def prepare(out):
    """Freeze all crossings, exact reused inputs and parent hashes before forwards."""
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Preserve previous runs")
    spec = read_json(SPEC)
    parent = ROOT / spec["parent"]
    old = parent_check(parent)
    if (
        read_json(parent / "verification.json")["status"]
        != "VERIFIED_FINITE_HM55_NOT_GLOBAL"
    ):
        raise ValueError("Verified HM55 required")
    if spec["regimes"] != list(M.REGIMES):
        raise ValueError("Declared four regimes required")
    targets = old["spec"]["targets"]
    selected = [
        r
        for r in read_json(parent / "00/rows.json")
        if r["role"] == "atom" or r["regime"] in M.REGIMES
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    rows, reused = [], {}
    for doc in selected:
        prefix = None
        for query in targets:
            text, ids, part, cues, actor = render(
                doc["body"], query, "A", 1, old["spec"]["system"], tokenizer
            )
            current_prefix = ids[: part["body"][-1] + 1]
            if prefix is not None and prefix != current_prefix:
                raise ValueError("Query changed actual preceding body tokens")
            prefix = current_prefix
            reuse = query == doc["target"]
            if reuse:
                if (
                    ids != doc["input_ids"]
                    or text != doc["rendered"]
                    or part != doc["partition"]
                ):
                    raise ValueError("Reused input differs from original")
                name = f"01/units/unit_{doc['unit']:04d}.json"
                meta, _ = load_arrays(parent / name)
                reused[str(doc["unit"])] = {
                    "meta": name,
                    "meta_sha256": digest(parent / name),
                    "array_sha256": meta["array_sha256"],
                }
            rows.append(
                {
                    **doc,
                    "unit": len(rows),
                    "parent_unit": doc["unit"],
                    "content": doc["target"],
                    "query": query,
                    "reused": reuse,
                    "rendered": text,
                    "input_ids": ids,
                    "partition": part,
                    "cue_slots": cues,
                    "actor_slot": actor,
                }
            )
    proofs = [
        dict(query=q, **p)
        for q in targets
        for p in token_proofs([r for r in rows if r["query"] == q], tokenizer)
    ]
    if (
        len(selected) != spec["expected_documents"]
        or len(rows) != spec["expected_readouts"]
        or len(reused) != spec["expected_reused"]
        or len(proofs) != 288
    ):
        raise ValueError("Incomplete 216-body/1296-readout/216-reuse design")
    for folder in ("00/code", "01/units", "03"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    write(out / "00/rows.json", rows)
    write(out / "00/token_proofs.json", proofs)
    code = {}
    for name in CODE:
        path = out / "00/code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
        code[name] = digest(path)
    write(
        out / "00/manifest.json",
        {
            "packet": "HM56",
            "spec": spec,
            "targets": targets,
            "layer": old["spec"]["layer"],
            "alphabet": old["alphabet"],
            "code": code,
            "reused": reused,
            "assets": {
                n: digest(out / n) for n in ("00/rows.json", "00/token_proofs.json")
            },
            "parent_files": {
                n: digest(parent / n)
                for n in ("00/manifest.json", "verification.json", "01/observer.json")
            },
            "model_config_sha256": digest(MODEL / "config.json"),
            "model_weights_not_separately_hashed": True,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "independent_human_units": 0,
            "fit_or_patch": False,
            "new_arrays": "endpoint only; parent full states retained",
        },
    )
    write(
        out / "run_status.json",
        {
            "status": "PREPARED",
            "forward_calls": 0,
            "attempted_forward_calls": 0,
            "completed_units": 0,
            "reused_units": 0,
            "model_seconds": 0.0,
            "in_flight": None,
        },
    )
    print(
        "PREPARED:1296 readings,216 exact reuses,1080 bounded new forwards", flush=True
    )


def check(out):
    """Check immutable code/material and reused observations without neural calls."""
    m = read_json(out / "00/manifest.json")
    for name, sha in m["code"].items():
        if digest(ROOT / name) != sha or digest(out / "00/code" / name) != sha:
            raise ValueError("Frozen code drift " + name)
    for name, sha in m["assets"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen input drift " + name)
    parent = ROOT / m["spec"]["parent"]
    for name, sha in m["parent_files"].items():
        if digest(parent / name) != sha:
            raise ValueError("Parent drift " + name)
    for item in m["reused"].values():
        if (
            digest(parent / item["meta"]) != item["meta_sha256"]
            or digest((parent / item["meta"]).with_suffix(".npz"))
            != item["array_sha256"]
        ):
            raise ValueError("Reused observation drift")
    if digest(MODEL / "config.json") != m["model_config_sha256"]:
        raise ValueError("Observer configuration drift")
    return m


def observation(out, row, manifest):
    """Load original or fresh readout with its exact content/query identity."""
    if row["reused"]:
        item = manifest["reused"][str(row["parent_unit"])]
        meta, arrays = load_arrays(ROOT / manifest["spec"]["parent"] / item["meta"])
        if (
            row["query"] != row["content"]
            or meta["unit"] != row["parent_unit"]
            or meta["target"] != row["content"]
        ):
            raise ValueError("Original capture identity mismatch")
        endpoint = arrays["full"][-1]
    else:
        meta, arrays = load_arrays(out / f"01/units/unit_{row['unit']:04d}.json")
        if any(meta[k] != row[k] for k in FIELDS):
            raise ValueError("Fresh capture identity mismatch")
        endpoint = arrays["endpoint"]
    if not np.isfinite(endpoint).all():
        raise ValueError("Nonfinite state")
    return meta["native"], endpoint


def collect(out):
    """Collect finite passive crossings once; checkpoint every attempted forward."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] == "COLLECTED":
        return
    if status["status"] == "STOPPED" or status["in_flight"] is not None:
        raise RuntimeError("No automatic retry of terminal or uncertain forwards")
    saved = list((out / "01/units").glob("unit_*.json"))
    if (
        len(saved) != status["completed_units"]
        or status["completed_units"] != status["forward_calls"]
    ):
        raise ValueError("Saved observation/status mismatch")
    started, previous = time.monotonic(), status["model_seconds"]
    try:
        observer = load_observer()
        actual = observer.manifest()
        prior = read_json(ROOT / m["spec"]["parent"] / "01/observer.json")
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
        for row in read_json(out / "00/rows.json"):
            if row["reused"]:
                continue
            path = out / f"01/units/unit_{row['unit']:04d}.json"
            if path.exists():
                observation(out, row, m)
                continue
            elapsed = previous + time.monotonic() - started
            if not call_allowed(status, m["spec"], elapsed):
                raise TimeoutError("1080-forward/900-second acquisition bound")
            status.update(
                status="COLLECTING", in_flight=row["unit"], model_seconds=elapsed
            )
            status["attempted_forward_calls"] += 1
            write(out / "run_status.json", status)
            native, arrays = observe_full(observer, row, m["alphabet"], m["layer"])
            status["forward_calls"] += 1
            save(
                path,
                {k: row[k] for k in FIELDS}
                | {"native": native, "finished_epoch": time.time()},
                {"endpoint": arrays["full"][-1]},
            )
            status.update(
                in_flight=None,
                completed_units=status["completed_units"] + 1,
                model_seconds=previous + time.monotonic() - started,
            )
            write(out / "run_status.json", status)
            if status["completed_units"] % 120 == 0:
                print(
                    f"ACTUAL:{status['completed_units']}/1080 new passive readings",
                    flush=True,
                )
        if (
            any(
                status[k] != m["spec"]["max_forwards"]
                for k in ("forward_calls", "attempted_forward_calls", "completed_units")
            )
            or status["model_seconds"] > m["spec"]["max_model_seconds"]
        ):
            raise ValueError("Incomplete acquisition")
        status.update(status="COLLECTED", reused_units=m["spec"]["expected_reused"])
        write(out / "run_status.json", status)
    except BaseException as error:
        status.update(
            status="STOPPED",
            error=repr(error),
            model_seconds=previous + time.monotonic() - started,
        )
        write(out / "run_status.json", status)
        raise


def analyze(out):
    """Test weight-only composition; no parameter learned for a future prediction."""
    m, status = check(out), read_json(out / "run_status.json")
    if status["status"] != "COLLECTED":
        raise ValueError("Complete finite crossings required")
    rows, targets = read_json(out / "00/rows.json"), m["targets"]
    values, atoms, native = {}, {}, []
    for row in rows:
        raw, _ = observation(out, row, m)
        v = readout(raw)
        values[row["unit"]] = v
        native.append({k: row[k] for k in FIELDS} | v | {"reused": row["reused"]})
        if row["role"] == "atom":
            atoms[row["content"], row["family"], row["polarity"], row["query"]] = v[
                "score"
            ]
    grouped = {}
    for row in rows:
        if row["role"] == "account":
            grouped.setdefault(row["parent_unit"], {})[row["query"]] = row
    profiles, certificates, components, vectors = [], [], [], {}
    for parent_unit, group in grouped.items():
        r = group[targets[0]]
        c, f, p = r["content"], r["family"], r["polarity"]
        z = np.array([values[group[q]["unit"]]["score"] for q in targets])
        x = np.array([atoms[c, f, p, q] for q in targets])
        y = np.array([atoms[c, 1 - f, 1 - p, q] for q in targets])
        b, n = [], []
        for q in targets:
            channel = channels(
                r["history"],
                {
                    (a, t): atoms[c, a, t, q]
                    for a, t in itertools.product((0, 1), repeat=2)
                },
            )
            profiles.append({k: group[q][k] for k in FIELDS} | channel)
            b.append(channel["event_profile"])
            n.append(channel["mention_profile"])
        cert = M.segment_certificate(z, x, y)
        base = {
            k: r[k]
            for k in ("parent_unit", "content", "family", "polarity", "order", "regime")
        }
        certificates.append(
            base | {k: v for k, v in cert.items() if not isinstance(v, np.ndarray)}
        )
        for j, q in enumerate(targets):
            components.append(
                base
                | {
                    "query": q,
                    "native": z[j],
                    "focal_atom": x[j],
                    "companion_atom": y[j],
                    "event_profile": b[j],
                    "mention_profile": n[j],
                    "segment_projection": cert["projection"][j],
                    "affine_projection": cert["affine_projection"][j],
                    "box_projection": cert["box_projection"][j],
                    "dual_witness": cert["witness"][j],
                }
            )
        vectors[c, f, p, r["order"], r["regime"]] = {
            "native": z,
            "focal": x,
            "companion": y,
            "event_profile": np.array(b),
            "mention_profile": np.array(n),
            "parent_unit": parent_unit,
        }
    updates, update_components, summaries = [], [], {}
    for c, f, p, order in itertools.product(targets, (0, 1), (0, 1), (0, 1)):
        for name, hi, lo in M.CONTRASTS:
            a, b = (vectors[c, f, p, order, k] for k in (hi, lo))
            change = a["native"] - b["native"]
            d = M.update_decomposition(change, a["focal"], a["companion"])
            base = {
                "content": c,
                "family": f,
                "polarity": p,
                "order": order,
                "contrast": name,
                "hi": a["parent_unit"],
                "lo": b["parent_unit"],
            }
            updates.append(
                base | {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
            )
            for j, q in enumerate(targets):
                update_components.append(
                    base
                    | {
                        "query": q,
                        "native_change": change[j],
                        "along_weight": d["along"][j],
                        "context_remainder": d["remainder"][j],
                        "event_change": a["event_profile"][j] - b["event_profile"][j],
                        "mention_change": a["mention_profile"][j]
                        - b["mention_profile"][j],
                    }
                )
    tol = m["spec"]["arithmetic_tolerance"]
    for regime in M.REGIMES:
        summaries[regime] = M.summarize_certificates(
            [r for r in certificates if r["regime"] == regime], tol
        )
    for name, _, _ in M.CONTRASTS:
        selected = [r for r in updates if r["contrast"] == name]
        total = sum(r["change_norm"] ** 2 for r in selected)
        summaries[name] = {
            "paired_cases": len(selected),
            "update_norm_rms": float(
                np.sqrt(np.mean([r["change_norm"] ** 2 for r in selected]))
            ),
            "remainder_norm_rms": float(
                np.sqrt(np.mean([r["remainder_norm"] ** 2 for r in selected]))
            ),
            "pooled_perpendicular_energy_share": sum(
                r["remainder_norm"] ** 2 for r in selected
            )
            / total
            if total
            else None,
            "nonparallel_numerical": sum(r["remainder_norm"] > tol for r in selected),
        }
    tables = {
        "native": native,
        "profiles": profiles,
        "certificates": certificates,
        "components": components,
        "updates": updates,
        "update_components": update_components,
    }
    for name, table in tables.items():
        write_csv(out / f"03/{name}.csv", table)
    summary = {
        "packet": "HM56",
        "status": "ANALYZED_FINITE_NOT_GLOBAL",
        "readouts": len(native),
        "accounts": len(certificates),
        "new_forwards": status["forward_calls"],
        "reused_forwards": status["reused_units"],
        "model_seconds": status["model_seconds"],
        "results": summaries,
        "no_fit_or_patch": True,
        "is_personality_accuracy": False,
        "original_scope_complete": False,
    }
    write(out / "03/summary.json", summary)
    write(
        out / "03/artifact_hashes.json",
        {
            str(p.relative_to(out)): digest(p)
            for p in sorted((out / "03").iterdir())
            if p.name != "artifact_hashes.json"
        },
    )
    print(summary)


def main():
    """Expose finite prepare/collect/analyze stages without automatic retries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.phase == "collect":
        with collector_guard(args.output):
            collect(args.output)
    else:
        {"prepare": prepare, "analyze": analyze}[args.phase](args.output)


if __name__ == "__main__":
    main()
