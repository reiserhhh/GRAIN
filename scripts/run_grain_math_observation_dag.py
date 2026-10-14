#!/usr/bin/env python3
"""HM-07 cold proposition-DAG experiment with untouched HM-03 source units."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_purpose_accounts import coordinates
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

from grain_research.construction import math_local_roles as R
from grain_research.construction import math_observation_dag as D
from grain_research.construction import math_purpose_accounts as P
from grain_research.source_event_graph import stanza_words

SPEC = ROOT / "configs/grain_math_observation_dag_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/observation_dag_r1"
ARMS = ("flat", "dag_vector", "dag_typed", "declared_typed")


def prepare(out, reuse=None):
    """Freeze parser rules, source coefficients and all new public expressions."""
    spec = json.loads(SPEC.read_text())
    source = ROOT / spec["source_packet"]
    if reuse is not None:
        spec = {
            **spec,
            "result_after_interface_repair": True,
            "reuses_primary": str(reuse),
            "independent_replication": False,
        }
        for name, digest in json.loads(
            (reuse / "04/artifact_hashes.json").read_text()
        ).items():
            if sha(reuse / name) != digest:
                raise ValueError("changed primary: " + name)
    hashes = json.loads((source / "04/artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(source / name) != digest:
            raise ValueError("changed source: " + name)
    bank = D.material_bank(read_bank(source / "00/singles.jsonl"))
    paths = [
        Path(__file__),
        SPEC,
        ROOT / "grain_research/construction/math_observation_dag.py",
        ROOT / "grain_research/construction/math_purpose_accounts.py",
        ROOT / "grain_research/construction/math_local_roles.py",
        ROOT / "grain_research/source_event_graph.py",
        ROOT / "scripts/run_grain_math_purpose_accounts.py",
        ROOT / "scripts/run_grain_math_response.py",
        ROOT / "scripts/run_grain_math_binding.py",
        ROOT / "grain_research/construction/math_binding.py",
        ROOT / "grain_research/construction/math_local.py",
    ]
    signature = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    signature["source_manifest"] = sha(source / "04/artifact_hashes.json")
    content = "".join(json.dumps(r) + "\n" for r in bank)
    protocol = out / "00/protocol_before_run.json"
    if protocol.exists():
        if (
            json.loads(protocol.read_text())["signature"] != signature
            or (out / "00/documents.jsonl").read_text() != content
        ):
            raise ValueError("HM-07 protocol changed; preserve r1")
        return source, bank
    for p in paths:
        dst = out / "00/code" / p.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    (out / "00/documents.jsonl").write_text(content)
    write_json(
        out / "00/reader_registry.json",
        json.loads((source / "00/reader_registry.json").read_text()),
    )
    write_json(
        protocol,
        {
            "signature": signature,
            "spec": spec,
            "materials": len(bank),
            "source_hashes": len(hashes),
            "rules_before_parse_or_encoding": True,
            "node_typing_uses_expected_action_or_role": False,
            "supplied_identity_after_observation": True,
            "new_objects_not_source_fitted": True,
            "source_fits": 0,
        },
    )
    write_json(out / "00/compression_witness.json", D.compression_witness())
    # Never replace an absent licensed cache with private or psychological data.
    write_json(
        out / "00/natural_inventory.json",
        {
            "inspected_cache": str(ROOT / "results/grain_vector_mechanisms/resources"),
            "licensed_text_corpus_found": False,
            "present_resources": [
                "stanza_111_ewt model weights",
                "stanza_py111 parser package",
            ],
            "natural_run": "pending independently licensed text cache; no private substitution",
            "candidate": "https://github.com/UniversalDependencies/UD_English-GUM",
            "candidate_license": "CC BY-NC-SA 4.0; source/author metadata must be inventoried",
            "ewt_caution": "EWT annotations CC BY-SA does not settle rights to all underlying texts",
            "private_reads": 0,
        },
    )
    print(f"HM-07 frozen {len(bank)} fresh public constructions", flush=True)
    return source, bank


def parse(out, bank, reuse=None):
    """Run actual UD parsing, retaining both legacy flat and new typed graphs."""
    from fetch_grain_syntax_reference import MODELS as SYNTAX
    from fetch_grain_syntax_reference import PACKAGES, PROCESSORS
    from run_grain_source_event_graph import verify_models

    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    if stanza.__version__ != "1.11.0":
        raise ValueError("frozen parser version changed")
    torch.set_num_threads(2)
    parsed, pipeline, calls, start = [], None, 0, time.monotonic()
    for i, row in enumerate(bank):
        path = out / f"01/parses/{i:04d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["text_sha256"] != R.text_digest(row["text"]):
                raise ValueError("parse text changed")
        else:
            if reuse is not None:
                prior = json.loads((reuse / f"01/parses/{i:04d}.json").read_text())
                if prior["text_sha256"] != R.text_digest(row["text"]):
                    raise ValueError("primary raw text changed")
                words = prior["words"]
            else:
                if pipeline is None:
                    pipeline = stanza.Pipeline(
                        "en",
                        dir=str(SYNTAX),
                        package=None,
                        processors=PROCESSORS,
                        use_gpu=False,
                        download_method=None,
                        verbose=False,
                    )
                words = stanza_words(pipeline(row["text"]), row["text"])
                calls += 1
            graph = D.observe_dag(words, row["text"])
            events = R.attach_event_identity(graph["events"], row["events"])
            flat = R.attach_event_identity(
                P.observe_modes(words, row["text"]), row["events"]
            )
            data = {
                "text_sha256": R.text_digest(row["text"]),
                "words": words,
                "graph": graph,
                "events": events,
                "flat": flat,
            }
            write_json(path, data)
        parsed.append(data)
        if (i + 1) % 128 == 0:
            print(
                f"HM-07 parse {i + 1}/{len(bank)} {time.monotonic() - start:.1f}s",
                flush=True,
            )
    write_json(
        out / "01/parser_accounting.json",
        {
            "new_calls": calls,
            "cached_rows": len(bank) - calls,
            "elapsed_seconds": time.monotonic() - start,
            "parser_sha256": sha(SYNTAX / "grain_inventory.json"),
        },
    )
    return parsed


def inputs(out, source, bank, parsed):
    """Encode full unchanged sentences and separately addressed action/object slots."""
    old = {
        r["text"]: i for i, r in enumerate(read_bank(source / "00/material_bank.jsonl"))
    }
    fresh, lookup = [], {}

    def ref(text, kind):
        if text in old:
            return ["source", old[text]]
        if text not in lookup:
            lookup[text] = len(fresh)
            fresh.append({"text": text, "kind": kind})
        return ["new", lookup[text]]

    def add(e, text=None):
        return {
            **e,
            "readout": ref(
                e["projection_text"] if text is None else text, "unchanged_expression"
            ),
            "action_ref": ref("Action: " + e["action"] + ".", "action_slot"),
            "object_ref": ref("Object: " + e["object"] + ".", "object_slot"),
        }

    indices = {"flat": [], "dag": [], "declared": []}
    for row, data in zip(bank, parsed, strict=True):
        indices["flat"].append([add(e) for e in data["flat"]])
        indices["dag"].append([add(e) for e in data["events"]])
        indices["declared"].append(
            [
                add(
                    {
                        **e,
                        "record_index": j,
                        "status": "ok",
                        "actor": "self" if e["actor"] == 1 else "other",
                        "conceiver": "self" if e["conceiver"] == 1 else "other",
                    },
                    row["text"][e["span"][0] : e["span"][1]],
                )
                for j, e in enumerate(row["events"])
            ]
        )
    content = "".join(json.dumps(r) + "\n" for r in fresh)
    path = out / "00/material_bank.jsonl"
    if path.exists() and path.read_text() != content:
        raise ValueError("frozen encoding bank changed")
    path.write_text(content)
    write_json(out / "01/observation_indices.json", indices)
    return indices


def truth(row, account):
    """Evaluate declared construction roles only after automatic observations."""
    values = []
    for e in row["events"]:
        mode, actor, conceiver = e["mode"], e["actor"], e["conceiver"]
        qualify = (
            mode == "desire" and conceiver == 1 and actor == -1
            if account == "desire_about_other"
            else mode == account
            and (
                conceiver == 1
                if mode == "state"
                else actor == 1
                and (conceiver == 1 or mode in {"actual", "hypothetical"})
            )
        )
        values.append(
            (1 if mode == "state" else e["bits"][1])
            if qualify and e["bits"][0] == 1
            else 0
        )
    return float(np.mean(values)), sum(v != 0 for v in values)


def evaluate(out, source, bank, indices):
    """Compare routing, fixed vector units and typed ownership as separate arms."""
    predictions, event_rows, transports = [], [], []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        models = {
            target: load_model(source / f"02/{reader}_{target}.npz")
            for target in {r["target"] for r in bank}
        }
        for i, row in enumerate(bank):
            for arm in ARMS:
                events = indices[
                    "declared"
                    if arm == "declared_typed"
                    else "flat"
                    if arm == "flat"
                    else "dag"
                ][i]
                z = coordinates(events, models[row["target"]], old, new)
                missing = R.missing_records(events, len(row["events"]))
                for account in D.ACCOUNTS:
                    if arm == "flat":
                        value, mass, eligible = P.purpose_values(
                            z,
                            [e["mode"] for e in events],
                            "desire" if account == "desire_about_other" else account,
                        )
                    else:
                        value, mass, eligible = D.readout(
                            z, events, account, typed_owner=arm != "dag_vector"
                        )
                    expected, expected_count = truth(row, account)
                    predictions.append(
                        {
                            "reader": reader,
                            "target": row["target"],
                            "material_id": row["material_id"],
                            "kind": row["kind"],
                            "family": row["family"],
                            "arm": arm,
                            "account": account,
                            "complete": missing == 0,
                            "unresolved": missing,
                            "unit": float(value.sum() / len(row["events"]))
                            if missing == 0
                            else None,
                            "numerator": float(value.sum()),
                            "soft_mass": float(mass.sum()),
                            "observed_events": int(eligible.sum())
                            if missing == 0
                            else None,
                            "profile": float(value[eligible].mean())
                            if missing == 0 and eligible.any()
                            else None,
                            "expected_unit": expected,
                            "expected_events": expected_count,
                        }
                    )
                    for j, e in enumerate(events):
                        event_rows.append(
                            {
                                "reader": reader,
                                "target": row["target"],
                                "material_id": row["material_id"],
                                "arm": arm,
                                "account": account,
                                "node_id": e.get("node_id"),
                                "record_index": e["record_index"],
                                "event_id": e["event_id"],
                                "mode": e["mode"],
                                "actor": e.get("actor"),
                                "conceiver": e.get("conceiver"),
                                "k": float(z[j, 0]),
                                "p": float(z[j, 1]),
                                "o": float(z[j, 2]),
                                "a": float(z[j, 3]),
                                "contribution": float(value[j]),
                                "mass": float(mass[j]),
                                "eligible": bool(eligible[j]),
                            }
                        )
                if arm == "dag_typed" and len(z):
                    # Keep a nonzero context channel, not a fake zero padding.
                    x = np.r_[z[:, 1], z[:, 0]]
                    projection = np.diag(np.r_[np.ones(len(z)), np.zeros(len(z))])
                    change = np.eye(len(x))
                    change[: len(z), len(z) :] = np.eye(len(z)) * 0.1
                    transports.append(
                        {
                            "reader": reader,
                            "material_id": row["material_id"],
                            **D.transported_readout(x, projection, change),
                        }
                    )
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/event_readouts.csv", event_rows)
    write_csv(out / "03/typed_transport.csv", transports)
    return predictions, transports


def summarize(out, bank, parsed, predictions, transports):
    """Report finite coverage, unit errors, retained context and author profiles."""
    groups = defaultdict(list)
    for p in predictions:
        groups[p["reader"], p["target"], p["kind"], p["arm"], p["account"]].append(p)
    metrics = []
    for key, rows in groups.items():
        good = [r for r in rows if r["complete"]]
        metrics.append(
            {
                **dict(
                    zip(
                        ("reader", "target", "kind", "arm", "account"), key, strict=True
                    )
                ),
                "n": len(rows),
                "complete": len(good),
                "unit_rmse": float(
                    np.sqrt(
                        np.mean([(r["unit"] - r["expected_unit"]) ** 2 for r in good])
                    )
                )
                if good
                else None,
                "fp": sum(
                    r["observed_events"] > 0 and r["expected_events"] == 0 for r in good
                ),
                "fn": sum(
                    r["observed_events"] == 0 and r["expected_events"] > 0 for r in good
                ),
            }
        )
    coverage, diagnostics = [], []
    for row, data in zip(bank, parsed, strict=True):
        dag, flat = data["events"], data["flat"]
        unresolved = R.missing_records(dag, len(row["events"]))
        coverage.append(
            {
                "material_id": row["material_id"],
                "kind": row["kind"],
                "expected": len(row["events"]),
                "flat_heads": len(flat),
                "dag_heads": len(dag),
                "flat_missing": R.missing_records(flat, len(row["events"])),
                "dag_missing": unresolved,
                "retained_nodes": len(data["graph"]["nodes"]),
                "context_nodes": sum(
                    n["type"].endswith("context") for n in data["graph"]["nodes"]
                ),
            }
        )
        if unresolved:
            diagnostics.append(
                {
                    "material_id": row["material_id"],
                    "text": row["text"],
                    "nodes": data["graph"]["nodes"],
                    "events": dag,
                }
            )
    write_csv(out / "03/metrics.csv", metrics)
    write_csv(out / "03/record_coverage.csv", coverage)
    write_json(out / "03/unresolved_examples.json", diagnostics)
    lookup = {
        (p["reader"], p["material_id"], p["arm"], p["account"]): p for p in predictions
    }
    comparisons, rng = [], np.random.default_rng(7108)
    for reader in MODELS:
        for target in sorted({r["target"] for r in bank}):
            for account in D.ACCOUNTS:
                by_family = defaultdict(list)
                for row in bank:
                    if row["target"] != target or row["kind"] != "single":
                        continue
                    a, b = (
                        lookup[reader, row["material_id"], arm, account]
                        for arm in ("flat", "dag_typed")
                    )
                    if a["complete"] and b["complete"]:
                        by_family[row["family"]].append(
                            (a["unit"] - a["expected_unit"]) ** 2
                            - (b["unit"] - b["expected_unit"]) ** 2
                        )
                f = np.array([np.mean(v) for v in by_family.values()])
                if len(f):
                    lo, hi = np.quantile(
                        f[rng.integers(0, len(f), (1000, len(f)))].mean(axis=1),
                        [0.025, 0.975],
                    )
                    comparisons.append(
                        {
                            "reader": reader,
                            "target": target,
                            "account": account,
                            "families": len(f),
                            "matched_mse_gain": float(f.mean()),
                            "ci_low": float(lo),
                            "ci_high": float(hi),
                            "scope": "fixed construction families, incomplete flat cases excluded",
                        }
                    )
    write_csv(out / "03/matched_comparisons.csv", comparisons)
    panels = []
    singles = {
        (
            r["target"],
            r["family"],
            r["events"][0]["mode"],
            r["style"],
            r["events"][0]["bits"][1],
            r["events"][0]["actor"],
        ): r
        for r in bank
        if r["kind"] == "single"
    }
    for reader in MODELS:
        for target in sorted({r["target"] for r in bank}):
            for author in range(16):
                for account in P.ACCOUNTS:
                    entries, expected, complete = [], [], True
                    for c, j in ((-1, 0), (-1, 1), (1, 0), (1, 1)):
                        family = (author + j) % 4
                        polarity = -1 if (author + j + (c == 1)) % 3 == 0 else 1
                        row = singles[target, family, account, int(c == 1), polarity, 1]
                        pred = lookup[reader, row["material_id"], "dag_typed", account]
                        complete &= pred["complete"]
                        if pred["complete"] and pred["observed_events"]:
                            entries.append(
                                {
                                    "event_id": f"{c}_{j}",
                                    "condition": c,
                                    "value": pred["profile"],
                                }
                            )
                        expected.append(
                            {
                                "event_id": f"{c}_{j}",
                                "condition": c,
                                "value": 1 if account == "state" else polarity,
                            }
                        )
                    profile, gold = (
                        P.account_profile(entries),
                        P.account_profile(expected),
                    )
                    panels.append(
                        {
                            "reader": reader,
                            "target": target,
                            "author_program": author,
                            "account": account,
                            "complete": bool(complete),
                            **profile,
                            "expected_level": gold["level"],
                            "expected_response": gold["response"],
                            "condition_meaning": "controlled wording/context program, not a human state change",
                        }
                    )
    write_csv(out / "03/author_profiles.csv", panels)
    summary = {
        "materials": len(bank),
        "kinds": dict(Counter(r["kind"] for r in bank)),
        "graph_nodes": sum(len(d["graph"]["nodes"]) for d in parsed),
        "node_types": dict(
            Counter(n["type"] for d in parsed for n in d["graph"]["nodes"])
        ),
        "flat_complete": sum(c["flat_missing"] == 0 for c in coverage),
        "dag_complete": sum(c["dag_missing"] == 0 for c in coverage),
        "scope_challenges": sum(r["kind"] == "scope_challenge" for r in bank),
        "metrics": metrics,
        "comparisons": comparisons,
        "author_profiles": len(panels),
        "transport_readouts": len(transports),
        "max_transport_error": max(
            abs(t["original"] - t["transported"]) for t in transports
        ),
        "fixed_mask_mean_absolute_change": float(
            np.mean([abs(t["original"] - t["fixed_mask"]) for t in transports])
        ),
        "natural_cache_run": "pending licensed source inventory",
        "source_refits": 0,
        "api_calls": 0,
        "private_reads": 0,
        "new_neural_training": 0,
        "full_goal_complete": False,
    }
    write_json(out / "03/summary.json", summary)
    return summary


def main():
    """Execute one cold bounded packet; never mutate earlier results or source maps."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reuse-packet", type=Path)
    args = ap.parse_args()
    source, bank = prepare(args.out, args.reuse_packet)
    parsed = parse(args.out, bank, args.reuse_packet)
    indices = inputs(args.out, source, bank, parsed)
    for reader in MODELS:
        if args.reuse_packet is None:
            encode(args.out, reader, args.device)
        else:
            import shutil

            if (args.out / "00/material_bank.jsonl").read_bytes() != (
                args.reuse_packet / "00/material_bank.jsonl"
            ).read_bytes():
                raise ValueError("reuse requires identical ordered encoder inputs")
            dest = args.out / "01B/embeddings" / f"{reader}.npy"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.reuse_packet / f"01B/embeddings/{reader}.npy", dest)
            write_json(
                args.out / f"01B/{reader}_reuse.json",
                {
                    "source": str(args.reuse_packet),
                    "sha256": sha(dest),
                    "new_encoder_inputs": 0,
                    "same_ordered_texts": True,
                },
            )
    predictions, transports = evaluate(args.out, source, bank, indices)
    summary = summarize(args.out, bank, parsed, predictions, transports)
    summary["version_programs"] = version_checks(args.out, bank, parsed)
    summary["old_failures_diagnostic"] = old_diagnostics(args.out)
    write_json(args.out / "03/summary.json", summary)
    write_json(
        args.out / "04/artifact_hashes.json",
        {
            str(p.relative_to(args.out)): sha(p)
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
            and not p.name.startswith("._")
            and p != args.out / "04/artifact_hashes.json"
        },
    )
    print(
        f"HM-07 finished: DAG {summary['dag_complete']}/{len(bank)}, flat {summary['flat_complete']}/{len(bank)}",
        flush=True,
    )


def version_checks(out, bank, parsed):
    """Exercise actual newly parsed graph snapshots through supplied revisions."""
    singles = {
        (
            r["target"],
            r["family"],
            r["events"][0]["mode"],
            r["style"],
            r["events"][0]["bits"][1],
            r["events"][0]["actor"],
        ): i
        for i, r in enumerate(bank)
        if r["kind"] == "single"
    }
    rows = []
    for target in sorted({r["target"] for r in bank}):
        for family in range(4):
            a = singles[target, family, "actual", 0, 1, 1]
            b = singles[target, family, "plan", 1, 1, 1]
            for change in ("alias", "revise", "retract"):
                common = {
                    "author_id": f"{target}_{family}",
                    "event_id": "e",
                    "condition": -1,
                }
                records = [
                    {
                        **common,
                        "record_id": "r1",
                        "operation": "new",
                        "graph": parsed[a]["graph"],
                    },
                    {
                        **common,
                        "record_id": "r2",
                        "operation": change,
                        "parent_id": "r1",
                        "graph": parsed[b if change == "revise" else a]["graph"],
                    },
                ]
                active, nodes = D.replay_graph_records(records)
                rows.append(
                    {
                        "target": target,
                        "family": family,
                        "change": change,
                        "active_records": [r["record_id"] for r in active],
                        "nodes": len(nodes),
                        "node_ids": [n["node_id"] for n in nodes],
                        "actual_main": sum(
                            n["candidate_mode"] == "actual"
                            and n["type"] == "main_proposition"
                            for n in nodes
                        ),
                        "plan_main": sum(
                            n["candidate_mode"] == "plan"
                            and n["type"] == "main_proposition"
                            for n in nodes
                        ),
                        "context_nodes": sum(
                            n["type"].endswith("context") for n in nodes
                        ),
                    }
                )
    write_csv(out / "03/version_dependencies.csv", rows)
    return len(rows)


def old_diagnostics(out):
    """Replay the ten known HM-06 failures as diagnostics, without reencoding."""
    root = ROOT / "results/grain_math_response_v1/purpose_accounts_r1"
    rows = []
    for i, row in enumerate(read_bank(root / "00/documents.jsonl")):
        if row["kind"] == "unsupported_scope":
            continue
        data = json.loads((root / f"01/parses/{i:04d}.json").read_text())
        if R.missing_records(data["events"], len(row["events"])) == 0:
            continue
        graph = D.observe_dag(data["words"], row["text"])
        e = R.attach_event_identity(graph["events"], row["events"])
        rows.append(
            {
                "material_id": row["material_id"],
                "expected": len(row["events"]),
                "old_heads": len(data["events"]),
                "new_heads": len(e),
                "new_missing": R.missing_records(e, len(row["events"])),
                "retained_nodes": len(graph["nodes"]),
                "scope": "previously observed failure, not cold replication",
            }
        )
    write_csv(out / "03/old_failures_diagnostic.csv", rows)
    return {
        "documents": len(rows),
        "complete": sum(r["new_missing"] == 0 for r in rows),
        "new_parser_or_encoder_calls": 0,
        "independent_replication": False,
    }


if __name__ == "__main__":
    main()
