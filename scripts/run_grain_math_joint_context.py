#!/usr/bin/env python3
"""HM-08 addressed context views with fixed HM-03 vector functions and units."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from run_grain_math_binding import load_model, read_bank
from run_grain_math_local_roles import vectors
from run_grain_math_observation_dag import truth
from run_grain_math_purpose_accounts import coordinates
from run_grain_math_response import MODELS, ROOT, encode, sha, write_csv, write_json

from grain_research.construction import math_binding as B
from grain_research.construction import math_joint_context as J
from grain_research.construction import math_local as L
from grain_research.construction import math_local_roles as R
from grain_research.construction import math_observation_dag as D
from grain_research.construction import math_purpose_accounts as P
from grain_research.source_event_graph import stanza_words

SPEC = ROOT / "configs/grain_math_joint_context_v1.json"
DEFAULT = ROOT / "results/grain_math_response_v1/joint_context_r1"
ARMS = ("strict", "legal_joint", "naive_joint", "conservative", "whole", "declared")


def prepare(out):
    """Freeze all materials/rules and verify source artifacts before observation."""
    spec = json.loads(SPEC.read_text())
    source = ROOT / spec["source_packet"]
    hashes = json.loads((source / "04/artifact_hashes.json").read_text())
    for name, digest in hashes.items():
        if sha(source / name) != digest:
            raise ValueError("changed HM-03 source: " + name)
    bank = J.material_bank(read_bank(source / "00/singles.jsonl"))
    if len(bank) != spec["new_materials"]:
        raise ValueError("material count disagrees with frozen spec")
    paths = [
        Path(__file__),
        SPEC,
        ROOT / "grain_research/construction/math_joint_context.py",
        ROOT / "grain_research/construction/math_observation_dag.py",
        ROOT / "grain_research/construction/math_purpose_accounts.py",
        ROOT / "grain_research/construction/math_binding.py",
        ROOT / "grain_research/construction/math_local_roles.py",
        ROOT / "grain_research/construction/math_local.py",
        ROOT / "grain_research/source_event_graph.py",
        ROOT / "scripts/run_grain_math_purpose_accounts.py",
        ROOT / "scripts/run_grain_math_local_roles.py",
        ROOT / "scripts/run_grain_math_binding.py",
        ROOT / "scripts/run_grain_math_response.py",
        ROOT / "scripts/run_grain_math_observation_dag.py",
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
            raise ValueError(
                "frozen HM-08 changed; preserve primary and use a new output"
            )
        return source, bank
    for p in paths:
        dest = out / "00/code" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
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
            "frozen_before_calls": True,
            "source_fits": 0,
            "view_selection_uses_target_score": False,
            "scope": "designed public expressions, not human personality outcomes",
        },
    )
    write_json(out / "00/factorization_witness.json", J.kernel_factorization_witness())
    print(f"HM-08 frozen {len(bank)} cold expressions; no source refits", flush=True)
    return source, bank


def parse(out, bank):
    """Run the actual frozen CPU parser and keep every context node and raw word."""
    from fetch_grain_syntax_reference import MODELS as SYNTAX
    from fetch_grain_syntax_reference import PACKAGES, PROCESSORS
    from run_grain_source_event_graph import verify_models

    verify_models()
    sys.path.insert(0, str(PACKAGES))
    import stanza
    import torch

    if stanza.__version__ != "1.11.0":
        raise ValueError("frozen Stanza version changed")
    torch.set_num_threads(2)
    parsed, pipeline, calls, start = [], None, 0, time.monotonic()
    for i, row in enumerate(bank):
        path = out / f"01/parses/{i:04d}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data["text_sha256"] != R.text_digest(row["text"]):
                raise ValueError("parse cache raw text changed")
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
            graph = D.observe_dag(words, row["text"])
            observed = J.object_views(words, row["text"], graph)
            events = R.attach_event_identity(observed, row["events"])
            data = {
                "text_sha256": R.text_digest(row["text"]),
                "words": words,
                "graph": graph,
                "events": events,
            }
            write_json(path, data)
            calls += 1
        parsed.append(data)
        if (i + 1) % 256 == 0:
            print(
                f"HM-08 parse {i + 1}/{len(bank)} {time.monotonic() - start:.1f}s",
                flush=True,
            )
    write_json(
        out / "01/parser_accounting.json",
        {
            "new_calls": calls,
            "cached_rows": len(bank) - calls,
            "elapsed_seconds": time.monotonic() - start,
            "parser_inventory_sha256": sha(SYNTAX / "grain_inventory.json"),
        },
    )
    return parsed


def inputs(out, source, bank, parsed):
    """Address actual strict/joint/whole candidates; gold appears only in a control."""
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

    indexed = []
    for row, data in zip(bank, parsed, strict=True):
        events = []
        for e in data["events"]:
            events.append(
                {
                    **e,
                    "readout": ref(e["projection_text"], "full_expression"),
                    "action_ref": ref("Action: " + e["action"] + ".", "action"),
                    "object_ref": ref("Object: " + e["object"] + ".", "strict_object"),
                    "legal_ref": ref(
                        "Object: " + e["legal_object"] + ".", "legal_object"
                    ),
                    "naive_ref": ref(
                        "Object: " + e["naive_object"] + ".", "naive_object"
                    ),
                    "candidate_refs": [
                        ref("Object: " + c["text"] + ".", "candidate_object")
                        for c in e["candidates"]
                    ],
                }
            )
        declared = [
            {
                **e,
                "status": "ok",
                "record_index": j,
                "actor": "self" if e["actor"] == 1 else "other",
                "conceiver": "self" if e["conceiver"] == 1 else "other",
                "readout": ref(row["text"][slice(*e["span"])], "full_expression"),
                "action_ref": ref("Action: " + e["action"] + ".", "action"),
                "object_ref": ref("Object: " + e["object"] + ".", "declared_reference"),
            }
            for j, e in enumerate(row["events"])
        ]
        indexed.append({"observed": events, "declared": declared})
    content = "".join(json.dumps(r) + "\n" for r in fresh)
    path = out / "00/material_bank.jsonl"
    if path.exists() and path.read_text() != content:
        raise ValueError("encoding bank changed")
    path.write_text(content)
    write_json(out / "01/observation_indices.json", indexed)
    return indexed


def batch_coordinates(indexed, selected, arm, model, old, new):
    """Batch fixed vector computations, retaining row/event addresses and envelopes."""
    events, slices = [], {}
    for i in selected:
        start = len(events)
        local = indexed[i]["declared" if arm == "declared" else "observed"]
        for e in local:
            field = (
                "legal_ref"
                if arm == "legal_joint"
                else "naive_ref"
                if arm == "naive_joint"
                else "object_ref"
            )
            events.append({**e, "object_ref": e[field]})
        slices[i] = (start, len(events))
    z = coordinates(events, model, old, new)
    env = [None] * len(events)
    if arm == "whole":
        z[:, 0] = L.affine_score(
            model["whole_factors"], vectors([e["readout"] for e in events], old, new)
        )[:, 0]
    if arm == "conservative":
        pairs, spans = [], []
        for e in events:
            a = len(pairs)
            pairs.extend((e["action_ref"], o) for o in e["candidate_refs"])
            spans.append((a, len(pairs)))
        scores = B.score_kernel(
            model["auto_tensor"],
            vectors([a for a, _ in pairs], old, new),
            vectors([o for _, o in pairs], old, new),
        )
        for j, (a, b) in enumerate(spans):
            env[j] = {
                **J.envelope(scores[a:b]),
                "candidate_scores": scores[a:b].tolist(),
            }
            z[j, 0] = env[j]["low"]
    return events, z, slices, env


def evaluate(out, source, bank, indexed):
    """Compare identical role routing under different frozen observation views."""
    predictions, event_rows, envelopes, contrasts = [], [], [], []
    for reader in MODELS:
        old = np.load(source / f"01B/embeddings/{reader}.npy").astype(float)
        new = np.load(out / f"01B/embeddings/{reader}.npy").astype(float)
        for target in sorted({r["target"] for r in bank}):
            model = load_model(source / f"02/{reader}_{target}.npz")
            selected = [i for i, r in enumerate(bank) if r["target"] == target]
            zs = {}
            for arm in ARMS:
                events, z, slices, env = batch_coordinates(
                    indexed, selected, arm, model, old, new
                )
                for i in selected:
                    a, b = slices[i]
                    local, zz, row = events[a:b], z[a:b], bank[i]
                    zs[i, arm] = zz
                    missing = R.missing_records(local, len(row["events"]))
                    for account in D.ACCOUNTS:
                        values, mass, eligible = D.readout(
                            zz, local, account, typed_owner=True
                        )
                        expected, count = truth(row, account)
                        predictions.append(
                            {
                                "reader": reader,
                                "target": target,
                                "material_id": row["material_id"],
                                "kind": row["kind"],
                                "family": row["family"],
                                "object_family": row["object_family"],
                                "variant": row["variant"],
                                "arm": arm,
                                "account": account,
                                "mode": row["events"][0]["mode"],
                                "complete": missing == 0,
                                "unresolved": missing,
                                "unit": float(values.sum() / len(row["events"]))
                                if not missing
                                else None,
                                "numerator": float(values.sum()),
                                "soft_mass": float(mass.sum()),
                                "observed_events": int(eligible.sum())
                                if not missing
                                else None,
                                "profile": float(values[eligible].mean())
                                if not missing and eligible.any()
                                else None,
                                "expected_unit": expected,
                                "expected_events": count,
                            }
                        )
                    for j, e in enumerate(local):
                        event_rows.append(
                            {
                                "reader": reader,
                                "target": target,
                                "material_id": row["material_id"],
                                "arm": arm,
                                "node_id": e.get("node_id"),
                                "event_id": e["event_id"],
                                "mode": e["mode"],
                                "actor": e.get("actor"),
                                "conceiver": e.get("conceiver"),
                                "k": float(zz[j, 0]),
                                "p": float(zz[j, 1]),
                                "o": float(zz[j, 2]),
                                "a": float(zz[j, 3]),
                                "object_text": e["object"],
                                "object_ref": json.dumps(e["object_ref"]),
                                "status": e["status"],
                            }
                        )
                        if arm == "conservative":
                            envelopes.append(
                                {
                                    "reader": reader,
                                    "target": target,
                                    "material_id": row["material_id"],
                                    "node_id": e["node_id"],
                                    "variant": row["variant"],
                                    **env[a + j],
                                }
                            )
                print(f"HM-08 scored {reader}/{target}/{arm}", flush=True)
            for i in selected:
                row = bank[i]
                if len(zs[i, "strict"]) != 1:
                    continue
                k = {arm: float(zs[i, arm][0, 0]) for arm in ARMS}
                strict, legal = indexed[i]["observed"][0], indexed[i]["observed"][0]
                contrasts.append(
                    {
                        "reader": reader,
                        "target": target,
                        "material_id": row["material_id"],
                        "variant": row["variant"],
                        "mode": row["events"][0]["mode"],
                        "expected_k": row["events"][0]["bits"][0],
                        **{f"k_{arm}": v for arm, v in k.items()},
                        "object_changed": strict["object_ref"] != legal["legal_ref"],
                        "action_changed": False,
                        "has_candidate": strict["has_attachment_alternative"],
                        "legal_minus_strict": k["legal_joint"] - k["strict"],
                        "naive_minus_strict": k["naive_joint"] - k["strict"],
                    }
                )
    write_csv(out / "02/predictions.csv", predictions)
    write_csv(out / "02/event_readouts.csv", event_rows)
    write_json(out / "02/candidate_envelopes.json", envelopes)
    write_csv(out / "03/target_contrasts.csv", contrasts)
    return predictions, contrasts, envelopes


def summarize(out, bank, parsed, predictions, contrasts, envelopes):
    """Separate observation coverage, candidate ambiguity and author-score loss."""
    coverage = [
        {
            "material_id": r["material_id"],
            "variant": r["variant"],
            "kind": r["kind"],
            "mode": r["events"][0]["mode"],
            "missing": R.missing_records(p["events"], len(r["events"])),
            "nodes": len(p["graph"]["nodes"]),
            "routed_events": len(p["events"]),
            "contexts": sum(len(e["contexts"]) for e in p["events"]),
            "candidate_events": sum(
                e["has_attachment_alternative"] for e in p["events"]
            ),
        }
        for r, p in zip(bank, parsed, strict=True)
    ]
    write_csv(out / "03/coverage.csv", coverage)
    write_json(
        out / "03/unresolved_examples.json",
        [
            {"material": r, "graph": p["graph"]}
            for r, p, c in zip(bank, parsed, coverage, strict=True)
            if c["missing"]
        ],
    )
    groups = defaultdict(list)
    for p in predictions:
        groups[p["reader"], p["target"], p["variant"], p["arm"], p["account"]].append(p)
    metrics = []
    for key, rows in groups.items():
        good = [r for r in rows if r["complete"]]
        metrics.append(
            {
                **dict(
                    zip(
                        ("reader", "target", "variant", "arm", "account"),
                        key,
                        strict=True,
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
    write_csv(out / "03/metrics.csv", metrics)
    lookup = {
        (p["reader"], p["material_id"], p["arm"], p["account"]): p for p in predictions
    }
    rng, comparisons = np.random.default_rng(8108), []
    for reader in MODELS:
        for target in sorted({r["target"] for r in bank}):
            for account in P.ACCOUNTS:
                for arm in ARMS[1:]:
                    families = defaultdict(list)
                    for r in bank:
                        if (
                            r["target"] != target
                            or r["kind"] != "single"
                            or r["events"][0]["mode"] != account
                        ):
                            continue
                        a, b = (
                            lookup[reader, r["material_id"], v, account]
                            for v in ("strict", arm)
                        )
                        if a["complete"] and b["complete"]:
                            families[r["family"], r["object_family"]].append(
                                (a["unit"] - a["expected_unit"]) ** 2
                                - (b["unit"] - b["expected_unit"]) ** 2
                            )
                    f = np.array([np.mean(v) for v in families.values()])
                    if len(f):
                        lo, hi = np.quantile(
                            f[rng.integers(0, len(f), (2000, len(f)))].mean(axis=1),
                            [0.025, 0.975],
                        )
                        comparisons.append(
                            {
                                "reader": reader,
                                "target": target,
                                "account": account,
                                "arm": arm,
                                "families": len(f),
                                "matched_mse_gain": float(f.mean()),
                                "ci_low": float(lo),
                                "ci_high": float(hi),
                                "scope": "fixed authored construction families, not human population CI",
                            }
                        )
    write_csv(out / "03/matched_comparisons.csv", comparisons)
    singles = {
        (
            r["target"],
            r["family"],
            r["object_family"],
            r["variant"],
            r["events"][0]["mode"],
            r["events"][0]["bits"][1],
            r["events"][0]["actor"],
        ): r
        for r in bank
        if r["kind"] == "single"
    }
    panels = []
    for reader, target, author, account, arm in product(
        MODELS, sorted({r["target"] for r in bank}), range(16), P.ACCOUNTS, ARMS
    ):
        entries, expected, complete = [], [], True
        for c, j in ((-1, 0), (-1, 1), (1, 0), (1, 1)):
            family = (author + j) % 4
            polarity = -1 if (author + j + (c == 1)) % 3 == 0 else 1
            variant = "reduced_positive" if c == -1 else "finite_positive"
            row = singles[target, family, j, variant, account, polarity, 1]
            pred = lookup[reader, row["material_id"], arm, account]
            complete &= pred["complete"]
            if pred["complete"] and pred["observed_events"]:
                entries.append(
                    {"event_id": f"{c}_{j}", "condition": c, "value": pred["profile"]}
                )
            expected.append(
                {
                    "event_id": f"{c}_{j}",
                    "condition": c,
                    "value": 1 if account == "state" else polarity,
                }
            )
        profile, gold = P.account_profile(entries), P.account_profile(expected)
        panels.append(
            {
                "reader": reader,
                "target": target,
                "author_program": author,
                "account": account,
                "arm": arm,
                "parser_complete": bool(complete),
                **profile,
                "expected_level": gold["level"],
                "expected_response": gold["response"],
            }
        )
    write_csv(out / "03/author_profiles.csv", panels)
    gate_groups = defaultdict(list)
    for r in contrasts:
        gate_groups[r["reader"], r["target"], r["variant"], r["mode"]].append(r)
    gate_rows = [
        {
            **dict(zip(("reader", "target", "variant", "mode"), key, strict=True)),
            "n": len(v),
            "changed_objects": sum(r["object_changed"] for r in v),
            "candidate_discoveries": sum(r["has_candidate"] for r in v),
            **{
                f"{arm}_gate_errors": sum(
                    (r[f"k_{arm}"] > 0) != (r["expected_k"] > 0) for r in v
                )
                for arm in ARMS
            },
            "joint_shift_mean": float(np.mean([r["legal_minus_strict"] for r in v])),
        }
        for key, v in gate_groups.items()
    ]
    write_csv(out / "03/gate_metrics.csv", gate_rows)
    summary = {
        "materials": len(bank),
        "new_source_fits": 0,
        "parser_complete": sum(c["missing"] == 0 for c in coverage),
        "routed_events": sum(c["routed_events"] for c in coverage),
        "nodes_retained": sum(c["nodes"] for c in coverage),
        "context_links_retained": sum(c["contexts"] for c in coverage),
        "candidate_events": sum(c["candidate_events"] for c in coverage),
        "candidate_gate_disagreements": sum(e["gate_disagreement"] for e in envelopes),
        "prediction_rows": len(predictions),
        "coordinate_rows": sum(len(p["events"]) for p in parsed) * 2 * 5
        + len(bank) * 2,
        "author_program_rows": len(panels),
        "author_complete": sum(p["parser_complete"] for p in panels),
        "author_scored": sum(
            p["level"] is not None and p["response"] is not None for p in panels
        ),
        "original_scope_complete": False,
        "goal_status": "active",
        "psychological_scope": "new controlled observation mechanisms, not human personality validity",
    }
    write_json(out / "03/summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    """Execute one bounded new-material packet, preserving resumable artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if (out / "04/artifact_hashes.json").exists():
        raise ValueError("completed output is frozen; use verification, not a rerun")
    source, bank = prepare(out)
    if args.prepare_only:
        return
    parsed = parse(out, bank)
    indexed = inputs(out, source, bank, parsed)
    for reader in MODELS:
        encode(out, reader, args.device)
    predictions, contrasts, envelopes = evaluate(out, source, bank, indexed)
    summarize(out, bank, parsed, predictions, contrasts, envelopes)
    hashes = {
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()
    }
    write_json(out / "04/artifact_hashes.json", hashes)
    print("HM-08 finite packet saved; full goal remains active", flush=True)


if __name__ == "__main__":
    main()
