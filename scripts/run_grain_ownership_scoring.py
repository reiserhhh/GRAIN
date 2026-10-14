#!/usr/bin/env python3
"""Source-blind quotation attribution and source-selected narrow scoring."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.source_ownership import (
    HYPOTHESES,
    WRAPPERS,
    classify_source,
    quote_premise,
    select_observations,
)
from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    digest,
    load_functions,
    read_json,
    write_json,
)
from scripts.run_grain_entailment_reference import MODEL as NLI_MODEL
from scripts.run_grain_entailment_reference import REVISION as NLI_REVISION
from scripts.run_grain_multiview_address import load_material
from scripts.run_grain_narrow_actions import REFERENCE
from scripts.run_grain_narrow_actions import verify as verify_old
from scripts.run_grain_narrow_functions import reference_view
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_upstream_routes import collector_guard

OLD = BASE / "all_plan_closure_v1/narrow_actions_r1"
GRAPH = BASE / "all_plan_closure_v1/source_event_r4"
DEFAULT = BASE / "all_plan_closure_v1/ownership_scoring_r1"
PLAN = ROOT / "docs/GRAIN_SOURCE_OWNERSHIP_SCORING_PLAN_V1.md"
CODE = (
    "scripts/run_grain_ownership_scoring.py",
    "grain_research/source_ownership.py",
    "scripts/run_grain_narrow_functions.py",
    "scripts/run_grain_navigation_baseline.py",
    "grain_research/vector_mechanisms/full_state_edit.py",
    "grain_research/vector_mechanisms/functional_profiles.py",
)


def natural_records():
    """Join first-person quotation addresses to frozen raw texts, without outcomes."""
    import json
    import re

    _, _, raw, _, material = load_material()
    summary = read_json(GRAPH / "summary.json")
    for name, expected in summary["files"].items():
        if digest(GRAPH / name) != expected:
            raise ValueError("Natural source graph changed")
    result = []
    for line in (GRAPH / "private_graphs.jsonl").open():
        graph = json.loads(line)
        source = graph["source_row"]
        text = raw[source]["body"]
        if hashlib.sha256(text.encode()).hexdigest() != graph["raw_sha256"]:
            raise ValueError("Natural text address changed")
        for qi, quote in enumerate(graph["quotes"]):
            body = text[quote["start"] : quote["end"]]
            if not quote["balanced"] or not re.search(
                r"\b(?:I|me|my)\b", body, re.IGNORECASE
            ):
                continue
            events = [
                e
                for e in graph["events"]
                if quote["start"] <= e["predicate_start"] < quote["end"]
            ]
            result.append(
                {
                    "record_id": f"natural_{source}_{qi}",
                    "domain": "natural",
                    "text": text,
                    "quote": body,
                    "source_row": source,
                    "start": quote["start"],
                    "end": quote["end"],
                    "syntax_routes": [e["subject_reference"] for e in events],
                    "event_ids": [e["event_id"] for e in events],
                    "expected_owner": None,
                }
            )
    return result, material


def prepare(out):
    """Freeze source-only inputs and500 query identities before detector inference."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if out.exists():
        raise FileExistsError("Use a fresh output packet")
    verify_old(OLD)
    definitions = read_json(REFERENCE / "queries.json")
    old_manifest = read_json(REFERENCE / "manifest.json")
    old_worlds = read_json(OLD / "private_worlds.json")
    old_design = {
        (d["world_id"], d["query"]): d["index"]
        for d in read_json(OLD / "private_design.json")
        if d["reader_role"] == "target"
    }
    records = []
    for w in old_worlds:
        records.append(
            {
                "record_id": "original:" + w["world_id"],
                "domain": "synthetic",
                "text": w["texts"][0],
                "quote": w["evidence_body"],
                "query": w["target"],
                "pair": w["pair"],
                "polarity": w["polarity"],
                "expected_owner": w["owner"],
                "wrapper": "original",
                "old_index": old_design[(w["world_id"], w["target"])],
                "old_world_id": w["world_id"],
            }
        )
        if w["pair"] == 0:
            for wrapper in ("written", "explained"):
                records.append(
                    {
                        "record_id": wrapper + ":" + w["world_id"],
                        "domain": "synthetic",
                        "text": WRAPPERS[wrapper][w["owner"]]
                        + '"'
                        + w["evidence_body"]
                        + '"',
                        "quote": w["evidence_body"],
                        "query": w["target"],
                        "pair": 0,
                        "polarity": w["polarity"],
                        "expected_owner": w["owner"],
                        "wrapper": wrapper,
                        "old_index": None,
                    }
                )
    if len(records) != 400 or len({r["record_id"] for r in records}) != 400:
        raise ValueError("Expected400 distinct authored source records")
    lookup = {
        (r["query"], r["wrapper"], r["polarity"], r["expected_owner"]): r["record_id"]
        for r in records
        if r["pair"] == 0
    }
    mixtures = []
    for query in definitions:
        for wrapper in WRAPPERS:
            for polarity, opposite in (("high", "low"), ("low", "high")):
                own, other = (
                    lookup[(query, wrapper, polarity, "self")],
                    lookup[(query, wrapper, opposite, "other")],
                )
                for order in ("self_first", "other_first"):
                    ids = [own, other] if order == "self_first" else [other, own]
                    mixtures.append(
                        {
                            "case_id": f"mix:{query}:{wrapper}:{polarity}:{order}",
                            "query": query,
                            "wrapper": wrapper,
                            "polarity": polarity,
                            "order": order,
                            "record_ids": ids,
                            "oracle_self_id": own,
                        }
                    )
    natural, material = natural_records()
    records.extend(natural)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    model_path = Path(
        snapshot_download(NLI_MODEL, revision=NLI_REVISION, local_files_only=True)
    )
    nli_tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    requests = []
    for r in records:
        premise = quote_premise(r["text"], r["quote"])
        lengths = [
            len(nli_tokenizer(premise, h, truncation=False)["input_ids"])
            for h in HYPOTHESES.values()
        ]
        requests.append(
            {
                "record_id": r["record_id"],
                "premise": premise,
                "eligible": max(lengths) <= 512,
                "max_tokens": max(lengths),
            }
        )
    record_map = {r["record_id"]: r for r in records}
    cases = [
        {"case_id": r["record_id"], "query": r["query"], "record_ids": [r["record_id"]]}
        for r in records
        if r["domain"] == "synthetic" and r["old_index"] is None
    ] + mixtures
    design = []
    for case in cases:
        _, _, encoded, mask = reference_view(
            tokenizer,
            [record_map[i]["text"] for i in case["record_ids"]],
            definitions[case["query"]]["question"],
        )
        if (
            np.asarray(encoded["input_ids"])[mask].tolist()
            != old_manifest["question_tokens"][case["query"]]
        ):
            raise ValueError("Old question coordinates changed")
        design.append(
            {
                "index": len(design),
                **case,
                "input_ids": encoded["input_ids"],
                "mask": mask.tolist(),
                "alphabet": old_manifest["alphabet"],
            }
        )
    if len(design) != 500 or len(mixtures) != 300:
        raise ValueError("Unexpected bounded packet size")
    out.mkdir(parents=True)
    (out / "private_nli").mkdir()
    (out / "private_packets").mkdir()
    for name, value in (
        ("private_records.json", records),
        ("private_nli_requests.json", requests),
        ("private_design.json", design),
        ("private_mixtures.json", mixtures),
    ):
        write_json(out / name, value)
    inputs = {
        n: digest(out / n)
        for n in (
            "private_records.json",
            "private_nli_requests.json",
            "private_design.json",
            "private_mixtures.json",
        )
    }
    sources = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            OLD / "manifest.json",
            OLD / "summary.json",
            OLD / "private_worlds.json",
            OLD / "private_design.json",
            REFERENCE / "reference.npz",
            REFERENCE / "queries.json",
            REFERENCE / "manifest.json",
            GRAPH / "summary.json",
            GRAPH / "private_graphs.jsonl",
        )
    }
    write_json(
        out / "manifest.json",
        {
            "protocol": "GRAIN_SOURCE_OWNERSHIP_SCORING_V1",
            "inputs": inputs,
            "sources": sources,
            "plan_sha256": digest(PLAN),
            "code": {n: digest(ROOT / n) for n in CODE},
            "nli": {
                "model": NLI_MODEL,
                "revision": NLI_REVISION,
                "path": str(model_path),
                "weight_sha256": digest(model_path / "model.safetensors"),
                "hypotheses": HYPOTHESES,
            },
            "model_sha256": old_manifest["model_sha256"],
            "material": material,
            "synthetic_records": 400,
            "natural_quotes": len(natural),
            "nli_eligible_records": sum(r["eligible"] for r in requests),
            "native_units": 500,
            "mixtures": 300,
            "labels_used_for_routing": False,
            "meps_used": False,
            "training": False,
        },
    )
    for stage in ("nli", "native"):
        write_json(
            out / f"{stage}_status.json",
            {"status": "PREPARED", "calls": 0, "seconds": 0, "units": 0},
        )
    print(
        {
            "status": "PREPARED",
            "records": len(records),
            "natural_quotes": len(natural),
            "nli_eligible": sum(r["eligible"] for r in requests),
            "native_units": len(design),
        },
        flush=True,
    )


def verify(out):
    """Refuse changed inputs or instruments; old sources remain unmodified."""
    m = read_json(out / "manifest.json")
    for root, key in ((out, "inputs"), (ROOT, "sources"), (ROOT, "code")):
        for name, expected in m[key].items():
            if digest(root / name) != expected:
                raise ValueError(f"Changed {key}: {name}")
    if digest(PLAN) != m["plan_sha256"]:
        raise ValueError("Frozen plan changed")
    return m


def nli(out, seconds):
    """Classify text-only requests; per-record journals survive bounded interruption."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    m = verify(out)
    prior = read_json(out / "nli_status.json")
    if prior["status"] == "COMPLETE":
        print(prior, flush=True)
        return
    model_path = Path(m["nli"]["path"])
    if digest(model_path / "model.safetensors") != m["nli"]["weight_sha256"]:
        raise ValueError("NLI weights changed")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
    ).eval()
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    if set(labels.values()) != {"entailment", "contradiction", "neutral"}:
        raise ValueError("NLI label mapping changed")
    requests = read_json(out / "private_nli_requests.json")
    eligible = [(i, r) for i, r in enumerate(requests) if r["eligible"]]
    start, calls, complete = time.monotonic(), 0, 0
    for begin in range(0, len(eligible), 8):
        batch = []
        for index, r in eligible[begin : begin + 8]:
            path = out / "private_nli" / f"record_{index:04d}.json"
            if path.exists():
                saved = read_json(path)
                if saved["record_id"] != r["record_id"]:
                    raise ValueError("NLI resume identity mismatch")
                complete += 1
            else:
                batch.append((index, r))
        if time.monotonic() - start >= seconds:
            write_json(
                out / "nli_status.json",
                {
                    "status": "PAUSED_RESOURCE_BOUNDARY",
                    "calls": prior["calls"] + calls,
                    "seconds": prior["seconds"] + time.monotonic() - start,
                    "units": complete,
                },
            )
            return
        if not batch:
            continue
        pairs = [(r["premise"], h) for _, r in batch for h in HYPOTHESES.values()]
        inputs = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError("No silent NLI truncation")
        with torch.inference_mode():
            values = torch.softmax(model(**inputs).logits, -1).numpy()
        calls += len(pairs)
        for j, (index, r) in enumerate(batch):
            probabilities = {
                owner: {labels[k]: float(v) for k, v in enumerate(values[2 * j + c])}
                for c, owner in enumerate(HYPOTHESES)
            }
            write_json(
                out / "private_nli" / f"record_{index:04d}.json",
                {
                    "record_id": r["record_id"],
                    "probabilities": probabilities,
                    **classify_source(probabilities),
                },
            )
            complete += 1
        status = {
            "status": "CAPTURING",
            "calls": prior["calls"] + calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": complete,
        }
        write_json(out / "nli_status.json", status)
        if complete % 80 == 0:
            print(status, flush=True)
    write_json(
        out / "nli_status.json",
        {
            "status": "COMPLETE",
            "calls": prior["calls"] + calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "units": complete,
        },
    )
    print(read_json(out / "nli_status.json"), flush=True)


def native(out, seconds):
    """Capture fixed-reader and affine inputs without changing any weights."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver

    m = verify(out)
    prior = read_json(out / "native_status.json")
    if prior["status"] == "COMPLETE":
        print(prior, flush=True)
        return
    if digest(MODEL / "model.safetensors") != m["model_sha256"]:
        raise ValueError("Native model changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    write_json(out / "observer.json", observer.manifest())
    start, complete = time.monotonic(), 0
    for done, d in enumerate(read_json(out / "private_design.json"), start=1):
        path = out / "private_packets" / f"unit_{d['index']:04d}.json"
        if path.exists():
            meta, _ = checked_arrays(path)
            if meta["case_id"] != d["case_id"] or meta["query"] != d["query"]:
                raise ValueError("Native resume identity mismatch")
        else:
            if time.monotonic() - start >= seconds:
                write_json(
                    out / "native_status.json",
                    {
                        "status": "PAUSED_RESOURCE_BOUNDARY",
                        "units": complete,
                        "calls": prior["calls"] + observer.calls,
                        "seconds": prior["seconds"] + time.monotonic() - start,
                    },
                )
                return
            state, _, value = capture_boundary(observer, d["input_ids"], d["alphabet"])
            mask = np.asarray(d["mask"], bool)
            save_unit(
                path,
                {
                    "index": d["index"],
                    "case_id": d["case_id"],
                    "query": d["query"],
                    "native": value,
                },
                P=state[0, mask].cpu().numpy(),
                Q=state[0, -1].cpu().numpy(),
            )
        complete = done
        status = {
            "status": "CAPTURING",
            "units": complete,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        }
        if complete % 50 == 0:
            write_json(out / "native_status.json", status)
            print(status, flush=True)
    write_json(out / "native_status.json", status | {"status": "COMPLETE"})
    print(read_json(out / "native_status.json"), flush=True)


def evaluate(out):
    """Score actual selections; oracle information is consulted only for evaluation."""
    verify(out)
    if any(
        read_json(out / f"{stage}_status.json")["status"] != "COMPLETE"
        for stage in ("nli", "native")
    ):
        raise ValueError("Both stages must finish")
    records = read_json(out / "private_records.json")
    requests = read_json(out / "private_nli_requests.json")
    decisions = {}
    rows = []
    for index, (r, request) in enumerate(zip(records, requests, strict=True)):
        if r["record_id"] != request["record_id"]:
            raise ValueError("Record/request identity mismatch")
        if request["eligible"]:
            saved = read_json(out / "private_nli" / f"record_{index:04d}.json")
            decision = classify_source(saved["probabilities"])
            if saved["record_id"] != r["record_id"] or any(
                saved[k] != v for k, v in decision.items()
            ):
                raise ValueError("Detector journal replay differs")
        else:
            decision = {
                "source_owner": "unresolved",
                "reason": "NLI_CONTEXT_LIMIT",
                "entailed_claims": [],
            }
        decisions[r["record_id"]] = decision
        rows.append(
            {
                "record_id": r["record_id"],
                "domain": r["domain"],
                "wrapper": r.get("wrapper"),
                "polarity": r.get("polarity"),
                "query": r.get("query"),
                "expected_owner": r["expected_owner"],
                "eligible": request["eligible"],
                "nli_tokens": request["max_tokens"],
                **decision,
                "syntax_routes": r.get("syntax_routes", []),
            }
        )
    decision_frame = pd.DataFrame(rows)
    decision_frame.to_csv(out / "source_decisions.csv", index=False)
    functions = load_functions(
        REFERENCE / "reference.npz", queries=list(read_json(REFERENCE / "queries.json"))
    )
    scored = {}

    def point(query, meta, states):
        return {
            "native": float(meta["native"]["conditional_score"]),
            "affine": functions[query].parts(states["P"], states["Q"])["score"],
        }

    for r in records:
        if r["domain"] == "synthetic" and r["old_index"] is not None:
            meta, states = checked_arrays(
                OLD / "private_packets" / f"unit_{r['old_index']:04d}.json"
            )
            if meta["world_id"] != r["old_world_id"] or meta["query"] != r["query"]:
                raise ValueError("Reused old endpoint identity mismatch")
            scored[r["record_id"]] = point(r["query"], meta, states)
    for d in read_json(out / "private_design.json"):
        meta, states = checked_arrays(
            out / "private_packets" / f"unit_{d['index']:04d}.json"
        )
        if meta["case_id"] != d["case_id"] or meta["query"] != d["query"]:
            raise ValueError("New endpoint identity mismatch")
        scored[d["case_id"]] = point(d["query"], meta, states)
    record_map = {r["record_id"]: r for r in records}
    mixed = []
    for case in read_json(out / "private_mixtures.json"):
        ids = case["record_ids"]
        selected = select_observations(
            [record_map[i]["text"] for i in ids], [decisions[i] for i in ids]
        )
        selected_ids = [ids[i] for i in selected["selected_indices"]]
        gated = (
            scored[selected_ids[0]]
            if len(selected_ids) == 1
            else scored[case["case_id"]]
            if len(selected_ids) == 2
            else None
        )
        # Designed source labels below evaluate, but never select, the measurement.
        oracle = scored[case["oracle_self_id"]]
        for readout in ("native", "affine"):
            mixed.append(
                {
                    k: case[k]
                    for k in ("case_id", "query", "wrapper", "polarity", "order")
                }
                | {
                    "readout": readout,
                    "unfiltered": scored[case["case_id"]][readout],
                    "selected": gated[readout] if gated else np.nan,
                    "oracle_self": oracle[readout],
                    "status": selected["status"],
                    "selected_count": len(selected_ids),
                    "selected_ids": selected_ids,
                    "context_ids": [ids[i] for i in selected["context_indices"]],
                    "correct_self_only": selected_ids == [case["oracle_self_id"]],
                    "excerpt_coverage": selected["excerpt_coverage"],
                }
            )
    frame = pd.DataFrame(mixed)
    frame.to_csv(out / "mixed_scores.csv", index=False)
    errors = []
    for identity, g in frame.groupby(["wrapper", "readout"], sort=True):
        ok = g.selected.notna()
        errors.append(
            {
                "wrapper": identity[0],
                "readout": identity[1],
                "cases": len(g),
                "observed": int(ok.sum()),
                "correct_self_only": int(g.correct_self_only.sum()),
                "raw_mae_all": float(np.abs(g.unfiltered - g.oracle_self).mean()),
                "raw_mae_matched": float(
                    np.abs(g.loc[ok, "unfiltered"] - g.loc[ok, "oracle_self"]).mean()
                )
                if ok.any()
                else None,
                "selected_mae_observed": float(
                    np.abs(g.loc[ok, "selected"] - g.loc[ok, "oracle_self"]).mean()
                )
                if ok.any()
                else None,
            }
        )
    pd.DataFrame(errors).to_csv(out / "scoring_summary.csv", index=False)
    responses, orders = [], []
    for keys, g in frame.groupby(["query", "wrapper", "readout", "order"], sort=True):
        g = g.set_index("polarity")
        responses.append(
            dict(zip(("query", "wrapper", "readout", "order"), keys, strict=True))
            | {
                name: float(g.loc["high", name] - g.loc["low", name])
                for name in ("unfiltered", "selected", "oracle_self")
            }
        )
    for keys, g in frame.groupby(
        ["query", "wrapper", "readout", "polarity"], sort=True
    ):
        g = g.set_index("order")
        orders.append(
            dict(zip(("query", "wrapper", "readout", "polarity"), keys, strict=True))
            | {
                name: float(g.loc["self_first", name] - g.loc["other_first", name])
                for name in ("unfiltered", "selected", "oracle_self")
            }
        )
    pd.DataFrame(responses).to_csv(out / "polarity_responses.csv", index=False)
    pd.DataFrame(orders).to_csv(out / "order_sensitivity.csv", index=False)
    synthetic = decision_frame[decision_frame.domain == "synthetic"]
    accuracy = []
    for wrapper, g in synthetic.groupby("wrapper", sort=True):
        accuracy.append(
            {
                "wrapper": wrapper,
                "records": len(g),
                "correct": int((g.source_owner == g.expected_owner).sum()),
                "abstained": int((g.source_owner == "unresolved").sum()),
                "wrong": int(
                    (
                        (g.source_owner != "unresolved")
                        & (g.source_owner != g.expected_owner)
                    ).sum()
                ),
            }
        )
    natural = decision_frame[decision_frame.domain == "natural"]
    files = (
        "source_decisions.csv",
        "mixed_scores.csv",
        "scoring_summary.csv",
        "polarity_responses.csv",
        "order_sensitivity.csv",
    )
    summary = {
        "status": "SOURCE_OWNERSHIP_SCORING_COMPLETE",
        "source_classification": accuracy,
        "natural_quotes": len(natural),
        "natural_eligible": int(natural.eligible.sum()),
        "natural_routes": natural.source_owner.value_counts().to_dict(),
        "scoring": errors,
        "nli_status": read_json(out / "nli_status.json"),
        "native_status": read_json(out / "native_status.json"),
        "files": {n: digest(out / n) for n in files},
        "scope": "source selection tested on authored mixtures; natural source decisions are not gold labels",
    }
    write_json(out / "summary.json", summary)
    print(
        {
            k: summary[k]
            for k in (
                "status",
                "source_classification",
                "natural_quotes",
                "natural_routes",
                "scoring",
            )
        },
        flush=True,
    )


def main():
    """Separate freeze, source detector, fixed observer and exact scoring stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "nli", "native", "evaluate"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1200 or not args.out.resolve().is_relative_to(
        (BASE / "all_plan_closure_v1").resolve()
    ):
        parser.error("Use a bounded local all-plan packet")
    with collector_guard(args.out):
        if args.mode == "prepare":
            prepare(args.out)
        elif args.mode == "nli":
            nli(args.out, args.seconds)
        elif args.mode == "native":
            native(args.out, args.seconds)
        else:
            evaluate(args.out)


if __name__ == "__main__":
    main()
