#!/usr/bin/env python3
"""M30: frozen user-only counseling queries and local questionnaire associations."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.counseling_queries import (
    SYSTEM,
    questions,
    render,
    select_evidence,
)
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write


def prepare(root, out, view, inventory):
    """Tokenize author material without reading linked questionnaire outcomes."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.context_carrier import query_region_mask
    from grain_research.vector_mechanisms.direct_traits import candidates
    from grain_research.vector_mechanisms.meps_material import load_text_material
    from scripts.run_grain_real_author_path import MODEL

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    views, source = load_text_material(root)
    authors = sorted({v["metadata"]["participant_code"] for v in views})
    design, coverage, qtokens = [], [], {}
    query_defs = questions()
    for uid in authors:
        chosen = [
            v
            for v in views
            if v["metadata"]["participant_code"] == uid
            and v["metadata"]["view_kind"]
            == ("free_chat" if view == "free_chat" else "direct_answer")
        ]
        chosen = sorted(chosen, key=lambda v: v["metadata"]["condition"])
        if not chosen:
            coverage.append({"participant_code": uid, "status": "no_user_view"})
            continue
        bodies = [v["variants"]["user_only"] for v in chosen]
        selected, audit = select_evidence(bodies, tokenizer)
        coverage.append(
            {
                "participant_code": uid,
                "status": "included",
                "source_spans": audit,
                "original_chars": sum(len(b) for b in bodies),
                "selected_chars": sum(len(b) for b in selected),
                "source_conditions": [v["metadata"]["condition"] for v in chosen],
                "ai_assistance_possible": any(
                    v["metadata"]["ai_assistance_possible"] for v in chosen
                ),
            }
        )
        for query in query_defs:
            user = render(selected, query, view)
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            if len(encoded["input_ids"]) > 3056:
                raise ValueError(
                    "Frozen evidence budget exceeds observer context; revise preparation before any inference"
                )
            start = text.index(user) + user.rindex("\nQuestion: ") + 1
            mask = query_region_mask(encoded["offset_mapping"], start)
            token_ids = np.asarray(encoded["input_ids"])[mask].tolist()
            if query in qtokens and qtokens[query] != token_ids:
                raise ValueError("Query tokens changed by participant")
            qtokens[query] = token_ids
            design.append(
                {
                    "index": len(design),
                    "participant_code": uid,
                    "view": view,
                    "query": query,
                    "input_ids": encoded["input_ids"],
                    "question_mask": mask.tolist(),
                }
            )
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    write(out / "private_design.json", design)
    write(out / "private_coverage.json", coverage)
    write(out / "queries.json", query_defs)
    write(out / "private_source_manifest.json", source)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_M30_EXTERNAL_V1",
            "view": view,
            "units": len(design),
            "participants": len({d["participant_code"] for d in design}),
            "questionnaire_read": False,
            "per_run_seconds": 1200,
            "selection": "complete_else_4_equispaced_char_spans_equal_source_budget2400",
            "input_sha256": sha(out / "private_design.json"),
            "query_sha256": sha(out / "queries.json"),
            "source_sha256": sha(out / "private_source_manifest.json"),
            "coverage_sha256": sha(out / "private_coverage.json"),
            "question_tokens": qtokens,
            "alphabet": candidates("neuroticism", tokenizer),
            "model_sha256": sha(MODEL / "model.safetensors"),
            "code_sha256": {
                p: sha(ROOT / p)
                for p in (
                    "scripts/run_grain_counseling_external.py",
                    "grain_research/vector_mechanisms/counseling_queries.py",
                )
            },
            "plan_sha256": sha(ROOT / "docs/GRAIN_MEPS_EXTERNAL_QUERIES_M30.md"),
            "criterion_cache_sha256": sha(inventory / "private_meps_scores.csv"),
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "completed_units": 0, "calls": 0, "seconds": 0},
    )
    print(
        {
            "status": "PREPARED",
            "view": view,
            "units": len(design),
            "authors": len(authors),
            "partial_coverage_authors": sum(
                r.get("selected_chars", 0) < r.get("original_chars", 0)
                for r in coverage
            ),
        },
        flush=True,
    )


def collect(out):
    """Collect native readouts plus P/Q states locally; never join human labels."""
    import torch

    from grain_research.vector_mechanisms.full_state_edit import capture_boundary
    from grain_research.vector_mechanisms.trace import DecoderObserver
    from scripts.run_grain_real_author_path import MODEL

    m, prior = read(out / "manifest.json"), read(out / "run_status.json")
    if prior["status"] == "COMPLETE":
        return
    for name, digest in m["code_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("M30 frozen code changed")
    if (
        sha(out / "private_design.json") != m["input_sha256"]
        or sha(MODEL / "model.safetensors") != m["model_sha256"]
    ):
        raise ValueError("Material/model changed")
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    manifest = observer.manifest()
    opath = out / "observer.json"
    if opath.exists():
        old = read(opath)
        for k in (
            "torch",
            "transformers",
            "dtype",
            "device",
            "attention",
            "template_sha256",
            "pad_to_length",
        ):
            if old[k] != manifest[k]:
                raise ValueError("Observer runtime changed")
    write(opath, manifest)
    start, complete = time.monotonic(), 0

    def status(state):
        """Persist aggregate, private-text-free progress at a completed unit."""
        value = {
            "status": state,
            "completed_units": complete,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "questionnaire_read": False,
        }
        write(out / "run_status.json", value)
        return value

    try:
        for d in read(out / "private_design.json"):
            path = out / "private_packets" / f"cell_{d['index']:04d}.json"
            if path.exists():
                if sha(path.with_suffix(".npz")) != read(path)["array_sha256"]:
                    raise ValueError("Completed packet changed")
                complete += 1
                continue
            if time.monotonic() - start > m["per_run_seconds"]:
                print(status("PAUSED_RESOURCE_BOUNDARY"), flush=True)
                return
            state, kwargs, native = capture_boundary(
                observer, d["input_ids"], m["alphabet"]
            )
            mask = np.asarray(d["question_mask"], bool)
            np.savez_compressed(
                path.with_suffix(".npz"),
                P=state[0, mask].cpu().numpy(),
                Q=state[0, -1].cpu().numpy(),
            )
            write(
                path,
                {k: d[k] for k in ("index", "participant_code", "view", "query")}
                | {
                    "native_score": native["conditional_score"],
                    "candidate_mass": native["candidate_mass"],
                    "array_sha256": sha(path.with_suffix(".npz")),
                },
            )
            complete += 1
            if complete % 22 == 0:
                print(status("COLLECTING"), flush=True)
            del state, kwargs
        print(status("COMPLETE"), flush=True)
    except Exception:
        status("ERROR")
        raise


def analyze(out, inventory):
    """Freeze native predictions before the local, participant-level criterion join."""
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Complete all planned participant-query units first")
    rows = []
    for d in read(out / "private_design.json"):
        path = out / "private_packets" / f"cell_{d['index']:04d}.json"
        p = read(path)
        if (
            any(p[k] != d[k] for k in ("index", "participant_code", "view", "query"))
            or sha(path.with_suffix(".npz")) != p["array_sha256"]
        ):
            raise ValueError("Prediction provenance mismatch")
        rows.append(p)
    scores = pd.DataFrame(rows)
    if scores.duplicated(["participant_code", "query"]).any():
        raise ValueError("Multiple observations per participant-query")
    scores.to_csv(out / "private_predictions.csv", index=False)
    write(
        out / "prediction_freeze.json",
        {
            "sha256": sha(out / "private_predictions.csv"),
            "human_labels_read_before_freeze": False,
        },
    )
    if (
        sha(inventory / "private_meps_scores.csv")
        != read(out / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Audited criterion cache changed after design freeze")
    targets = pd.read_csv(inventory / "private_meps_scores.csv")
    if targets.duplicated(["participant_code", "scale"]).any():
        raise ValueError("Duplicate criterion")
    result = []
    defs = read(out / "queries.json")
    for q, qp in scores.groupby("query"):
        for criterion, yc in targets.groupby("scale"):
            joined = qp.merge(yc, on="participant_code", validate="one_to_one")
            if len(joined) != len(qp):
                raise ValueError("Missing participant criterion")
            result.append(
                {
                    "view": qp.view.iloc[0],
                    "query": q,
                    "criterion": criterion,
                    "intended": q == criterion,
                    "phase": defs[q]["phase"],
                    "family": defs[q]["phase"] if q == criterion else "off_diagonal",
                    "prediction_sd": float(joined.native_score.std()),
                    "candidate_mass_mean": float(joined.candidate_mass.mean()),
                    **association(joined.native_score, joined.score),
                }
            )
            if q == criterion:
                rng, rs = np.random.default_rng(20260909), []
                x, y = joined.native_score.to_numpy(), joined.score.to_numpy()
                for _ in range(1000):
                    ix = rng.integers(len(x), size=len(x))
                    if np.ptp(x[ix]) > 1e-14 and np.ptp(y[ix]) > 1e-14:
                        rs.append(float(np.corrcoef(x[ix], y[ix])[0, 1]))
                result[-1].update(
                    bootstrap_r_lower=float(np.quantile(rs, 0.025)) if rs else None,
                    bootstrap_r_upper=float(np.quantile(rs, 0.975)) if rs else None,
                )
    result = pd.DataFrame(result)
    result["q_bh_family"] = result.groupby("family").p.transform(bh_adjust)
    result.to_csv(out / "external_associations.csv", index=False)
    result[result.intended].to_csv(out / "intended_associations.csv", index=False)
    write(
        out / "external_summary.json",
        {
            "status": "COMPLETE_EXPLORATORY_EXTERNAL_LINK",
            "participants": scores.participant_code.nunique(),
            "queries": scores["query"].nunique(),
            "all_pairs": len(result),
            "training": False,
            "source_scale_sha256": sha(inventory / "private_meps_scores.csv"),
            "privacy": "local_only_aggregate_report",
            "score_kind": "native conditional expectation; new functional maps not yet constructed",
        },
    )
    print(
        result[result.intended][["query", "n", "pearson_r", "p", "q_bh_family"]]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meps-root", type=Path)
    parser.add_argument(
        "--view", choices=("free_chat", "direct_answers"), default="free_chat"
    )
    parser.add_argument("--inventory-dir", type=Path)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Private outputs must stay under ignored mechanism results")
    if args.mode == "prepare":
        if args.meps_root is None or args.inventory_dir is None:
            parser.error("Explicit MEPS root and audited inventory required")
        prepare(args.meps_root, args.output_dir, args.view, args.inventory_dir)
    elif args.mode == "collect":
        collect(args.output_dir)
    else:
        if args.inventory_dir is None:
            parser.error("Audited M27 inventory required")
        analyze(args.output_dir, args.inventory_dir)
