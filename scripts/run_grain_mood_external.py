#!/usr/bin/env python3
"""M30C: a finite local expressed-state extension on the two fixed MEPS views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.counseling_queries import (
    SYSTEM,
    render,
    select_evidence,
)
from grain_research.vector_mechanisms.factor_catalog import association, bh_adjust
from grain_research.vector_mechanisms.mood_catalog import (
    DESCRIPTIONS,
    audit_mood,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.run_grain_counseling_external import collect


def prepare(root, out):
    """Freeze all276 inputs and audit items; no correlations or fitted readout."""
    from transformers import AutoTokenizer

    from grain_research.vector_mechanisms.context_carrier import query_region_mask
    from grain_research.vector_mechanisms.direct_traits import candidates
    from grain_research.vector_mechanisms.meps_material import load_text_material
    from scripts.run_grain_real_author_path import MODEL

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    material, sources = load_text_material(root)
    people = sorted({v["metadata"]["participant_code"] for v in material})
    definitions = {
        q: {
            "question": "Based only on the participant's own expressed experience in these excerpts, estimate the degree of "
            + descriptor
            + ". Do not rate an imagined character, diagnose a condition, or infer a lifelong trait. Use 0 for very low, 9 for very high, and intermediate digits for intermediate levels. Return only one digit from 0 to 9.",
            "kind": "adapted expressed-state composite",
            "items_basis": q,
        }
        for q, descriptor in DESCRIPTIONS.items()
    }
    design, coverage, qtokens = [], [], {}
    for view in ("free_chat", "direct_answers"):
        for uid in people:
            chosen = sorted(
                [
                    v
                    for v in material
                    if v["metadata"]["participant_code"] == uid
                    and v["metadata"]["view_kind"]
                    == ("free_chat" if view == "free_chat" else "direct_answer")
                ],
                key=lambda v: v["metadata"]["condition"],
            )
            texts = [v["variants"]["user_only"] for v in chosen]
            evidence, audit = select_evidence(texts, tokenizer)
            coverage.append(
                {
                    "participant_code": uid,
                    "view": view,
                    "source_spans": audit,
                    "ai_assistance_possible": any(
                        v["metadata"]["ai_assistance_possible"] for v in chosen
                    ),
                }
            )
            prefix = render(evidence, "pre_trust_positive", view).rsplit(
                "\n\nQuestion: ", 1
            )[0]
            for q, definition in definitions.items():
                user = prefix + "\n\nQuestion: " + definition["question"] + "\nAnswer:"
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
                start = text.index(user) + user.rindex("\nQuestion: ") + 1
                mask = query_region_mask(encoded["offset_mapping"], start)
                ids = np.asarray(encoded["input_ids"])
                if len(ids) > 3056:
                    raise ValueError("State-query input exceeds frozen context")
                if q in qtokens and qtokens[q] != ids[mask].tolist():
                    raise ValueError("State query tokens vary by view/person")
                qtokens[q] = ids[mask].tolist()
                design.append(
                    {
                        "index": len(design),
                        "participant_code": uid,
                        "view": view,
                        "query": q,
                        "input_ids": ids.tolist(),
                        "question_mask": mask.tolist(),
                    }
                )
    if len(people) != 46 or len(design) != 276:
        raise ValueError("Expected complete46 x2 x3 state design")
    out.mkdir(parents=True, exist_ok=False)
    (out / "private_packets").mkdir()
    scores, audit, source_path = audit_mood(root, authorized=True)
    scores.to_csv(out / "private_mood_scores.csv", index=False)
    audit.to_csv(out / "mood_scoring_audit.csv", index=False)
    write(out / "private_design.json", design)
    write(out / "private_coverage.json", coverage)
    write(out / "private_source_manifest.json", sources)
    write(out / "queries.json", definitions)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_M30C_ADAPTED_MOOD_V1",
            "units": len(design),
            "participants": len(people),
            "per_run_seconds": 1200,
            "question_tokens": qtokens,
            "input_sha256": sha(out / "private_design.json"),
            "criterion_cache_sha256": sha(out / "private_mood_scores.csv"),
            "mood_source_sha256": sha(source_path),
            "model_sha256": sha(MODEL / "model.safetensors"),
            "alphabet": candidates("neuroticism", tokenizer),
            "code_sha256": {
                p: sha(ROOT / p)
                for p in (
                    "scripts/run_grain_mood_external.py",
                    "scripts/run_grain_counseling_external.py",
                    "grain_research/vector_mechanisms/mood_catalog.py",
                    "grain_research/vector_mechanisms/counseling_queries.py",
                )
            },
            "plan_sha256": sha(ROOT / "docs/GRAIN_MEPS_MOOD_EXTENSION_M30C.md"),
            "questionnaires_read_for_scoring_audit": True,
            "outcome_correlations_computed": False,
        },
    )
    write(
        out / "run_status.json",
        {"status": "PREPARED", "completed_units": 0, "calls": 0, "seconds": 0},
    )
    print(
        {
            "status": "PREPARED",
            "units": len(design),
            "mood_phase_composites": len(audit),
            "missing_scores": int(scores.score.isna().sum()),
        },
        flush=True,
    )


def bootstrap_r(x, y, draws=1000):
    """Participant resampling of a fixed score/criterion association."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    rng, values = np.random.default_rng(20260909), []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        if np.ptp(x[ix]) > 1e-14 and np.ptp(y[ix]) > 1e-14:
            values.append(float(np.corrcoef(x[ix], y[ix])[0, 1]))
    return {
        "bootstrap_r_lower": float(np.quantile(values, 0.025)) if values else None,
        "bootstrap_r_upper": float(np.quantile(values, 0.975)) if values else None,
    }


def analyze(out):
    """Join frozen scores to all phase criteria, with endpoints and changes labeled."""
    if read(out / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Finish fixed state queries first")
    rows = []
    for d in read(out / "private_design.json"):
        path = out / "private_packets" / f"cell_{d['index']:04d}.json"
        p = read(path)
        if (
            any(p[k] != d[k] for k in ("participant_code", "view", "query", "index"))
            or sha(path.with_suffix(".npz")) != p["array_sha256"]
        ):
            raise ValueError("Prediction provenance mismatch")
        rows.append(p)
    scores = pd.DataFrame(rows)
    if scores.duplicated(["participant_code", "query", "view"]).any():
        raise ValueError("Duplicate participant-view-query")
    scores.to_csv(out / "private_predictions.csv", index=False)
    write(
        out / "prediction_freeze.json",
        {
            "sha256": sha(out / "private_predictions.csv"),
            "correlation_computed_before_freeze": False,
        },
    )
    if (
        sha(out / "private_mood_scores.csv")
        != read(out / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Mood scoring cache changed")
    criteria = pd.read_csv(out / "private_mood_scores.csv").pivot(
        index="participant_code", columns="scale", values="score"
    )
    result = []
    for (view, q), group in scores.groupby(["view", "query"]):
        group = group.set_index("participant_code").sort_index()
        target = criteria.reindex(group.index)
        end, start = ("post", "mid") if view == "free_chat" else ("mid", "pre")
        for name in target:
            intended = name == f"{end}_{q}"
            result.append(
                {
                    "view": view,
                    "query": q,
                    "criterion": name,
                    "kind": "endpoint" if intended else "temporal_or_cross_composite",
                    **association(group.native_score, target[name]),
                }
            )
            if intended:
                result[-1].update(bootstrap_r(group.native_score, target[name]))
        change = target[f"{end}_{q}"] - target[f"{start}_{q}"]
        result.append(
            {
                "view": view,
                "query": q,
                "criterion": f"{end}_minus_{start}_{q}",
                "kind": "observational_change",
                **association(group.native_score, change),
                **bootstrap_r(group.native_score, change),
            }
        )
    table = pd.DataFrame(result)
    table["q_bh_view_kind"] = table.groupby(["view", "kind"]).p.transform(bh_adjust)
    table.to_csv(out / "external_associations.csv", index=False)
    table[table.kind == "endpoint"].to_csv(
        out / "intended_associations.csv", index=False
    )
    write(
        out / "external_summary.json",
        {
            "status": "COMPLETE_M30C_EXPLORATORY_STATE_LINK",
            "participants": 46,
            "queries": 3,
            "views": 2,
            "links": len(table),
            "new_fits": 0,
            "canonical_scale_claim": False,
        },
    )
    print(
        table[table.kind != "temporal_or_cross_composite"]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meps-root", type=Path)
    a = parser.parse_args()
    if not a.output_dir.resolve().is_relative_to(BASE.resolve()):
        parser.error("Use ignored local mechanism output")
    if a.mode == "prepare":
        if a.meps_root is None:
            parser.error("Explicit authorized MEPS root required")
        prepare(a.meps_root, a.output_dir)
    elif a.mode == "collect":
        collect(a.output_dir)
    else:
        analyze(a.output_dir)
