#!/usr/bin/env python3
"""Logic NEXT01/02/03: cached source-disjoint natural material transport."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.cohort import ID_SEED
from grain_research.vector_mechanisms.direct_traits import tokenize_prompt
from grain_research.vector_mechanisms.factor_path_routes import historical_overlap
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.natural_transport import (
    estimand_rows,
    fit_operator,
    paired_summary,
    removal_coordinate,
    response_channels,
)
from grain_research.vector_mechanisms.readout_cache_cpu import BASE, read, sha, write
from scripts.analyze_grain_conditional_edits import load_panel
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

PLAN = ROOT / "docs/GRAIN_LOGIC_NATURAL_TRANSPORT_PLAN_V1.md"
FROZEN = ROOT / "results/v7_operator_boundary_audit/e0_full_20260714"
SOURCE = BASE / "all_factor_source_m29_v1"
S1 = BASE / "multiquery_sensitivity_s1_r1"
LEVEL = BASE / "complete_level_response_v1"
QUERIES = ("TF_cont", "neuroticism")
SEMANTIC_QUERIES = (*QUERIES, "confirmation_seeking")
OPERATORS = ("whole", "native", "fixed_128_cross")
FACTORS = [f"{op}:F{i:02d}" for op in OPERATORS for i in range(1, 5)]
KINDS = ("between", "within", "transition")


def load_inputs():
    """Bind exact cached materials, independent author roles and query functions."""
    native, states, lm, hashes = load_panel()
    sm, s1m = read(SOURCE / "manifest.json"), read(S1 / "manifest.json")
    if len({sm["model_sha256"], s1m["model_sha256"], lm["model_sha256"]}) != 1:
        raise ValueError("Native/source checkpoint mismatch")
    if sha(S1 / "private_material.json") != s1m["material_sha256"]:
        raise ValueError("S1 material changed")
    if sha(SOURCE / "private_design.json") != sm["input_sha256"]:
        raise ValueError("Source query design changed")
    functions = Path(sm["functions"]) / "nine_query_functions.npz"
    if sha(functions) != sm["functions_sha256"]:
        raise ValueError("Frozen functions changed")
    fp = BASE / "factor_path_routes_v1/source_r2/private_factor_views.csv"
    pp = BASE / "dynamic_path_channels_v1/private_path_views.csv"
    factors, paths = pd.read_csv(fp), pd.read_csv(pp)
    if factors.duplicated(["author_id", "view"]).any():
        raise ValueError("Duplicated factor observation")
    if (
        set(native.author_id) & set(factors.author_id)
        or factors.author_id.nunique() != 276
    ):
        raise ValueError("Source/evaluation authors not disjoint or incomplete")
    past = pd.read_csv(FROZEN / "scores_native.csv", dtype={"user_id": str})
    if historical_overlap(past.user_id, set(native.author_id), seed=ID_SEED):
        raise ValueError("Held author in factor construction")
    for d in read(SOURCE / "private_design.json"):
        if d["query"] not in QUERIES:
            continue
        for view in d["views"]:
            mask = np.asarray(view["question_mask"], bool)
            if (
                np.asarray(view["input_ids"])[mask].tolist()
                != lm["question_tokens"][d["query"]]
            ):
                raise ValueError("Source/evaluation query token mismatch")
    material = {
        r["author_id"]: r
        for r in read(S1 / "private_material.json")
        if r["view"] == 0 and r["author_id"] in set(native.author_id)
    }
    if len(material) != 12:
        raise ValueError("Require all twelve exact held materials")
    packets = {}
    for p in sorted((S1 / "private_packets").glob("panel_*.json")):
        r = read(p)
        if r["view"] != 0 or r["author_id"] not in material:
            continue
        if sha(p.with_suffix(".npz")) != r["state_sha256"]:
            raise ValueError("S1 native packet changed")
        if (
            r["author_id"] in packets
            or r["panel_index"] != material[r["author_id"]]["panel_index"]
        ):
            raise ValueError("Invalid native/material alignment")
        packets[r["author_id"]] = r
        hashes[str(p.relative_to(ROOT))] = sha(p)
        hashes[str(p.with_suffix(".npz").relative_to(ROOT))] = r["state_sha256"]
    if len(packets) != 12:
        raise ValueError("Missing held native packet")
    files = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/natural_transport.py",
        SOURCE / "manifest.json",
        S1 / "manifest.json",
        S1 / "private_material.json",
        fp,
        pp,
        functions,
        FROZEN / "artifacts/common_source_comment_representation.joblib",
    ]
    files += [
        FROZEN / "operator_bundles" / f"{op}_factor_bundle.json" for op in OPERATORS
    ]
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in files})
    return native, states, factors, paths, material, packets, functions, s1m, hashes


def score_material(material, model_dir):
    """Score frozen GRAIN posteriors and verify exact native full/E1 input tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )
    lookup = {
        (d["author_id"], d["query"]): d for d in read(LEVEL / "private_design.json")
    }
    raw, meta = [], []
    for uid, row in sorted(material.items()):
        for edit in ("full", "e1", "e2"):
            texts = [
                t
                for j, t in enumerate(row["texts"])
                if edit == "full" or j not in row[edit]
            ]
            if edit != "e2":
                for q in QUERIES:
                    ids, _, _ = tokenize_prompt(texts, q, tokenizer)
                    stem = "full" if edit == "full" else "edited"
                    if ids != lookup[(uid, q)][f"{stem}_ids"]:
                        raise ValueError(
                            "GRAIN material does not match cached neural input"
                        )
            key = f"{uid}::{edit}"
            meta.append(
                {"user_id": key, "author_id": uid, "edit": edit, "comments": len(texts)}
            )
            raw.extend({"author": key, "body": t} for t in texts)
    comments = canonicalize_comments(
        pd.DataFrame(raw), order_col=None, condition_col=None, min_tokens=12
    )
    comments["split"] = "confirmation"
    runtime = joblib.load(
        FROZEN / "artifacts/common_source_comment_representation.joblib"
    )
    scored = pd.DataFrame(meta)
    support = []
    for op in OPERATORS:
        bundle = read_factor_bundle(
            FROZEN / "operator_bundles" / f"{op}_factor_bundle.json"
        )
        units, _, scores = score_operator_from_comments(
            comments,
            spec=ObservationSpec(**bundle.operator),
            representation=runtime,
            bundle=bundle,
        )
        if len(scores) != len(scored) or not scores.score_status.eq("SCORED").all():
            raise ValueError("Missing factor support in held material")
        cols = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
        named = scores[["user_id", *cols]].rename(
            columns={c: f"{op}:F{i:02d}" for i, c in enumerate(cols, 1)}
        )
        scored = scored.merge(named, on="user_id", validate="one_to_one")
        support.append(
            {"operator": op, "units": len(units), "observations": len(scores)}
        )
    return scored, support


def source_maps(factors, paths):
    """Fit fixed source-only maps and retain grouped source diagnostic predictions."""
    maps, diagnostics = {}, []
    for q in QUERIES:
        joined = factors.merge(
            paths[paths["query"] == q], on=["author_id", "view"], validate="one_to_one"
        )
        joined = joined.sort_values(["author_id", "view"])
        counts = joined.groupby("author_id").size()
        if not counts.eq(3).all() or len(counts) != (63 if q == "TF_cont" else 213):
            raise ValueError("Incorrect source panel support")
        x = joined[FACTORS].to_numpy().reshape(-1, 3, len(FACTORS))
        y = joined["score"].to_numpy().reshape(-1, 3)
        ids = joined.author_id.unique()
        for kind in KINDS:
            maps[q, kind] = fit_operator(x, y, kind)
            for fold, (train, test) in enumerate(
                KFold(5, shuffle=True, random_state=20260910).split(x)
            ):
                fit = fit_operator(x[train], y[train], kind)
                if kind == "transition":
                    a = (
                        np.diff(x[test], axis=1)
                        - np.asarray(fit["transition_x_center"])
                    ).reshape(-1, len(FACTORS))
                    b = (
                        np.diff(y[test], axis=1)
                        - np.asarray(fit["transition_y_center"])
                    ).ravel()
                else:
                    a, b = estimand_rows(x[test], y[test], kind)
                prediction = a @ fit["coef"] + fit["intercept"]
                repeated = np.repeat(
                    ids[test], 1 if kind == "between" else 3 if kind == "within" else 2
                )
                diagnostics.extend(
                    {
                        "author_id": uid,
                        "query": q,
                        "kind": kind,
                        "fold": fold,
                        "actual": t,
                        "predicted": p,
                    }
                    for uid, t, p in zip(repeated, b, prediction, strict=True)
                )
    return maps, pd.DataFrame(diagnostics)


def compute(out):
    """Execute one cache-only test and retain every predeclared alternative."""
    native, states, factors, paths, material, packets, functions, s1m, hashes = (
        load_inputs()
    )
    scored, support = score_material(material, s1m["model_dir"])
    maps, diagnostics = source_maps(factors, paths)
    scored.to_csv(out / "private_factor_scores.csv", index=False)
    diagnostics.to_csv(out / "private_source_oof.csv", index=False)
    write(
        out / "source_operators.json",
        {f"{q}::{kind}": v for (q, kind), v in maps.items()},
    )
    print("Frozen factors scored; source-only operators fitted", flush=True)
    z = np.load(functions, allow_pickle=False)
    fun = {
        q: FunctionalQuery(*(z[f"{q}__{k}"] for k in ("Z", "h", "G", "g", "b")))
        for q in QUERIES
    }
    direction_path = BASE / "dynamic_path_channels_v1/native_direction"
    dm = read(direction_path / "manifest.json")
    direction_file = direction_path / "private_directions.npz"
    if sha(direction_file) != dm["direction_sha256"]:
        raise ValueError("Positive-control direction changed")
    with np.load(direction_file, allow_pickle=False) as d:
        fg = np.concatenate((z["TF_cont__G"].ravel(), z["TF_cont__g"]))
        scalar = float(fg @ d["real"] / dm["change_scale"])
    hashes[str(direction_file.relative_to(ROOT))] = sha(direction_file)
    hashes[str((direction_path / "manifest.json").relative_to(ROOT))] = sha(
        direction_path / "manifest.json"
    )
    lookup = scored.set_index(["author_id", "edit"])
    rows, semantics = [], []
    for n, item in native.iterrows():
        uid, q = item["author_id"], item["query"]
        full = lookup.loc[(uid, "full"), FACTORS].to_numpy(float)
        f_full = fun[q].parts(states[n]["full_P"], states[n]["full_Q"])["score"]
        f_edit = fun[q].parts(states[n]["edited_P"], states[n]["edited_Q"])["score"]
        readouts = packets[uid]["queries"][q]["readouts"]
        if (
            max(
                abs(readouts[k]["conditional_score"] - item[c])
                for k, c in (("full", "teacher_full"), ("e1", "teacher_edit"))
            )
            > 2e-6
        ):
            raise ValueError("S1 and multi-position native targets differ")
        for edit in ("e1", "e2"):
            kept = lookup.loc[(uid, edit), FACTORS].to_numpy(float)
            delta_g = kept - full
            row = {
                "author_id": uid,
                "query": q,
                "edit": edit,
                "native_full": float(item.teacher_full),
                "native_delta": float(
                    readouts[edit]["conditional_score"] - item.teacher_full
                ),
                "functional_full": f_full,
                "functional_delta": f_edit - f_full if edit == "e1" else np.nan,
            }
            for kind in KINDS:
                model = maps[q, kind]
                row[kind + "_delta"] = float(delta_g @ model["coef"])
                if kind == "between":
                    row["between_full"] = float(
                        full @ model["coef"] + model["intercept"]
                    )
            row["scalar_F02_delta"] = (
                float(delta_g[1] * scalar) if q == "TF_cont" else np.nan
            )
            rows.append(row)
            m = material[uid]
            lengths = np.asarray(m["lengths"], float)
            signed = np.asarray(m["signed_relevance"], float)
            for j, semantic_q in enumerate(SEMANTIC_QUERIES):
                semantics.append(
                    dict(
                        author_id=uid,
                        query=q,
                        edit=edit,
                        semantic_query=semantic_q,
                        native_delta=row["native_delta"],
                        functional_delta=row["functional_delta"],
                        removed_length_fraction=float(
                            lengths[m[edit]].sum() / lengths.sum()
                        ),
                        matched_length=bool(m["matching"]["within_tolerance"]),
                        **removal_coordinate(signed[:, j], m[edit]),
                    )
                )
    z.close()
    predictions, semantic = pd.DataFrame(rows), pd.DataFrame(semantics)
    predictions.to_csv(out / "private_predictions.csv", index=False)
    semantic.to_csv(out / "private_semantic_coordinates.csv", index=False)
    measures, moments, channel_rows = [], [], []
    for (q, edit), frame in predictions.groupby(["query", "edit"], sort=True):
        for target in ("native_delta", "functional_delta"):
            if frame[target].isna().any():
                continue
            arms = [k + "_delta" for k in KINDS] + (
                ["scalar_F02_delta"] if q == "TF_cont" else []
            )
            if target == "native_delta" and edit == "e1":
                arms += ["functional_delta"]
            for arm in arms:
                measures.append(
                    dict(
                        query=q,
                        edit=edit,
                        target=target,
                        arm=arm,
                        **paired_summary(frame[target], frame[arm]),
                    )
                )
        if edit == "e1":
            for target in ("native_full", "functional_full"):
                measures.append(
                    dict(
                        query=q,
                        edit="full",
                        target=target,
                        arm="between_full",
                        **paired_summary(frame[target], frame.between_full),
                    )
                )
            for kind in KINDS:
                ch, audit = response_channels(
                    frame.native_delta, frame.functional_delta, frame[kind + "_delta"]
                )
                moments.append(dict(query=q, kind=kind, **audit))
                for uid, a in zip(frame.author_id, ch, strict=True):
                    channel_rows.append(
                        {
                            "author_id": uid,
                            "query": q,
                            "kind": kind,
                            "grain": a[0],
                            "query_remainder": a[1],
                            "reader_gap": a[2],
                        }
                    )
    for (q, edit, semantic_q), frame in semantic.groupby(
        ["query", "edit", "semantic_query"]
    ):
        for target in ("native_delta", "functional_delta"):
            if frame[target].isna().any():
                continue
            for coord in ("delta", "fraction", "contrast", "removed_length_fraction"):
                measures.append(
                    dict(
                        query=q,
                        edit=edit,
                        target=target,
                        arm=f"semantic::{semantic_q}::{coord}",
                        **paired_summary(frame[target], frame[coord], calibrated=False),
                    )
                )
    pd.DataFrame(measures).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(
        out / "private_response_channels.csv", index=False
    )
    write(out / "channel_moments.json", moments)
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_LOGIC_NATURAL_TRANSPORT_V1",
            "hashes": hashes,
            "source_authors": 276,
            "held_authors": 12,
            "queries": list(QUERIES),
            "factors": FACTORS,
            "alpha": 100,
            "support": support,
            "model_sha256": s1m["model_sha256"],
            "new_neural_calls": 0,
            "psychological_labels_read": False,
            "scalar_F02_raw_coefficient": scalar,
            "semantic_mixture_max_error": float(semantic.identity_error.max()),
            "full_e1_materials_token_verified": True,
            "e2_verified_against_same_s1_material_and_packet": True,
            "e2_PQ_available": False,
            "matched_authors": int(
                semantic.groupby("author_id").matched_length.first().sum()
            ),
        },
    )
    write(
        out / "status.json",
        {"status": "COMPLETE", "rows": len(predictions), "metrics": len(measures)},
    )
    print(
        pd.DataFrame(measures)
        .query(
            "arm in ['between_delta','transition_delta','functional_delta','semantic::TF_cont::delta','semantic::neuroticism::delta']"
        )[["query", "edit", "target", "arm", "r", "r_lower", "r_upper"]]
        .to_string(index=False),
        flush=True,
    )


def main():
    """Create a new private packet; never overwrite an earlier experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "logic_natural_transport_v1"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Use the private result tree")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "status.json", {"status": "RUNNING", "started": time.time()})
    try:
        with threadpool_limits(limits=2):
            compute(out)
    except Exception as exc:
        write(
            out / "status.json", {"status": "FAILED", "error_type": type(exc).__name__}
        )
        raise


if __name__ == "__main__":
    main()
