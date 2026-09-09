#!/usr/bin/env python3
"""Complete NAV10's reader-source factorial for the same independent A/B programs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.affine_program import compile_affine, score_affine
from grain_research.vector_mechanisms.context_program import score_context_charts
from grain_research.vector_mechanisms.functional_profiles import FunctionalQuery
from grain_research.vector_mechanisms.navigation import (
    BASE,
    MODEL,
    OUTPUT,
    QUERIES,
    digest,
    load_functions,
    quality,
    read_json,
    write_json,
)
from grain_research.vector_mechanisms.projection_program import score_token_ids
from grain_research.vector_mechanisms.state_transport import pack_state
from scripts.run_grain_navigation_baseline import checked_arrays, save_unit
from scripts.run_grain_projection_program import load_program as load_token
from scripts.run_grain_source_span_extension import DEFAULT as SPAN
from scripts.run_grain_source_span_extension import OLD, load_extension
from scripts.run_grain_state_transport import (
    CONTEXT,
    input_units,
    load_operators,
    load_runtime,
)
from scripts.run_grain_upstream_routes import collector_guard

DEFAULT = BASE / "navigation_v1/NAV-10/reference_routes_r1"
TOKEN = BASE / "navigation_v1/NAV-03/input_pullback_r1"
REFERENCES = {"R_real": "real_reference.npz", "R_synth": "synthetic_reference.npz"}


def construct(out):
    """Change the reader source only; keep chart origins and all hyperparameters fixed."""
    import torch

    from grain_research.vector_mechanisms.context_program import (
        capture_context_jacobians,
    )
    from grain_research.vector_mechanisms.trace import DecoderObserver
    from grain_research.vector_mechanisms.upstream_routes import (
        capture_differentiable_prefix,
    )

    if out.exists():
        raise FileExistsError("Choose a fresh reference-control run")
    parent = read_json(OUTPUT / "manifest.json")
    refs = read_json(OUTPUT / "reference_source_manifest.json")
    if digest(MODEL / "model.safetensors") != parent["model_sha256"]:
        raise ValueError("Observer changed")
    functions = {}
    for name, path in REFERENCES.items():
        if digest(OUTPUT / path) != refs[name]:
            raise ValueError("Reader changed")
        functions[name] = load_functions(OUTPUT / path)
    if digest(OUTPUT / "design.json") != parent["design_sha256"]:
        raise ValueError("Origin design changed")
    design = {(d["world_id"], d["query"]): d for d in read_json(OUTPUT / "design.json")}
    _, models, _ = load_operators(OLD)
    expanded = load_extension(SPAN)
    out.mkdir(parents=True)
    files = {}
    for reference, fs in functions.items():
        for method, operators in (("A_linear", models), ("A_source", expanded)):
            for q in QUERIES:
                path = out / f"{reference}_{method}_{q}.npz"
                np.savez_compressed(path, **compile_affine(operators[q], fs[q]))
                files[path.name] = digest(path)
    torch.set_num_threads(2)
    obs = DecoderObserver.load(
        str(MODEL), device="mps", dtype="float32", max_tokens=3072
    )
    obs.model.requires_grad_(False)
    start = time.monotonic()
    for q, origins in parent["synthetic_references"].items():
        f = functions["R_real"][q]
        g = f.config_coef + f.mean_coef / len(f.p_ref)
        pieces = (
            FunctionalQuery(f.p_ref, f.q_ref, g, np.zeros_like(f.g_final), 0),
            FunctionalQuery(f.p_ref, f.q_ref, np.zeros_like(g), f.g_final, 0),
        )
        for i, wid in enumerate(origins):
            d = design[(wid, q)]
            meta = {"query": q, "world_id": wid, "source_split": "construction"}
            values = capture_context_jacobians(
                obs, d["input_ids"], d["mask"], f, cuts=(3,)
            )[3]
            save_unit(out / f"context_{q}_{i}.json", meta, **values)
            select = ~np.asarray(d["mask"], bool)
            select[-1] = False
            captures = [
                capture_differentiable_prefix(obs, d["input_ids"], d["mask"], p)
                for p in pieces
            ]
            tokens = torch.tensor(np.asarray(d["input_ids"])[select], device="mps")
            with torch.no_grad():
                embedding = obs.model.model.embed_tokens(tokens).float().cpu().numpy()
            save_unit(
                out / f"token_{q}_{i}.json",
                meta,
                embeddings=embedding,
                gradients=np.asarray([c["input_gradient"][select] for c in captures]),
                projection_origin=np.asarray([c["score"] for c in captures]),
            )
    for path in out.glob("*.json"):
        files[path.name] = digest(path)
        if path.with_suffix(".npz").exists():
            files[path.with_suffix(".npz").name] = digest(path.with_suffix(".npz"))
    write_json(
        out / "construction_freeze.json",
        {
            "protocol": "NAV10_REFERENCE_ROUTE_FACTORIAL_V1",
            "files": files,
            "code_sha256": digest(Path(__file__)),
            "readers": {r: refs[r] for r in REFERENCES},
            "state_source_sha256": digest(OLD / "program.json"),
            "expanded_source_sha256": digest(SPAN / "construction_freeze.json"),
            "origins": parent["synthetic_references"],
            "question_tokens": parent["question_tokens"],
            "anchors": {
                r: {q: fs[q].anchor for q in QUERIES} for r, fs in functions.items()
            },
            "token_program_sha256": digest(TOKEN / "program.json"),
            "context_program_sha256": digest(CONTEXT / "program.json"),
            "human_labels_used": False,
            "MEPS_construction": False,
            "reference_factor": "reader R_real/R_synth; identical synthetic pullback origins",
            "methods": ["A_linear", "A_source", "B_token", "B_context4"],
            "calls": obs.calls,
            "seconds": time.monotonic() - start,
        },
    )
    print(
        {
            "status": "CONSTRUCTED",
            "calls": obs.calls,
            "seconds": time.monotonic() - start,
        },
        flush=True,
    )


def load_frozen(out):
    """Load both reader-source arms and reject changed charts or coefficients."""
    m = read_json(out / "construction_freeze.json")
    if digest(Path(__file__)) != m["code_sha256"]:
        raise ValueError("Frozen implementation changed")
    for name, sha in m["files"].items():
        if digest(out / name) != sha:
            raise ValueError("Frozen coefficient changed")
    for folder, key in (
        (TOKEN, "token_program_sha256"),
        (CONTEXT, "context_program_sha256"),
    ):
        if digest(folder / "program.json") != m[key]:
            raise ValueError("Comparator program changed")
    _, table, synth = load_token(TOKEN)
    affine, token, context = {}, {}, {}
    for ref in REFERENCES:
        token[ref], context[ref] = {}, {}
        for q in QUERIES:
            token[ref][q], context[ref][q] = [], []
            for method in ("A_linear", "A_source"):
                with np.load(out / f"{ref}_{method}_{q}.npz", allow_pickle=False) as z:
                    affine[(ref, method, q)] = dict(z)
            for i in range(2):
                cp = (
                    out / f"context_{q}_{i}.json"
                    if ref == "R_real"
                    else CONTEXT / "charts" / f"{q}_{i}_3.json"
                )
                meta, c = checked_arrays(cp)
                if meta["query"] != q or meta["source_split"] != "construction":
                    raise ValueError("Chart identity changed")
                context[ref][q].append(c)
                if ref == "R_real":
                    meta, t = checked_arrays(out / f"token_{q}_{i}.json")
                    if meta["query"] != q or meta["source_split"] != "construction":
                        raise ValueError("Token chart identity changed")
                    token[ref][q].append(t)
                else:
                    token[ref][q].append(synth[q][i])
    return m, affine, token, context, table


def evaluate(out, mode, seconds):
    """Replay eight fixed candidates per input with one shared four-block execution."""
    m, affine, token, context, table = load_frozen(out)
    units, sources = input_units(mode)
    folder = out / mode
    manifest = {
        "source_hashes": sources,
        "construction_sha256": digest(out / "construction_freeze.json"),
        "mode": mode,
    }
    if folder.exists():
        if read_json(folder / "manifest.json") != manifest:
            raise ValueError("Evaluation design changed")
        if (folder / "prediction_freeze.json").exists():
            print({"status": "COMPLETE", "mode": mode, "new_calls": 0}, flush=True)
            return
    else:
        (folder / "private_units").mkdir(parents=True)
        write_json(folder / "manifest.json", manifest)
    _, runtime = load_runtime(OLD)
    start, rows = time.monotonic(), []
    prior = (
        read_json(folder / "run_status.json")
        if (folder / "run_status.json").exists()
        else {"calls": 0, "seconds": 0}
    )
    for u in units:
        ident = {k: v for k, v in u.items() if k != "views"}
        path = folder / "private_units" / f"unit_{u['index']:04d}.json"
        if path.exists():
            saved = read_json(path)
            if saved["identity"] != ident:
                raise ValueError("Input unit changed")
            rows.extend(saved["rows"])
            continue
        if time.monotonic() - start > seconds:
            write_json(
                folder / "run_status.json",
                {
                    "status": "PAUSED",
                    "calls": prior["calls"] + runtime.calls,
                    "seconds": prior["seconds"] + time.monotonic() - start,
                },
            )
            return
        q, current = u["query"], []
        for view in u["views"]:
            ids, mask = view["input_ids"], np.asarray(view["question_mask"], bool)
            if np.asarray(ids)[mask].tolist() != m["question_tokens"][q]:
                raise ValueError("Query coordinate mismatch")
            h = runtime.capture(ids)[3]
            x = pack_state(h[mask], h[-1])
            for ref in REFERENCES:
                anchor = m["anchors"][ref][q]
                values = {
                    "A_linear": score_affine(affine[(ref, "A_linear", q)], x),
                    "A_source": score_affine(affine[(ref, "A_source", q)], x),
                    "B_token": score_token_ids(
                        ids, mask, table, token[ref][q], anchor, "semantic_order"
                    ),
                    "B_context4": score_context_charts(
                        context[ref][q], h, mask, anchor
                    )["mean"],
                }
                current.extend(
                    {"reference": ref, "method": method, "score": value["score"]}
                    for method, value in values.items()
                )
        averaged = (
            pd.DataFrame(current)
            .groupby(["reference", "method"], as_index=False)
            .score.mean()
        )
        result = [{**ident, **r} for r in averaged.to_dict("records")]
        write_json(path, {"identity": ident, "rows": result})
        rows.extend(result)
        if (u["index"] + 1) % 60 == 0:
            print(
                {
                    "mode": mode,
                    "units": u["index"] + 1,
                    "total": len(units),
                    "seconds": time.monotonic() - start,
                },
                flush=True,
            )
    pd.DataFrame(rows).to_csv(folder / "predictions.csv", index=False)
    write_json(
        folder / "prediction_freeze.json",
        {
            "sha256": digest(folder / "predictions.csv"),
            "labels_read_before_freeze": False,
            "construction_sha256": manifest["construction_sha256"],
        },
    )
    write_json(
        folder / "run_status.json",
        {
            "status": "COMPLETE",
            "units": len(units),
            "calls": prior["calls"] + runtime.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
        },
    )
    print(read_json(folder / "run_status.json"), flush=True)


def analyze(out):
    """Read criteria only after all predictions; keep human, function and response results separate."""
    from scripts.analyze_grain_state_transport import external_quality

    load_frozen(out)
    frames = {}
    for mode in ("worlds", "natural", "external"):
        path = out / mode / "predictions.csv"
        if digest(path) != read_json(path.parent / "prediction_freeze.json")["sha256"]:
            raise ValueError("Predictions changed")
        frames[mode] = pd.read_csv(path, dtype={"author_id": str})
    folder = out / "analysis"
    if folder.exists():
        raise FileExistsError("Preserve earlier analysis")
    folder.mkdir()
    metrics = []
    target_world = pd.read_csv(OUTPUT / "native_and_function.csv")
    for mode in ("worlds", "natural"):
        if mode == "worlds":
            target = target_world[
                ["world_id", "query", "reference", "score", "native_score"]
            ].rename(columns={"score": "target"})
        else:
            source = BASE / "navigation_v1/NAV-04/natural_evidence_r1"
            path = source / "predictions.csv"
            if digest(path) != read_json(source / "prediction_freeze.json")["sha256"]:
                raise ValueError("Natural target freeze changed")
            target = natural_targets(pd.read_csv(path))
        frame = frames[mode].merge(
            target, on=["world_id", "query", "reference"], validate="many_to_one"
        )
        if len(frame) != len(frames[mode]):
            raise ValueError("Function targets dropped observations")
        if mode == "worlds":
            splits = {
                w["world_id"]: w["split"] for w in read_json(OUTPUT / "worlds.json")
            }
            frame = frame[frame.world_id.map(splits) == "held_scene"]
        for (q, ref, method), g in frame.groupby(["query", "reference", "method"]):
            metrics.append(
                {
                    "mode": mode,
                    "query": q,
                    "reference": ref,
                    "method": method,
                    **quality(g.target, g.score),
                    "native_r": quality(g.native_score, g.score)["r"],
                }
            )
        frame.to_csv(folder / f"private_{mode}.csv", index=False)
    pd.DataFrame(metrics).to_csv(folder / "function_metrics.csv", index=False)
    from scripts.analyze_grain_evidence_packets import (
        compare_responses,
        read_packet,
        response_rows,
    )

    natural_source = BASE / "navigation_v1/NAV-04/natural_evidence_r1"
    base = read_packet(natural_source)
    own = frames["natural"].copy()
    own["readout"] = own.reference + "_" + own.method
    own = own.drop(columns=["reference", "method"]).merge(
        pd.DataFrame(read_json(natural_source / "private_worlds.json")).drop(
            columns="texts"
        ),
        on="world_id",
        validate="many_to_one",
    )
    paired, response_metrics = compare_responses(
        response_rows(pd.concat([base, own], ignore_index=True), "natural")
    )
    paired.to_csv(folder / "private_natural_responses.csv", index=False)
    response_metrics.to_csv(folder / "response_metrics.csv", index=False)
    refdir = BASE / "navigation_v1/NAV-10/reference_transfer_r1"
    source = pd.read_csv(
        refdir / "private_source_external.csv", dtype={"author_id": str}
    )
    source = source[
        ["author_id", "query", "reference", "score", "y_true", "prediction"]
    ].rename(columns={"score": "full_function", "prediction": "native"})
    frame = frames["external"]
    keys = ["author_id", "query", "reference"]
    joined = frame[frame.dataset == "PANDORA_source"].merge(
        source, on=keys, validate="many_to_one"
    )
    mood = pd.read_csv(
        refdir / "private_counseling_predictions.csv", dtype={"participant_code": str}
    ).rename(
        columns={
            "participant_code": "author_id",
            "view": "dataset",
            "score": "full_function",
            "native_score": "native",
        }
    )
    chat = frame[frame.dataset != "PANDORA_source"].merge(
        mood[keys + ["dataset", "full_function", "native"]],
        on=keys + ["dataset"],
        validate="many_to_one",
    )
    criterion = BASE / "counseling_mood_m30c_v1/private_mood_scores.csv"
    if (
        digest(criterion)
        != read_json(criterion.parent / "manifest.json")["criterion_cache_sha256"]
    ):
        raise ValueError("Criteria changed")
    y = pd.read_csv(criterion, dtype={"participant_code": str}).pivot(
        index="participant_code", columns="scale", values="score"
    )
    chat["y_true"] = [
        float(
            (
                y.loc[
                    r.author_id,
                    f"{'post' if r.dataset == 'free_chat' else 'mid'}_adapted_dejected5",
                ]
                - 1
            )
            / 5
        )
        for r in chat.itertuples()
    ]
    joined = pd.concat([joined, chat], ignore_index=True)
    if (
        len(joined) != len(frame)
        or joined.duplicated(keys + ["dataset", "method"]).any()
    ):
        raise ValueError("External membership changed")
    human = []
    for (dataset, q, ref, method), g in joined.groupby(
        ["dataset", "query", "reference", "method"]
    ):
        human.append(
            {
                "dataset": dataset,
                "query": q,
                "reference": ref,
                "method": method,
                "full_function_r": quality(g.full_function, g.score)["r"],
                **external_quality(g.score, g.native, g.y_true, q == "TF_cont"),
            }
        )
    pd.DataFrame(human).to_csv(folder / "external_metrics.csv", index=False)
    joined.to_csv(folder / "private_external.csv", index=False)
    write_json(
        folder / "summary.json",
        {
            "status": "COMPLETE",
            "construction_sha256": digest(out / "construction_freeze.json"),
            "criterion_sha256": digest(criterion),
            "methods": 4,
            "reference_sources": 2,
            "human_fit": False,
            "source_factor": "reader provenance; not origin-source crossover",
            "all_predictions_frozen_before_criteria": True,
        },
    )
    print(
        pd.DataFrame(human)[
            [
                "dataset",
                "query",
                "reference",
                "method",
                "n",
                "function_external_r",
                "function_auc",
            ]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


def natural_targets(frame):
    """Align each reader arm to its own target and the same native observation."""
    keys = ["world_id", "query"]
    if frame.duplicated(keys + ["readout"]).any():
        raise ValueError("Duplicate target readings")
    native = frame[frame.readout == "native"][keys + ["score"]].rename(
        columns={"score": "native_score"}
    )
    targets = frame[frame.readout.isin(REFERENCES)][keys + ["readout", "score"]].rename(
        columns={"readout": "reference", "score": "target"}
    )
    joined = targets.merge(native, on=keys, validate="many_to_one")
    if len(joined) != 2 * len(native) or len(joined) != len(targets):
        raise ValueError("Reader-source factorial is incomplete")
    return joined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("construct", "evaluate", "analyze"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument(
        "--mode", choices=("worlds", "natural", "external"), default="worlds"
    )
    parser.add_argument("--seconds", type=int, default=1200)
    a = parser.parse_args()
    with collector_guard(a.out), threadpool_limits(limits=2):
        if a.action == "construct":
            construct(a.out)
        elif a.action == "evaluate":
            evaluate(a.out, a.mode, a.seconds)
        else:
            analyze(a.out)
