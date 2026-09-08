#!/usr/bin/env python3
"""M23B: frozen query-coordinate readers on compatible independent natural edits."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.conditional_coordinates import FAMILIES
from grain_research.vector_mechanisms.readout_cache_cpu import (
    BASE,
    corr,
    read,
    sha,
    write,
)

LEVEL = BASE / "complete_level_response_v1"
PREFIX = BASE / "prefix_source_connection_v1"
PLAN = ROOT / "docs/GRAIN_CONDITIONAL_EDIT_CONNECTION_M23B.md"


def compatible_observers(a, b):
    """Collection time is provenance, not a change to the measurement operator."""
    return {k: v for k, v in a.items() if k != "unix_time"} == {
        k: v for k, v in b.items() if k != "unix_time"
    }


def load_panel():
    """Verify existing native edits without importing a neural runtime."""
    manifest = read(LEVEL / "manifest.json")
    if sha(LEVEL / "private_design.json") != manifest["input_sha256"]:
        raise ValueError("Edit design changed")
    if read(LEVEL / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("Native edit panel incomplete")
    if manifest["model_sha256"] != read(PREFIX / "manifest.json")["model_sha256"]:
        raise ValueError("Prefix and native checkpoints differ")
    if not compatible_observers(
        read(LEVEL / "observer.json"), read(PREFIX / "observer.json")
    ):
        raise ValueError("Prefix and native observers differ")
    rows, arrays, hashes = [], [], {}
    for d in read(LEVEL / "private_design.json"):
        if d["query"] not in ("TF_cont", "neuroticism"):
            continue
        path = LEVEL / "private_packets" / f"unit_{d['index']:02d}_{d['query']}.json"
        r = read(path)
        if any(r[k] != d[k] for k in ("index", "author_id", "query")):
            raise ValueError("Native author/query mismatch")
        if sha(path.with_suffix(".npz")) != r["arrays_sha256"]:
            raise ValueError("Native states changed")
        hashes[str(path.relative_to(ROOT))] = sha(path)
        hashes[str(path.with_suffix(".npz").relative_to(ROOT))] = r["arrays_sha256"]
        tokens = manifest["question_tokens"][d["query"]]
        for stem in ("full", "edited"):
            if (
                list(np.asarray(d[f"{stem}_ids"])[np.asarray(d[f"{stem}_mask"], bool)])
                != tokens
            ):
                raise ValueError("Native query token identity differs")
        if (
            abs(r["full"] - d["source_full"]) > 2e-6
            or abs(r["edited"] - d["source_edit"]) > 2e-6
        ):
            raise ValueError("Teacher target does not replay")
        rows.append(
            {
                "author_id": d["author_id"],
                "query": d["query"],
                "teacher_full": r["full"],
                "teacher_edit": r["edited"],
            }
        )
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            arrays.append(
                {
                    k: z[k].astype(float)
                    for k in ("full_P", "full_Q", "edited_P", "edited_Q")
                }
            )
    meta = pd.DataFrame(rows)
    if (
        len(meta) != 24
        or meta.author_id.nunique() != 12
        or meta.duplicated(["author_id", "query"]).any()
    ):
        raise ValueError("Require complete12-author two-query panel")
    return meta, arrays, manifest, hashes


def paired_metrics(y, p, draws=1000):
    """Describe both direction and magnitude; bootstrap authors, not token rows."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.ndim != 1 or y.shape != p.shape or not np.isfinite([y, p]).all():
        raise ValueError("Finite paired response vectors required")
    rng = np.random.default_rng(923)
    boots = []
    for _ in range(draws):
        ix = rng.integers(len(y), size=len(y))
        value = corr(y[ix], p[ix])
        if value is not None:
            boots.append(value)
    energy = float(np.mean(y * y))
    return {
        "authors": len(y),
        "pearson_r": corr(y, p),
        "r_lower": float(np.quantile(boots, 0.025)) if boots else None,
        "r_upper": float(np.quantile(boots, 0.975)) if boots else None,
        "mse_over_zero": float(np.mean((y - p) ** 2) / energy)
        if energy > 1e-20
        else None,
        "sign_match": float(np.mean(np.sign(y) == np.sign(p))),
        "predicted_rms": float(np.sqrt(np.mean(p * p))),
        "teacher_rms": float(np.sqrt(energy)),
    }


def main():
    """Apply saved readers only; preserve all families and functional comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.readout_dir.resolve(), args.output_dir.resolve()
    if not source.is_relative_to(BASE) or not out.is_relative_to(BASE):
        raise ValueError("Use private result paths")
    if read(source / "run_status.json")["status"] != "COMPLETE":
        raise ValueError("M23 readers incomplete")
    if (
        sha(source / "private_oof_before_outcomes.csv")
        != read(source / "oof_frozen.json")["sha256"]
    ):
        raise ValueError("M23 OOF changed")
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    rows, contribution_rows, metrics, links, part_metrics = [], [], [], [], []
    hashes = {}
    try:
        with threadpool_limits(limits=2):
            meta, arrays, lm, hashes = load_panel()
            evaluation_authors = set(meta.author_id)
            for query in ("TF_cont", "neuroticism"):
                ix = np.flatnonzero(meta["query"].to_numpy() == query)
                group = meta.iloc[ix]
                states = {k: np.asarray([arrays[i][k] for i in ix]) for k in arrays[0]}
                for family in FAMILIES:
                    before, after, deltas = [], [], []
                    for fold in range(5):
                        path = (
                            source
                            / "private_models"
                            / f"{query}_{family}_{fold}.joblib"
                        )
                        artifact = joblib.load(path)
                        hashes[str(path.relative_to(ROOT))] = sha(path)
                        if set(artifact["train_authors"]) & evaluation_authors:
                            raise ValueError(
                                "Native edit authors overlap fitting authors"
                            )
                        if artifact["question_tokens"] != lm["question_tokens"][query]:
                            raise ValueError("Saved reader query identity differs")
                        r = artifact["reader"]
                        before.append(r.predict(states["full_P"], states["full_Q"]))
                        after.append(r.predict(states["edited_P"], states["edited_Q"]))
                        if family == "p_full_q":
                            a, _, _ = r.full_parts(states["full_P"], states["full_Q"])
                            b, _, _ = r.full_parts(
                                states["edited_P"], states["edited_Q"]
                            )
                            deltas.append({k: b[k] - a[k] for k in a})
                    full, edit = np.mean(before, axis=0), np.mean(after, axis=0)
                    teacher_delta = (group.teacher_edit - group.teacher_full).to_numpy()
                    for j, uid in enumerate(group.author_id):
                        rows.append(
                            {
                                "author_id": uid,
                                "query": query,
                                "family": family,
                                "teacher_full": float(group.teacher_full.iloc[j]),
                                "teacher_delta": float(teacher_delta[j]),
                                "prediction_full": float(full[j]),
                                "prediction_delta": float(edit[j] - full[j]),
                            }
                        )
                    for mode, t, p in (
                        ("level", group.teacher_full.to_numpy(), full),
                        ("response", teacher_delta, edit - full),
                    ):
                        metrics.append(
                            {
                                "query": query,
                                "family": family,
                                "mode": mode,
                                **paired_metrics(t, p),
                            }
                        )
                    if deltas:
                        parts = {
                            k: np.mean([d[k] for d in deltas], axis=0)
                            for k in deltas[0]
                        }
                        if np.max(np.abs(sum(parts.values()) - (edit - full))) > 1e-7:
                            raise ValueError("Natural response accounting failed")
                        for k, v in parts.items():
                            part_metrics.append(
                                {
                                    "query": query,
                                    "part": k,
                                    **paired_metrics(teacher_delta, v),
                                }
                            )
                        contribution_rows.extend(
                            {
                                "author_id": uid,
                                "query": query,
                                "teacher_delta": float(teacher_delta[j]),
                                "prediction_delta": float(edit[j] - full[j]),
                                **{k: float(v[j]) for k, v in parts.items()},
                            }
                            for j, uid in enumerate(group.author_id)
                        )
            frame = pd.DataFrame(rows)
            for family, g in frame.groupby("family"):
                for mode in ("full", "delta"):
                    for target in ("teacher", "prediction"):
                        p = g.pivot(
                            index="author_id",
                            columns="query",
                            values=f"{target}_{mode}",
                        )
                        a, b = p.TF_cont.to_numpy(), p.neuroticism.to_numpy()
                        links.append(
                            {
                                "family": family,
                                "target": target,
                                "mode": mode,
                                "authors": len(p),
                                "cross_query_r": corr(a, b),
                                "cross_query_covariance": float(
                                    np.cov(a, b, ddof=0)[0, 1]
                                ),
                            }
                        )
        for name, data in (
            ("private_predictions", rows),
            ("private_contributions", contribution_rows),
            ("metrics", metrics),
            ("component_response_metrics", part_metrics),
            ("functional_links", links),
        ):
            pd.DataFrame(data).to_csv(out / f"{name}.csv", index=False)
        write(
            out / "manifest.json",
            {
                "protocol": "GRAIN_M23B_CONDITIONAL_EDIT_CONNECTION",
                "plan_sha256": sha(PLAN),
                "script_sha256": sha(__file__),
                "source_manifest_sha256": sha(source / "manifest.json"),
                "edit_manifest_sha256": sha(LEVEL / "manifest.json"),
                "private_input_hashes": hashes,
                "authors": 12,
                "new_neural_calls": 0,
                "fits": 0,
                "human_labels_read": False,
            },
        )
        write(
            out / "summary.json",
            {
                "status": "COMPLETE",
                "seconds": time.monotonic() - start,
                "authors": 12,
                "families": len(FAMILIES),
                "new_neural_calls": 0,
                "source_author_overlap": 0,
                "fits": 0,
            },
        )
        print(pd.DataFrame(metrics).query("mode == 'response'").to_string(index=False))
    except BaseException as exc:
        write(out / "error_log.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
