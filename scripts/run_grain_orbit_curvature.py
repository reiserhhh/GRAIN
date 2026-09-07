#!/usr/bin/env python3
"""S8: finite directional curvature and held-radius orbit response on frozen authors."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.direct_traits import SYSTEM
from grain_research.vector_mechanisms.trace import DecoderObserver
from scripts.analyze_grain_multiquery_sensitivity import (
    cluster_r_interval,
    corr,
    load_panel,
)
from scripts.run_grain_mean_null_path import mean_null_direction
from scripts.run_grain_multiquery_sensitivity import (
    QUERIES,
    conditional_torch,
    native_alphabet,
    query_prompt,
    token_groups,
)
from scripts.run_grain_orbit_response import apply_permutation
from scripts.run_grain_real_author_path import BASE, MODEL, read, sha, write

RADII = (-1.0, -0.5, -0.125, -0.0625, 0.0, 0.0625, 0.125, 0.5)


def radial_state(h, groups, selected, seed, radius):
    """Interpolate from the group-mean center to a fixed permutation and beyond."""
    center = h + mean_null_direction(h, groups, selected)
    displaced = apply_permutation(h, groups, selected, seed)
    return center + radius * (displaced - center)


def coefficients(values, epsilon):
    """Return the central first derivative and half-second-derivative estimate."""
    if epsilon <= 0 or any(k not in values for k in (0.0, epsilon, -epsilon)):
        raise ValueError("Require complete symmetric nodes and positive epsilon")
    linear = (values[epsilon] - values[-epsilon]) / (2 * epsilon)
    quadratic = (values[epsilon] + values[-epsilon] - 2 * values[0.0]) / (
        2 * epsilon**2
    )
    return float(linear), float(quadratic)


def radial_score(observer, ids, alphabet, groups, selected, seed, radius):
    """Evaluate the fixed suffix along a body-only mean-null radial displacement."""

    def boundary(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        leaf = radial_state(h[0], groups, selected, seed, radius)[None]
        return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

    handle = observer.model.model.layers[16].register_forward_hook(boundary)
    try:
        tokens = torch.tensor([ids], device=next(observer.model.parameters()).device)
        with torch.no_grad():
            logits = observer.model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            ).logits[0, -1]
            value = float(conditional_torch(logits, alphabet).cpu())
        observer.calls += 1
        if not np.isfinite(value):
            raise ValueError("Nonfinite radial score")
        return value
    finally:
        handle.remove()


def paired_quality(actual, prediction, authors):
    """Keep fidelity, absolute error, and zero-response reference in one record."""
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    energy = float(np.mean(actual**2))
    mse = float(np.mean((actual - prediction) ** 2))
    return {
        **cluster_r_interval(actual, prediction, authors, draws=500),
        "actual_rms": energy**0.5,
        "prediction_rms": float(np.sqrt(np.mean(prediction**2))),
        "rmse": mse**0.5,
        "mse_over_zero": mse / energy if energy > 1e-15 else None,
    }


def summarize(out):
    """Compare fixed small-radius coefficients at held radii without fitting there."""
    rows = [
        r
        for p in sorted((out / "private_packets").glob("panel_*.json"))
        for r in read(p)["rows"]
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "private_radial_scores.csv", index=False)
    coeff, predictions = [], []
    for (author, query, edit), g in frame.groupby(["author_id", "query", "edit"]):
        values = dict(zip(g.radius, g.score, strict=True))
        large, small = coefficients(values, 0.125), coefficients(values, 0.0625)
        specs = {
            "primary_0125": large,
            "small_00625": small,
            "richardson": tuple(
                (4 * a - b) / 3 for a, b in zip(small, large, strict=True)
            ),
        }
        coeff.append(
            {
                "author_id": author,
                "query": query,
                "edit": edit,
                "linear": large[0],
                "quadratic": large[1],
                "small_linear": small[0],
                "small_quadratic": small[1],
            }
        )
        for name, (linear, quadratic) in specs.items():
            for radius in (0.5, 1.0):
                even = (values[radius] + values[-radius]) / 2 - values[0.0]
                odd = (values[radius] - values[-radius]) / 2
                predictions.append(
                    {
                        "author_id": author,
                        "query": query,
                        "edit": edit,
                        "method": name,
                        "radius": radius,
                        "actual": values[radius] - values[0.0],
                        "prediction": linear * radius + quadratic * radius**2,
                        "even": even,
                        "even_prediction": quadratic * radius**2,
                        "odd": odd,
                        "odd_prediction": linear * radius,
                    }
                )
    c, p = pd.DataFrame(coeff), pd.DataFrame(predictions)
    c.to_csv(out / "private_coefficients.csv", index=False)
    p.to_csv(out / "private_radius_predictions.csv", index=False)
    metrics = []
    for (query, method, radius), g in p.groupby(["query", "method", "radius"]):
        for part in ("full", "even", "odd"):
            actual, pred = (
                ("actual", "prediction")
                if part == "full"
                else (part, part + "_prediction")
            )
            metrics.append(
                {
                    "query": query,
                    "method": method,
                    "radius": radius,
                    "part": part,
                    **paired_quality(g[actual], g[pred], g.author_id),
                }
            )
    stability = [
        {
            "query": q,
            "linear_r": corr(g.linear, g.small_linear),
            "quadratic_r": corr(g.quadratic, g.small_quadratic),
            "quadratic_delta_rms": float(
                np.sqrt(np.mean((g.quadratic - g.small_quadratic) ** 2))
            ),
            "quadratic_rms": float(np.sqrt(np.mean(g.quadratic**2))),
        }
        for q, g in c.groupby("query")
    ]
    shared = []
    for measure in ("linear", "quadratic"):
        pivot = c.pivot(index=["author_id", "edit"], columns="query", values=measure)
        pivot = pivot - pivot.groupby(level="edit").transform("mean")
        shared.append(
            {
                "measure": measure,
                **cluster_r_interval(
                    pivot.neuroticism,
                    pivot.confirmation_seeking,
                    pivot.index.get_level_values("author_id"),
                    draws=500,
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(stability).to_csv(out / "coefficient_stability.csv", index=False)
    pd.DataFrame(shared).to_csv(out / "shared_coefficients.csv", index=False)
    write(
        out / "summary.json",
        {
            "status": "COMPLETE",
            "manifest_sha256": sha(out / "manifest.json"),
            "score_sha256": sha(out / "private_radial_scores.csv"),
            "metrics": metrics,
            "stability": stability,
            "shared_N_confirmation": shared,
            "scope": "one seeded orbit direction per condition; not a full covariance trace estimate",
            "questionnaire_labels_read": False,
        },
    )
    report = pd.DataFrame(metrics)
    print(
        report[
            (report.method == "primary_0125") & (report.part.isin(["full", "even"]))
        ].to_string(index=False),
        flush=True,
    )
    print(pd.DataFrame(stability).to_string(index=False), flush=True)
    print(pd.DataFrame(shared).to_string(index=False), flush=True)


def main():
    """Acquire the predeclared directional panel and save complete-author units."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--orbit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(BASE):
        raise ValueError("Private outputs must remain ignored")
    _, records = load_panel(args.panel)
    previous = read(args.orbit / "summary.json")
    input_path = args.orbit / "private_permutations.csv"
    if previous["status"] != "COMPLETE" or sha(input_path) != previous["rows_sha256"]:
        raise ValueError("Frozen S7 endpoints required")
    fixed = pd.read_csv(input_path, dtype={"author_id": str})
    fixed = fixed[fixed["repeat"] == 0].set_index(["author_id", "query", "edit"])
    selected = [
        (m, p) for m, p, _ in records if m["view"] == 0 and m["panel_index"] < 8
    ]
    if len(selected) != 8 or not fixed.index.is_unique:
        raise ValueError("Expected eight fixed authors and unique endpoints")
    manifest = {
        "protocol": "GRAIN_S8_DIRECTIONAL_CURVATURE_V1",
        "script_sha256": sha(__file__),
        "source_sha256": sha(args.panel / "manifest.json"),
        "orbit_sha256": sha(args.orbit / "summary.json"),
        "model_sha256": sha(MODEL / "model.safetensors"),
        "device": args.device,
        "queries": list(QUERIES),
        "authors": 8,
        "permutation_repeat": 0,
        "radii": list(RADII),
        "max_forwards": 384,
        "max_seconds": 1800,
        "questionnaire_labels_read": False,
    }
    if out.exists():
        if read(out / "manifest.json") != manifest:
            raise ValueError("Frozen experiment changed")
    else:
        out.mkdir(parents=True)
        (out / "private_packets").mkdir()
        write(out / "manifest.json", manifest)
    state_file = out / "run_status.json"
    prior = read(state_file) if state_file.exists() else {"calls": 0, "seconds": 0}
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    torch.set_num_threads(2)
    observer = DecoderObserver.load(
        str(MODEL), device=args.device, dtype="float32", max_tokens=3072
    )
    observer.model.requires_grad_(False)
    start = time.monotonic()

    def status(value):
        record = {
            "status": value,
            "calls": prior["calls"] + observer.calls,
            "seconds": prior["seconds"] + time.monotonic() - start,
            "backwards": 0,
            "complete_authors": len(
                list((out / "private_packets").glob("panel_*.json"))
            ),
        }
        write(state_file, record)
        print(record, flush=True)

    try:
        for material, _ in selected:
            path = out / "private_packets" / f"panel_{material['panel_index']:02d}.json"
            if path.exists():
                if read(path)["author_id"] != material["author_id"]:
                    raise ValueError("Resumed author mismatch")
                continue
            if prior["seconds"] + time.monotonic() - start > 1800:
                status("PAUSED_TIME_AT_AUTHOR_BOUNDARY")
                return
            rows = []
            for query in QUERIES:
                alphabet = native_alphabet(query, observer.tokenizer)
                user, spans = query_prompt(material["texts"], query)
                ids = observer.prompt_ids(SYSTEM, user)
                groups = token_groups(observer, user, spans, ids)
                for edit in ("e1", "e2"):
                    endpoint = fixed.loc[(material["author_id"], query, edit)]
                    seed = int(endpoint.seed)
                    values = {1.0: float(endpoint.shuffled)}
                    for radius in RADII:
                        values[radius] = radial_score(
                            observer,
                            ids,
                            alphabet,
                            groups,
                            material[edit],
                            seed,
                            radius,
                        )
                    if abs(values[0.0] - endpoint.collapsed) > 1e-5:
                        raise ValueError("Center changed since S6/S7")
                    rows.extend(
                        {
                            "author_id": material["author_id"],
                            "query": query,
                            "edit": edit,
                            "seed": seed,
                            "radius": radius,
                            "score": value,
                        }
                        for radius, value in values.items()
                    )
            write(path, {"author_id": material["author_id"], "rows": rows})
            status("RUNNING")
            if prior["calls"] + observer.calls > 384:
                raise RuntimeError("Forward budget exceeded")
        summarize(out)
        status("COMPLETE")
    except BaseException as exc:
        status("FAILED")
        write(out / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
