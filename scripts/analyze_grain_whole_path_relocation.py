#!/usr/bin/env python3
"""NEXT55 post-result zero-native crossing of complete suffix readers/tangents."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.response_repair import allocate, summarize
from grain_research.vector_mechanisms.stage_file_guard import install
from scripts.run_grain_whole_path_assembly import (
    BASE,
    QUERIES,
    apply_map,
    arrays,
    body_directions,
    fidelity,
    forbidden,
    frozen,
    paired_queries,
    read,
    ref,
    sha,
    verify,
    verify_hashes,
    write,
)

CELLS = ("II", "IJ", "JI", "JJ")
NOTE = ROOT / "docs/GRAIN_WHOLE_PATH_RELOCATION_NOTE_V1.md"


def predict(out, name):
    """Cross only saved candidate gradients and fixed tangents, without native targets."""
    m, pf = verify(out), frozen(out, "predictions")
    jobs = read(out / "private_jobs.json")
    dst = out / name
    dst.mkdir(exist_ok=False)
    reads = [out / n for n in m["prepared"]] + [
        out / "predictions" / n for n in pf["files"]
    ]
    paths = [dst / (k + ".npz") for j in jobs for k in j["query_keys"]]
    meta = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_sha256": sha(Path(__file__)),
        "note_sha256": sha(NOTE),
    }
    denied = install(
        reads,
        paths + [dst / "freeze.json"],
        forbidden(),
        [ROOT / "scripts", ROOT / "grain_research"],
    )
    started, error = time.monotonic(), 0.0
    for j in jobs:
        model = arrays(out / "private_models" / f"joint_{ref(j)}_pointwise.npz")
        for key in j["query_keys"]:
            data = arrays(out / "private_inputs" / (key + ".npz"))
            x, d = data["state"].astype(float), body_directions(data)
            mx, md = apply_map(x, d, model)[:2]
            p = arrays(out / "predictions" / (key + ".npz"))
            g = p["gradient"][[0, 2]].astype(float)
            directions = np.stack([d, md]).astype(np.float32).astype(float)
            cells = np.einsum("sth,dfth->sdf", g, directions)
            error = max(
                error,
                float(abs(cells[0, 0] - p["tau"][0]).max()),
                float(abs(cells[1, 1] - p["tau"][2]).max()),
            )
            dg = g[1] - g[0]
            delta = mx - x
            geometry = np.asarray(
                [
                    np.linalg.norm(g[1]) / np.linalg.norm(g[0]),
                    np.sum(g[1] * g[0]) / (np.linalg.norm(g[1]) * np.linalg.norm(g[0])),
                    np.linalg.norm(dg) / np.linalg.norm(g[0]),
                    np.linalg.norm(md - d) / np.linalg.norm(d),
                    np.linalg.norm(md) / np.linalg.norm(d),
                    np.mean(delta.mean(0) ** 2) / np.mean(delta**2),
                ]
            )
            np.savez(dst / (key + ".npz"), cells=cells, geometry=geometry)
    elapsed = time.monotonic() - started
    if elapsed > 60 or error > 1e-11:
        raise ValueError("Cached crossing bound/diagonal parity failed")
    r = {
        "status": "COMPLETE",
        **meta,
        "seconds": elapsed,
        "diagonal_error": error,
        "off_diagonal_vectors": 48,
        "native_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "denial_probes": 4,
        "extra_denials": len(denied) - 4,
        "files": {p.name: sha(p) for p in paths},
    }
    write(dst / "freeze.json", r)
    print({k: v for k, v in r.items() if k != "files"}, flush=True)


def evaluate(out, name, prediction):
    """Keep exact signed response/error accounts and both cohort outcomes."""
    verify(out)
    pf = frozen(out, prediction)
    if pf["code_sha256"] != sha(Path(__file__)) or pf["note_sha256"] != sha(NOTE):
        raise ValueError("Post-result crossing definition changed")
    summary = read(out / "analysis/summary.json")
    verify_hashes(out / "analysis", summary["files"])
    native = (
        pd.read_csv(out / "analysis/private_actions.csv")
        .query("candidate == 'identity'")
        .set_index(["author_key", "query_name", "factor"])
    )
    dst = out / name
    dst.mkdir(exist_ok=False)
    rows, geometry = [], []
    for j in read(out / "private_jobs.json"):
        for q, key in zip(QUERIES, j["query_keys"], strict=True):
            p = arrays(out / prediction / (key + ".npz"))
            geometry.append(
                {"author_key": j["key"], "cohort": j["cohort"], "query_name": q}
                | dict(
                    zip(
                        (
                            "gradient_norm_ratio",
                            "gradient_cosine",
                            "gradient_change_relative",
                            "tangent_change_relative",
                            "tangent_norm_ratio",
                            "constant_token_correction_energy_share",
                        ),
                        p["geometry"],
                        strict=True,
                    )
                )
            )
            for ci, cell in enumerate(CELLS):
                for f in range(4):
                    factor = f"whole:F{f + 1:02d}"
                    rows.append(
                        {
                            "author_key": j["key"],
                            "cohort": j["cohort"],
                            "query_name": q,
                            "cell": cell,
                            "factor": factor,
                            "actual": float(
                                native.loc[(j["key"], q, factor), "actual"]
                            ),
                            "predicted": p["cells"][ci // 2, ci % 2, f],
                        }
                    )
    df = pd.DataFrame(rows)
    cf = paired_queries(
        df, ["author_key", "cohort", "cell", "factor"], ["actual", "predicted"]
    )
    files = []

    def save(tab, n):
        tab.to_csv(dst / n, index=False)
        files.append(n)

    save(df, "private_cells.csv")
    save(cf, "private_contrasts.csv")
    geo = pd.DataFrame(geometry)
    save(geo, "private_geometry.csv")
    save(
        geo.drop(columns="author_key")
        .groupby(["cohort", "query_name"])
        .mean()
        .reset_index(),
        "geometry.csv",
    )
    for tab, groups, stem in (
        (df, ["cohort", "query_name"], "query"),
        (cf, ["cohort"], "contrast"),
    ):
        unit = ["author_key", *groups, "factor"]
        effects = []
        for key, g in tab.groupby(unit, sort=True):
            p = (
                g.set_index("cell")
                .loc[list(CELLS), "predicted"]
                .to_numpy()
                .reshape(2, 2)
            )
            a = allocate(p, np.asarray(g.actual.iloc[0]))
            effects.append(
                dict(zip(unit, key, strict=True)) | {k: float(v) for k, v in a.items()}
            )
        ef = pd.DataFrame(effects)
        save(ef, "private_" + stem + "_effects.csv")
        fields = list(allocate(np.zeros((2, 2)), np.asarray(0.0)))
        for extra, label in (([], stem), (["factor"], stem + "_factor")):
            grouping = groups + extra
            save(
                pd.DataFrame(
                    [
                        dict(zip(grouping + ["cell"], k, strict=True))
                        | fidelity(g.actual.to_numpy(), g.predicted.to_numpy())
                        | {"authors": g.author_key.nunique()}
                        for k, g in tab.groupby(grouping + ["cell"], sort=True)
                    ]
                ),
                label + "_cells.csv",
            )
            result = []
            for k, g in ef.groupby(grouping, sort=True):
                k = k if isinstance(k, tuple) else (k,)
                result.append(
                    dict(zip(grouping, k, strict=True))
                    | summarize({f: g[f].to_numpy() for f in fields})
                    | {"authors": g.author_key.nunique()}
                )
            save(pd.DataFrame(result), label + "_effects.csv")
    r = {
        "status": "COMPLETE",
        "rows": len(df),
        "contrasts": len(cf),
        "native_calls": 0,
        "fits": 0,
        "files": {n: sha(dst / n) for n in files},
    }
    write(dst / "summary.json", r)
    print({k: v for k, v in r.items() if k != "files"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("predict", "evaluate"))
    p.add_argument("--name")
    p.add_argument("--predictions", default="relocation")
    a = p.parse_args()
    out = BASE / "logic_whole_path_assembly_v1"
    with threadpool_limits(limits=2):
        if a.stage == "predict":
            predict(out, a.name or "relocation")
        else:
            evaluate(out, a.name or "relocation_analysis", a.predictions)
