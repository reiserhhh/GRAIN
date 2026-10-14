#!/usr/bin/env python3
"""Run native MEM/text references without new neural training or private uploads."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.predecessors import MeaningExtraction
from scripts.prepare_grain_predecessor_reference import DEFAULT, digest, write

AUTHOR_PCA = "https://www.ryanboyd.io/software/meh/Supplemental/PCA-Template-for-MEM.R"


def compare_coordinates(left, right):
    """Separate rotated-coordinate agreement from agreement of the whole subspace."""
    left, right = np.asarray(left), np.asarray(right)
    if (
        left.shape != right.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Finite equal-sized score matrices required")
    corr = np.corrcoef(left.T, right.T)[: left.shape[1], left.shape[1] :]
    rows, columns = linear_sum_assignment(-np.abs(corr))
    rotation, _ = orthogonal_procrustes(left, right)
    return {
        "minimum_matched_absolute_r": float(np.min(np.abs(corr[rows, columns]))),
        "median_matched_absolute_r": float(np.median(np.abs(corr[rows, columns]))),
        "relative_procrustes_error": float(
            np.linalg.norm(left @ rotation - right) / np.linalg.norm(right)
        ),
        "scope": "coordinate/subspace parity, not personality validity",
    }


def run(args):
    """Prepare source-fold vocabulary, execute native references and bind provenance."""
    out = args.output
    if out.exists():
        raise ValueError("Use a new output directory")
    out.mkdir(parents=True)
    manifest = json.loads((args.material / "material_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(args.material / name) != sha:
            raise ValueError("Changed source material")
    data = json.loads((args.material / "private_material.json").read_text())
    with np.load(args.material / "private_arrays.npz", allow_pickle=False) as a:
        folds = a["folds"]
    documents = np.asarray(data["documents"])
    train, held = folds != 0, folds == 0
    mem = MeaningExtraction().fit(documents[train])
    vocabulary = mem.vectorizer_.get_feature_names_out()
    for name, mask in (("train", train), ("held", held)):
        pd.DataFrame(
            mem.vectorizer_.transform(documents[mask]).toarray(), columns=vocabulary
        ).to_csv(out / f"mem_{name}.csv", index=False)
        pd.DataFrame(mem.transform(documents[mask])).to_csv(
            out / f"python_mem_{name}_scores.csv", index=False
        )
    (out / "mem_rank.txt").write_text(str(mem.pca_.n_components_) + "\n")
    response = requests.get(AUTHOR_PCA, timeout=45)
    response.raise_for_status()
    source = out / "PCA-Template-for-MEM.R"
    source.write_bytes(response.content)
    rng = np.random.default_rng(731)
    x, y = rng.normal(size=(20, 16)), rng.normal(size=(20, 16))
    pd.DataFrame(x).to_csv(out / "similarity_x.csv", index=False)
    pd.DataFrame(y).to_csv(out / "similarity_y.csv", index=False)
    write(
        out / "contract.json",
        {
            "author_pca_url": AUTHOR_PCA,
            "author_pca_sha256": digest(source),
            "upstream_modifications": "none; only four parsed function definitions evaluated",
            "input": "local training-fold binary vocabulary; not original MEH preprocessing",
            "train_authors": int(train.sum()),
            "held_authors": int(held.sum()),
            "personality_labels_used": False,
            "public_text_example_grid": "3x2 nested CV; PCA5; penalties .1,1",
            "source_manifest_sha256": digest(args.material / "material_manifest.json"),
            "runner_sha256": digest(Path(__file__)),
            "R_sha256": digest(ROOT / "scripts/run_grain_known_native.R"),
        },
    )
    with (out / "native.log").open("w") as log:
        result = subprocess.run(
            [
                "/usr/local/bin/Rscript",
                str(ROOT / "scripts/run_grain_known_native.R"),
                str(out.resolve()),
                str(args.rlib.resolve()),
                str(source.resolve()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    if result.returncode:
        write(
            out / "summary.json",
            {
                "status": "FAILED_NATIVE",
                "returncode": result.returncode,
                "log": "native.log",
            },
        )
        raise RuntimeError("Native reference failed; inspect native.log")
    summary = {"status": "COMPLETE", "mem": {}}
    for name in ("train", "held"):
        summary["mem"][name] = compare_coordinates(
            pd.read_csv(out / f"python_mem_{name}_scores.csv"),
            pd.read_csv(out / f"native_mem_{name}_scores.csv"),
        )
    raw = (
        pd.read_csv(out / "native_text_raw_cosine.csv")
        .select_dtypes("number")
        .to_numpy()
        .reshape(-1)
    )
    expected = np.einsum("ij,ij->i", x, y) / (
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    )
    if raw.shape != expected.shape:
        raise ValueError("Native text cosine output layout changed")
    summary["raw_cosine_max_error"] = float(np.max(np.abs(raw - expected)))
    summary["example_results"] = pd.read_csv(
        out / "native_text_example_results.csv"
    ).to_dict("records")
    summary["scope"] = (
        "native author routines and numerical parity, not exact paper datasets/results"
    )
    write(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--material", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rlib", type=Path, required=True)
    with threadpool_limits(limits=2):
        run(p.parse_args())
