#!/usr/bin/env python3
"""Test a genuine frozen GRAIN observer against sealed same-text vector action."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction import math_observer_parity as M
from grain_research.vector_mechanisms.dynamic_channels import path_descriptors
from suica_core.suica import TOKEN_RE
from suica_core.v7_observations import ObservationSpec, canonicalize_comments
from suica_core.v7_operator_controls import score_operator_from_comments
from suica_core.v7_psychometric import read_factor_bundle

SPEC = ROOT / "configs/grain_observer_parity_hm96.json"
FC = [f"SU7-FC-{i:04d}@v1_z" for i in range(1, 5)]
CODE = (
    "configs/grain_observer_parity_hm96.json",
    "docs/GRAIN_OBSERVER_PARITY_PLAN_HM96.md",
    "grain_research/construction/math_observer_parity.py",
    "scripts/run_grain_observer_parity_hm96.py",
    "scripts/verify_grain_observer_parity_hm96.py",
    "tests/test_grain_math_observer_parity.py",
    "suica_core/suica.py",
    "suica_core/v7_observations.py",
    "suica_core/v7_psychometric.py",
    "suica_core/v7_operator_controls.py",
    "grain_research/vector_mechanisms/dynamic_channels.py",
)


def load(path):
    """Read an explicitly whitelisted JSON artifact."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hash the exact input bytes rather than trusting a status headline."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Preserve prior/partial outputs and forbid nonfinite JSON values."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def table(path, rows):
    """Write every declared record to an exclusive new CSV."""
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seal(base, folder):
    """Seal substantive files, excluding macOS resource-fork sidecars."""
    return {
        str(p.relative_to(base)): sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.name.startswith("._")
    }


def verify(base, hashes):
    """Fail on changed source or code, without silently refreezing."""
    for name, digest in hashes.items():
        if sha(base / name) != digest:
            raise ValueError("Frozen file changed: " + name)


def settings():
    """Return only the declared fixed experiment configuration."""
    spec = load(SPEC)
    return spec, ROOT / spec["source"], ROOT / spec["output"]


def check(out):
    """Authenticate the new plan and explicitly admitted historical inputs."""
    manifest = load(out / "00/manifest.json")
    verify(ROOT, manifest["code"] | manifest["source"])
    return manifest


def prepare():
    """Freeze the pre-outcome comparison, including all input-only exclusions."""
    spec, source, out = settings()
    if out.exists():
        raise FileExistsError("Preserve earlier HM96 output")
    if (
        load(source / "verification.json")["status"]
        != "VERIFIED_FINITE_HM93_NOT_GLOBAL"
    ):
        raise ValueError("Verified finite HM93 kernel required")
    verify(source, load(source / "00/manifest.json")["inputs"])
    verify(
        source,
        {
            n: h
            for n, h in load(source / "02/output_seal.json").items()
            if not Path(n).name.startswith("._")
        },
    )
    files = [
        source / p
        for p in (
            "00/manifest.json",
            "00/rows.json",
            "00/application_paths.json",
            "00/evaluation_truth.json",
            "02/kernels.npz",
            "02/output_seal.json",
            "verification.json",
        )
    ]
    files.extend(ROOT / spec[p] for p in ("bundle", "runtime"))
    rows = load(source / "00/rows.json")
    if len(rows) != 288:
        raise ValueError("All public HM93 bodies required")
    coverage = [
        {
            "unit": r["unit"],
            "kind": r["kind"],
            "lexicon": r["lexicon"],
            "frame": r["frame"],
            "family": r["family"],
            "lexical_tokens": len(TOKEN_RE.findall(r["body"])),
            "quartet_supported": bool(M.four_slices(r["body"])),
            "event_supported": bool(M.event_slices(r["body"])),
        }
        for r in rows
    ]
    (out / "00").mkdir(parents=True)
    table(out / "00/coverage.csv", coverage)
    write(
        out / "00/manifest.json",
        {
            "spec": spec,
            "code": {n: sha(ROOT / n) for n in CODE},
            "source": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "new_neural_calls": 0,
            "new_predictor_fits": 0,
            "historical_bundle_local_only": True,
            "scope_complete": False,
        },
    )
    print(
        "PREPARED HM96",
        len(rows),
        "bodies",
        sum(r["quartet_supported"] for r in coverage),
        "quartet-supported",
        flush=True,
    )


def score():
    """Run the inherited official scoring path on all public observation units."""
    spec, source, out = settings()
    check(out)
    started = time.monotonic()
    dest = out / "01"
    dest.mkdir()
    rows = load(source / "00/rows.json")
    records, meta = [], []
    for row in rows:
        views = {
            "whole": [row["body"]],
            "event": M.event_slices(row["body"]),
            "quartet": M.four_slices(row["body"]),
        }
        for view, texts in views.items():
            for index, text in enumerate(texts):
                uid = f"u{len(records):05d}"
                records.append({"author": uid, "body": text})
                meta.append(
                    {"user_id": uid, "unit": row["unit"], "view": view, "slice": index}
                )
    frame = canonicalize_comments(
        pd.DataFrame(records), order_col=None, condition_col=None, min_tokens=12
    )
    frame["split"] = "confirmation"
    runtime = joblib.load(ROOT / spec["runtime"])
    bundle = read_factor_bundle(ROOT / spec["bundle"])
    units, features, scores = score_operator_from_comments(
        frame,
        spec=ObservationSpec(**bundle.operator),
        representation=runtime,
        bundle=bundle,
    )
    if (
        len(units) != len(records)
        or not scores.score_status.eq("SCORED").all()
        or not scores.n_units.eq(1).all()
    ):
        raise ValueError(
            "Every declared supported text must yield exactly one official unit"
        )
    ids = [r["user_id"] for r in meta]
    units = units.set_index("user_id").loc[ids].reset_index()
    scores = scores.set_index("user_id").loc[ids].reset_index()
    features = features.set_index("user_id").loc[ids].reset_index()
    sparse = runtime.vectorizer.transform(units["text"])
    vectors = runtime.svd.transform(sparse)
    np.savez_compressed(
        dest / "coordinates.npz", vectors=vectors, factors=scores[FC].to_numpy(float)
    )
    save_npz(dest / "tfidf.npz", sparse)
    write(
        dest / "units.json",
        [{**r, "text": str(text)} for r, text in zip(meta, units["text"], strict=True)],
    )
    scores.to_csv(dest / "official_scores.csv", index=False)
    features.to_csv(dest / "official_features.csv", index=False)
    write(
        dest / "summary.json",
        {
            "public_bodies": len(rows),
            "scored_units": len(units),
            "bundle_id": bundle.bundle_id,
            "runtime_spec": runtime.spec.to_dict(),
            "fit_calls": 0,
            "neural_calls": 0,
            "seconds": time.monotonic() - started,
        },
    )
    write(out / "01_seal.json", seal(out, dest))
    print("SCORED HM96", len(units), "official units", flush=True)


def analyze():
    """Evaluate exchange and author-cycle retention without fitting an inverse."""
    spec, source, out = settings()
    check(out)
    verify(out, load(out / "01_seal.json"))
    started = time.monotonic()
    dest = out / "02"
    dest.mkdir()
    rows = load(source / "00/rows.json")
    units = load(out / "01/units.json")
    factors = np.load(out / "01/coordinates.npz")["factors"]
    index = {(r["unit"], r["view"], r["slice"]): i for i, r in enumerate(units)}
    groups = list(itertools.product(range(2), spec["frames"], spec["targets"]))
    lookup = {
        (r["lexicon"], r["frame"], r["family"], r["previous"], r["current"]): r["unit"]
        for r in rows
        if r["kind"] == "pair"
    }
    arms = (
        "whole",
        "event_mean",
        "event_step",
        "event_area",
        "quartet_mean",
        "quartet_dispersion",
        "quartet_order",
    )
    kernels, addresses, support = {a: [] for a in arms}, {a: [] for a in arms}, []
    event_exchange = []
    for g, group in enumerate(groups):
        complete = all(
            (lookup[*group, i, j], "quartet", 0) in index
            for i, j in itertools.product(range(3), repeat=2)
        )
        arrays = {
            "whole": np.zeros((3, 3, 4)),
            "event_mean": np.zeros((3, 3, 4)),
            "event_step": np.zeros((3, 3, 4)),
            "event_area": np.zeros((3, 3, 6)),
        }
        if complete:
            arrays.update(
                {
                    "quartet_mean": np.zeros((3, 3, 4)),
                    "quartet_dispersion": np.zeros((3, 3, 4)),
                    "quartet_order": np.zeros((3, 3, 14)),
                }
            )
        events = np.zeros((3, 3, 3, 4))
        for i, j in itertools.product(range(3), repeat=2):
            unit = lookup[*group, i, j]
            arrays["whole"][i, j] = factors[index[unit, "whole", 0]]
            events[i, j] = np.array(
                [factors[index[unit, "event", k]] for k in range(3)]
            )
            mean, step, area = M.event_observations(events[i, j])
            (
                arrays["event_mean"][i, j],
                arrays["event_step"][i, j],
                arrays["event_area"][i, j],
            ) = mean, step, area
            if complete:
                x = np.array([factors[index[unit, "quartet", k]] for k in range(4)])
                sd, ordered = path_descriptors(x, groups=((0, 1, 2, 3),))
                (
                    arrays["quartet_mean"][i, j],
                    arrays["quartet_dispersion"][i, j],
                    arrays["quartet_order"][i, j],
                ) = x.mean(0), sd, ordered
        for a in arms:
            support.append(
                {
                    "group": g,
                    "lexicon": group[0],
                    "frame": group[1],
                    "family": group[2],
                    "arm": a,
                    "supported": a in arrays,
                    "reason": "COMPLETE"
                    if a in arrays
                    else "INCOMPLETE_QUARTET_PAIR_CELLS",
                }
            )
            if a in arrays:
                kernels[a].append(arrays[a])
                addresses[a].append(g)
        for i, j in itertools.combinations(range(3), 2):
            delta = events[i, j] - events[j, i][[1, 0, 2]]
            event_exchange.append(
                {
                    "group": g,
                    "previous": i,
                    "current": j,
                    "point_exchange_max": float(abs(delta).max()),
                }
            )
    arrays = {a: np.array(v) for a, v in kernels.items()}
    with np.load(source / "02/kernels.npz") as z:
        target = z["kernel"].reshape(24, 3, 3, 3, 6)
    arrays["target"] = target.reshape(24, 3, 3, 18)
    addresses["target"] = list(range(24))
    np.savez_compressed(dest / "kernels.npz", **arrays)
    write(dest / "addresses.json", addresses)
    table(dest / "support.csv", support)
    table(dest / "event_exchange.csv", event_exchange)
    parity_rows = []
    for arm, kset in arrays.items():
        for g, kernel in zip(addresses[arm], kset, strict=True):
            even, odd = M.parity(kernel)
            contrast = M.cycle_action(kernel)
            for k in range(kernel.shape[-1]):
                parity_rows.append(
                    {
                        "group": g,
                        "arm": arm,
                        "channel": k,
                        "odd_rms": float(np.sqrt(np.mean(odd[..., k] ** 2))),
                        "odd_max": float(abs(odd[..., k]).max()),
                        "cycle_contrast": float(contrast[k]),
                        "identity_max": float(
                            abs(kernel[..., k] - even[..., k] - odd[..., k]).max()
                        ),
                    }
                )
    table(dest / "parity.csv", parity_rows)
    from scipy.sparse import load_npz

    tfidf = load_npz(out / "01/tfidf.npz")
    vectors = np.load(out / "01/coordinates.npz")["vectors"]
    swap = []
    for g, group in enumerate(groups):
        for i, j in itertools.combinations(range(3), 2):
            left = index[lookup[*group, i, j], "whole", 0]
            right = index[lookup[*group, j, i], "whole", 0]
            diff = tfidf[left] - tfidf[right]
            swap.append(
                {
                    "group": g,
                    "previous": i,
                    "current": j,
                    "tfidf_diff_nnz": int(diff.nnz),
                    "tfidf_max": float(abs(diff.data).max()) if diff.nnz else 0.0,
                    "svd_max": float(abs(vectors[left] - vectors[right]).max()),
                    "factor_max": float(abs(factors[left] - factors[right]).max()),
                    "target_max": float(
                        abs(arrays["target"][g, i, j] - arrays["target"][g, j, i]).max()
                    ),
                }
            )
    table(dest / "whole_exchange.csv", swap)
    paths = load(source / "00/application_paths.json")
    application = []
    for arm, kset in arrays.items():
        for g, kernel in zip(addresses[arm], kset, strict=True):
            for author in paths:
                values = M.integrate(author["states"], kernel)
                application.extend(
                    {
                        "author": author["author"],
                        "group": g,
                        "arm": arm,
                        "channel": k,
                        "value": float(v),
                    }
                    for k, v in enumerate(values)
                )
    table(dest / "application.csv", application)
    truth = {r["author"]: r for r in load(source / "00/evaluation_truth.json")}
    paired = {}
    for r in application:
        t = truth[r["author"]]
        paired.setdefault(
            (r["group"], r["arm"], r["channel"], t["seed"], t["replicate"]), {}
        )[t["direction"]] = r["value"]
    differences = []
    for key, values in paired.items():
        differences.append(
            {
                "group": key[0],
                "arm": key[1],
                "channel": key[2],
                "seed": key[3],
                "replicate": key[4],
                "plus_minus": values[1] - values[-1],
                "midpoint_error": values[0] - (values[1] + values[-1]) / 2,
            }
        )
    table(dest / "contrasts.csv", differences)
    summary = {
        "status": "EXECUTED_FINITE_HM96_PENDING_VERIFIER",
        "public_bodies": len(rows),
        "actual_official_units": len(units),
        "tfidf_exact_equal_pairs": sum(r["tfidf_diff_nnz"] == 0 for r in swap),
        "exchange_pairs": len(swap),
        "tfidf_exchange_max": max(r["tfidf_max"] for r in swap),
        "event_point_exchange_max": max(
            r["point_exchange_max"] for r in event_exchange
        ),
        "new_neural_calls": 0,
        "new_predictor_fits": 0,
        "numerical_ledger_reuse": len(paths),
        "scope_complete": False,
        "seconds": time.monotonic() - started,
        "arms": {},
    }
    for arm in arrays:
        selected = [r for r in parity_rows if r["arm"] == arm]
        counts = [
            max(abs(r["cycle_contrast"]) for r in selected if r["group"] == g)
            for g in addresses[arm]
        ]
        summary["arms"][arm] = {
            "supported_groups": len(addresses[arm]),
            "odd_max": max(r["odd_max"] for r in selected),
            "cycle_abs_min_max": [min(counts), max(counts)],
            "nonzero_cycle_groups_at_frozen_tolerance": sum(
                c > spec["tolerance"] for c in counts
            ),
        }
    write(dest / "summary.json", summary)
    write(out / "02_seal.json", seal(out, dest))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "score", "analyze"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"prepare": prepare, "score": score, "analyze": analyze}[args.stage]()
