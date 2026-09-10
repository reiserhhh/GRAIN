#!/usr/bin/env python3
"""NEXT44: cached source-reader exchange, fixed recipient targets, zero neural calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms.compiled_query_transfer import (
    MODES,
    QUERIES,
    borrow,
    contrast_accounting,
    contrasts,
    cosine,
    fidelity,
    validate_sources,
)
from grain_research.vector_mechanisms.stage_file_guard import install

BASE = ROOT / "results/grain_vector_mechanisms"
SOURCE = BASE / "logic_gated_update_v1"
NATIVE = BASE / "logic_layer_accumulation_v1"
PLAN = ROOT / "docs/GRAIN_COMPILED_QUERY_TRANSFER_PLAN_V1.md"


def read(path):
    """Read private metadata without logging identities or vector values."""
    return json.loads(Path(path).read_text())


def write(path, data):
    """Write strict, replayable JSON metadata."""
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    """Hash source, code or numerical artifacts before use."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_hashes(root, files):
    """Fail if any frozen file is absent or changed."""
    for name, digest in files.items():
        if sha(root / name) != digest:
            raise ValueError("Frozen file differs: " + name)


def arrays(path):
    """Load pickle-free arrays; callers verify file lineage before use."""
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def verify(out):
    """Verify this packet's code, source metadata and prepared pure numerical inputs."""
    m = read(out / "manifest.json")
    verify_hashes(ROOT, m["inputs"])
    verify_hashes(out, m["prepared"])
    return m


def prepare(out):
    """Export only d/w after exact same-material and source-exclusion checks."""
    old = read(SOURCE / "manifest.json")
    verify_hashes(ROOT, old["inputs"])
    verify_hashes(SOURCE, old["prepared"])
    cap = read(SOURCE / "capture_freeze.json")
    verify_hashes(SOURCE, cap["files"])
    old_predictions = read(SOURCE / "prediction_freeze.json")
    if any(f["runtime"]["status"] != "COMPLETE" for f in (cap, old_predictions)):
        raise ValueError("Completed NEXT43 required")
    wp = SOURCE / "private_compiled_readouts.npz"
    if sha(wp) != read(SOURCE / "replay_audit.json")["compiled_readouts_sha256"]:
        raise ValueError("Compiled reader changed")
    bank = arrays(wp)
    if set(bank) != {"w", "key", "mode"} or bank["w"].shape != (48, 1536):
        raise ValueError("Unexpected source-reader schema")
    slots = list(zip(bank["key"], bank["mode"], strict=True))
    if len(set(slots)) != 48:
        raise ValueError("Duplicated source-reader address")
    grouped = {}
    jobs = read(SOURCE / "private_jobs.json")
    if len(jobs) != 24 or len({j["key"] for j in jobs}) != 24:
        raise ValueError("Expected exactly24 fixed units")
    for j in jobs:
        group = grouped.setdefault((j["cohort"], j["index"]), {})
        if j["query"] in group:
            raise ValueError("Duplicate author-query slot")
        group[j["query"]] = j
    expected = {(c, i) for c, n in (("source4", 4), ("new8", 8)) for i in range(n)}
    if set(grouped) != expected:
        raise ValueError("Frozen author inventory differs")
    out.mkdir(exist_ok=False)
    (out / "private_inputs").mkdir()
    pair_jobs, audit, prepared = [], [], []
    source_paths = [
        SOURCE / n
        for n in (
            "manifest.json",
            "capture_freeze.json",
            "prediction_freeze.json",
            "replay_audit.json",
            "private_jobs.json",
        )
    ] + [wp]
    for (cohort, index), group in sorted(grouped.items()):
        if set(group) != set(QUERIES):
            raise ValueError("Missing query pair")
        hs, ds, ws, source_lists = [], [], [], []
        keys = [group[q]["key"] for q in QUERIES]
        for key in keys:
            hp = SOURCE / "private_inputs" / (key + ".npz")
            dp = SOURCE / "private_capture" / (key + ".npz")
            mp = SOURCE / "private_predictions" / (key + ".json")
            if sha(mp) != old_predictions["files"][str(mp.relative_to(SOURCE))]:
                raise ValueError("Source-membership metadata differs")
            members = read(mp)["source_indices"]
            validate_sources(cohort, index, members)
            source_lists.append(members)
            h, boundary = arrays(hp), arrays(dp)
            if set(boundary) != {"x", "d"} or boundary["d"].shape != (4, 1536):
                raise ValueError("Unexpected current input schema")
            hs.append(h)
            ds.append(boundary["d"].astype(float))
            ws.append([bank["w"][slots.index((key, mode))] for mode in MODES])
            source_paths += [hp, dp, mp]
        same = all(
            np.array_equal(hs[0][field], hs[1][field])
            for field in ("grain", "positions")
        )
        same_body = np.array_equal(
            hs[0]["state"][hs[0]["positions"]], hs[1]["state"][hs[1]["positions"]]
        )
        if not same or not same_body or source_lists[0] != source_lists[1]:
            raise ValueError("Same-material or source membership alignment failed")
        key = f"{cohort}_a{index:02d}"
        p = out / "private_inputs" / (key + ".npz")
        np.savez_compressed(p, w=np.asarray(ws), d=np.asarray(ds))
        prepared.append(p)
        pair_jobs.append(
            {"key": key, "cohort": cohort, "index": index, "query_keys": keys}
        )
        audit.append(
            {
                "key": key,
                "same_body": same_body,
                "same_grain_positions": same,
                "source_indices": source_lists,
            }
        )
    write(out / "private_jobs.json", pair_jobs)
    write(out / "private_alignment.json", audit)
    prepared += [out / n for n in ("private_jobs.json", "private_alignment.json")]
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/compiled_query_transfer.py",
        ROOT / "grain_research/vector_mechanisms/stage_file_guard.py",
        ROOT / "tests/test_grain_compiled_query_transfer.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_COMPILED_QUERY_TRANSFER_V1",
            "queries": list(QUERIES),
            "modes": list(MODES),
            "max_predictions": 384,
            "max_seconds": 60,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "prediction_inputs": "pure source w and current d; no x, weights, targets or raw text",
            "native_target": "NEXT42 updates[:,17] contracted with recipient NEXT43 fixed t",
            "inputs": {
                str(p.relative_to(ROOT)): sha(p)
                for p in sorted(set(code + source_paths))
            },
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print(
        {
            "status": "PREPARED",
            "aligned_author_pairs": len(pair_jobs),
            "neural_calls": 0,
        }
    )


def predict(out):
    """Write the full borrowing matrix while targets and model assets are denied."""
    m = verify(out)
    if (out / "prediction_freeze.json").exists():
        verify_hashes(out, read(out / "prediction_freeze.json")["files"])
        print("Already frozen; zero additional predictions")
        return
    if (out / "prediction_status.json").exists():
        raise ValueError("Inspect partial/error packet before retry")
    jobs = read(out / "private_jobs.json")
    mh = sha(out / "manifest.json")
    before = not (out / "analysis").exists()
    files = [
        out / n
        for n in (
            "private_predictions.npz",
            "prediction_status.json",
            "prediction_freeze.json",
        )
    ]
    probes = [
        NATIVE / "private_predictions/new8_a00_neuroticism.npz",
        SOURCE / "private_predictions/new8_a00_neuroticism.npz",
        SOURCE / "analysis/private_actions.csv",
        BASE / "logic_fixed_seed_attention_v1/private_packets/new8_a00_neuroticism.npz",
    ]
    denied = install(
        [out / "private_inputs" / (j["key"] + ".npz") for j in jobs],
        files,
        probes,
        [ROOT / "grain_research", ROOT / "scripts"],
    )
    start, done, predictions = time.monotonic(), [], []

    def status(kind):
        result = {
            "status": kind,
            "seconds": time.monotonic() - start,
            "pairs": len(done),
            "predictions": len(done) * 32,
            "neural_calls": 0,
            "fits": 0,
            "human_labels_read": False,
            "denial_probes": len(probes),
            "extra_denials": len(denied) - len(probes),
        }
        write(out / "prediction_status.json", result)
        return result

    try:
        for j in jobs:
            if time.monotonic() - start > m["max_seconds"]:
                raise RuntimeError("Bounded numeric runtime exceeded")
            a = arrays(out / "private_inputs" / (j["key"] + ".npz"))
            if set(a) != {"w", "d"}:
                raise ValueError("Non-numeric input field present")
            predictions.append(borrow(a["w"], a["d"]))
            done.append(j["key"])
        a = np.asarray(predictions)
        if a.size != m["max_predictions"] or not np.isfinite(a).all():
            raise ValueError("Incomplete fixed borrowing matrix")
        np.savez_compressed(files[0], predicted=a, key=np.asarray(done))
        final = status("COMPLETE")
        if final["extra_denials"] or final["seconds"] > m["max_seconds"]:
            raise RuntimeError("Execution boundary exceeded")
        write(
            files[2],
            {
                "manifest_sha256": mh,
                "before_evaluation": before,
                "runtime": final,
                "files": {files[0].name: sha(files[0])},
            },
        )
        print(final)
    except BaseException:
        status("ERROR")
        raise


def evaluate(out, name):
    """Open recipient targets only after all predictions are fixed; retain every cell."""
    verify(out)
    freeze = read(out / "prediction_freeze.json")
    if (
        not freeze["before_evaluation"]
        or freeze["manifest_sha256"] != sha(out / "manifest.json")
        or freeze["runtime"]["status"] != "COMPLETE"
    ):
        raise ValueError("Complete pre-evaluation freeze required")
    verify_hashes(out, freeze["files"])
    target_freeze = read(NATIVE / "prediction_freeze.json")
    previous_freeze = read(SOURCE / "prediction_freeze.json")
    jobs, saved = (
        read(out / "private_jobs.json"),
        arrays(out / "private_predictions.npz"),
    )
    if saved["key"].tolist() != [j["key"] for j in jobs]:
        raise ValueError("Prediction key order differs")
    dest = out / name
    dest.mkdir(exist_ok=False)
    actions, contrast_rows, geometry, target_hashes = [], [], [], {}
    replay_error, contrast_error = 0.0, 0.0
    for row_index, j in enumerate(jobs):
        a = arrays(out / "private_inputs" / (j["key"] + ".npz"))
        p = saved["predicted"][row_index]
        c = contrasts(a["w"], a["d"])
        truth = []
        for recipient, key in enumerate(j["query_keys"]):
            path = NATIVE / "private_predictions" / (key + ".npz")
            prior = SOURCE / "private_predictions" / (key + ".npz")
            model = SOURCE / "private_models" / (key + ".npz")
            for base, f, fr in (
                (NATIVE, path, target_freeze),
                (SOURCE, prior, previous_freeze),
            ):
                if sha(f) != fr["files"][str(f.relative_to(base))]:
                    raise ValueError("Frozen evaluation target differs")
            source_manifest = read(SOURCE / "manifest.json")
            if (
                sha(model)
                != source_manifest["prepared"][str(model.relative_to(SOURCE))]
            ):
                raise ValueError("Recipient readout changed")
            native, previous, means = (
                arrays(path)["updates"][:, 17].astype(float),
                arrays(prior),
                arrays(model),
            )
            target_hashes.update(
                {str(f.relative_to(ROOT)): sha(f) for f in (path, prior, model)}
            )
            truth.append(
                np.asarray(
                    [native @ means[mode + "_t"].astype(float) for mode in MODES]
                )
            )
            for mode_index, mode in enumerate(MODES):
                replay_error = max(
                    replay_error,
                    float(
                        abs(
                            p[recipient, recipient, mode_index]
                            - previous[mode + "_source_BN"]
                        ).max()
                    ),
                )
                for donor, query in enumerate(QUERIES):
                    for f in range(4):
                        actions.append(
                            {
                                "key": j["key"],
                                "cohort": j["cohort"],
                                "index": j["index"],
                                "recipient": QUERIES[recipient],
                                "donor": query,
                                "mode": mode,
                                "factor": f"whole:F{f + 1:02d}",
                                "actual": float(truth[-1][mode_index, f]),
                                "predicted": float(p[recipient, donor, mode_index, f]),
                            }
                        )
        actual_contrast = truth[0] - truth[1]
        contrast_error = max(
            contrast_error,
            float(abs(c["contrast"] - c["reader"] - c["transport"]).max()),
            float(abs(c["contrast"] - (p[0, 0] - p[1, 1])).max()),
        )
        for mode_index, mode in enumerate(MODES):
            for f in range(4):
                contrast_rows.append(
                    {
                        "key": j["key"],
                        "cohort": j["cohort"],
                        "index": j["index"],
                        "mode": mode,
                        "factor": f"whole:F{f + 1:02d}",
                        "actual": float(actual_contrast[mode_index, f]),
                        **{k: float(v[mode_index, f]) for k, v in c.items()},
                    }
                )
            geometry.append(
                {
                    "key": j["key"],
                    "cohort": j["cohort"],
                    "kind": "reader",
                    "coordinate": mode,
                    "cosine": cosine(*a["w"][:, mode_index]),
                    "N_norm": float(np.linalg.norm(a["w"][0, mode_index])),
                    "C_norm": float(np.linalg.norm(a["w"][1, mode_index])),
                }
            )
        for f in range(4):
            geometry.append(
                {
                    "key": j["key"],
                    "cohort": j["cohort"],
                    "kind": "direction",
                    "coordinate": f"whole:F{f + 1:02d}",
                    "cosine": cosine(*a["d"][:, f]),
                    "N_norm": float(np.linalg.norm(a["d"][0, f])),
                    "C_norm": float(np.linalg.norm(a["d"][1, f])),
                }
            )
    if replay_error > 1e-12 or contrast_error > 1e-12:
        raise ValueError("Own-reader replay or bilinear identity differs")
    actions, con, geom = (
        pd.DataFrame(actions),
        pd.DataFrame(contrast_rows),
        pd.DataFrame(geometry),
    )
    comparisons, factors, contrast_summary, contrast_factors = [], [], [], []
    keys = ["cohort", "recipient", "donor", "mode"]
    for key, group in actions.groupby(keys, sort=True):
        meta = dict(zip(keys, key, strict=True))
        centered = group[["actual", "predicted"]] - group.groupby("factor")[
            ["actual", "predicted"]
        ].transform("mean")
        comparisons.append(
            meta
            | {
                "authors": group["index"].nunique(),
                **fidelity(group.actual, group.predicted),
                "factor_centered_action_r": fidelity(
                    centered.actual, centered.predicted
                )["action_r"],
            }
        )
        for factor, g in group.groupby("factor", sort=True):
            factors.append(meta | {"factor": factor, **fidelity(g.actual, g.predicted)})

    def summarize(g):
        return {
            **fidelity(g.actual, g.contrast),
            **contrast_accounting(g.reader, g.transport),
            "reader_vs_native_r": fidelity(g.actual, g.reader)["action_r"],
            "transport_vs_native_r": fidelity(g.actual, g.transport)["action_r"],
        }

    for (cohort, mode), group in con.groupby(["cohort", "mode"], sort=True):
        meta = {"cohort": cohort, "mode": mode}
        contrast_summary.append(
            meta | {"authors": group["index"].nunique(), **summarize(group)}
        )
        for factor, g in group.groupby("factor", sort=True):
            contrast_factors.append(meta | {"factor": factor, **summarize(g)})
    geometry_summary = geom.groupby(["cohort", "kind", "coordinate"], sort=True)[
        ["cosine", "N_norm", "C_norm"]
    ].agg(["mean", "min", "max"])
    geometry_summary.columns = ["_".join(c) for c in geometry_summary.columns]
    frames = {
        "private_actions": actions,
        "private_contrasts": con,
        "private_geometry": geom,
        "comparison": pd.DataFrame(comparisons),
        "factor_actions": pd.DataFrame(factors),
        "contrast_summary": pd.DataFrame(contrast_summary),
        "contrast_factors": pd.DataFrame(contrast_factors),
        "geometry_summary": geometry_summary.reset_index(),
    }
    if (
        len(actions) != 384
        or actions.duplicated(["key", "recipient", "donor", "mode", "factor"]).any()
        or len(con) != 96
    ):
        raise ValueError("Fixed matrix coverage differs")
    for filename, frame in frames.items():
        frame.to_csv(dest / (filename + ".csv"), index=False)
    summary = {
        "pairs": len(jobs),
        "predictions": len(actions),
        "paired_contrasts": len(con),
        "own_diagonal_replay_max": replay_error,
        "contrast_identity_max": contrast_error,
        "neural_calls": 0,
        "human_labels_read": False,
        "fits": 0,
        "runtime": freeze["runtime"],
        "target_hashes": target_hashes,
        "files": {
            p.name: sha(p)
            for p in sorted(dest.glob("*.csv"))
            if not p.name.startswith("._")
        },
    }
    write(dest / "summary.json", summary)
    print({k: v for k, v in summary.items() if k not in ("target_hashes", "files")})


def main():
    """Separate preparation, target-denied numeric prediction and frozen evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict", "evaluate"))
    parser.add_argument("--name", default="logic_compiled_query_transfer_v1")
    parser.add_argument("--analysis-name", default="analysis")
    args = parser.parse_args()
    if any(
        Path(n).name != n or n in (".", "..") for n in (args.name, args.analysis_name)
    ):
        raise ValueError("Simple output names required")
    with threadpool_limits(limits=2):
        if args.stage == "evaluate":
            evaluate(BASE / args.name, args.analysis_name)
        else:
            {"prepare": prepare, "predict": predict}[args.stage](BASE / args.name)


if __name__ == "__main__":
    main()
