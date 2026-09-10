#!/usr/bin/env python3
"""NEXT45: observed query-transport formation from cached signed native derivatives."""

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
from grain_research.vector_mechanisms.compiled_query_transfer import (
    MODES,
    cosine,
    fidelity,
)
from grain_research.vector_mechanisms.query_transport_formation import (
    STAGES,
    UPDATE_LABELS,
    WINDOWS,
    decompose,
    moment_accounting,
    prefix_updates,
)
from scripts.run_grain_compiled_query_transfer import (
    BASE,
    arrays,
    read,
    sha,
    verify,
    verify_hashes,
    write,
)

SOURCE = BASE / "logic_compiled_query_transfer_v1"
NATIVE = BASE / "logic_layer_accumulation_v1"
PLAN = ROOT / "docs/GRAIN_QUERY_TRANSPORT_FORMATION_PLAN_V1.md"


def prepare(out):
    """Freeze the exact observed prefix, shared readouts and previous contrast linkage."""
    verify(SOURCE)
    nf = read(NATIVE / "prediction_freeze.json")
    nm = read(NATIVE / "manifest.json")
    sf = read(SOURCE / "analysis/summary.json")
    verify_hashes(SOURCE / "analysis", sf["files"])
    if (
        nf["runtime"]["status"] != "COMPLETE"
        or tuple(nm["stages"][:17]) != UPDATE_LABELS
    ):
        raise ValueError("Completed aligned native prefix required")
    jobs = read(SOURCE / "private_jobs.json")
    if len(jobs) != 12:
        raise ValueError("Exact12 paired authors required")
    out.mkdir(exist_ok=False)
    (out / "private_inputs").mkdir()
    prepared = []
    references = [
        SOURCE / "manifest.json",
        SOURCE / "private_jobs.json",
        SOURCE / "analysis/summary.json",
        SOURCE / "analysis/private_contrasts.csv",
        NATIVE / "manifest.json",
        NATIVE / "prediction_freeze.json",
    ]
    c = pd.read_csv(SOURCE / "analysis/private_contrasts.csv")
    for j in jobs:
        p = SOURCE / "private_inputs" / (j["key"] + ".npz")
        previous = arrays(p)
        initial, updates = [], []
        for key in j["query_keys"]:
            f = NATIVE / "private_predictions" / (key + ".npz")
            if sha(f) != nf["files"][str(f.relative_to(NATIVE))]:
                raise ValueError("Native prefix changed")
            a = arrays(f)
            initial.append(a["initial"].astype(float))
            updates.append(prefix_updates(a["updates"], nm["stages"]))
            references.append(f)
        reference = {}
        for name in ("reader", "transport", "contrast", "actual"):
            rows = []
            for mode in MODES:
                g = c[(c.key == j["key"]) & (c["mode"] == mode)].sort_values("factor")
                if g.factor.tolist() != [f"whole:F{i + 1:02d}" for i in range(4)]:
                    raise ValueError("Factor linkage differs")
                rows.append(g[name].to_numpy())
            reference[name] = np.asarray(rows)
        dst = out / "private_inputs" / (j["key"] + ".npz")
        np.savez_compressed(
            dst,
            initial=np.asarray(initial),
            updates=np.asarray(updates),
            shared=previous["w"].mean(0),
            boundary=previous["d"],
            **reference,
        )
        prepared.append(dst)
        references.append(p)
    write(out / "private_jobs.json", jobs)
    prepared.append(out / "private_jobs.json")
    code = [
        PLAN,
        Path(__file__),
        ROOT / "grain_research/vector_mechanisms/query_transport_formation.py",
        ROOT / "grain_research/vector_mechanisms/compiled_query_transfer.py",
        ROOT / "scripts/run_grain_compiled_query_transfer.py",
        ROOT / "tests/test_grain_query_transport_formation.py",
    ]
    write(
        out / "manifest.json",
        {
            "protocol": "GRAIN_QUERY_TRANSPORT_FORMATION_V1",
            "type": "explanatory_decomposition_not_prediction",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "max_seconds": 60,
            "neural_calls": 0,
            "human_labels_read": False,
            "fits": 0,
            "stages": list(STAGES),
            "windows": list(WINDOWS),
            "later_updates_excluded": True,
            "inputs": {
                str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code + references))
            },
            "prepared": {str(p.relative_to(out)): sha(p) for p in prepared},
        },
    )
    print(
        {
            "status": "PREPARED",
            "pairs": 12,
            "updates_per_query": 17,
            "later_updates_copied": False,
        }
    )


def signed_metrics(g):
    """Read a signed component against the final observed transport, without dropping zeros."""
    y, p = g.target.to_numpy(), g.value.to_numpy()
    energy = np.mean(y * y)
    return {
        **fidelity(y, p),
        "mean": float(p.mean()),
        "variance": float(p.var()),
        "projected_share": float(np.mean(p * y) / energy) if energy else np.nan,
        "N_rms": float(np.sqrt(np.mean(g.N * g.N))),
        "C_rms": float(np.sqrt(np.mean(g.C * g.C))),
        "query_action_cosine": cosine(g.N, g.C),
    }


def analyze(out, name):
    """Explain stored responses; all native labels and previously observed target names stay explicit."""
    m = verify(out)
    dst = out / name
    dst.mkdir(exist_ok=False)
    start = time.monotonic()
    stage_rows, window_rows, kind_rows, checks, links = [], [], [], [], []
    for j in read(out / "private_jobs.json"):
        a = arrays(out / "private_inputs" / (j["key"] + ".npz"))
        v = decompose(a["initial"], a["updates"], a["shared"], a["boundary"])
        link_error = float(abs(v["target"] - a["transport"]).max())
        if link_error > 1e-12:
            raise ValueError("NEXT44 transport definition changed")
        checks.append(
            {
                "key": j["key"],
                "vector_relative": v["vector_relative"].tolist(),
                "vector_absolute": v["vector_absolute"].tolist(),
                "scalar_rounding_max": float(abs(v["scalar_error"]).max()),
                "scalar_bound_max": float(v["scalar_bound"].max()),
                "NEXT44_transport_replay": link_error,
                "initial_norm": float(np.linalg.norm(a["initial"])),
            }
        )
        for mi, mode in enumerate(MODES):
            for fi in range(4):
                meta = {
                    "key": j["key"],
                    "cohort": j["cohort"],
                    "index": j["index"],
                    "mode": mode,
                    "factor": f"whole:F{fi + 1:02d}",
                    "target": v["target"][mi, fi],
                }
                for si, stage in enumerate(STAGES):
                    vectors = v["parts"][:, fi, si]
                    stage_rows.append(
                        meta
                        | {
                            "stage": stage,
                            "stage_index": si,
                            "value": v["phi"][mi, fi, si],
                            "N": v["query_actions"][0, mi, fi, si],
                            "C": v["query_actions"][1, mi, fi, si],
                            "N_vector_norm": float(np.linalg.norm(vectors[0])),
                            "C_vector_norm": float(np.linalg.norm(vectors[1])),
                            "difference_vector_norm": float(
                                np.linalg.norm(vectors[0] - vectors[1])
                            ),
                            "vector_cosine": cosine(*vectors),
                            "final_difference_norm": v["difference_norm"][fi],
                        }
                    )
                for window, lo, hi in WINDOWS:
                    for kind, begin in (("group", lo), ("cumulative", 0)):
                        sl = slice(begin, hi)
                        window_rows.append(
                            meta
                            | {
                                "window": window,
                                "kind": kind,
                                "value": v["phi"][mi, fi, sl].sum(),
                                "N": v["query_actions"][0, mi, fi, sl].sum(),
                                "C": v["query_actions"][1, mi, fi, sl].sum(),
                            }
                        )
                for kind in ("initial", "attention", "mlp"):
                    keep = [
                        si
                        for si, s in enumerate(STAGES)
                        if (
                            s == "initial"
                            if kind == "initial"
                            else s.endswith("_" + kind)
                        )
                    ]
                    kind_rows.append(
                        meta
                        | {
                            "kind": kind,
                            "value": v["phi"][mi, fi, keep].sum(),
                            "N": v["query_actions"][0, mi, fi, keep].sum(),
                            "C": v["query_actions"][1, mi, fi, keep].sum(),
                        }
                    )
                links.append(
                    meta
                    | {
                        "native_block25_contrast": a["actual"][mi, fi],
                        "previous_reader": a["reader"][mi, fi],
                        "previous_compiled": a["contrast"][mi, fi],
                        "reconstructed_transport": v["phi"][mi, fi].sum(),
                        "reconstructed_compiled": v["phi"][mi, fi].sum()
                        + a["reader"][mi, fi],
                    }
                )
        if time.monotonic() - start > m["max_seconds"]:
            raise RuntimeError("Numeric cap reached")
    s, w, k, l = map(pd.DataFrame, (stage_rows, window_rows, kind_rows, links))
    if len(s) != 1728 or len(w) != 576 or len(k) != 288 or len(l) != 96:
        raise ValueError("Incomplete fixed analysis coverage")

    def aggregate(df, cols, by_factor=False):
        rows = []
        group_cols = ["cohort", "mode"] + cols + (["factor"] if by_factor else [])
        for key, g in df.groupby(group_cols, sort=True):
            r = dict(zip(group_cols, key, strict=True)) | {
                "authors": g["index"].nunique(),
                **signed_metrics(g),
            }
            if "vector_cosine" in g:
                r |= {
                    f + "_mean": float(g[f].mean())
                    for f in (
                        "N_vector_norm",
                        "C_vector_norm",
                        "difference_vector_norm",
                        "vector_cosine",
                        "final_difference_norm",
                    )
                }
            rows.append(r)
        return pd.DataFrame(rows)

    covariance, accounts, link_summary = [], [], []
    for (cohort, mode), g in s.groupby(["cohort", "mode"], sort=True):
        x = g.pivot(
            index=["key", "factor"], columns="stage_index", values="value"
        ).sort_index()
        targets = (
            g.groupby(["key", "factor"]).target.first().reindex(x.index).to_numpy()
        )
        b = moment_accounting(x.to_numpy(), targets)
        meta = {"cohort": cohort, "mode": mode, "authors": g["index"].nunique()}
        accounts.append(
            meta
            | {n: v for n, v in b.items() if np.isscalar(v)}
            | {"share_sum": float(np.sum(b["share"]))}
        )
        for row in range(18):
            for col in range(18):
                covariance.append(
                    meta
                    | {
                        "stage_a": STAGES[row],
                        "stage_b": STAGES[col],
                        "second_moment": b["second"][row, col],
                        "covariance": b["covariance"][row, col],
                        "normalized_second_moment": b["second"][row, col]
                        / b["target_energy"]
                        if b["target_energy"]
                        else np.nan,
                    }
                )
        z = l[(l.cohort == cohort) & (l["mode"] == mode)]
        for label, col in (
            ("transport_vs_NEXT44_T", "reconstructed_transport"),
            ("compiled_vs_native_block25_contrast", "reconstructed_compiled"),
        ):
            target = (
                z.target if label.startswith("transport") else z.native_block25_contrast
            )
            link_summary.append(
                meta | {"comparison": label, **fidelity(target, z[col])}
            )
    frames = {
        "private_stages": s,
        "private_windows": w,
        "private_kinds": k,
        "private_links": l,
        "stage_summary": aggregate(s, ["stage_index", "stage"]),
        "factor_stages": aggregate(s, ["stage_index", "stage"], True),
        "window_summary": aggregate(w, ["window", "kind"]),
        "factor_windows": aggregate(w, ["window", "kind"], True),
        "kind_summary": aggregate(k, ["kind"]),
        "covariance": pd.DataFrame(covariance),
        "accounting": pd.DataFrame(accounts),
        "linkage": pd.DataFrame(link_summary),
    }
    for filename, frame in frames.items():
        frame.to_csv(dst / (filename + ".csv"), index=False)
    write(dst / "private_checks.json", checks)
    seconds = time.monotonic() - start
    if seconds > m["max_seconds"]:
        raise RuntimeError("Numeric cap exceeded")
    summary = {
        "status": "COMPLETE",
        "seconds": seconds,
        "pairs": 12,
        "signed_stage_values": len(s),
        "neural_calls": 0,
        "fits": 0,
        "human_labels_read": False,
        "vector_relative_max": max(max(c["vector_relative"]) for c in checks),
        "vector_absolute_max": max(max(c["vector_absolute"]) for c in checks),
        "scalar_rounding_max": max(c["scalar_rounding_max"] for c in checks),
        "scalar_bound_max": max(c["scalar_bound_max"] for c in checks),
        "NEXT44_transport_replay_max": max(
            c["NEXT44_transport_replay"] for c in checks
        ),
        "initial_norm_max": max(c["initial_norm"] for c in checks),
        "files": {
            p.name: sha(p)
            for p in sorted(dst.glob("*.csv"))
            if not p.name.startswith("._")
        },
    }
    write(dst / "summary.json", summary)
    print({n: v for n, v in summary.items() if n != "files"})


def audit(out):
    """Check exact table replay and independent signed projections and covariance sums."""
    verify(out)
    summary = read(out / "analysis/summary.json")
    for folder in ("analysis", "analysis_replay"):
        verify_hashes(out / folder, summary["files"])
    s = pd.read_csv(out / "analysis/private_stages.csv")
    table = pd.read_csv(out / "analysis/stage_summary.csv")
    projection_error = 0.0
    for _, row in table.iterrows():
        g = s[
            (s.cohort == row.cohort)
            & (s["mode"] == row["mode"])
            & (s.stage == row.stage)
        ]
        expected = float(np.dot(g.value, g.target) / np.dot(g.target, g.target))
        projection_error = max(projection_error, abs(expected - row.projected_share))
    covariance = pd.read_csv(out / "analysis/covariance.csv")
    accounts = pd.read_csv(out / "analysis/accounting.csv")
    moment_error = 0.0
    for _, r in accounts.iterrows():
        g = covariance[
            (covariance.cohort == r.cohort) & (covariance["mode"] == r["mode"])
        ]
        moment_error = max(
            moment_error,
            abs(g.second_moment.sum() - r.reconstructed_energy),
            abs(g.covariance.sum() - r.reconstructed_variance),
        )
    if max(projection_error, moment_error) > 1e-12:
        raise ValueError("Independent accounting failed")
    report = {
        "exact_csv_replays": len(summary["files"]),
        "projection_replay_max": projection_error,
        "moment_replay_max": moment_error,
        "source_pairs": 12,
        "later_stage_rows": int(
            s.stage.isin(
                ["block25_mlp", "block26_attention", "block26_mlp", "block27_attention"]
            ).sum()
        ),
        "neural_calls": 0,
    }
    if report["later_stage_rows"]:
        raise ValueError("Later target updates entered prefix")
    write(out / "replay_audit.json", report)
    print(report)


def main():
    """Run a bounded descriptive mechanism packet, not a new label-prediction search."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "analyze", "audit"))
    p.add_argument("--name", default="logic_query_transport_formation_v1")
    p.add_argument("--analysis-name", default="analysis")
    a = p.parse_args()
    if any(Path(n).name != n or n in (".", "..") for n in (a.name, a.analysis_name)):
        raise ValueError("Simple output names required")
    with threadpool_limits(limits=2):
        if a.stage == "analyze":
            analyze(BASE / a.name, a.analysis_name)
        else:
            {"prepare": prepare, "audit": audit}[a.stage](BASE / a.name)


if __name__ == "__main__":
    main()
