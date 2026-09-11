#!/usr/bin/env python3
"""One post-outcome, zero-fit account of NEXT85 amplitude and source coverage."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms import coordinate_support as fn
from scripts import run_grain_evidence_coordinate_reader as run


def compute():
    """Use only frozen fitted functions and already exposed evaluation packets."""
    baseline = run.old.arrays(run.BASELINE)
    jobs = [
        j for j in run.old.read(run.OUT / "private_jobs.json") if j["cohort"] == "new8"
    ]
    rows, points = [], []
    for j in jobs:
        rows.append(
            run.old.arrays(run.OUT / "analysis" / ("private_" + j["key"] + ".npz"))
        )
        data = [
            run.old.arrays(run.ep.STATE / "private_inputs" / (k + ".npz"))
            for k in j["query_keys"]
        ]
        points.append(
            run.fn.base.flatten(
                run.fn.base.features(data), baseline["scale"], "grain_field"
            )
        )
    points = np.stack(points)
    support, scale, components = [], [], []
    for vi, variant in enumerate(run.fn.VARIANTS):
        m = run.old.arrays(run.OUT / "models" / (variant + ".npz"))
        for cohort, x in (("source_original", m["x"][:8:2]), ("new8", points)):
            a = fn.account(m, x)
            support.append(
                {
                    "variant": variant,
                    "cohort": cohort,
                    "authors": len(x),
                    **{
                        k + "_" + label: float(op(a[k]))
                        for k in (
                            "coverage_remainder",
                            "nearest_distance",
                            "max_original_kernel",
                        )
                        for label, op in (
                            ("mean", np.mean),
                            ("min", np.min),
                            ("max", np.max),
                        )
                    },
                }
            )
        a = fn.account(m, points)
        predicted = np.stack([r["coordinate_values"][vi] for r in rows])
        np.testing.assert_allclose(a["parts"].sum(axis=1), predicted, atol=1e-10)
        frames = (
            np.stack([r["frame"] for r in rows])
            if vi
            else np.broadcast_to(m["fixed_frame"], (8, *m["fixed_frame"].shape))
        )
        native = np.stack([r["native_r"][0] - r["native_r"][1] for r in rows])
        actual = np.einsum("nkh,nh->nk", frames, native - baseline["d0"])
        for f in range(5):
            sr, yr, pr = [
                float(np.sqrt(np.mean(x**2)))
                for x in (m["target"][:8:2, f], actual[:, f], predicted[:, f])
            ]
            meta = {
                "variant": variant,
                "authors": 8,
                "coordinate": "scalar" if f == 0 else f"total_F{f:02d}",
            }
            scale.append(
                meta
                | {
                    "source_authors": 4,
                    "source_original_rms": sr,
                    "recipient_target_rms": yr,
                    "predicted_rms": pr,
                    "recipient_over_source_rms": yr / sr if sr else np.nan,
                    "predicted_over_actual_rms": pr / yr if yr else np.nan,
                }
            )
            energy = float(np.sum(actual[:, f] ** 2))
            for gi, group in enumerate(run.fn.base.GROUPS):
                part = a["parts"][:, gi, f]
                components.append(
                    meta
                    | {
                        "group": group,
                        "component_rms": float(np.sqrt(np.mean(part**2))),
                        "signed_target_projection": float(part @ actual[:, f]) / energy
                        if energy
                        else 0.0,
                    }
                )
    return [pd.DataFrame(x) for x in (support, scale, components)]


def main():
    """Recompute the same accounting twice; publish cohort-only deterministic tables."""
    run.old.verify(run.OUT)
    if run.old.read(run.OUT / "audit.json")["status"] != "PASS":
        raise ValueError("Audited primary experiment required")
    start = time.process_time()
    with threadpool_limits(limits=1):
        tables, replay = compute(), compute()
    for a, b in zip(tables, replay, strict=True):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    names = ("support.csv", "coordinate_scale.csv", "coordinate_components.csv")
    if [len(x) for x in tables] != [4, 10, 30]:
        raise ValueError("Incomplete finite accounting")
    dst = ROOT / "reports/grain_vector_mechanisms/coordinate_support_v1"
    dst.mkdir(exist_ok=False)
    for n, t in zip(names, tables, strict=True):
        if not set(t.authors) <= {4, 8}:
            raise ValueError("Cohort summaries only")
        t.to_csv(dst / n, index=False)
    run.old.write(
        dst / "provenance.json",
        {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": run.old.sha(Path(__file__)),
            "module_sha256": run.old.sha(Path(fn.__file__)),
            "primary_manifest_sha256": run.old.sha(run.OUT / "manifest.json"),
            "primary_audit_sha256": run.old.sha(run.OUT / "audit.json"),
            "primary_stage_freezes": {
                f: run.old.sha(run.OUT / f / "freeze.json")
                for f in ("teachers", "models", "predictions", "analysis")
            },
            "files": {n: run.old.sha(dst / n) for n in names},
            "account_cpu_seconds_including_replay": time.process_time() - start,
            "post_outcome": True,
            "fits": 0,
            "neural_calls": 0,
            "human_labels_read": False,
            "replay": "exact",
            "scope": "Existing source4/new8 coordinate mechanism account. Kernel coverage is not calibrated uncertainty. No model/threshold change or recipient-selected fit.",
        },
    )
    print({"status": "COMPLETE", "tables": 3, "fits": 0}, flush=True)


if __name__ == "__main__":
    main()
