#!/usr/bin/env python3
"""One post-freeze, zero-fit account of component accuracy versus composition."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.vector_mechanisms import teaching_composition as fn
from scripts import run_grain_conditional_teaching as run


def compute():
    """Read only pinned, already-evaluated vectors; publish no author rows."""
    old, ep = run.old, run.ep
    jobs = [j for j in old.read(run.OUT / "private_jobs.json") if j["cohort"] == "new8"]
    rows = [
        old.arrays(run.OUT / "analysis" / ("private_" + j["key"] + ".npz"))
        for j in jobs
    ]
    z = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    programs = []
    for j in jobs:
        pair = [
            old.arrays(ep.OUT / "predictions" / (k + ".npz")) for k in j["query_keys"]
        ]
        programs.append(
            {k: np.stack([p[k] for p in pair]) for k in ("primal", "vectors")}
        )
    p = {k: np.stack([x[k] for x in programs]) for k in programs[0]}
    baseline = old.arrays(run.previous.BASELINE)
    output = []
    for vi, variant in enumerate(run.fn.VARIANTS):
        for arm, name in enumerate(ep.fn.ARMS[:4]):
            for oi, observable in enumerate(("scalar", *ep.fn.COMPONENTS)):
                current = (
                    p["primal"][:, :, arm, None, :]
                    if oi == 0
                    else p["vectors"][:, :, arm, oi - 1]
                )
                native = (
                    z["native_primal"][:, :, None, :]
                    if oi == 0
                    else z["native_vectors"][:, :, oi - 1]
                )
                a = fn.decompose(
                    current, native, z["readers"][:, vi], z["native_r"], baseline["m"]
                )
                expected = (
                    (z["scalar"][:, vi, 0, arm, 0] - z["scalar"][:, vi, 1, arm, 0])[
                        :, None
                    ]
                    if oi == 0
                    else (
                        z["actions"][:, vi, 0, arm, oi - 1, 0]
                        - z["actions"][:, vi, 1, arm, oi - 1, 0]
                    )
                )
                actual = (
                    (z["target_scalar"][:, 0, 0] - z["target_scalar"][:, 1, 0])[:, None]
                    if oi == 0
                    else (
                        z["target_actions"][:, 0, oi - 1, 0]
                        - z["target_actions"][:, 1, oi - 1, 0]
                    )
                )
                np.testing.assert_allclose(a["prediction"], expected, atol=1e-10)
                np.testing.assert_allclose(a["target"], actual, atol=1e-10)
                output.append(
                    {
                        "variant": variant,
                        "authors": 8,
                        "arm": name,
                        "observable": observable,
                        **fn.energies(a),
                    }
                )
    return pd.DataFrame(output)


def main():
    """Run exact double computation after primary audit; never change models."""
    run.old.verify(run.OUT)
    audit = run.old.read(run.OUT / "audit.json")
    if audit["status"] != "PASS":
        raise ValueError("Completed primary analysis required")
    for folder in ("models", "predictions", "analysis"):
        run.ep.freeze(run.OUT, folder)
    start = time.process_time()
    with threadpool_limits(limits=1):
        a, b = compute(), compute()
    elapsed = time.process_time() - start
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    if len(a) != 96 or elapsed > 30:
        raise ValueError("Bounded post-account contract differs")
    dst = ROOT / "reports/grain_vector_mechanisms/teaching_composition_v1"
    dst.mkdir(exist_ok=False)
    a.to_csv(dst / "signed_errors.csv", index=False)
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
            "stage_freezes": {
                f: run.old.sha(run.OUT / f / "freeze.json")
                for f in ("models", "predictions", "analysis")
            },
            "files": {"signed_errors.csv": run.old.sha(dst / "signed_errors.csv")},
            "post_outcome": True,
            "fits": 0,
            "neural_calls": 0,
            "human_labels_read": False,
            "cpu_seconds_including_replay": elapsed,
            "replay": "exact",
            "scope": "Existing eight discovery authors. Native arrays are evaluation-only. Three signed algebraic errors and oracle account, not a deployable oracle or causal variance shares.",
        },
    )
    print({"status": "PUBLISHED", "rows": len(a), "cpu_seconds": elapsed}, flush=True)


if __name__ == "__main__":
    main()
