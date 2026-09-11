#!/usr/bin/env python3
"""Post-NEXT89 zero-fit account: new paired mean, preserved learned contrast."""

import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_joint_query_functional_reader as run


def account():
    """Compare frozen arrays only; do not fit, tune or alter the primary result."""
    old, out = run.old, run.OUT
    old.verify(out)
    if old.read(out / "audit.json")["status"] != "PASS":
        raise ValueError("Complete NEXT89 required")
    for root in (out, run.previous.OUT):
        for stage in ("teachers", "models", "analysis"):
            run.ep.freeze(root, stage)
    start = time.process_time()
    m = old.arrays(out / "models" / (run.fn.VARIANT + ".npz"))
    prior = old.arrays(
        run.previous.OUT / "models/original12_compatible_coordinates.npz"
    )
    checks = {}

    def check(name, left, right, tolerance=1e-9):
        error = float(np.max(abs(np.asarray(left) - np.asarray(right))))
        if not np.isfinite(error) or error > tolerance:
            raise ValueError(f"{name}: {error} exceeds {tolerance}")
        checks[name] = max(checks.get(name, 0.0), error)

    for key in ("target", "fitted", "beta"):
        pair = m[key].reshape(12, 2, 5)
        check("source_" + key + "_contrast_max", pair[:, 0] - pair[:, 1], prior[key])

    mean_before, mean_after, contrast_before, contrast_after = [], [], [], []
    common_shifts, new_common_derivatives = [], []
    for job in old.read(out / "private_jobs.json"):
        if job["cohort"] != "new8":
            continue
        filename = "private_" + job["key"] + ".npz"
        b = old.arrays(run.previous.OUT / "analysis" / filename)
        a = old.arrays(out / "analysis" / filename)
        c = a["coordinate_values"][0]
        check("coordinate_contrast_max", c[0] - c[1], b["coordinate_values"][1])
        dc = a["coordinate_derivatives"][0]
        check(
            "coordinate_derivative_contrast_max",
            dc[0] - dc[1],
            b["coordinate_derivatives"][1],
        )
        # Only the primary five-row arm is constrained. K and V separately are not.
        before = np.concatenate(
            (b["scalar"][1, :, 3, 0, None], b["actions"][1, :, 3, 2, 0]), axis=1
        )
        after = np.concatenate(
            (a["scalar"][0, :, 3, 0, None], a["actions"][0, :, 3, 2, 0]), axis=1
        )
        actual = np.concatenate(
            (a["target_scalar"][:, 0, None], a["target_actions"][:, 2, 0]), axis=1
        )
        check(
            "primary_five_observation_contrast_max",
            after[0] - after[1],
            before[0] - before[1],
        )
        check(
            "equal_shift_both_queries_max", after[0] - before[0], after[1] - before[1]
        )
        for key in ("diagonal", "correction"):
            check(
                "primary_" + key + "_contrast_max",
                a[key][0, 0, 3, 0] - a[key][0, 1, 3, 0],
                b[key][1, 0, 3, 0] - b[key][1, 1, 3, 0],
            )
        eb, ea = before - actual, after - actual
        mean_before.append(eb.mean(axis=0))
        mean_after.append(ea.mean(axis=0))
        contrast_before.append(eb[0] - eb[1])
        contrast_after.append(ea[0] - ea[1])
        common_shifts.append(after.mean(axis=0) - before.mean(axis=0))
        new_common_derivatives.append(a["reader_derivatives"][0].mean(axis=0))
    arrays = [
        np.asarray(x)
        for x in (
            mean_before,
            mean_after,
            contrast_before,
            contrast_after,
            common_shifts,
        )
    ]
    if any(a.shape != (8, 5) for a in arrays):
        raise ValueError("Eight complete authors and five observations required")
    mb, ma, db, da, shift = arrays
    rows = []
    for k, name in enumerate(
        ("scalar", "total_F01", "total_F02", "total_F03", "total_F04")
    ):
        before = float(2 * np.sum(mb[:, k] ** 2) + np.sum(db[:, k] ** 2) / 2)
        after = float(2 * np.sum(ma[:, k] ** 2) + np.sum(da[:, k] ** 2) / 2)
        mean_gain = float(2 * np.sum(mb[:, k] ** 2 - ma[:, k] ** 2))
        check("paired_error_gain_is_mean_gain_max", before - after, mean_gain)
        rows.append(
            {
                "observable": name,
                "authors": 8,
                "pair_error2_before": before,
                "pair_error2_after": after,
                "pair_error_reduction_fraction": (before - after) / before,
                "mean_channel_error_reduction2": mean_gain,
                "shared_observation_shift_rms": float(
                    np.sqrt(np.mean(shift[:, k] ** 2))
                ),
            }
        )
    result = {
        "status": "PASS",
        "timing": "post-primary zero-fit mathematical account",
        "primary_commit": old.read(out / "manifest.json")["commit"],
        "account_code_sha256": old.sha(Path(__file__)),
        "source_solves": 0,
        "native_calls": 0,
        "compiled_calls": 0,
        "human_labels_read": False,
        "checks": checks,
        "observations": rows,
        "new_common_reader_derivative_rms": float(
            np.sqrt(np.mean(np.asarray(new_common_derivatives) ** 2))
        ),
        "numeric_cpu_seconds": time.process_time() - start,
        "interpretation": "The identity-output kernel preserves the old contrast; paired error gains come from newly observed common levels, not learned output coupling.",
    }
    dst = ROOT / "reports/grain_vector_mechanisms/joint_query_innovation_account_v1"
    dst.mkdir(exist_ok=False)
    old.write(dst / "account.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        account()
