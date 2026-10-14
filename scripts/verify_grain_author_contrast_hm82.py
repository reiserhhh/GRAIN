#!/usr/bin/env python3
"""Alternative QR/nested-sum verification of the finite author-contrast packet."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import qr
from threadpoolctl import threadpool_limits


def main():
    """Check every scientific score/contrast without production projection helpers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    for name, expected in json.loads((out / "output_seal.json").read_text()).items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != expected:
            raise ValueError("primary output seal drift: " + name)
    data = np.load(out / "source.npz", allow_pickle=False)
    p, targets = data["reference"], data["targets"]
    bundle = json.loads((out / "bundle.json").read_text())
    truth = {
        a["author_key"]: a
        for a in json.loads((out / "truth_terminal_only.json").read_text())
    }
    observed = {
        a["author_key"]: a["events"]
        for a in (
            json.loads(line)
            for line in (out / "observations.jsonl").read_text().splitlines()
        )
    }
    errors, compiled, ranks, bases = [], {}, {}, {}
    with threadpool_limits(1):
        for arm, original in bundle["compiled"].items():
            b = data[arm + "_features"]
            b = b - np.sum(p[:, None] * b, axis=0)
            scales = np.sqrt(np.sum(p[:, None] * b**2, axis=0))
            b /= np.where(scales > 1e-12, scales, 1)
            weighted = np.sqrt(p[:, None]) * b
            orth, triangular, _ = qr(weighted, mode="economic", pivoting=True)
            singular = np.linalg.svd(weighted, compute_uv=False)
            errors.append(float(np.max(abs(singular - data[arm + "_singular"]))))
            tolerance = 100 * max(weighted.shape) * np.finfo(float).eps * singular[0]
            rank = int(np.sum(abs(np.diag(triangular)) > tolerance))
            svd_rank = int(np.sum(singular > tolerance))
            if rank != svd_rank:
                raise ValueError(f"rank disagreement: {arm}: {rank} != {svd_rank}")
            orth = orth[:, :rank]
            values = (
                orth @ (orth.T @ (np.sqrt(p[:, None]) * targets)) / np.sqrt(p[:, None])
            )
            compiled[arm], ranks[arm], bases[arm] = values, rank, orth
            errors.append(float(np.max(abs(values - original))))
        with (out / "source_target_visibility.csv").open() as stream:
            for row in csv.DictReader(stream):
                j, arm = bundle["targets"].index(row["target"]), row["arm"]
                if int(row["rank"]) != ranks[arm]:
                    raise ValueError("source rank table mismatch: " + arm)
                values = compiled[arm][:, j]
                errors.extend(
                    [
                        abs(float(row["visible_norm2"]) - sum(p * values**2)),
                        abs(
                            float(row["lost_norm2"])
                            - sum(p * (targets[:, j] - values) ** 2)
                        ),
                        abs(sum(p * values * (targets[:, j] - values))),
                    ]
                )
        with (out / "author_scores.csv").open() as stream:
            for row in csv.DictReader(stream):
                author = truth[row["author_key"]]
                probability = np.array(author["probability"])
                j = bundle["targets"].index(row["target"])
                values = compiled[row["arm"]][:, j]
                exact, true = (
                    sum(probability * values),
                    sum(probability * targets[:, j]),
                )
                events = observed[row["author_key"]]
                if int(row["n"]) != len(events):
                    raise ValueError("observed sample size mismatch")
                sample = sum(values[events]) / len(events)
                errors.extend(
                    [
                        abs(float(row["observable_expectation"]) - exact),
                        abs(float(row["true_declared_target"]) - true),
                        abs(float(row["sampled_score"]) - sample),
                        abs(float(row["omitted_target"]) - (true - exact)),
                        abs(float(row["sampling_error"]) - (sample - exact)),
                    ]
                )
        with (out / "contrast_accounts.csv").open() as stream:
            for row in csv.DictReader(stream):
                base = f"s{row['seed']}_{row['world']}_p{row['pair']}_"
                f = (
                    np.array(truth[base + "+1"]["probability"])
                    - truth[base + "-1"]["probability"]
                ) / p
                arm, j = row["arm"], bundle["targets"].index(row["target"])
                orth = bases[arm]
                visible = orth @ (orth.T @ (np.sqrt(p) * f)) / np.sqrt(p)
                tv = compiled[arm][:, j]
                condition = data["conditions"]
                choice = np.array(
                    [
                        sum(p[condition == value] * f[condition == value])
                        / sum(p[condition == value])
                        for value in condition
                    ]
                )
                values = {
                    "total_norm2": sum(p * f**2),
                    "visible_norm2": sum(p * visible**2),
                    "lost_norm2": sum(p * (f - visible) ** 2),
                    "choice_norm2": sum(p * choice**2),
                    "conditional_norm2": sum(p * (f - choice) ** 2),
                    "true_contrast": sum(p * f * targets[:, j]),
                    "visible_contrast": sum(p * visible * tv),
                    "lost_contrast": sum(p * (f - visible) * (targets[:, j] - tv)),
                }
                for name, value in values.items():
                    errors.append(abs(float(row[name]) - value))
                errors.extend(
                    [
                        abs(sum(p * choice * (f - choice))),
                        abs(
                            values["total_norm2"]
                            - values["visible_norm2"]
                            - values["lost_norm2"]
                        ),
                        abs(
                            values["true_contrast"]
                            - values["visible_contrast"]
                            - values["lost_contrast"]
                        ),
                    ]
                )
    audit = json.loads((out / "cold_audit.json").read_text())
    if (
        set(audit["opened"]) != set(audit["allowed"])
        or audit["truth_or_source_npz_access"]
    ):
        raise ValueError("cold process read undeclared scientific inputs")
    # Direct joint-probability sums, independently written for the five controls.
    c, x = np.array([-1, -1, 1, 1]), np.array([-1, 1, -1, 1])
    channels = {
        "identity": np.eye(4),
        "readdress": np.eye(4)[[3, 1, 0, 2]],
        "forget_condition": np.eye(2)[[0, 1, 0, 1]],
        "forget_content": np.eye(2)[[0, 0, 1, 1]],
        "flip_content_02": np.array(
            [[0.8, 0.2, 0, 0], [0.2, 0.8, 0, 0], [0, 0, 0.8, 0.2], [0, 0, 0.2, 0.8]]
        ),
    }
    with (out / "channel_controls.csv").open() as stream:
        for row in csv.DictReader(stream):
            f, k = (
                {"choice": c, "level": x, "response": c * x}[row["target"]],
                channels[row["channel"]],
            )
            means, output = [], []
            for y in range(k.shape[1]):
                mass = sum(k[i, y] / 4 for i in range(4))
                means.append(sum(f[i] * k[i, y] / 4 for i in range(4)) / mass)
                output.append(mass)
            visible = sum(m * value**2 for m, value in zip(output, means, strict=True))
            lost = sum(
                k[i, y] / 4 * (f[i] - means[y]) ** 2
                for i in range(4)
                for y in range(k.shape[1])
            )
            errors.extend(
                [
                    abs(visible - float(row["visible_norm2"])),
                    abs(lost - float(row["lost_norm2"])),
                ]
            )
    if max(errors) > 1e-9:
        raise ValueError(f"alternate discrepancy {max(errors)}")
    result = {
        "status": "VERIFIED_FINITE_HM82_NOT_GLOBAL",
        "alternate_scalar_checks": len(errors),
        "max_error": max(errors),
        "QR_source_ranks": ranks,
        "cold_files": len(audit["opened"]),
        "same_agent_alternate_not_independent_rebuilder": True,
        "original_scope_complete": False,
    }
    with (out / "verification.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
