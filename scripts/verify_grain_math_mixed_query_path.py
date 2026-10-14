#!/usr/bin/env python3
"""Alternate HM35 array/dot/quadrature replay without a new model invocation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_query_context as OLD
from scripts.run_grain_math_position_support import array_unit, check
from scripts.run_grain_math_query_response import load_model
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/mixed_query_path_r1"
VERIFY = ROOT / "results/grain_math_response_v1/mixed_query_path_verification_r1"


def verify(out, destination):
    """Recompute direct signed contractions and fixed integrals on all saved nodes."""
    if destination.exists():
        raise FileExistsError("preserve previous verification")
    m = check(out)
    status = read_json(out / "run_status.json")
    if status["status"] != "EVALUATED" or status["in_flight"] is not None:
        raise ValueError("complete actual acquisition required")
    for kind, expected in (("forward", 288), ("first", 288), ("second", 576)):
        if (
            status[kind + "_attempted"] != expected
            or status[kind + "_completed"] != expected
        ):
            raise ValueError("actual derivative counts differ from complete design")
    hashes = read_json(out / "02/artifact_hashes.json")
    for name, sha in hashes.items():
        if digest(out / name) != sha:
            raise ValueError("output changed " + name)
    barrier = out / "01/source_barrier.json"
    b = read_json(barrier)
    for name, sha in b["files"].items():
        if digest(out / name) != sha:
            raise ValueError("source checkpoint changed")
    rows = {
        (int(r["path"]), r["arm"]): r for r in table(out / "02/path_comparisons.csv")
    }
    errors = {
        k: 0.0
        for k in (
            "mixed",
            "split",
            "integral",
            "gradient",
            "endpoint",
            "projection",
            "metric",
        )
    }
    originals = {
        (int(r["unit"]), r["arm"]): r
        for r in table(ROOT / m["spec"]["parent"] / "02/actions.csv")
    }
    projectors = {}
    for query in m["spec"]["queries"]:
        model = load_model(ROOT / m["spec"]["operator_parent"], query)
        basis, singular, _ = np.linalg.svd(model["D"], full_matrices=False)
        projectors[query] = basis[:, singular**2 > model["tolerance"]]
    for path in read_json(out / "00/design.json"):
        banks, mixed = [], []
        for node in range(3):
            file = out / "01/nodes" / f"path_{path['path']:03d}_node_{node}.json"
            meta = array_unit(file)
            if meta["address"] != path or meta["node"] != node or meta["t"] != node / 2:
                raise ValueError("semantic path or node address changed")
            if any(meta["audit"].values()):
                raise ValueError("H/P/Q patch or B preservation failed")
            if path["role"] == "source":
                if (
                    meta["source_sha256"] is not None
                    or meta["acquired_at"] >= b["frozen_at"]
                ):
                    raise ValueError("source order changed")
            elif (
                meta["source_sha256"] != digest(barrier)
                or meta["acquired_at"] <= b["frozen_at"]
            ):
                raise ValueError("application before source checkpoint")
            with np.load(file.with_suffix(".npz"), allow_pickle=False) as a:
                arrays = {k: a[k].copy() for k in a.files}
            basis = projectors[path["query"]]
            expected_support = basis @ (basis.T @ arrays["delta_p"].ravel())
            errors["projection"] = max(
                errors["projection"],
                float(np.max(abs(expected_support - arrays["supported"].ravel()))),
            )
            if not np.array_equal(
                arrays["delta_p"],
                arrays["p0"].astype(float) - arrays["p1"].astype(float),
            ):
                raise ValueError("actual P displacement changed")
            expected_p = (
                arrays["p1"]
                if node == 0
                else arrays["p0"]
                if node == 2
                else (
                    (arrays["p0"].astype(float) + arrays["p1"].astype(float)) / 2
                ).astype(np.float32)
            )
            if not np.array_equal(arrays["p"], expected_p):
                raise ValueError("P path changed")
            v = arrays["p_covectors"].astype(float).reshape(2, -1)
            d = np.stack(
                [arrays[k].ravel() for k in ("delta_p", "supported", "unsupported")],
                axis=1,
            )
            actual = v @ d
            errors["mixed"] = max(
                errors["mixed"], float(np.max(abs(actual - arrays["mixed_actions"])))
            )
            errors["split"] = max(
                errors["split"],
                float(np.max(abs(actual[:, 0] - actual[:, 1] - actual[:, 2]))),
            )
            if node in (0, 2):
                row = path["right"] if node == 0 else path["left"]
                parent = ROOT / m["spec"]["parent"]
                ref, _ = OLD.paths(parent, row, read_json(parent / "00/manifest.json"))
                ref_meta = array_unit(ref)
                with np.load(ref.with_suffix(".npz"), allow_pickle=False) as bank:
                    err = float(
                        np.max(
                            abs(
                                arrays["gradient"].astype(float)
                                - bank["gradient"].astype(float)
                            )
                        )
                    )
                if meta["native"] != ref_meta["native"]:
                    raise ValueError("native endpoint score replay failed")
                errors["endpoint"] = max(
                    errors["endpoint"],
                    err,
                    abs(meta["gradient_readout"] - ref_meta["gradient_readout"]),
                )
            banks.append(arrays)
            mixed.append(actual)
        for i, arm in enumerate(m["spec"]["arms"]):
            r = rows[path["path"], arm]
            endpoint = float(
                np.dot(
                    (
                        banks[2]["gradient"].astype(float)
                        - banks[0]["gradient"].astype(float)
                    ).ravel(),
                    banks[0]["delta_h"][i].ravel(),
                )
                * 0.5
            )
            native = (
                float(originals[path["left"]["unit"], arm]["odd_action"])
                - float(originals[path["right"]["unit"], arm]["odd_action"])
            ) * 0.5
            errors["gradient"] = max(
                errors["gradient"],
                abs(endpoint - float(r["actual_gradient_halfcontrast"])),
                abs(native - float(r["actual_finite_halfcontrast"])),
            )
            for rule, weights in (
                ("simpson", (1 / 12, 1 / 3, 1 / 12)),
                ("trapezoid", (1 / 8, 1 / 4, 1 / 8)),
            ):
                values = np.einsum("t,tc->c", weights, np.array(mixed)[:, i])
                for j, component in enumerate(("full", "supported", "unsupported")):
                    errors["integral"] = max(
                        errors["integral"],
                        abs(values[j] - float(r[rule + "_" + component])),
                    )
                errors["integral"] = max(
                    errors["integral"],
                    abs(values[0] - endpoint - float(r[rule + "_gradient_error"])),
                    abs(values[0] - native - float(r[rule + "_finite_error"])),
                    abs(2 * values[1] * values[2] - float(r[rule + "_support_cross"])),
                )
    for metric in table(out / "02/metrics.csv"):
        r = [
            v
            for v in rows.values()
            if (v["role"], v["query"], v["arm"])
            == (metric["role"], metric["query"], metric["arm"])
        ]
        energy = lambda key, rows=r: sum(float(x[key]) ** 2 for x in rows) / len(rows)
        ref = energy("actual_gradient_halfcontrast") ** 0.5
        for rule in ("simpson", "trapezoid"):
            value = (
                energy(rule + "_gradient_error")
                / energy("actual_gradient_halfcontrast")
            ) ** 0.5
            errors["metric"] = max(
                errors["metric"],
                abs(value - float(metric[rule + "_relative_gradient_rmse"])),
            )
        errors["metric"] = max(
            errors["metric"], abs(ref - float(metric["gradient_rms"]))
        )
    if max(errors.values()) > m["spec"]["alternate_absolute_tolerance"]:
        raise ValueError("fixed numerical tolerance failed " + repr(errors))
    destination.mkdir(parents=True)
    write_json(
        destination / "summary.json",
        {
            "status": "VERIFIED_FINITE_HM35_NOT_GLOBAL",
            "maximum_errors": errors,
            "output_hashes": len(hashes),
            "inherited_hashes": len(m["files"]),
            "paths": 96,
            "nodes": 288,
            "comparisons": len(rows),
            "new_model_calls": 0,
        },
    )
    print(read_json(destination / "summary.json"), flush=True)


def main():
    """Run an alternate finite replay, preserving every previous outcome."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verification", type=Path, default=VERIFY)
    a = parser.parse_args()
    verify(a.out, a.verification)


if __name__ == "__main__":
    main()
