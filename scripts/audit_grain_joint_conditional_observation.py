#!/usr/bin/env python3
"""Independent array, calibration and person-function audit for M04/M05.

No production calibration/scoring functions are imported. Outputs distinguish
numeric formula checks from independent-person psychological validation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(p):
    """Read a retained artifact."""
    return json.loads(p.read_text())


def sha(p):
    """Verify frozen bytes without deserializing them."""
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main(out):
    """Recompute source anchors, all raw functions, individual scores and metrics."""
    root = Path(__file__).resolve().parents[1]
    for name, digest in read(out / "freeze.json").items():
        assert sha(root / name) == digest, name
    follow = out / "paired_followup"
    for name, digest in read(follow / "freeze.json").items():
        assert sha(Path(name)) == digest, name
    cfg = read(out / "config.json")
    bank = pd.read_csv(out / "public_bank.csv").sort_values("case_id")
    outputs = pd.concat(
        [
            pd.read_csv(out / d / "observations.csv")
            for d in ["cold_source", "cold_application"]
        ]
    )
    frame = bank.merge(outputs, on="case_id", validate="one_to_one").sort_values(
        "case_id"
    )
    values = frame[cfg["channels"]].to_numpy(float)
    augmented = np.c_[values, values[:, 8] * values[:, 9]]
    raw_models = {
        q: dict(
            np.load(
                out.parent / "query_moment_r1/models" / (q + ".npz"), allow_pickle=False
            )
        )
        for q in ["neuroticism", "confirmation_seeking"]
    }
    worst = {
        "raw_scalar": 0.0,
        "calibration": 0.0,
        "meter_resolution": 0.0,
        "function": 0.0,
        "metric": 0.0,
        "truth_event_sampling": 0.0,
    }
    checks = 0

    def compare(key, a, b, tol=1e-7):
        nonlocal checks
        a, b = np.asarray(a), np.asarray(b)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        worst[key] = max(worst[key], err)
        np.testing.assert_allclose(a, b, atol=tol, rtol=1e-9)
        checks += 1

    for folder in ["cold_source", "cold_application"]:
        rt = read(out / folder / "runtime.json")
        assert rt["prefix_calls"] == (32 if folder == "cold_source" else 40)
        assert rt["full_teacher_calls"] == 0 and len(rt["denied"]) == 2
        for trace in rt["traces"]:
            assert (
                trace["completed_blocks"] == list(range(17))
                and not trace["forbidden_calls"]
            )
        nli = pd.read_csv(out / folder / "nli_pairs.csv")
        for row in frame[frame.source == (folder == "cold_source")].itertuples():
            raw = np.load(out / folder / (row.case_id + ".npz"), allow_pickle=False)
            g, h = raw["g"], raw["h"]
            compare("raw_scalar", g.mean(axis=0), values[int(row.case_id[1:]), :4])
            _, index = np.unique(g, axis=0, return_inverse=True)
            residual = h.copy()
            for cell in np.unique(index):
                residual[index == cell] -= h[index == cell].mean(axis=0)
            for qi, query in enumerate(["neuroticism", "confirmation_seeking"]):
                model = raw_models[query]
                z = (g - model["g_mean"]) / model["g_scale"]
                terms = np.column_stack(
                    [
                        np.ones(len(z)),
                        *z.T,
                        *[z[:, i] * z[:, j] for i in range(4) for j in range(i, 4)],
                    ]
                )
                old = float(
                    np.sum((terms.T @ h / len(h)) * model["beta2"])
                    + model["offsets"][1]
                )
                r = residual @ model["query_basis"]
                addition = float(np.sum((r.T @ r / len(h)) * model["within_matrix"].T))
                compare("raw_scalar", old, values[int(row.case_id[1:]), 4 + qi])
                compare("raw_scalar", addition, values[int(row.case_id[1:]), 6 + qi])
                f = nli[(nli.case_id == row.case_id) & (nli["query"] == qi)].set_index(
                    "pole"
                )
                compare(
                    "raw_scalar",
                    f.loc[0, "p_entailment"] - f.loc[1, "p_entailment"],
                    values[int(row.case_id[1:]), 8 + qi],
                )
    for folder, data in [(out, values), (follow, augmented)]:
        assert (
            sha(folder / "fit/meters.json")
            == read(folder / "fit/seal.json")["meters_sha256"]
        )
        fit = read(folder / "fit/calibration.json")
        for cell in fit["cells"]:
            source = frame.source.to_numpy() & (
                frame.condition.to_numpy() == cell["condition"]
            )
            yy = data[source]
            poles = frame.loc[source, ["worry", "reassurance"]].to_numpy()
            x = np.column_stack([np.ones(len(yy)), poles, poles[:, 0] * poles[:, 1]])
            coef = x.T @ yy / len(yy)
            compare("calibration", coef[0], cell["offset"])
            compare("calibration", coef[1:].T, cell["loading"])
            residual = yy - x @ coef
            empirical = residual.T @ residual / (len(yy) - 4)
            compare("calibration", empirical, cell["empirical_covariance"])
            diagonal = np.maximum(
                np.diag(empirical), np.maximum(yy.var(axis=0), 1e-20) * 1e-8
            )
            covariance = 0.9 * empirical + 0.1 * np.diag(diagonal)
            compare("calibration", covariance, cell["covariance"])
        for m in read(folder / "fit/meters.json").values():
            l, r, b = (np.array(m[k]) for k in ["loading", "covariance", "raw_offset"])
            # Cholesky whitening and lstsq, instead of correlation-eigen whitening.
            c = np.linalg.cholesky(r)
            w = np.linalg.lstsq(
                np.linalg.solve(c, l), np.linalg.solve(c, np.eye(len(b))), rcond=1e-8
            )[0]
            compare("meter_resolution", w @ l, m["resolution"], tol=2e-7)
            compare("calibration", w, np.array(m["weights"]), tol=2e-7)
            compare("calibration", -w @ b, m["offset"], tol=2e-7)
        assert (
            sha(folder / "apply/functions.npz")
            == read(folder / "apply/seal.json")["functions_sha256"]
        )
        assert len(read(folder / "fit/guard.json")["denied"]) == 2
        assert len(read(folder / "apply/guard.json")["denied"]) == 1
    ev = np.load(out / "generate/evidence.npz", allow_pickle=False)
    truth = np.load(out / "generate/truth.npz", allow_pickle=False)
    main_scores = np.load(out / "apply/functions.npz", allow_pickle=False)
    pair_scores = np.load(follow / "apply/functions.npz", allow_pickle=False)
    main_meters, pair_meters = (read(p / "fit/meters.json") for p in [out, follow])
    main_metrics = pd.read_csv(out / "evaluate/metrics_by_seed.csv").set_index(
        ["seed", "style", "arm", "mode", "target", "moment"]
    )
    pair_metrics = pd.read_csv(follow / "evaluate/metrics_by_seed.csv").set_index(
        ["seed", "style", "arm", "target", "moment"]
    )
    for name in ev.files:
        ids = ev[name]
        run = name.removesuffix("_case_indices")
        seed = int(run.split("_")[0][1:])
        style = run.split("_", 1)[1]
        for is_pair, meters, stored, arms in [
            (
                False,
                main_meters,
                main_scores,
                list(cfg["arms"]) + ["joint_all_diagonal"],
            ),
            (
                True,
                pair_meters,
                pair_scores,
                ["nli_product", "joint_product", "joint_product_unpaired"],
            ),
        ]:
            for arm in arms:
                observations = values[ids]
                if is_pair:
                    right = (
                        np.roll(observations[..., 9], 1, axis=2)
                        if arm.endswith("unpaired")
                        else observations[..., 9]
                    )
                    observations = np.concatenate(
                        [observations, (observations[..., 8] * right)[..., None]],
                        axis=3,
                    )
                estimated = []
                for zi, z in enumerate(cfg["conditions"]):
                    meter = meters[f"{arm.removesuffix('_unpaired')}__{z}"]
                    estimate = (
                        observations[:, zi][:, :, np.array(meter["channels"])].mean(
                            axis=1
                        )
                        @ np.array(meter["weights"]).T
                        + meter["offset"]
                    )
                    estimated.append(estimate)
                a = (estimated[0] + estimated[2]) / 2
                b = (estimated[2] - estimated[0]) / 2
                for mode in ["joint"] if is_pair else ["joint", "level", "slope"]:
                    aa = np.zeros_like(a) if mode == "slope" else a
                    bb = np.zeros_like(b) if mode == "level" else b
                    for target, result in [
                        ("level", aa),
                        ("finite_change", 2 * bb),
                        ("new_condition", aa + 0.5 * bb),
                    ]:
                        pkey = (
                            f"{run}__{arm}__"
                            + ("" if is_pair else mode + "__")
                            + target
                        )
                        compare("function", result, stored[pkey], tol=1e-8)
                        actual = (
                            truth[f"s{seed}_level"]
                            if target == "level"
                            else 2 * truth[f"s{seed}_slope"]
                            if target == "finite_change"
                            else truth[f"s{seed}_condition_moments"][:, 1]
                        )
                        for j, moment in enumerate(cfg["targets"]):
                            row = (
                                pair_metrics.loc[(seed, style, arm, target, moment)]
                                if is_pair
                                else main_metrics.loc[
                                    (seed, style, arm, mode, target, moment)
                                ]
                            )
                            compare(
                                "metric",
                                np.mean((result[:, j] - actual[:, j]) ** 2),
                                row.mse,
                                tol=1e-7,
                            )
        poles = frame[["worry", "reassurance"]].to_numpy()
        perfect = np.column_stack([poles, poles[:, 0] * poles[:, 1]])[ids].mean(axis=2)
        quantities = {
            "level": (perfect[:, 0] + perfect[:, 2]) / 2,
            "finite_change": perfect[:, 2] - perfect[:, 0],
            "new_condition": 0.25 * perfect[:, 0] + 0.75 * perfect[:, 2],
        }
        for target, result in quantities.items():
            actual = (
                truth[f"s{seed}_level"]
                if target == "level"
                else 2 * truth[f"s{seed}_slope"]
                if target == "finite_change"
                else truth[f"s{seed}_condition_moments"][:, 1]
            )
            for j, moment in enumerate(cfg["targets"]):
                row = pair_metrics.loc[
                    (seed, style, "perfect_event_mean", target, moment)
                ]
                compare(
                    "truth_event_sampling",
                    np.mean((result[:, j] - actual[:, j]) ** 2),
                    row.mse,
                )
    # Identify observer drift, independently from real changes in conditional event moments.
    drift = []
    for style, variants in [
        ("source_style", cfg["source_variants"]),
        ("held_style", cfg["held_variants"]),
    ]:
        for is_pair, meters, data in [
            (False, main_meters, values),
            (True, pair_meters, augmented),
        ]:
            for arm in (
                list(cfg["arms"]) + ["joint_all_diagonal"]
                if not is_pair
                else ["nli_product", "joint_product"]
            ):
                maps = []
                for z in [-1, 1]:
                    chosen = frame.condition.eq(z) & frame.variant.isin(variants)
                    poles = frame.loc[chosen, ["worry", "reassurance"]].to_numpy()
                    x = np.column_stack(
                        [np.ones(chosen.sum()), poles, poles[:, 0] * poles[:, 1]]
                    )
                    coef = x.T @ data[chosen] / chosen.sum()
                    m = meters[f"{arm}__{z}"]
                    w = np.array(m["weights"])
                    ind = np.array(m["channels"])
                    maps.append((w @ coef[1:, ind].T, w @ coef[0, ind] + m["offset"]))
                cminus, dminus = maps[0]
                cplus, dplus = maps[1]
                for label, matrix in [
                    ("level_to_level", (cminus + cplus) / 2),
                    ("slope_to_level", (cplus - cminus) / 2),
                    ("level_to_slope", (cplus - cminus) / 2),
                    ("slope_to_slope", (cplus + cminus) / 2),
                ]:
                    for i in range(3):
                        for j in range(3):
                            drift.append(
                                {
                                    "style": style,
                                    "arm": arm,
                                    "map": label,
                                    "output": cfg["targets"][i],
                                    "input": cfg["targets"][j],
                                    "coefficient": matrix[i, j],
                                }
                            )
                for i, target in enumerate(cfg["targets"]):
                    drift.append(
                        {
                            "style": style,
                            "arm": arm,
                            "map": "offset_to_level",
                            "output": target,
                            "input": "constant",
                            "coefficient": (dminus[i] + dplus[i]) / 2,
                        }
                    )
                    drift.append(
                        {
                            "style": style,
                            "arm": arm,
                            "map": "offset_to_slope",
                            "output": target,
                            "input": "constant",
                            "coefficient": (dplus[i] - dminus[i]) / 2,
                        }
                    )
    pd.DataFrame(drift).to_csv(
        out / "evaluate/meter_drift_function_map.csv", index=False
    )
    result = {
        "status": "PASS",
        "formula_checks": checks,
        "public_texts": 72,
        "actual_prefix_calls": 72,
        "actual_nli_pairs": 288,
        "worst_absolute_errors": worst,
        "post_result_product_followup": True,
        "independent_equations_not_independent_researcher": True,
    }
    (out / "evaluate/independent_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/grain_vector_mechanisms/mechanism_todo_v1/joint_conditional_observation_r1",
    )
    main(parser.parse_args().output)
