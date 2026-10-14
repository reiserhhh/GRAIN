#!/usr/bin/env python3
"""Second-formula audit of response geometry and direct author expectations."""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.construction.math_cold_evidence import rotate
from scripts import run_grain_author_evidence_merge_hm102 as H
from scripts.run_grain_response_observability_hm127 import budget, check, load_banks


def direct_read(k, v, endpoint, bundle):
    """Use explicit head/token dot products and stable exponential weights."""
    directions = np.asarray(bundle["directions"])
    ratio = directions.shape[1] // v.shape[1]
    outputs = np.zeros((2, len(directions)))
    for target in range(len(directions)):
        q = rotate(
            np.asarray(bundle["raw_queries"])[target],
            endpoint + bundle["gaps"][target],
            bundle["frequencies"],
            bundle["scaling"],
        )
        for head in range(directions.shape[1]):
            logits = np.array([float(token[head // ratio] @ q[head]) for token in k])
            weight = np.exp(logits - max(logits))
            weight /= sum(weight)
            content = np.array(
                [float(token[head // ratio] @ directions[target, head]) for token in v]
            )
            outputs[0, target] += sum(content) / len(content)
            outputs[1, target] += weight @ content
    return outputs


def main(out):
    """Read only sealed observations; do not rerun their encoder/constructor."""
    if H.load(out / "status.json")["state"] != "ANALYZED":
        raise ValueError("Analyze once before audit")
    check(out)
    H.verify_hashes(out, H.load(out / "02/output_seal.json"))
    with budget(out, "readback") as state:
        banks = load_banks(out)
        with np.load(out / "02/witness.npz", allow_pickle=False) as z:
            theta = z["theta"]
            saved = {name: z[name] for name in z.files}
        with np.load(out / "02/cold_input.npz", allow_pickle=False) as z:
            data = dict(z)
        with np.load(out / "cold_output.npz", allow_pickle=False) as z:
            cold = dict(z)
        bundle = H.load(out / "02/cold_bundle.json")
        direct = np.array(
            [
                direct_read(k, v, int(data["endpoint"]), bundle)
                for k, v in zip(data["k"], data["v"], strict=True)
            ]
        )
        error = float(np.max(abs(direct - cold["readings"])))
        absolute = np.array(
            [
                sum(p * reading for p, reading in zip(law, direct, strict=True))
                for law in data["probabilities"]
            ]
        )
        error = max(error, float(np.max(abs(absolute - cold["absolute"]))))
        alias = H.load(out / "02/ambiguity.json")
        checked = 0
        manual_fields = {}
        for (view, wording), bank in banks.items():
            manual_fields[view, wording] = {}
            for channel, values in bank.items():
                gain = (values[:, :, 1, 1, 0] - values[:, :, 0, 1, 0]) / 2
                other_focus_gain = (values[:, :, 1, 1, 1] - values[:, :, 0, 1, 1]) / 2
                parts = {
                    "own_A": gain,
                    "own_B": other_focus_gain,
                    "own_focus_even": (gain + other_focus_gain) / 2,
                    "own_focus_odd": (gain - other_focus_gain) / 2,
                }
                manual_fields[view, wording][channel] = parts
                if view < 3 and channel == "weighted":
                    key = f"v{view}_w{wording}__weighted"
                    error = max(
                        error,
                        float(np.max(abs(gain.reshape(12, -1).T - saved[key]))),
                    )
                    raw = values[:, :, :, 1, 0].reshape(12, 2, -1)
                    p, m = saved["positive_law"], saved["negative_law"]
                    contrast = np.einsum("jb,jbq->q", p - m, raw)
                    expected = np.array(
                        alias["applications"][f"v{view}_w{wording}"]["weighted"][
                            "contrast"
                        ]
                    )
                    error = max(error, float(np.max(abs(contrast - expected))))
                # Explicit Walsh sum reconstructs both actors and both focuses.
                for actor_power, focus_power in np.ndindex(2, 2):
                    signed = sum(
                        (2 * o - 1) ** actor_power
                        * (1 - 2 * r) ** focus_power
                        * (2 * b - 1)
                        * values[:, :, b, o, r]
                        / 8
                        for b, o, r in np.ndindex(2, 2, 2)
                    )
                    reconstruction = np.zeros_like(signed)
                    for o, r in np.ndindex(2, 2):
                        pair_gain = (values[:, :, 1, o, r] - values[:, :, 0, o, r]) / 2
                        reconstruction += (
                            (2 * o - 1) ** actor_power
                            * (1 - 2 * r) ** focus_power
                            * pair_gain
                            / 4
                        )
                    error = max(error, float(np.max(abs(signed - reconstruction))))
                    checked += signed.size
                    key = f"actor_{'odd' if actor_power else 'even'}_focus_{'odd' if focus_power else 'even'}"
                    parts[key] = signed
            if view < 3:
                columns = {
                    n: manual_fields[view, wording][n]["own_A"].reshape(12, -1).T
                    for n in ("value", "uniform", "weighted")
                }
                indexed = np.zeros((12, 12))
                for j in range(12):
                    indexed[6 * (j % 2) : 6 * (j % 2 + 1), j] = columns["weighted"][
                        :, j
                    ]
                maps = {
                    "weighted": columns["weighted"],
                    "uniform": columns["uniform"],
                    "value_vectors": columns["value"],
                    "joint_uniform_weighted": np.concatenate(
                        (columns["uniform"], columns["weighted"])
                    ),
                    "condition_indexed_weighted": indexed,
                }
                for name, r in maps.items():
                    error = max(
                        error,
                        float(np.max(abs(r - saved[f"v{view}_w{wording}__{name}"]))),
                    )
                    expected = alias["applications"][f"v{view}_w{wording}"][name]
                    actual = 2 * r @ theta / 12
                    error = max(
                        error,
                        abs(float(np.linalg.norm(actual)) - expected["contrast_norm"]),
                    )
        correspondence = H.load(out / "02/correspondence.json")
        matrix_checks = 0
        permutations = list(itertools.permutations(range(6)))
        for name, reports in correspondence.items():
            stratum, key = name.split("__v")
            view, wording = map(int, key.split("_w"))
            indices = [0, 1] if stratum == "all_conditions" else [0]
            for channel_sector, expected in reports.items():
                channel, sector = channel_sector.split("__")
                x = manual_fields[0, 0][channel][sector][:, indices]
                y = manual_fields[view, wording][channel][sector][:, indices]
                a, b = x.reshape(6 * len(indices), -1), y.reshape(6 * len(indices), -1)
                gx = np.array([[float(u @ v) for v in a] for u in a])
                gy = np.array([[float(u @ v) for v in b] for u in b])
                alignment = float(
                    np.sum(gx * gy) / np.sqrt(np.sum(gx * gx) * np.sum(gy * gy))
                )
                error = max(
                    error,
                    abs(alignment - expected["gram_alignment"]),
                    abs(float(np.sum(a * a)) - expected["source_spectrum"]["trace"]),
                    abs(float(np.sum(b * b)) - expected["target_spectrum"]["trace"]),
                )
                cross = np.array(
                    [
                        [
                            np.mean(
                                [
                                    float(x[f, c] @ y[g, c])
                                    / (
                                        np.linalg.norm(x[f, c])
                                        * np.linalg.norm(y[g, c])
                                    )
                                    for c in range(len(indices))
                                ]
                            )
                            for g in range(6)
                        ]
                        for f in range(6)
                    ]
                )
                mean = float(np.trace(cross) / 6)
                tails = [
                    sum(cross[f, perm[f]] for f in range(6)) / 6
                    for perm in permutations
                ]
                rank = sum(v >= mean - 1e-12 for v in tails)
                if rank != expected["matching_rank"]:
                    raise ValueError("Family permutation rank failed independent loop")
                error = max(error, abs(mean - expected["matched_mean_cosine"]))
                matrix_checks += 1
        source = banks[0, 0]["weighted"][:, :, :, 1, 0].reshape(12, 2, 6)
        p, m = saved["positive_law"], saved["negative_law"]
        by_condition = np.array(
            [np.einsum("jb,jbq->q", (p - m)[c::2], source[c::2]) for c in range(2)]
        )
        error = max(
            error,
            float(
                np.max(
                    abs(
                        by_condition
                        - (
                            cold["condition_contributions"][:, 0]
                            - cold["condition_contributions"][:, 1]
                        )
                    )
                )
            ),
        )
        if error > H.load(out / "00/manifest.json")["spec"]["numeric_atol"]:
            raise ValueError("Alternate arithmetic audit failed")
        H.write(
            out / "verification.json",
            {
                "status": "VERIFIED_FINITE_HM127_NOT_GLOBAL",
                "alternate_max_error": error,
                "signed_entries_checked": checked,
                "geometry_and720matching_checks": matrix_checks,
                "same_agent_second_formula_not_independent_reviewer": True,
                "source_probability_global_margin_error": float(
                    np.max(abs(p.sum(0) - m.sum(0)))
                ),
                "no_new_encoder_or_fits_in_audit": True,
            },
        )
        H.write(
            out / "audit_seal.json",
            {
                "verification.json": H.sha(out / "verification.json"),
                "cold_output.npz": H.sha(out / "cold_output.npz"),
            },
        )
        state["state"] = "VERIFIED"
    print("HM127 readback verified", error, checked, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/grain_math_response_v1/response_observability_r1",
    )
    main(parser.parse_args().out)
