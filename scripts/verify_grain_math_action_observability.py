#!/usr/bin/env python3
"""HM18 alternate explicit-dyad and compact-covector arithmetic verification."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_action_observability import DEFAULT, check
from scripts.run_grain_math_context_operator_decomposition import inputs
from scripts.run_grain_math_response import write_json
from scripts.run_grain_upstream_routes import collector_guard

OUTPUT = ROOT / "results/grain_math_response_v1/action_observability_verification_r1"
CHANNELS = ("linear", "symmetric", "skew", "finite_step")


def rows(path):
    """Read a specified CSV, without expanding filesystem metadata sidecars."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def measure(families):
    """Rebuild fixed family mass independently of the primary weight helper."""
    counts = Counter(families)
    return np.array([1 / len(counts) / counts[f] for f in families])


def dot_cos(value, aa, bb):
    """Do not replace missing similarities with a successful zero/perfect score."""
    return float(value / (aa * bb) ** 0.5) if aa > 0 and bb > 0 else None


def stats(v, w, multiplier):
    """Recompute weighted channel geometry by explicit pairwise contractions."""
    means = np.einsum("i,ij->j", w, v)
    g = np.einsum("i,ij,ik->jk", w, v, v)
    centered = v - means
    cov = np.einsum("i,ij,ik->jk", w, centered, centered)
    eigen = np.linalg.eigvalsh((g[1:3, 1:3] + g[1:3, 1:3].T) / 2)
    tol = multiplier * np.finfo(float).eps * max(1.0, eigen[-1]) * 2
    rank = int(sum(eigen > tol))
    rms = np.sqrt(np.maximum(0, np.diag(g)))
    direct = float(np.sqrt(sum(w * (v[:, 1] + v[:, 2]) ** 2)))
    transpose = float(np.sqrt(sum(w * (v[:, 1] - v[:, 2]) ** 2)))
    values = {
        **{p + "_rms": float(rms[i]) for i, p in enumerate(CHANNELS)},
        "sh_inner": float(g[1, 2]),
        "sh_cosine": dot_cos(g[1, 2], g[1, 1], g[2, 2]),
        "sh_correlation": dot_cos(cov[1, 2], cov[1, 1], cov[2, 2]),
        "direct_rms": direct,
        "transpose_rms": transpose,
        "transpose_direct_ratio": transpose / direct if direct else None,
        "step_skew_ratio": float(rms[3] / rms[2]) if rms[2] else None,
        "sh_eigen_min": float(eigen[0]),
        "sh_eigen_max": float(eigen[-1]),
        "sh_rank": rank,
        "sh_tolerance": float(tol),
        "sh_condition_number": float(eigen[-1] / eigen[0]) if rank == 2 else None,
        "sh_min_trace_ratio": float(eigen[0] / np.trace(g[1:3, 1:3]))
        if np.trace(g[1:3, 1:3])
        else None,
        "numerical_null_dimension": 2 - rank,
    }
    return values, g, cov, means


def verify(packet, out):
    """Verify every diagnostic, spectrum, address and fixed-channel result."""
    if out.exists():
        raise FileExistsError("preserve earlier independent arithmetic replay")
    manifest = check(packet, geometry=True)
    spec = manifest["spec"]
    ledger = read_json(packet / "02/artifact_hashes.json")
    for name, expected in ledger.items():
        if digest(packet / name) != expected:
            raise ValueError(f"HM18 output changed: {name}")
    parent = ROOT / spec["operator_packet"]
    _, old, parameters, support, views, alpha, baseline = inputs(parent)
    x, c, midpoint, shuffled, _ = views
    kappa = float(parameters["delta_scale"] / parameters["context_scale"])
    with np.load(parent / "01/source_support.npz") as arrays:
        q = arrays["basis"].copy()
    xq = x @ q
    contexts = {
        "receiver_tensor": c,
        "endpoint_potential": midpoint,
        "shuffled_context": shuffled,
    }
    metadata = read_json(packet / "00/metadata.json")
    first = sorted(
        (r for r in metadata if r["query"] == spec["queries"][0]),
        key=lambda r: r["state_index"],
    )
    by_state = {r["state_index"]: r for r in first}
    native = np.empty(len(metadata))
    for role in ("source", "held"):
        with np.load(old / f"00/{role}_targets.npz") as arrays:
            native[arrays["indices"]] = arrays["y"]
    errors, checks = defaultdict(float), Counter()

    def compare(actual, expected, group):
        for field, value in expected.items():
            saved = actual[field]
            if value is None:
                if saved != "":
                    raise ValueError(f"{group}/{field}: undefined value filled")
            elif isinstance(value, (bool, np.bool_)):
                if saved != str(bool(value)):
                    raise ValueError(f"{group}/{field}: Boolean differs")
            elif isinstance(value, (int, float, np.number)):
                err = abs(float(saved) - float(value))
                errors[group] = max(errors[group], err)
                if not np.isclose(float(saved), float(value), atol=2e-9, rtol=1e-7):
                    raise ValueError(f"{group}/{field}: {saved} != {value}")
                if (
                    field in ("sh_rank", "observable_rank", "numerical_null_dimension")
                    and err
                ):
                    raise ValueError(f"{group}/{field}: rank changed")
            elif str(saved) != str(value):
                raise ValueError(f"{group}/{field}: address differs")
        checks[group] += 1

    actions, operators = {}, {}
    for query in spec["queries"]:
        actions[query, "full_linear"] = np.column_stack(
            (x @ baseline[query], np.zeros((len(x), 3)))
        )
        for kind, ctx in contexts.items():
            cq = ctx @ q
            b = alpha[query, kind] @ xq[support]
            operator = (xq[support].T * alpha[query, kind]) @ cq[support]
            if kind == "endpoint_potential":
                operator = (operator + operator.T) / 2
            s, h = (operator + operator.T) / 2, (operator - operator.T) / 2
            # Full covectors avoid amplifying the source-basis truncation in tiny actions.
            xs, cs = x[support], ctx[support]
            direct_covector = ((ctx @ cs.T) * alpha[query, kind]) @ xs
            transpose_covector = ((ctx @ xs.T) * alpha[query, kind]) @ cs
            direct = np.einsum("ij,ij->i", x, direct_covector)
            transpose = np.einsum("ij,ij->i", x, transpose_covector)
            step_covector = ((x @ xs.T) * alpha[query, kind]) @ cs
            parts = np.column_stack(
                (
                    x @ (alpha[query, kind] @ xs),
                    (direct + transpose) / 2,
                    (direct - transpose) / 2,
                    kappa * np.einsum("ij,ij->i", x, step_covector) / 2,
                )
            )
            if kind == "endpoint_potential":
                parts[:, 2:] = 0
            actions[query, kind] = parts
            operators[query, kind] = {"A": operator, "S": s, "H": h, "b": b}
    component_rows = rows(parent / "02/components.csv")
    original = {r["row"]: r for r in metadata}
    for r in component_rows:
        meta = original[int(r["row"])]
        compare(
            r,
            {
                k: meta[k]
                for k in (
                    "row",
                    "query",
                    "family",
                    "role",
                    "arm",
                    "state_index",
                    "unit_path",
                    "reference_path",
                    "reference_state_sha256",
                    "target_state_sha256",
                )
            },
            "address",
        )
        v = actions[r["query"], r["model"]][int(r["state_index"])]
        compare(
            r,
            {
                **dict(zip(CHANNELS, v, strict=True)),
                "observed_action": native[meta["row"]],
                "predicted_action": float(sum(v[:3])),
            },
            "component_replay",
        )
    with np.load(packet / "01/source_actions.npz") as arrays:
        if not np.array_equal(arrays["states"], support):
            raise ValueError("source states changed")
        for (query, kind), v in actions.items():
            error = float(np.max(abs(v[support] - arrays[query + "__" + kind])))
            errors["source_actions"] = max(errors["source_actions"], error)
            if error > 1e-10:
                raise ValueError("source action freeze failed")

    spectral = {
        (r["design"], r["context"], r["part"], r["measure"]): r
        for r in rows(packet / "01/observation_spectra.csv")
    }
    dimension = {
        "A": q.shape[1] ** 2,
        "S": q.shape[1] * (q.shape[1] + 1) // 2,
        "H": q.shape[1] * (q.shape[1] - 1) // 2,
    }
    for design in spec["source_designs"]:
        states = np.array(
            [
                s
                for s in support
                if design == "all_states" or by_state[s]["arm"] == design
            ]
        )
        families = [by_state[s]["family"] for s in states]
        weights = {
            "uniform": np.full(len(states), 1 / len(states)),
            "family_balanced": measure(families),
        }
        for kind, ctx in contexts.items():
            # Explicit small-coordinate dyads provide an alternate to Hadamard Grams.
            d = np.einsum("ij,ik->ijk", xq[states], (ctx @ q)[states])
            features = {
                "A": d,
                "S": (d + d.transpose(0, 2, 1)) / 2,
                "H": (d - d.transpose(0, 2, 1)) / 2,
            }
            with np.load(
                packet / "01" / f"{design}_{kind}_observation_grams.npz"
            ) as arrays:
                if not np.array_equal(arrays["state_indices"], states):
                    raise ValueError("source design addresses changed")
                for part, features_part in features.items():
                    f = features_part.reshape(len(states), -1)
                    raw = f @ f.T
                    error = float(np.max(abs(raw - arrays[part + "_raw"])))
                    errors["explicit_gram"] = max(errors["explicit_gram"], error)
                    if error > 1e-9:
                        raise ValueError("explicit source dyad Gram differs")
                    for name, w in weights.items():
                        np.testing.assert_allclose(
                            w, arrays[name + "_weights"], atol=1e-15
                        )
                        weighted = np.sqrt(w)[:, None] * f
                        eig = np.linalg.eigvalsh(weighted @ weighted.T)
                        tol = (
                            spec["machine_tolerance_multiplier"]
                            * np.finfo(float).eps
                            * max(1.0, eig[-1])
                            * len(states)
                        )
                        positive = eig[eig > tol]
                        errors["eigenvalues"] = max(
                            errors["eigenvalues"],
                            float(
                                np.max(
                                    abs(
                                        eig - arrays[part + "_" + name + "_eigenvalues"]
                                    )
                                )
                            ),
                        )
                        compare(
                            spectral[design, kind, part, name],
                            {
                                "observations": len(states),
                                "families": len(set(families)),
                                "support_rank": q.shape[1],
                                "operator_dimension": dimension[part],
                                "observable_rank": len(positive),
                                "numerical_null_dimension": dimension[part]
                                - len(positive),
                                "tolerance": tol,
                                "smallest_positive": float(positive[0])
                                if len(positive)
                                else None,
                                "largest": float(eig[-1]),
                                "eigen_min": float(eig[0]),
                            },
                            "source_spectrum",
                        )

    source_diag = {
        (r["design"], r["query"], r["model"]): r
        for r in rows(packet / "01/source_action_diagnostics.csv")
    }
    source_gram = {
        (r["design"], r["query"], r["model"], r["part_a"], r["part_b"]): r
        for r in rows(packet / "01/source_channel_grams.csv")
    }
    diag = {
        (r["role"], r["design"], r["query"], r["model"]): r
        for r in rows(packet / "02/action_diagnostics.csv")
    }
    sharing = {
        (r["role"], r["design"], r["model"], r["query_a"], r["query_b"], r["part"]): r
        for r in rows(packet / "02/target_sharing.csv")
    }
    cov_table = {
        (r["role"], r["design"], r["model"], r["query_a"], r["query_b"]): r
        for r in rows(packet / "02/native_compiled_covariance.csv")
    }
    query_row = {(r["query"], r["state_index"]): r["row"] for r in metadata}
    for role in ("source", "validation", "application"):
        for design in ("all_states", *sorted({r["arm"] for r in first})):
            members = [
                r
                for r in first
                if r["role"] == role and (design == "all_states" or r["arm"] == design)
            ]
            states = np.array([r["state_index"] for r in members])
            w = measure([r["family"] for r in members])
            info = {
                "observations": len(states),
                "families": len({r["family"] for r in members}),
            }
            for kind in spec["models"]:
                parts = {}
                for query in spec["queries"]:
                    v = actions[query, kind][states]
                    st, g, cov, _ = stats(v, w, spec["machine_tolerance_multiplier"])
                    compare(
                        diag[role, design, query, kind],
                        {**info, **st},
                        "action_diagnostic",
                    )
                    if role == "source" and design in spec["source_designs"]:
                        compare(
                            source_diag[design, query, kind],
                            {**info, **st},
                            "source_diagnostic",
                        )
                        for i, j in itertools.product(range(4), repeat=2):
                            compare(
                                source_gram[
                                    design, query, kind, CHANNELS[i], CHANNELS[j]
                                ],
                                {
                                    **info,
                                    "inner_product": g[i, j],
                                    "covariance": cov[i, j],
                                },
                                "source_channel_gram",
                            )
                    parts[query] = {
                        **dict(zip(CHANNELS, v.T, strict=True)),
                        "predicted_action": v[:, :3].sum(axis=1),
                        "coupling": v[:, 1:3].sum(axis=1),
                        "observed_action": np.array(
                            [native[query_row[query, s]] for s in states]
                        ),
                    }
                for qa, qb in itertools.combinations(spec["queries"], 2):
                    for part in (
                        *CHANNELS,
                        "predicted_action",
                        "observed_action",
                        "coupling",
                    ):
                        v = np.column_stack((parts[qa][part], parts[qb][part]))
                        _, g, cov, _ = stats(
                            np.column_stack((v, np.zeros_like(v))),
                            w,
                            spec["machine_tolerance_multiplier"],
                        )
                        op_part = {
                            "linear": "b",
                            "symmetric": "S",
                            "skew": "H",
                            "coupling": "A",
                        }.get(part)
                        fc = None
                        if kind != "full_linear" and op_part:
                            a, b = (
                                operators[qa, kind][op_part],
                                operators[qb, kind][op_part],
                            )
                            fc = dot_cos(np.sum(a * b), np.sum(a * a), np.sum(b * b))
                        ac = dot_cos(g[0, 1], g[0, 0], g[1, 1])
                        compare(
                            sharing[role, design, kind, qa, qb, part],
                            {
                                **info,
                                "inner_product": g[0, 1],
                                "covariance": cov[0, 1],
                                "rms_a": np.sqrt(g[0, 0]),
                                "rms_b": np.sqrt(g[1, 1]),
                                "action_cosine": ac,
                                "action_correlation": dot_cos(
                                    cov[0, 1], cov[0, 0], cov[1, 1]
                                ),
                                "frobenius_cosine": fc,
                                "cosine_sign_reversal": fc is not None
                                and ac is not None
                                and fc * ac < 0,
                            },
                            "sharing",
                        )
                    expected = {**info}
                    for label, part in (
                        ("native", "observed_action"),
                        ("compiled", "predicted_action"),
                    ):
                        v = np.column_stack((parts[qa][part], parts[qb][part]))
                        _, _, cov, means = stats(
                            np.column_stack((v, np.zeros_like(v))),
                            w,
                            spec["machine_tolerance_multiplier"],
                        )
                        expected.update(
                            {
                                label + "_covariance": cov[0, 1],
                                label + "_correlation": dot_cos(
                                    cov[0, 1], cov[0, 0], cov[1, 1]
                                ),
                                label + "_mean_a": means[0],
                                label + "_mean_b": means[1],
                                label + "_sd_a": np.sqrt(cov[0, 0]),
                                label + "_sd_b": np.sqrt(cov[1, 1]),
                            }
                        )
                    compare(
                        cov_table[role, design, kind, qa, qb], expected, "covariance"
                    )
    expected_counts = {
        "address": 8688,
        "component_replay": 8688,
        "source_spectrum": 36,
        "source_diagnostic": 24,
        "source_channel_gram": 384,
        "action_diagnostic": 288,
        "sharing": 2016,
        "covariance": 288,
    }
    if dict(checks) != expected_counts:
        raise ValueError(f"incomplete numerical scope: {dict(checks)}")
    write_json(out / "frozen_primary_artifact_hashes.json", ledger)
    summary = {
        "status": "VERIFIED_FINITE_HM18_NOT_GLOBAL_SCOPE_COMPLETE",
        "checked_rows": dict(checks),
        "max_errors": dict(errors),
        "new_source_fits": 0,
        "new_model_calls": 0,
        "api_calls": 0,
        "same_agent_alternate_arithmetic_not_independent_researcher": True,
        "primary_manifest_sha256": digest(packet / "00/manifest.json"),
        "verifier_sha256": digest(Path(__file__)),
        "original_18_protocols_100_cards_44_questions_complete": False,
    }
    write_json(out / "summary.json", summary)
    print(summary, flush=True)


def main():
    """Run a bounded alternate replay; never create a replacement research queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with collector_guard(args.out):
        verify(args.packet, args.out)


if __name__ == "__main__":
    main()
