#!/usr/bin/env python3
"""Alternate HM39 scalar/Jacobian composition and frozen-HM38 replay verification."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support as ARR
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_active_reading_pairing import centered
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/readout_curvature_r1"
DEST = ROOT / "results/grain_math_response_v1/readout_curvature_verification_r1"
COMPONENTS = ("link", "decoder", "roundoff")


def scalar_metrics(rows):
    """Independently reconstruct signed moments by compensated scalar summation."""
    n = len(rows)
    out = {"count": n}
    for k in (*COMPONENTS, "total"):
        out[k + "_mean"] = math.fsum(r[k] for r in rows) / n
        out[k + "_rms"] = math.sqrt(math.fsum(r[k] ** 2 for r in rows) / n)
    for i, a in enumerate(COMPONENTS):
        for b in COMPONENTS[i + 1 :]:
            out[a + "__" + b + "_cross"] = 2 * math.fsum(r[a] * r[b] for r in rows) / n
    den = math.fsum(r["total"] ** 2 for r in rows) / n
    out["link_ms_over_total_ms"] = (
        math.fsum(r["link"] ** 2 for r in rows) / n / den if den else None
    )
    out["decoder_ms_over_total_ms"] = (
        math.fsum(r["decoder"] ** 2 for r in rows) / n / den if den else None
    )
    out["link_decoder_cross_over_total_ms"] = (
        out["link__decoder_cross"] / den if den else None
    )
    out["sum_identity_error"] = max(
        abs(math.fsum(r[k] for k in COMPONENTS) - r["total"]) for r in rows
    )
    return out


def verify(out, destination):
    """Check every addressed derivative, projection, component and signed metric."""
    started = time.monotonic()
    if destination.exists():
        raise FileExistsError("preserve previous verification")
    destination.mkdir(parents=True)
    status = read_json(out / "run_status.json")
    spec = read_json(out / "00/manifest.json")["spec"]
    remaining = spec["maximum_analysis_seconds"] - status["analysis_seconds"]
    if status["status"] != "EVALUATED" or remaining <= 0:
        raise ValueError("complete evaluated packet with CPU budget required")

    def stop(_signal, _frame):
        raise TimeoutError("combined120-second analysis/verification cap")

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(max(1, math.floor(remaining)))
    checked, maximum, softmax_max, gradient_sum_max = 0, 0.0, 0.0, 0.0

    def close(a, b, tolerance=None):
        nonlocal checked, maximum
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite or wrong-shaped numerical comparison")
        error = float(np.max(abs(a - b))) if a.size else 0.0
        checked += a.size
        if tolerance is None:
            maximum = max(maximum, error)
        if error > (spec["absolute_tolerance"] if tolerance is None else tolerance):
            raise ValueError(f"numerical mismatch {error}")
        return error

    try:
        manifest = check(out)
        hashes = read_json(out / "02/artifact_hashes.json")
        for name, sha in hashes.items():
            if digest(out / name) != sha:
                raise ValueError("changed output " + name)
        design = read_json(out / "00/design.json")
        barrier = read_json(out / "01/source_barrier.json")
        barrier_sha = digest(out / "01/source_barrier.json")
        if (
            len(design) != 96
            or barrier["source_nodes"] != 48
            or barrier["application_nodes_loaded"] != 0
        ):
            raise ValueError("complete source-first protocol required")
        for name, sha in barrier["files"].items():
            if digest(out / name) != sha:
                raise ValueError("source checkpoint changed")
        for kind, count in (("forward", 96), ("first", 1056), ("second", 192)):
            if (
                status[f"{kind}_attempted"] != count
                or status[f"{kind}_completed"] != count
            ):
                raise ValueError("actual calls missing")
        if (
            status["model_seconds"] > 900
            or status["in_flight"] is not None
            or status["committed_nodes"] != 96
        ):
            raise ValueError("resource/stage cap violated")
        projection = {}
        for query in spec["queries"]:
            _, bank = ARR.load(
                ROOT / spec["basis_parent"] / "01/models" / f"{query}_V.json"
            )
            u = bank["basis"]
            for path in design:
                if path["query"] != query:
                    continue
                file = out / "01/nodes" / f"path_{path['path']:03d}_node_1.json"
                meta, a = ARR.load(file)
                oldmeta, old = ARR.load(ROOT / meta["parent_file"])
                if (
                    meta["address"] != path
                    or oldmeta["address"] != path
                    or meta["parent_sha256"] != digest(ROOT / meta["parent_file"])
                ):
                    raise ValueError("semantic/token/parent address changed")
                if (
                    path["role"] == "source"
                    and (
                        meta["source_sha256"] is not None
                        or meta["acquired_at"] >= barrier["frozen_at"]
                    )
                ) or (
                    path["role"] == "application"
                    and (
                        meta["source_sha256"] != barrier_sha
                        or meta["acquired_at"] <= barrier["frozen_at"]
                    )
                ):
                    raise ValueError("source/application barrier order violated")
                if (
                    max(meta["audit"].values()) != 0
                    or meta["native"] != oldmeta["native"]
                ):
                    raise ValueError("score or patch replay changed")
                close(meta["score"], oldmeta["gradient_readout"])
                close(a["gradient"], old["gradient"])
                close(a["parent_total"], old["p_covectors"])
                close(a["delta_p"], old["delta_p"])
                close(a["delta_h"], old["delta_h"])
                close(a["logits"], oldmeta["native"]["candidate_logits"])
                exp = np.exp(a["logits"].astype(float) - float(np.max(a["logits"])))
                probability = exp / math.fsum(exp)
                output_gradient = probability * (
                    np.array(manifest["alphabet"]["values"])
                    - probability @ np.array(manifest["alphabet"]["values"])
                )
                # These are forward float32 quantities, not a changed score definition.
                softmax_max = max(
                    softmax_max,
                    close(
                        a["probabilities"], probability, 8 * np.finfo(np.float32).eps
                    ),
                    close(
                        a["output_gradient"],
                        output_gradient,
                        8 * np.finfo(np.float32).eps,
                    ),
                )
                p, g = (
                    a["probabilities"].astype(float),
                    a["output_gradient"].astype(float),
                )
                h = np.array(
                    [
                        [
                            (g[i] if i == j else 0.0) - p[i] * g[j] - g[i] * p[j]
                            for j in range(10)
                        ]
                        for i in range(10)
                    ]
                )
                close(a["hessian"], h)
                directional = np.array(
                    [
                        [
                            np.sum(a["h_jacobian"][j].astype(float) * d)
                            for j in range(10)
                        ]
                        for d in a["delta_h"]
                    ]
                )
                close(a["h_directional"], directional)
                jp = a["p_jacobian"].astype(float)
                link = np.stack(
                    [
                        sum(
                            sum(h[i, j] * directional[arm, j] for j in range(10))
                            * jp[i]
                            for i in range(10)
                        )
                        for arm in range(2)
                    ]
                )
                close(a["link"], link)
                close(a["total"], link + a["decoder"])
                close(a["total"], old["p_covectors"])
                close(a["roundoff"], old["p_covectors"] - a["total"])
                close(a["total"] + a["roundoff"], a["parent_total"])
                gradient_sum = sum(
                    g[j] * a["h_jacobian"][j].astype(float) for j in range(10)
                )
                gradient_sum_max = max(
                    gradient_sum_max,
                    close(
                        a["gradient"],
                        gradient_sum,
                        max(
                            1e-8,
                            8
                            * np.finfo(np.float32).eps
                            * float(np.max(abs(a["gradient"]))),
                        ),
                    ),
                )
                norms = np.sqrt(np.sum(old["delta_h"] ** 2, axis=(1, 2)))
                close(a["norms"], norms)
                z = u.T @ old["delta_p"].ravel()
                result = {}
                for name in (*COMPONENTS, "parent"):
                    v = (
                        (old["p_covectors"] if name == "parent" else a[name])
                        .astype(float)
                        .reshape(2, -1)
                    )
                    coef = (u.T @ v.T).T
                    theta = coef / norms[:, None]
                    eta = norms[:, None] * z[None, :]
                    full = np.array([np.dot(row, old["delta_p"].ravel()) for row in v])
                    supported = np.array(
                        [np.dot(t, e) for t, e in zip(theta, eta, strict=True)]
                    )
                    result[name] = {
                        "theta": theta,
                        "eta": eta,
                        "full": full,
                        "supported": supported,
                        "complement": full - supported,
                    }
                projection[path["path"]] = result
        references = {}
        for query in spec["queries"]:
            _, reference = ARR.load(out / "01/references" / f"{query}.json")
            refs = {}
            for cell in range(8):
                paths = sorted(
                    [
                        p
                        for p in design
                        if p["query"] == query
                        and p["role"] == "source"
                        and p["H"] * 4 + p["B"] * 2 + p["Q"] == cell
                    ],
                    key=lambda p: p["wording"],
                )
                for ci, name in enumerate(COMPONENTS):
                    theta = np.stack(
                        [projection[p["path"]][name]["theta"] for p in paths]
                    )
                    close(reference["theta"][cell, :, ci], theta)
                    refs[cell, name] = np.average(theta, axis=0)
                    close(reference["reference"][cell, ci], refs[cell, name])
            references[query] = refs
        actions = table(out / "02/actions.csv")
        for row in actions:
            ai = spec["arms"].index(row["arm"])
            pr = projection[int(row["path"])]
            for name in COMPONENTS:
                close(float(row[name]), pr[name][row["observable"]][ai])
            close(float(row["total"]), pr["parent"][row["observable"]][ai])
        expected_pairs, expected_words = {}, {}
        for query in spec["queries"]:
            for role in ("source", "application"):
                for cell in range(8):
                    paths = sorted(
                        [
                            p
                            for p in design
                            if p["query"] == query
                            and p["role"] == role
                            and p["H"] * 4 + p["B"] * 2 + p["Q"] == cell
                        ],
                        key=lambda p: p["wording"],
                    )
                    _, oldpair = ARR.load(
                        ROOT
                        / spec["parent"]
                        / "01/arrays"
                        / f"{query}_{role}_cell_{cell}.json"
                    )
                    for name in (*COMPONENTS, "parent"):
                        projs = [projection[p["path"]][name] for p in paths]
                        for ai, arm in enumerate(spec["arms"]):
                            theta, eta, full = [
                                np.stack([p[k][ai] for p in projs])
                                for k in ("theta", "eta", "full")
                            ]
                            ref = (
                                oldpair["reference"][1, ai]
                                if name == "parent"
                                else references[query][cell, name][ai]
                            )
                            pair, words = centered(theta, eta, full, ref)
                            key = query, role, cell, arm, name
                            expected_pairs[key] = pair
                            for wording, word in zip(
                                spec["wordings"], words, strict=True
                            ):
                                expected_words[(*key, wording)] = word
                            if name == "parent":
                                close(theta, oldpair["theta"][1, ai])
                                close(eta, oldpair["eta"][1, ai])
                                close(full, oldpair["full"][1, ai])
        pairs = table(out / "02/pair_decomposition.csv")
        words = table(out / "02/word_decomposition.csv")
        if (len(actions), len(pairs), len(words)) != (576, 384, 768):
            raise ValueError("complete addressed decomposition tables required")
        for rows, expected, word in (
            (pairs, expected_pairs, False),
            (words, expected_words, True),
        ):
            seen = set()
            for row in rows:
                key = (
                    row["query"],
                    row["role"],
                    int(row["cell"]),
                    row["arm"],
                    row["component"],
                )
                if word:
                    key = (*key, int(row["wording"]))
                if key in seen or key not in expected:
                    raise ValueError("duplicate or absent decomposition address")
                seen.add(key)
                for field, value in expected[key].items():
                    if value is None:
                        if row[field] not in ("", "None"):
                            raise ValueError("null denominator changed")
                    else:
                        close(float(row[field]), value)
        metrics = table(out / "02/metrics.csv")
        for row in metrics:
            head = {k: row[k] for k in ("query", "role", "arm")}
            scope, term = row["measure"].split("_", 1)
            rows = {"action": actions, "pair": pairs, "word": words}[scope]
            selected = [r for r in rows if all(r[k] == v for k, v in head.items())]
            if scope == "action":
                values = [
                    {k: float(r[k]) for k in (*COMPONENTS, "total")}
                    for r in selected
                    if r["observable"] == term
                ]
            else:
                group = {}
                for r in selected:
                    key = (r["cell"], r.get("wording", "-1"))
                    group.setdefault(key, {})[
                        "total" if r["component"] == "parent" else r["component"]
                    ] = float(r[term])
                values = list(group.values())
            expected = scalar_metrics(values)
            for field, value in expected.items():
                if value is None:
                    if row[field] not in ("", "None"):
                        raise ValueError("zero-denominator policy changed")
                else:
                    close(float(row[field]), value)
        elapsed = time.monotonic() - started
        if elapsed + status["analysis_seconds"] > 120:
            raise TimeoutError("combined CPU budget exhausted")
        write_json(
            destination / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM39_NOT_GLOBAL",
                "verified_at": now(),
                "nodes": 96,
                "numerical_values_checked": checked,
                "maximum_alternate_error": maximum,
                "softmax_float32_error": softmax_max,
                "digit_gradient_sum_error": gradient_sum_max,
                "output_hashes": len(hashes),
                "ancestry_hashes": len(manifest["files"]),
                "verification_seconds": elapsed,
                "combined_analysis_seconds": elapsed + status["analysis_seconds"],
                "forward_calls": 96,
                "first_gradient_calls": 1056,
                "second_gradient_calls": 192,
                "new_fits": 0,
                "new_model_calls_in_verification": 0,
            },
        )
        print(read_json(destination / "summary.json"), flush=True)
    except Exception as exc:
        write_json(
            destination / "failure.json",
            {
                "status": "VERIFICATION_FAILED_NOT_ACCEPTED",
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - started,
                "verified_at": now(),
            },
        )
        raise
    finally:
        signal.alarm(0)


def main():
    """Verify an evaluated finite packet without any neural calls or refitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--destination", type=Path, default=DEST)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify(args.out, args.destination)


if __name__ == "__main__":
    main()
