#!/usr/bin/env python3
"""HM40 independent tall-SVD geometry, direct projection and signed-moment replay."""

from __future__ import annotations

import argparse
import itertools
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
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_json
from scripts.run_grain_math_target_active_support import load
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/cancellation_transport_r1"
DEST = ROOT / "results/grain_math_response_v1/cancellation_transport_verification_r1"
FIELDS = (
    "full",
    "u",
    "w",
    "remaining",
    "missing_u",
    "random_w",
    "random_remaining",
    "link_complement",
    "decoder_complement",
    "roundoff_complement",
    "c_action",
    "e_action",
)


def alternate(v, l, d, r, dp, u, w, wr):
    """Reconstruct each projection as an explicit vector before contracting the input."""
    vp = (u @ (u.T @ v.T)).T
    vw = (w @ (w.T @ v.T)).T
    vr = (wr @ (wr.T @ v.T)).T
    lp = l - (u @ (u.T @ l.T)).T
    dd = d - (u @ (u.T @ d.T)).T
    rr = r - (u @ (u.T @ r.T)).T
    c = (lp - dd) * 0.5
    e = lp - c
    dot = lambda x: np.einsum("ij,j->i", x, dp)
    square = lambda x: np.einsum("ij,ij->i", x, x)
    full, au, aw, ar = dot(v), dot(vp), dot(vw), dot(vr)
    norms = np.sqrt(square(lp) * square(dd))
    cd = np.einsum("ij,ij->i", lp, dd)
    result = {
        "full": full,
        "u": au,
        "w": aw,
        "remaining": dot(v - vp - vw),
        "missing_u": dot(v - vp),
        "random_w": ar,
        "random_remaining": dot(v - vp - vr),
        "link_complement": dot(lp),
        "decoder_complement": dot(dd),
        "roundoff_complement": dot(rr),
        "c_action": dot(c),
        "e_action": dot(e),
        "c_norm": np.sqrt(square(c)),
        "e_norm": np.sqrt(square(e)),
        "link_complement_norm": np.sqrt(square(lp)),
        "decoder_complement_norm": np.sqrt(square(dd)),
        "component_dot": cd,
        "component_cosine": [
            float(x / n) if n else None for x, n in zip(cd, norms, strict=True)
        ],
        "covector_energy": square(v),
        "u_covector_energy": square(vp),
        "w_covector_energy": square(vw),
        "random_covector_energy": square(vr),
        "input_energy": np.full(len(v), float(dp @ dp)),
        "u_input_energy": np.full(len(v), float(np.linalg.norm(u @ (u.T @ dp)) ** 2)),
        "w_input_energy": np.full(len(v), float(np.linalg.norm(w @ (w.T @ dp)) ** 2)),
        "random_input_energy": np.full(
            len(v), float(np.linalg.norm(wr @ (wr.T @ dp)) ** 2)
        ),
        "component_identity_max": np.max(abs(v - (l + d + r)), axis=1),
    }
    return result


def scalar_moments(rows):
    """Use compensated scalar reductions, preserving unresolved denominators."""
    n = len(rows)
    ms = {k: math.fsum(r[k] * r[k] for r in rows) / n for k in FIELDS}
    result = {
        "count": n,
        "full_denominator_resolved": ms["full"] > 1e-16,
        "missing_u_denominator_resolved": ms["missing_u"] > 1e-16,
    }
    for k in FIELDS:
        result[k + "_mean"] = math.fsum(r[k] for r in rows) / n
        result[k + "_rms"] = math.sqrt(ms[k])
    for a, b in itertools.combinations(("u", "w", "remaining"), 2):
        result[a + "__" + b + "_cross"] = 2 * math.fsum(r[a] * r[b] for r in rows) / n
    result["w_remaining_cross"] = result["w__remaining_cross"]
    result["missing_ms_reduction"] = ms["missing_u"] - ms["remaining"]
    result["random_missing_ms_reduction"] = ms["missing_u"] - ms["random_remaining"]
    for k in ("missing_u", "remaining", "random_remaining", "w"):
        result[k + "_over_full_rms"] = (
            math.sqrt(ms[k] / ms["full"])
            if result["full_denominator_resolved"]
            else None
        )
    for prefix in ("", "random_"):
        result[prefix + "missing_ms_reduction_fraction"] = (
            result[prefix + "missing_ms_reduction"] / ms["missing_u"]
            if result["missing_u_denominator_resolved"]
            else None
        )
    result["sum_identity_error"] = max(
        abs(r["full"] - r["u"] - r["w"] - r["remaining"]) for r in rows
    )
    return result


def verify(out, dest):
    """Replay every scope row from the frozen parent and direct geometric alternatives."""
    started = time.monotonic()
    if dest.exists():
        raise FileExistsError("preserve every verifier result")
    status = read_json(out / "run_status.json")
    if status["status"] != "EVALUATED" or status["new_model_calls"] != 0:
        raise ValueError("complete CPU-only result required")
    remaining = 300 - status["analysis_seconds"]
    if remaining <= 0:
        raise TimeoutError("no original CPU budget remains")
    dest.mkdir(parents=True)
    old = signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(
            TimeoutError("fixed combined300-second budget")
        ),
    )
    signal.setitimer(signal.ITIMER_REAL, remaining)
    maximum, checked = 0.0, 0

    def close(a, b):
        nonlocal maximum, checked
        if a is None or b is None:
            if a is not None or b is not None:
                raise ValueError("unresolved/observed mismatch")
            return
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.shape != y.shape or not np.isfinite(x + y).all():
            raise ValueError("finite aligned numeric comparisons required")
        error = float(np.max(abs(x - y))) if x.size else 0.0
        maximum = max(maximum, error)
        checked += x.size
        if error > 1e-8:
            raise ValueError("physical numerical replay differs " + str(error))

    def row_close(row, expected):
        for key, value in expected.items():
            if isinstance(value, str):
                if row[key] != value:
                    raise ValueError("address changed " + key)
            elif isinstance(value, bool):
                if row[key] != str(value):
                    raise ValueError("resolution flag changed " + key)
            else:
                close(None if row[key] == "" else float(row[key]), value)

    try:
        m = check(out)
        spec = m["spec"]
        ledger = read_json(out / "02/artifact_hashes.json")
        for name, sha in ledger.items():
            if digest(out / name) != sha:
                raise ValueError("changed output " + name)
        barrier = read_json(out / "01/source_barrier.json")
        if barrier["source_nodes"] != 48 or barrier["application_nodes_loaded"] != 0:
            raise ValueError("source-first checkpoint lost")
        for name, sha in barrier["files"].items():
            if digest(out / name) != sha:
                raise ValueError("source barrier changed " + name)
        design = read_json(out / "00/design.json")
        actual = table(out / "02/observations.csv")
        observations = {(int(r["path"]), r["arm"]): r for r in actual}
        if len(actual) != 192 or len(observations) != 192 or len(design) != 96:
            raise ValueError("incomplete observations")
        regenerated = []
        spectra = table(out / "01/spectra.csv")
        specs = {(r["query"], int(r["index"])): r for r in spectra}
        ranks = []
        for qi, query in enumerate(spec["queries"]):
            _, oldbank = load(
                ROOT / spec["basis_parent"] / "01/models" / f"{query}_V.json"
            )
            meta, bank = load(out / "01/models" / f"{query}.json")
            u, w, wr = oldbank["basis"], bank["basis"], bank["random_basis"]
            if (
                meta["role"] != "source"
                or meta["application_nodes_loaded"] != 0
                or meta["random_seed"] != 4007 + qi
            ):
                raise ValueError("source-only matched control required")
            close(w.T @ w, np.eye(meta["rank"]))
            close(u.T @ w, np.zeros((u.shape[1], meta["rank"])))
            close(wr.T @ wr, np.eye(meta["rank"]))
            close(u.T @ wr, np.zeros((u.shape[1], meta["rank"])))
            columns, proxy = [], []
            for path in design:
                if path["query"] != query or path["role"] != "source":
                    continue
                file = (
                    ROOT
                    / spec["parent"]
                    / "01/nodes"
                    / f"path_{path['path']:03d}_node_1.json"
                )
                nm = array_unit(file)
                if nm["address"] != path or nm["node"] != 1:
                    raise ValueError("source address mismatch")
                with np.load(file.with_suffix(".npz"), allow_pickle=False) as data:
                    l, d, r = (
                        data[k].astype(float).reshape(2, -1).T
                        for k in ("link", "decoder", "roundoff")
                    )
                raw = (l - d) / 2
                raw = raw - u @ (u.T @ raw)
                raw = raw - u @ (u.T @ raw)
                columns.extend(raw.T)
                proxy.extend((r - u @ (u.T @ r)).T)
            x = np.stack(columns, axis=1)
            p = np.stack(proxy, axis=1)
            close(bank["raw"], x)
            close(bank["roundoff_proxy"], p)
            scales = np.linalg.norm(x, axis=0)
            close(bank["scales"], scales)
            normalized = x[:, scales > 0] / scales[scales > 0]
            alt, singular, _ = np.linalg.svd(normalized, full_matrices=False)
            tolerance = (
                100
                * np.finfo(float).eps
                * 32
                * (singular[0] ** 2 if len(singular) else 0.0)
            )
            rank = int((singular**2 > tolerance).sum())
            if rank != meta["rank"]:
                raise ValueError("alternate source rank differs")
            alt = alt[:, :rank]
            close(alt - w @ (w.T @ alt), np.zeros_like(alt))
            close(bank["singular_values"], singular)
            raw_singular = np.linalg.svd(x, compute_uv=False)
            close(bank["raw_singular_values"], raw_singular)
            nf = float(
                np.linalg.norm(
                    np.linalg.norm(p, axis=0)[scales > 0] / scales[scales > 0]
                )
            )
            pf = float(np.linalg.norm(p))
            close(meta["normalized_roundoff_proxy_floor"], nf)
            close(meta["physical_roundoff_proxy_floor"], pf)
            if (
                meta["resolved_against_proxy"] != int((singular[:rank] > nf).sum())
                or meta["proxy_is_certified_component_error_bound"] is not False
            ):
                raise ValueError("floor labels changed or falsely certified")
            for j, s in enumerate(singular):
                row_close(
                    specs[query, j],
                    {
                        "query": query,
                        "index": j,
                        "normalized_singular": float(s),
                        "raw_singular": float(raw_singular[j]),
                        "machine_rank_retained": j < rank,
                        "resolved_against_proxy": bool(s > nf),
                        "resolved_against_physical_tolerance": bool(
                            raw_singular[j] > 1e-8
                        ),
                        "normalized_proxy_floor": nf,
                        "physical_proxy_floor": pf,
                    },
                )
            # Compare the seeded random span through a direct tall SVD, not QR.
            random = np.random.default_rng(4007 + qi).standard_normal(wr.shape)
            random -= u @ (u.T @ random)
            random -= u @ (u.T @ random)
            ar = np.linalg.svd(random, full_matrices=False)[0]
            close(ar - wr @ (wr.T @ ar), np.zeros_like(ar))
            ranks.append(
                {
                    "query": query,
                    "rank": rank,
                    "resolved_proxy": meta["resolved_against_proxy"],
                }
            )
            for path in design:
                if path["query"] != query:
                    continue
                file = (
                    ROOT
                    / spec["parent"]
                    / "01/nodes"
                    / f"path_{path['path']:03d}_node_1.json"
                )
                nm = array_unit(file)
                if nm["address"] != path:
                    raise ValueError("evaluation address mismatch")
                with np.load(file.with_suffix(".npz"), allow_pickle=False) as data:
                    v, l, d, r = (
                        data[k].astype(float).reshape(2, -1)
                        for k in ("parent_total", "link", "decoder", "roundoff")
                    )
                    dp = data["delta_p"].astype(float).ravel()
                calc = alternate(v, l, d, r, dp, u, w, wr)
                for ai, arm in enumerate(spec["arms"]):
                    head = {
                        k: path[k]
                        for k in ("query", "role", "wording", "path", "H", "B", "Q")
                    }
                    head.update(cell=4 * path["H"] + 2 * path["B"] + path["Q"], arm=arm)
                    row = {
                        **head,
                        **{
                            k: float(value[ai]) if value[ai] is not None else None
                            for k, value in calc.items()
                        },
                    }
                    row_close(observations[path["path"], arm], row)
                    close(
                        row["missing_u"],
                        2 * row["e_action"] + row["roundoff_complement"],
                    )
                    regenerated.append(row)
        pairs = table(out / "02/paired_actions.csv")
        indexed = {
            (r["query"], r["role"], int(r["cell"]), r["arm"], r["measure"]): r
            for r in pairs
        }
        grouped = {}
        for row in regenerated:
            key = tuple(row[k] for k in ("query", "role", "cell", "arm"))
            grouped.setdefault(key, {})[row["wording"]] = row
        rebuilt_pairs = []
        if len(pairs) != 192 or len(indexed) != 192:
            raise ValueError("incomplete mean/contrast pairs")
        for key, group in grouped.items():
            a, b = (group[w] for w in (0, 2))
            for measure in ("pair_mean", "wording_contrast"):
                sign = 1 if measure == "pair_mean" else -1
                row = dict(
                    zip(("query", "role", "cell", "arm"), key, strict=True),
                    measure=measure,
                )
                row.update({k: math.fsum((a[k], sign * b[k])) / 2 for k in FIELDS})
                row_close(indexed[(*key, measure)], row)
                rebuilt_pairs.append(row)
        metrics = table(out / "02/action_metrics.csv")
        if len(metrics) != 36:
            raise ValueError("all36moment groups required")
        for saved in metrics:
            data = (
                regenerated
                if saved["measure"] == "wording"
                else [r for r in rebuilt_pairs if r["measure"] == saved["measure"]]
            )
            rows = [
                r
                for r in data
                if all(r[k] == saved[k] for k in ("query", "role", "arm"))
            ]
            row_close(saved, scalar_moments(rows))
        cover = table(out / "02/coverage.csv")
        if len(cover) != 12:
            raise ValueError("all12coverage groups required")
        for saved in cover:
            rows = [
                r
                for r in regenerated
                if all(r[k] == saved[k] for k in ("query", "role", "arm"))
            ]
            calc = {"count": len(rows)}
            for kind in ("covector", "input"):
                den = math.fsum(r[kind + "_energy"] for r in rows) / len(rows)
                calc[kind + "_energy"] = den
                calc[kind + "_denominator_resolved"] = den > 1e-16
                for space in ("u", "w", "random"):
                    energy = math.fsum(
                        r[space + "_" + kind + "_energy"] for r in rows
                    ) / len(rows)
                    calc[space + "_" + kind + "_energy"] = energy
                    calc[space + "_" + kind + "_fraction"] = (
                        energy / den if den > 1e-16 else None
                    )
            row_close(saved, calc)
        for name, expected in (
            ("source_nodes_loaded", 48),
            ("source_nodes_evaluated", 48),
            ("application_nodes_loaded", 48),
            ("learned_bases", 3),
            ("random_bases", 3),
        ):
            if status[name] != expected:
                raise ValueError("scope counter changed " + name)
        elapsed = time.monotonic() - started
        if elapsed + status["analysis_seconds"] > 300:
            raise TimeoutError("combined CPU budget exceeded")
        write_json(
            dest / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM40_NOT_GLOBAL",
                "verified_at": now(),
                "ranks": ranks,
                "output_hashes": len(ledger),
                "ancestry_hashes": len(m["files"]),
                "numeric_values_checked": int(checked),
                "maximum_alternate_error": maximum,
                "verification_seconds": elapsed,
                "combined_analysis_seconds": elapsed + status["analysis_seconds"],
                "new_model_calls": 0,
                "observations": 192,
                "paired_actions": 192,
                "metric_groups": 36,
                "coverage_groups": 12,
            },
        )
        print(read_json(dest / "summary.json"), flush=True)
    except Exception as exc:
        write_json(
            dest / "failure.json",
            {
                "status": "VERIFICATION_FAILED_NOT_ACCEPTED",
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - started,
                "primary_unchanged": True,
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def main():
    """One bounded alternate numerical replay, no model calls or automatic retries."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dest", type=Path, default=DEST)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        verify(args.out, args.dest)


if __name__ == "__main__":
    main()
