#!/usr/bin/env python3
"""HM41 direct ordered contraction, inclusion-exclusion and every-table replay."""

from __future__ import annotations

import argparse
import itertools
import math
import signal
import sys
import time
from pathlib import Path

PROCESS_STARTED = time.monotonic()

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts.run_grain_math_position_support import array_unit, check, now
from scripts.run_grain_math_response import write_json
from scripts.run_grain_math_target_active_support import load
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/curvature_factor_transport_r1"
DEST = (
    ROOT / "results/grain_math_response_v1/curvature_factor_transport_verification_r1"
)
TERMS = tuple(f"m{i}" for i in range(8))
FIELDS = (
    "full",
    "link",
    "decoder",
    "roundoff",
    "source_mean",
    "source_paired",
    "pairing_loss",
    "delta_link",
    "error_mean",
    "error_paired",
    "error_C",
    "error_H",
    "error_P",
    "error_CH",
    "error_CP",
    "error_HP",
    "full_error_mean",
    "full_error_paired",
    *TERMS,
)


def read(parent, path):
    """Check exact original node identity before independent array loading."""
    file = parent / "01/nodes" / f"path_{path['path']:03d}_node_1.json"
    meta = array_unit(file)
    if meta["address"] != path or meta["node"] != 1 or meta["t"] != 0.5:
        raise ValueError("changed physical address")
    keys = (
        "hessian",
        "h_jacobian",
        "p_jacobian",
        "delta_h",
        "delta_p",
        "parent_total",
        "decoder",
        "roundoff",
        "link",
        "probabilities",
        "output_gradient",
    )
    with np.load(file.with_suffix(".npz"), allow_pickle=False) as a:
        return {k: a[k].astype(float) for k in keys}


def ordered(c, h, p, directions):
    """Contract C into the H-direction columns before applying the P-transpose."""
    hh, pp = h.reshape(len(c), -1), p.reshape(len(c), -1)
    dd = directions.reshape(len(directions), -1)
    return (c @ (hh @ dd.T)).T @ pp


def inclusion(cube):
    """Direct subset inclusion-exclusion, not the production fast transform."""
    return np.stack(
        [
            sum(
                ((-1) ** ((mask ^ sub).bit_count())) * cube[sub]
                for sub in range(8)
                if sub & mask == sub
            )
            for mask in range(8)
        ]
    )


def scalar_rows(a, cube, terms, paired):
    """Pair physical inputs before scalar reductions, independent of vector sums."""
    dp = a["delta_p"].ravel()
    vals = cube @ dp
    mts = terms @ dp
    ps = paired @ dp
    full, dec, rnd = [
        np.einsum("ap,p->a", a[k].reshape(2, -1), dp)
        for k in ("parent_total", "decoder", "roundoff")
    ]
    rows = []
    for i in range(2):
        r = {
            "full": float(full[i]),
            "link": float(vals[7, i]),
            "decoder": float(dec[i]),
            "roundoff": float(rnd[i]),
            "source_mean": float(vals[0, i]),
            "source_paired": float(ps[i]),
            "pairing_loss": float(ps[i] - vals[0, i]),
            "delta_link": float(vals[7, i] - vals[0, i]),
            "error_mean": float(vals[7, i] - vals[0, i]),
            "error_paired": float(vals[7, i] - ps[i]),
            "full_error_mean": float(full[i] - vals[0, i]),
            "full_error_paired": float(full[i] - ps[i]),
        }
        r.update({f"m{mask}": float(mts[mask, i]) for mask in range(8)})
        for name, mask in (
            ("C", 1),
            ("H", 2),
            ("P", 4),
            ("CH", 3),
            ("CP", 5),
            ("HP", 6),
        ):
            r["error_" + name] = float(vals[7, i] - vals[mask, i])
        rows.append(r)
    return rows, vals, mts


def moments(rows):
    """Use compensated scalar moments and preserve all45signed cross terms."""
    n, result = len(rows), {"count": len(rows)}
    for k in FIELDS:
        x = [r[k] for r in rows]
        result[k + "_mean"] = math.fsum(x) / n
        result[k + "_ms"] = math.fsum(t * t for t in x) / n
        result[k + "_rms"] = math.sqrt(result[k + "_ms"])
    components = (*TERMS, "decoder", "roundoff")
    for a, b in itertools.combinations(components, 2):
        result[a + "__" + b + "_cross"] = math.fsum(2 * r[a] * r[b] for r in rows) / n
    result["full_identity_error"] = max(
        abs(r["full"] - math.fsum(r[k] for k in components)) for r in rows
    )
    result["link_delta_identity_error"] = max(
        abs(r["delta_link"] - math.fsum(r[k] for k in TERMS[1:])) for r in rows
    )
    for v in ("paired", "C", "H", "P", "CH", "CP", "HP"):
        result["error_" + v + "_ms_reduction"] = (
            result["error_mean_ms"] - result["error_" + v + "_ms"]
        )
    return result


def descriptors(a):
    """Recompute actual state descriptors from the original saved float32 values."""
    p, c, g = (a[k] for k in ("probabilities", "hessian", "output_gradient"))
    norm = lambda x: math.sqrt(float(np.einsum("i,i->", x.ravel(), x.ravel())))
    ev = np.linalg.eigvalsh((c + c.T) / 2)
    r = {
        "entropy": -math.fsum(float(x) * math.log(float(x)) for x in p if x > 0),
        "top_probability": float(max(p)),
        "probability_sum": math.fsum(p),
        "g_norm": norm(g),
        "g_gauge_sum": math.fsum(g),
        "C_norm": norm(c),
        "C_gauge_norm": norm(np.sum(c, axis=1)),
        "C_asymmetry": float(np.max(abs(c - c.T))),
        "J_H_norm": norm(a["h_jacobian"]),
        "J_P_norm": norm(a["p_jacobian"]),
    }
    r.update({f"C_eigen_{i}": float(x) for i, x in enumerate(ev)})
    r.update({f"H_norm_{i}": norm(x) for i, x in enumerate(a["delta_h"])})
    return r


def verify(out, dest):
    """Verify all source triples, crossed arrays, paired rows and metric tables."""
    if dest.exists():
        raise FileExistsError("preserve every verification result")
    status = read_json(out / "run_status.json")
    if (
        status["status"] != "EVALUATED"
        or status["new_model_calls"]
        or status["new_fits"]
    ):
        raise ValueError("complete CPU-only result required")
    remaining = 300 - status["analysis_seconds"] - (time.monotonic() - PROCESS_STARTED)
    if remaining <= 0:
        raise TimeoutError("no original budget remains")
    dest.mkdir(parents=True)
    old = signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("fixed CPU cap"))
    )
    signal.setitimer(signal.ITIMER_REAL, remaining)
    maximum, checked = 0.0, 0

    def close(a, b):
        nonlocal maximum, checked
        x, y = np.asarray(a, float), np.asarray(b, float)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("finite aligned numerical replay required")
        err = float(np.max(abs(x - y))) if x.size else 0.0
        maximum, checked = max(maximum, err), checked + x.size
        if err > 1e-8:
            raise ValueError("unchanged physical replay tolerance exceeded " + str(err))

    def row_close(saved, calc):
        for k, value in calc.items():
            if isinstance(value, str):
                if saved[k] != value:
                    raise ValueError("changed address " + k)
            elif isinstance(value, bool):
                if saved[k] != str(value):
                    raise ValueError("changed diagnostic label")
            else:
                close(float(saved[k]), value)

    try:
        m = check(out)
        spec, design = m["spec"], read_json(out / "00/design.json")
        parent = ROOT / spec["parent"]
        ledger = read_json(out / "02/artifact_hashes.json")
        for rel, sha in ledger.items():
            if digest(out / rel) != sha:
                raise ValueError("changed output " + rel)
        barrier = read_json(out / "01/source_barrier.json")
        if (
            barrier["source_nodes"],
            barrier["source_bundles"],
            barrier["application_nodes_loaded"],
        ) != (48, 24, 0):
            raise ValueError("source-first construction required")
        for rel, sha in barrier["files"].items():
            if digest(out / rel) != sha:
                raise ValueError("source factor seal changed")
        tables = {
            name: table(out / "02" / f"{name}.csv")
            for name in (
                "observations",
                "crossings",
                "paired_actions",
                "action_metrics",
                "factor_descriptors",
            )
        }
        if [len(tables[k]) for k in tables] != [192, 1536, 192, 36, 96]:
            raise ValueError("complete tables required")
        observations = {(int(r["path"]), r["arm"]): r for r in tables["observations"]}
        crossings = {
            (int(r["path"]), r["arm"], int(r["mask"])): r for r in tables["crossings"]
        }
        descriptions = {int(r["path"]): r for r in tables["factor_descriptors"]}
        if (len(observations), len(crossings), len(descriptions)) != (192, 1536, 96):
            raise ValueError("unique addresses required")
        regenerated = []
        for query in spec["queries"]:
            refs = {}
            for name in sorted(
                {
                    f"{query}_cell_{p['H']}{p['B']}{p['Q']}"
                    for p in design
                    if p["query"] == query
                }
            ):
                meta, ref = load(out / "01/models" / f"{name}.json")
                paths = sorted(
                    [
                        p
                        for p in design
                        if p["query"] == query
                        and p["role"] == "source"
                        and f"{query}_cell_{p['H']}{p['B']}{p['Q']}" == name
                    ],
                    key=lambda p: p["wording"],
                )
                if (
                    meta["paths"] != [p["path"] for p in paths]
                    or meta["application_nodes_loaded"] != 0
                ):
                    raise ValueError("source identity changed")
                for i, p in enumerate(paths):
                    a = read(parent, p)
                    for key, arr in (
                        ("c", "hessian"),
                        ("h", "h_jacobian"),
                        ("p", "p_jacobian"),
                    ):
                        close(
                            ref["source_" + key][i],
                            a[arr].reshape(ref["source_" + key][i].shape),
                        )
                for key in ("c", "h", "p"):
                    close(
                        ref[key],
                        (
                            ref["source_" + key][0].astype(float)
                            + ref["source_" + key][1].astype(float)
                        )
                        * 0.5,
                    )
                refs[name] = ref
            for path in [p for p in design if p["query"] == query]:
                name = f"{query}_cell_{path['H']}{path['B']}{path['Q']}"
                ref, a = refs[name], read(parent, path)
                src = tuple(ref[k] for k in ("c", "h", "p"))
                act = tuple(a[k] for k in ("hessian", "h_jacobian", "p_jacobian"))
                cube = np.stack(
                    [
                        ordered(
                            *(act[j] if mask & (1 << j) else src[j] for j in range(3)),
                            a["delta_h"],
                        )
                        for mask in range(8)
                    ]
                )
                terms = inclusion(cube)
                paired = (
                    sum(
                        ordered(
                            *(ref["source_" + k][i] for k in ("c", "h", "p")),
                            a["delta_h"],
                        )
                        for i in range(2)
                    )
                    * 0.5
                )
                nm, bank = load(out / "02/nodes" / f"path_{path['path']:03d}.json")
                if nm["address"] != path or nm["source_cell"] != name:
                    raise ValueError("node source binding changed")
                if nm["source_barrier_sha256"] != digest(
                    out / "01/source_barrier.json"
                ):
                    raise ValueError("node predates source seal")
                close(bank["crossed"], cube)
                close(bank["paired_source"], paired)
                close(cube[7], a["link"].reshape(2, -1))
                close(np.sum(terms, axis=0), cube[7])
                scalar, vals, mts = scalar_rows(a, cube, terms, paired)
                head = {
                    k: path[k]
                    for k in ("path", "query", "role", "wording", "H", "B", "Q")
                }
                row_close(descriptions[path["path"]], {**head, **descriptors(a)})
                for i, arm in enumerate(spec["arms"]):
                    r = {**head, "arm": arm, **scalar[i]}
                    regenerated.append(r)
                    row_close(observations[(path["path"], arm)], r)
                    for mask in range(8):
                        row_close(
                            crossings[(path["path"], arm, mask)],
                            {
                                **head,
                                "arm": arm,
                                "mask": mask,
                                "is_local_factor_diagnostic": bool(mask),
                                "crossed_action": float(vals[mask, i]),
                                "mobius_action": float(mts[mask, i]),
                                "direct_delta_action": float(mts[mask, i]),
                            },
                        )
        keys = ("query", "role", "H", "B", "Q", "arm")
        paired = []
        saved_pairs = {
            tuple(r[k] for k in (*keys, "measure")): r for r in tables["paired_actions"]
        }
        for key in sorted({tuple(r[k] for k in keys) for r in regenerated}):
            rr = sorted(
                [r for r in regenerated if tuple(r[k] for k in keys) == key],
                key=lambda r: r["wording"],
            )
            for measure, sign in (("mean", 1), ("contrast", -1)):
                r = {
                    **dict(zip(keys, key, strict=True)),
                    "measure": measure,
                    **{k: math.fsum((rr[0][k], sign * rr[1][k])) * 0.5 for k in FIELDS},
                }
                skey = tuple(str(r[k]) for k in (*keys, "measure"))
                row_close(saved_pairs[skey], r)
                paired.append(r)
        for saved in tables["action_metrics"]:
            source = (
                regenerated
                if saved["measure"] == "word"
                else [r for r in paired if r["measure"] == saved["measure"]]
            )
            rr = [
                r
                for r in source
                if all(r[k] == saved[k] for k in ("query", "role", "arm"))
            ]
            row_close(saved, moments(rr))
        for k, v in (
            ("source_nodes_loaded", 48),
            ("source_nodes_evaluated", 48),
            ("application_nodes_loaded", 48),
            ("source_bundles", 24),
        ):
            if status[k] != v:
                raise ValueError("scope counter changed " + k)
        elapsed = time.monotonic() - PROCESS_STARTED
        if elapsed + status["analysis_seconds"] > 300:
            raise TimeoutError("combined CPU budget exceeded")
        write_json(
            dest / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM41_NOT_GLOBAL",
                "verified_at": now(),
                "output_hashes": len(ledger),
                "ancestry_hashes": len(m["files"]),
                "numeric_values_checked": int(checked),
                "maximum_alternate_error": maximum,
                "verification_seconds": elapsed,
                "combined_analysis_seconds": elapsed + status["analysis_seconds"],
                "nodes": 96,
                "observations": 192,
                "crossings": 1536,
                "paired_actions": 192,
                "metric_groups": 36,
                "source_bundles": 24,
                "new_model_calls": 0,
                "new_fits": 0,
            },
        )
        print(read_json(dest / "summary.json"), flush=True)
    except Exception as exc:
        write_json(
            dest / "failure.json",
            {
                "status": "VERIFICATION_FAILED_NOT_ACCEPTED",
                "error": repr(exc),
                "elapsed_seconds": time.monotonic() - PROCESS_STARTED,
                "primary_unchanged": True,
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def main():
    """One complete finite alternate replay, without model requests or retries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        verify(args.out, args.dest)


if __name__ == "__main__":
    main()
