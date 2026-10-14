#!/usr/bin/env python3
"""HM38 alternate centered-sum verification from immutable HM37 parent values."""

from __future__ import annotations

import argparse
import itertools
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.vector_mechanisms.navigation import digest, read_json
from scripts import run_grain_math_target_active_support as SUPPORT
from scripts.run_grain_math_position_support import check, now
from scripts.run_grain_math_response import write_json
from scripts.verify_grain_math_crossed_field import table

OUT = ROOT / "results/grain_math_response_v1/active_reading_pairing_r1"
DEST = ROOT / "results/grain_math_response_v1/active_reading_pairing_verification_r1"
PAIR = ("carrier", "shared_transport", "coupling")
WORD = (*PAIR, "coefficient_contrast", "input_contrast")


def centered(theta, eta, full, reference):
    """Use centered products, not the production two-half-difference formula."""
    tm, em = np.average(theta, axis=0), np.average(eta, axis=0)
    tc, ec = theta - tm, eta - em
    supported = np.array(
        [
            math.fsum(float(x * y) for x, y in zip(t, e, strict=True))
            for t, e in zip(theta, eta, strict=True)
        ]
    )
    predicted = eta @ reference
    covariance = sum(np.outer(t, e) for t, e in zip(tc, ec, strict=True)) / 2
    common, coupling = float(tm @ em), float(np.trace(covariance))
    shift = tm - reference
    transport = float(shift @ em)
    pair = {
        "actual": float(np.average(full)),
        "supported": float(np.average(supported)),
        "predicted": float(np.average(predicted)),
        "common": common,
        "coupling": coupling,
        "carrier": float(np.average(full - supported)),
        "shared_transport": transport,
        "error": float(np.average(full - predicted)),
        "supported_contrast": float((supported[0] - supported[1]) / 2),
        "reading_contrast": float(
            ((supported[0] - predicted[0]) - (supported[1] - predicted[1])) / 2
        ),
        "coefficient_contrast": float(tc[0] @ em),
        "input_contrast": float(shift @ ec[0]),
        "covariance_frobenius": float(np.sqrt(np.sum(covariance * covariance))),
        "pairing_identity_error": abs(float(np.average(supported)) - common - coupling),
    }
    pair["coupling_direction_cosine"] = (
        coupling / pair["covariance_frobenius"]
        if pair["covariance_frobenius"]
        else None
    )
    pair["error_identity_error"] = abs(pair["error"] - sum(pair[k] for k in PAIR))
    words = []
    for i in range(2):
        word = {
            "actual": float(full[i]),
            "supported": float(supported[i]),
            "predicted": float(predicted[i]),
            "carrier": float(full[i] - supported[i]),
            "shared_transport": transport,
            "coupling": coupling,
            "coefficient_contrast": float(tc[i] @ em),
            "input_contrast": float(shift @ ec[i]),
            "error": float(full[i] - predicted[i]),
        }
        word["error_identity_error"] = abs(word["error"] - sum(word[k] for k in WORD))
        word["supported_identity_error"] = abs(
            word["supported"] - common - coupling - float(tc[i] @ em + tm @ ec[i])
        )
        words.append(word)
    return pair, words


def metrics(rows, components):
    """Recalculate all component moments by independent scalar summation."""
    n = len(rows)
    actual_ms = math.fsum(r["actual"] ** 2 for r in rows) / n
    error_ms = math.fsum(r["error"] ** 2 for r in rows) / n
    result = {
        "count": n,
        "actual_rms": math.sqrt(actual_ms),
        "error_rms": math.sqrt(error_ms),
        "relative_error": math.sqrt(error_ms / actual_ms) if actual_ms else None,
    }
    rebuilt = 0.0
    for k in components:
        ms = math.fsum(r[k] ** 2 for r in rows) / n
        result[k + "_rms"] = math.sqrt(ms)
        result[k + "_mean"] = math.fsum(r[k] for r in rows) / n
        rebuilt += ms
    for a, b in itertools.combinations(components, 2):
        cross = 2 * math.fsum(r[a] * r[b] for r in rows) / n
        result[a + "__" + b + "_cross"] = cross
        rebuilt += cross
    result["second_moment_identity_error"] = abs(error_ms - rebuilt)
    return result


def verify(out, destination):
    """Replay every pair, wording and half integral with the same fixed tolerance."""
    started = time.monotonic()
    if destination.exists():
        raise FileExistsError("preserve previous verification")
    destination.mkdir(parents=True)
    status = read_json(out / "run_status.json")
    spec = read_json(out / "00/manifest.json")["spec"]
    remaining = spec["maximum_analysis_seconds"] - status["analysis_seconds"]
    if remaining <= 0 or status["status"] != "EVALUATED":
        raise ValueError("evaluated diagnostic with remaining combined budget required")

    def stop(_signal, _frame):
        raise TimeoutError("combined HM38 analysis/verification budget reached")

    signal.signal(signal.SIGALRM, stop)
    signal.alarm(max(1, math.floor(remaining)))
    maximum, checked = 0.0, 0

    def close(a, b):
        nonlocal maximum, checked
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite or wrong-shaped reconstruction")
        error = float(np.max(abs(a - b))) if a.size else 0.0
        maximum = max(maximum, error)
        checked += a.size
        if error > 1e-8:
            raise ValueError(f"original numerical tolerance failed: {error}")

    def row_close(row, expected):
        for key, value in expected.items():
            if value is None:
                if row[key] not in ("", None):
                    raise ValueError("null ratio was not preserved")
            elif isinstance(value, str):
                if row[key] != value:
                    raise ValueError("wrong address " + key)
            else:
                close(float(row[key]), value)

    def indexed(name, keys):
        rows = table(out / "02" / f"{name}.csv")
        result = {tuple(r[k] for k in keys): r for r in rows}
        if len(rows) != len(result):
            raise ValueError("duplicate observation " + name)
        return result

    try:
        check(out)
        hashes = read_json(out / "02/artifact_hashes.json")
        for name, sha in hashes.items():
            if digest(out / name) != sha:
                raise ValueError("changed artifact " + name)
        parent = ROOT / spec["parent"]
        design = read_json(out / "00/design.json")
        groups = {}
        for path in design:
            key = path["query"], path["role"], path["H"], path["B"], path["Q"]
            groups.setdefault(key, []).append(path)
        pair_nodes = indexed("paired_nodes", ("group", "node", "arm"))
        pair_integrals = indexed("paired_integrals", ("group", "rule", "arm"))
        word_nodes = indexed("wording_nodes", ("group", "node", "arm", "wording"))
        word_integrals = indexed(
            "wording_integrals", ("group", "rule", "arm", "wording")
        )
        if tuple(
            map(len, (pair_nodes, pair_integrals, word_nodes, word_integrals))
        ) != (288, 192, 576, 384):
            raise ValueError("incomplete fixed diagnostic output")
        parent_nodes = {
            (int(r["path"]), int(r["node"]), r["arm"]): r
            for r in table(parent / "02/nodes.csv")
            if r["law"] == "full"
        }
        parent_integrals = {
            (int(r["path"]), r["rule"], r["arm"]): r
            for r in table(parent / "02/integrals.csv")
            if r["law"] == "full"
        }
        provenance = {int(r["path"]): r for r in table(out / "01/provenance.csv")}
        if len(groups) != 48 or len(provenance) != 96:
            raise ValueError("all48original pairs and96provenance addresses required")
        regenerated = {
            k: []
            for k in ("pair_nodes", "pair_integrals", "word_nodes", "word_integrals")
        }
        for key, paths in groups.items():
            query, role, h, b, q = key
            paths.sort(key=lambda p: p["wording"])
            if [p["wording"] for p in paths] != [0, 2]:
                raise ValueError("incorrect pairing")
            cell = 4 * h + 2 * b + q
            group = f"{query}_{role}_cell_{cell}"
            meta, saved = SUPPORT.load(out / "01/arrays" / f"{group}.json")
            address = {
                "group": group,
                "query": query,
                "role": role,
                "cell": cell,
                "H": h,
                "B": b,
                "Q": q,
                "pair": paths[0]["pair"],
            }
            if meta["address"] != address or meta["paths"] != [
                p["path"] for p in paths
            ]:
                raise ValueError("stored pair address drift")
            _, model = SUPPORT.load(parent / "01/models" / f"{query}.json")
            ref = model["laws"][3, cell]
            close(saved["reference"], ref)
            values = []
            for wi, path in enumerate(paths):
                pf = parent / "01/predictions" / f"path_{path['path']:03d}.json"
                of = parent / "02/observations" / pf.name
                pm, pred = SUPPORT.load(pf)
                om, obs = SUPPORT.load(of)
                if pm["address"] != path or om["address"] != path:
                    raise ValueError("parent address mismatch")
                row_close(
                    provenance[path["path"]],
                    {
                        "role": role,
                        "query": query,
                        "cell": cell,
                        "wording": path["wording"],
                        "prediction_file": str(pf.relative_to(ROOT)),
                        "prediction_sha256": digest(pf),
                        "observation_file": str(of.relative_to(ROOT)),
                        "observation_sha256": digest(of),
                    },
                )
                n = pred["h_norms"]
                theta = obs["coefficients"] / n[..., None]
                eta = n[..., None] * pred["z"][:, None, :]
                for name, value in {
                    "theta": theta,
                    "eta": eta,
                    "full": obs["actual"],
                    "supported": obs["supported"],
                    "predicted": pred["predicted"][:, 3],
                    "norms": n,
                    "direction_error": pred["h_direction_error"],
                }.items():
                    close(saved[name][:, :, wi], value)
                values.append((theta, eta, obs["actual"], pred))
            for ai, arm in enumerate(spec["arms"]):
                pr, wr = [], [[], []]
                for node in range(3):
                    theta = np.stack([v[0][node, ai] for v in values])
                    eta = np.stack([v[1][node, ai] for v in values])
                    full = np.array([v[2][node, ai] for v in values])
                    pair, words = centered(theta, eta, full, ref[node, ai])
                    row = {**address, "arm": arm, "node": node, "t": node / 2, **pair}
                    row_close(pair_nodes[group, str(node), arm], row)
                    pr.append(row)
                    regenerated["pair_nodes"].append(row)
                    for wi, word in enumerate(words):
                        prediction = values[wi][3]
                        row = {
                            **address,
                            "arm": arm,
                            "node": node,
                            "t": node / 2,
                            "wording": paths[wi]["wording"],
                            "path": paths[wi]["path"],
                            "H_norm": prediction["h_norms"][node, ai],
                            "H_direction_difference": prediction["h_direction_error"][
                                node, ai
                            ],
                            **word,
                        }
                        row_close(
                            word_nodes[
                                group, str(node), arm, str(paths[wi]["wording"])
                            ],
                            row,
                        )
                        original = parent_nodes[paths[wi]["path"], node, arm]
                        row_close(
                            original,
                            {k: word[k] for k in ("actual", "supported", "predicted")},
                        )
                        close(
                            word["error"],
                            float(original["actual"]) - float(original["predicted"]),
                        )
                        wr[wi].append(row)
                        regenerated["word_nodes"].append(row)
                pair_keys = (
                    "actual",
                    "supported",
                    "predicted",
                    "common",
                    *PAIR,
                    "error",
                    "supported_contrast",
                    "reading_contrast",
                    "coefficient_contrast",
                    "input_contrast",
                )
                word_keys = ("actual", "supported", "predicted", *WORD, "error")
                for rule in spec["rules"]:
                    weights = (1, 4, 1) if rule == "simpson" else (1, 2, 1)
                    denominator = 12 if rule == "simpson" else 8
                    row = {
                        k: math.fsum(w * r[k] for w, r in zip(weights, pr, strict=True))
                        / denominator
                        for k in pair_keys
                    }
                    row["pairing_identity_error"] = abs(
                        row["supported"] - row["common"] - row["coupling"]
                    )
                    row["error_identity_error"] = abs(
                        row["error"] - sum(row[k] for k in PAIR)
                    )
                    row.update(address, arm=arm, rule=rule)
                    row_close(pair_integrals[group, rule, arm], row)
                    regenerated["pair_integrals"].append(row)
                    for wi in range(2):
                        row = {
                            k: math.fsum(
                                w * r[k] for w, r in zip(weights, wr[wi], strict=True)
                            )
                            / denominator
                            for k in word_keys
                        }
                        row["error_identity_error"] = abs(
                            row["error"] - sum(row[k] for k in WORD)
                        )
                        row.update(
                            address,
                            arm=arm,
                            rule=rule,
                            wording=paths[wi]["wording"],
                            path=paths[wi]["path"],
                        )
                        row_close(
                            word_integrals[group, rule, arm, str(paths[wi]["wording"])],
                            row,
                        )
                        original = parent_integrals[paths[wi]["path"], rule, arm]
                        row_close(
                            original,
                            {k: row[k] for k in ("actual", "supported", "predicted")},
                        )
                        close(
                            row["error"],
                            float(original["actual"]) - float(original["predicted"]),
                        )
                        regenerated["word_integrals"].append(row)
        ms = table(out / "02/metrics.csv")
        if len(ms) != 72:
            raise ValueError("incomplete moment table")
        seen = set()
        for metric in ms:
            tag = tuple(metric[k] for k in ("view", "role", "query", "arm", "scale"))
            if tag in seen:
                raise ValueError("duplicate metric group")
            seen.add(tag)
            view = "pair" if metric["view"] == "pair" else "word"
            rows = (
                regenerated[view + "_nodes"]
                if metric["scale"] == "nodes"
                else [
                    r
                    for r in regenerated[view + "_integrals"]
                    if r["rule"] == metric["scale"]
                ]
            )
            selected = [
                r
                for r in rows
                if all(r[k] == metric[k] for k in ("role", "query", "arm"))
            ]
            row_close(metric, metrics(selected, PAIR if view == "pair" else WORD))
        elapsed = time.monotonic() - started
        combined = status["analysis_seconds"] + elapsed
        if combined > spec["maximum_analysis_seconds"]:
            raise TimeoutError("combined diagnostic budget exceeded")
        write_json(
            destination / "summary.json",
            {
                "status": "VERIFIED_FINITE_HM38_NOT_GLOBAL",
                "verified_at": now(),
                "manifest_sha256": digest(out / "00/manifest.json"),
                "artifact_ledger_sha256": digest(out / "02/artifact_hashes.json"),
                "hashes_checked": len(hashes),
                "numerical_values_checked": checked,
                "maximum_absolute_error": maximum,
                "pair_groups": len(groups),
                "row_counts": {k: len(v) for k, v in regenerated.items()},
                "metric_groups": len(ms),
                "verification_seconds": elapsed,
                "combined_analysis_seconds": combined,
                "new_fits": 0,
                "new_model_calls": 0,
                "scope": spec["scope"],
            },
        )
        print(read_json(destination / "summary.json"), flush=True)
    except Exception as error:
        write_json(
            destination / "failure.json",
            {
                "at": now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "verification_seconds": time.monotonic() - started,
                "new_fits": 0,
                "new_model_calls": 0,
            },
        )
        raise
    finally:
        signal.alarm(0)


def main():
    """Verify one immutable result without automatic overwrite or replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--destination", type=Path, default=DEST)
    args = parser.parse_args()
    verify(args.out, args.destination)


if __name__ == "__main__":
    main()
