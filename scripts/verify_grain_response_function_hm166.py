#!/usr/bin/env python3
"""Alternate raw-response and pairwise-moment replay; same agent, not blind validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_observed_text_anchor_hm164 as A
from scripts import run_grain_response_function_transport_hm166 as R
from scripts import run_grain_writer_history_bridge_hm165 as H


def raw_response(row, order):
    """Compute semantic response from native logits with an independent link formula."""
    ab = row["ab_logits"]
    ell = ab[order] - ab[1 - order]
    return np.array([(1 + math.tanh(ell / 2)) / 2, ell])


def pairwise_covariance(x, y, denominator):
    """Compute cross-moment from all pairs, without the primary centering implementation."""
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return np.einsum("ij...,ijf->...f", dy, dx) / denominator


def main():
    """Verify real source/application addressing, frozen moments and cold contractions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=R.DEFAULT)
    args = parser.parse_args()
    out, errors = args.out, {}
    manifest = R.check(out)
    R.artifact(out, "source_program_seal.json")
    R.artifact(out, "cold_seal.json")
    for name, expected in A.read(out / "analysis_seal.json").items():
        if A.digest(out / name) != expected:
            raise ValueError("Analysis changed before independent arithmetic replay")
    previous = ROOT / manifest["config"]["predecessor"]
    old_cap, old_req = H.results(previous), A.read(previous / "requests.json")
    old_lookup = {
        (r["pair_id"], r["history"], r["opportunity"], r["order"]): r
        for r in old_req
        if r["policy"] == 0 and r["history"] and r["unit_index"] in old_cap
    }
    source_articles = A.read(out / "source_articles.json")
    with np.load(out / "source.npz", allow_pickle=False) as s:
        x, y, groups = s["features"], s["responses"], s["groups"]
    errors["source_native"] = 0.0
    for i, a in enumerate(source_articles):
        for c, name in enumerate(manifest["config"]["opportunities"]):
            for o in (0, 1):
                req = old_lookup[a["pair_id"], a["pole"], name, o]
                expected = raw_response(old_cap[req["unit_index"]], o)
                errors["source_native"] = max(
                    errors["source_native"],
                    float(np.max(abs(y[i, :, c, o] - expected))),
                )
    with np.load(out / "source_program.npz", allow_pickle=False) as p:
        origin, coefficients = p["origin"], p["coefficients"]
    total = pairwise_covariance(x, y, 2 * len(x) ** 2)
    within = np.zeros_like(total)
    for key in np.unique(groups):
        ids = groups == key
        within += pairwise_covariance(x[ids], y[ids], 2 * len(x) * ids.sum())
    errors["total_pairwise"] = float(np.max(abs(coefficients[0] - total)))
    errors["within_pairwise"] = float(np.max(abs(coefficients[2] - within)))
    errors["between_pairwise"] = float(np.max(abs(coefficients[3] - (total - within))))
    for shift in range(1, 6):
        shuffled = y.copy()
        for key in np.unique(groups):
            ids = np.flatnonzero(groups == key)
            shuffled[ids] = y[[ids[(j - shift) % len(ids)] for j in range(len(ids))]]
        expected = pairwise_covariance(x, shuffled, 2 * len(x) ** 2)
        errors[f"relink{shift}_pairwise"] = float(
            np.max(abs(coefficients[3 + shift] - expected))
        )
    with (
        np.load(out / "application_features.npz", allow_pickle=False) as f,
        np.load(out / "cold_readings.npz", allow_pickle=False) as p,
    ):
        expected = np.tensordot(f["features"] - origin, coefficients, axes=([1], [-1]))
        errors["cold_contraction"] = float(np.max(abs(p["readings"] - expected)))
    cap, requests = R.captures(out), A.read(out / "requests.json")
    observed, _, complete, _ = R.application_responses(out)
    articles = A.read(out / "application_articles.json")
    article_index = {(a["pair_id"], a["pole"]): i for i, a in enumerate(articles)}
    errors["application_native"] = 0.0
    units = A.read(out / "units.json")
    draw_errors = 0
    for r in requests:
        if r["unit_index"] not in cap:
            continue
        row = cap[r["unit_index"]]
        c = manifest["config"]["opportunities"].index(r["opportunity"])
        i = article_index[r["pair_id"], r["history"]]
        errors["application_native"] = max(
            errors["application_native"],
            float(
                np.max(
                    abs(raw_response(row, r["order"]) - observed[i, :, c, r["order"]])
                )
            ),
        )
    for i, row in cap.items():
        key = hashlib.sha256(f"166|{units[i]['unit_id']}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        qa = float(raw_response(row, 0)[0])
        draw_errors += int(int(rng.random() < qa) != row["draw_a"])
        ids_hash = hashlib.sha256(
            np.asarray(units[i]["prompt_ids"], dtype="<i8").tobytes()
        ).hexdigest()
        if ids_hash != row["unit_id"] or row["unit_id"] != units[i]["unit_id"]:
            raise ValueError("Native response input hash mismatch")
    with np.load(out / "paired_accounts.npz", allow_pickle=False) as p:
        raw = np.array([observed[hi] - observed[lo] for _, hi, lo in complete])
        transformed = np.stack(
            ((raw[..., 0] + raw[..., 1]) * 0.5, (raw[..., 0] - raw[..., 1]) * 0.5), -1
        )
        errors["order_and_pair_account"] = float(
            np.max(abs(transformed - p["observed"]))
        )
    if max(errors.values()) > 1e-10 or draw_errors:
        raise ValueError("Alternate arithmetic does not reproduce the sealed account")
    audit = {
        "errors": errors,
        "draw_errors": draw_errors,
        "native_captures": len(cap),
        "complete_pairs": len(complete),
        "source_profiles": len(x),
        "scope": "same-agent alternate arithmetic;not independent model/human validation",
        "new_forwards": 0,
    }
    A.write(out / "alternate_audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
