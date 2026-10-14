#!/usr/bin/env python3
"""Same-agent alternative finite accounts for HM167; no neural forward or fit."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_context_response_hm167 as R


def maximum(a, b):
    """Compare equally shaped finite arithmetic without fitting scale or signs."""
    return float(np.max(abs(np.asarray(a) - np.asarray(b))))


def verify(out):
    """Reconstruct captures, all functions and context accounts independently."""
    from transformers import AutoTokenizer

    manifest = R.check(out)
    parent = ROOT / manifest["config"]["predecessor"]
    for name, digest in R.A.read(out / "analysis_seal.json").items():
        if R.A.digest(out / name) != digest:
            raise ValueError("New analysis changed: " + name)
    R.R.artifact(out, "source_program_seal.json")
    R.R.artifact(out, "cold_seal.json")
    old_requests = R.A.read(
        ROOT / manifest["config"]["history_source"] / "requests.json"
    )
    old_by_index = {
        r["unit_index"]: r
        for r in old_requests
        if r["policy"] == 0 and r["history"] != 0
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(R.H.OLD.MODEL), local_files_only=True, trust_remote_code=False
    )
    profiles, errors, addresses, new_count, draw_errors = {}, {}, 0, 0, 0
    for stage in ("source", "application"):
        units = R.A.read(out / (stage + "_units.json"))
        articles = R.A.read(out / (stage + "_articles.json"))
        captures = R.captures(out, stage)
        y = np.full((len(articles), 2, 4, 2), np.nan)
        for address in R.A.read(out / (stage + "_addresses.json")):
            prior = old_by_index[address["original_unit_index"]]
            messages = [m.copy() for m in prior["messages"]]
            assert (
                hashlib.sha256(messages[2]["content"].encode()).hexdigest()
                == address["body_sha256"]
            )
            messages[1]["content"] = manifest["common_task"]
            ids = R.H.M.rendered_token_ids(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            unit = units[address["unit_index"]]
            assert ids == unit["prompt_ids"]
            assert (
                hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
                == unit["unit_id"]
            )
            addresses += 1
            if address["unit_index"] not in captures:
                continue
            row = captures[address["unit_index"]]
            a, b = row["ab_logits"]
            ell = (a - b) * (1 if address["order"] == 0 else -1)
            q = (1 + np.tanh(ell / 2)) / 2
            y[address["article"], :, address["opportunity"], address["order"]] = q, ell
        for i, row in captures.items():
            if units[i]["reuse_old_unit"] is None:
                a, b = row["ab_logits"]
                q = (1 + np.tanh((a - b) / 2)) / 2
                key = hashlib.sha256(
                    f"{manifest['config']['draw_seed']}|{row['unit_id']}".encode()
                ).digest()
                rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
                draw_errors += int(int(rng.random() < q) != row["draw_a"])
                new_count += 1
        profiles[stage] = y
        native, _ = R.profiles(out, stage)
        finite = np.isfinite(native)
        errors[stage + "_raw_tanh"] = maximum(y[finite], native[finite])
    with np.load(out / "source.npz", allow_pickle=False) as saved:
        x, old_y, groups = saved["features"], saved["responses"], saved["groups"]
    source_y = profiles["source"]

    def components(y):
        """Alternative second-moment formula, unlike the primary centered contraction."""
        full = np.einsum("n...,nf->...f", y, x) / len(x) - np.einsum(
            "...,f->...f", y.mean(0), x.mean(0)
        )
        within = np.zeros_like(full)
        for key in np.unique(groups):
            ids = groups == key
            local = np.einsum("n...,nf->...f", y[ids], x[ids]) / ids.sum() - np.einsum(
                "...,f->...f", y[ids].mean(0), x[ids].mean(0)
            )
            within += local * ids.sum() / len(x)
        return full, within, full - within

    full, within, between = components(source_y)
    common = [
        full,
        full * np.array([s.bit_count() <= 1 for s in R.R.S.feature_sectors()]),
        within,
        between,
    ]
    for shift in range(1, 6):
        ids = np.arange(len(x))
        for g in np.unique(groups):
            block = np.flatnonzero(groups == g)
            for offset, index in enumerate(block):
                ids[index] = block[(offset - shift) % len(block)]
        common.append(components(source_y[ids])[0])
    with np.load(parent / "source_program.npz", allow_pickle=False) as original:
        old_coeff = original["coefficients"]
    delta = np.array(common)[[0, 2, 3]] - old_coeff[[0, 2, 3]]
    alternate = np.concatenate((old_coeff, common, delta))
    with np.load(out / "source_program.npz", allow_pickle=False) as program:
        coefficients = program["coefficients"]
        errors["all21_source_functions"] = maximum(alternate, coefficients)
        errors["delta_covariance"] = maximum(delta[0], components(source_y - old_y)[0])
        with np.load(out / "application_features.npz", allow_pickle=False) as app:
            cold = np.tensordot(
                app["features"] - program["origin"], coefficients, axes=([-1], [-1])
            )
    with np.load(out / "cold_readings.npz", allow_pickle=False) as saved:
        errors["numeric_cold"] = maximum(cold, saved["readings"])
    original, _, _, _ = R.R.application_responses(parent)
    articles = R.A.read(out / "application_articles.json")
    indices = {(a["pair_id"], a["pole"]): i for i, a in enumerate(articles)}
    accounts = list(csv.DictReader((out / "context_pair_accounts.csv").open()))
    account_error = 0.0
    for row in accounts:
        lo, hi = indices[row["pair_id"], -1], indices[row["pair_id"], 1]
        c, o = manifest["opportunities"].index(row["opportunity"]), int(row["order"])
        a, b = original[[lo, hi], 1, c, o], profiles["application"][[lo, hi], 1, c, o]
        old_gap, new_gap = a[1] - a[0], b[1] - b[0]
        expected = {
            "joint_character": (new_gap - old_gap) / 4,
            "body_character": (new_gap + old_gap) / 4,
            "context_character": ((b[0] - a[0]) + (b[1] - a[1])) / 4,
            "log_odds_gap_change": new_gap - old_gap,
        }
        qa, qb = 1 / (1 + np.exp(-a)), 1 / (1 + np.exp(-b))
        offset = (b - a).sum() / 2
        qo = 1 / (1 + np.exp(-a - offset))
        expected.update(
            {
                "probability_gap_change": (qb[1] - qb[0]) - (qa[1] - qa[0]),
                "offset_only_probability_gap_change": (qo[1] - qo[0]) - (qa[1] - qa[0]),
                "joint_probability_remainder": (qb[1] - qb[0]) - (qo[1] - qo[0]),
                "body_only_two_context_floor": ((qb - qa) @ (qb - qa)) / 8,
            }
        )
        for key, value in expected.items():
            account_error = max(account_error, abs(float(row[key]) - value))
    errors["all_context_accounts"] = float(account_error)
    if max(errors.values()) > 1e-10 or draw_errors:
        raise ValueError({"errors": errors, "draw_errors": draw_errors})
    summary = {
        "errors": errors,
        "same_body_context_addresses": addresses,
        "new_native_captures": new_count,
        "context_pair_order_accounts": len(accounts),
        "draw_errors": draw_errors,
        "new_forwards": 0,
        "scope": "same-agent alternative arithmetic;not blind replication",
    }
    R.A.write(out / "alternate_audit.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    verify(R.DEFAULT)
