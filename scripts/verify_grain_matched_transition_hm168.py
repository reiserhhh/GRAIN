#!/usr/bin/env python3
"""Alternate HM168 moment/action arithmetic; no neural forward or scientific rewrite."""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_grain_matched_transition_hm168 as R


def verify(out):
    """Read sealed captures and replay every registered operation and numeric score."""
    from transformers import AutoTokenizer

    manifest = R.check(out)
    for name, expected in R.A.read(out / "analysis_seal.json").items():
        if R.A.digest(out / name) != expected:
            raise ValueError("Analysis changed")
    for seal in ("source_program_seal.json", "cold_seal.json"):
        R.R.artifact(out, seal)
    tokenizer = AutoTokenizer.from_pretrained(
        str(R.H.OLD.MODEL), local_files_only=True, trust_remote_code=False
    )
    requests = R.A.read(ROOT / manifest["config"]["history_source"] / "requests.json")
    old = {
        r["unit_index"]: r for r in requests if r["policy"] == 0 and r["history"] != 0
    }
    replay, operations, addresses, draws, new = {}, {}, 0, 0, 0
    for stage in ("source", "application"):
        articles = R.A.read(out / (stage + "_articles.json"))
        units = R.A.read(out / (stage + "_units.json"))
        captured = R.P.captures(out, stage)
        y = np.full((len(articles), 2, 4, 2), np.nan)
        keys = []
        for a in R.A.read(out / (stage + "_addresses.json")):
            request = old[a["original_unit_index"]]
            messages = [m.copy() for m in request["messages"]]
            assert (
                hashlib.sha256(messages[2]["content"].encode()).hexdigest()
                == a["body_sha256"]
            )
            operation = R.T.transition_address(
                messages, manifest["common_task"], manifest["destination_task"]
            )
            assert operation == a["operation_id"]
            keys.append(operation)
            messages[1]["content"] = manifest["common_task"]
            ids = R.H.M.rendered_token_ids(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            unit = units[a["unit_index"]]
            assert ids == unit["prompt_ids"]
            assert (
                hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
                == unit["unit_id"]
            )
            addresses += 1
            if a["unit_index"] in captured:
                row = captured[a["unit_index"]]
                ell = (row["ab_logits"][0] - row["ab_logits"][1]) * (
                    1 if a["order"] == 0 else -1
                )
                y[a["article"], :, a["opportunity"], a["order"]] = (
                    (1 + np.tanh(ell / 2)) / 2,
                    ell,
                )
        operations[stage], replay[stage] = keys, y
        for i, row in captured.items():
            if units[i]["reuse_old_unit"] is None:
                q = (1 + np.tanh((row["ab_logits"][0] - row["ab_logits"][1]) / 2)) / 2
                seed = hashlib.sha256(
                    f"{manifest['config']['draw_seed']}|{row['unit_id']}".encode()
                ).digest()
                draw = int(
                    np.random.default_rng(int.from_bytes(seed[:8], "little")).random()
                    < q
                )
                draws += draw != row["draw_a"]
                new += 1
    assert (
        R.T.support_account(operations["source"], operations["application"])[
            "unsupported_application_units"
        ]
        == 0
    )
    with np.load(out / "source.npz", allow_pickle=False) as data:
        x, end, g = data["features"], data["responses"], data["groups"]
    action = end - replay["source"]

    def moments(y):
        """Use uncentered second moments, unlike the primary centered products."""
        full = np.einsum("n...,nf->...f", y, x) / len(x) - np.einsum(
            "...,f->...f", y.mean(0), x.mean(0)
        )
        within = np.zeros_like(full)
        for key in np.unique(g):
            mask = g == key
            within += (
                np.einsum("n...,nf->...f", y[mask], x[mask]) / mask.sum()
                - np.einsum("...,f->...f", y[mask].mean(0), x[mask].mean(0))
            ) * mask.mean()
        return full, within, full - within

    coefficients = list(moments(action))
    for shift in range(1, 6):
        permutation = np.arange(len(g))
        for group in np.unique(g):
            block = np.flatnonzero(g == group)
            permutation[block] = np.roll(block, shift)
        coefficients.append(moments(action[permutation])[0])
    context = ROOT / manifest["config"]["context_predecessor"]
    with np.load(context / "source_program.npz", allow_pickle=False) as saved:
        coefficients.extend(saved["coefficients"][-3:])
    with np.load(out / "source_program.npz", allow_pickle=False) as program:
        coefficient_error = float(
            np.max(abs(np.array(coefficients) - program["coefficients"]))
        )
        with np.load(out / "application_features.npz", allow_pickle=False) as data:
            cold = np.tensordot(
                data["features"] - program["origin"],
                np.array(coefficients),
                axes=([-1], [-1]),
            )
    with np.load(out / "cold_readings.npz", allow_pickle=False) as saved:
        cold_error = float(np.max(abs(cold - saved["readings"])))
    end, _ = R.P.profiles(context, "application")
    articles = R.A.read(out / "application_articles.json")
    index = {(a["pair_id"], a["pole"]): i for i, a in enumerate(articles)}
    rows = list(csv.DictReader((out / "transition_pair_accounts.csv").open()))
    account_error = 0.0
    for row in rows:
        pair = [index[row["pair_id"], p] for p in (-1, 1)]
        c, order = (
            manifest["opportunities"].index(row["opportunity"]),
            int(row["order"]),
        )
        first, last = replay["application"][pair, 1, c, order], end[pair, 1, c, order]
        joint = ((last[1] - last[0]) - (first[1] - first[0])) / 4
        account_error = max(account_error, abs(joint - float(row["joint_character"])))
    errors = {
        "source_second_moment": coefficient_error,
        "numeric_cold": cold_error,
        "native_joint": float(account_error),
    }
    if max(errors.values()) > 1e-10 or draws:
        raise ValueError({"errors": errors, "draw_errors": draws})
    result = {
        "errors": errors,
        "operation_addresses": addresses,
        "new_captures": new,
        "pair_order_accounts": len(rows),
        "draw_errors": int(draws),
        "new_forwards": 0,
        "scope": "same-agent alternative arithmetic;not independent blind replication",
    }
    R.A.write(out / "alternate_audit.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=R.DEFAULT)
    verify(parser.parse_args().out.resolve())
