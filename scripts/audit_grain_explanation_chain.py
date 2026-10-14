#!/usr/bin/env python3
"""Post-result M13 numeric replay and rendered condition cards; no neural calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grain_research.explanation_chain import decompose_query
from grain_research.vector_mechanisms.interpretation_evidence import validate_packet
from scripts.run_grain_explanation_chain import BASE, CRITERION, OLD, sha, write
from scripts.run_grain_full_state_edit_connection import token_view
from scripts.run_grain_query_moment_operator import load_models
from scripts.run_grain_real_author_path import MODEL


def audit(out):
    """Replay frozen data and add explicit endpoint explanations without rescoring."""
    from transformers import AutoTokenizer

    dst = out / "interpretation_audit"
    dst.mkdir(exist_ok=False)
    freeze = json.loads((out / "freeze.json").read_text())
    manifest = json.loads((out / "artifact_manifest.json").read_text())
    for path, digest in freeze["files"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in manifest.items():
        assert sha(out / path) == digest, path
    model = load_models(OLD)["confirmation_seeking"]
    bank = json.loads((out / "public_bank.json").read_text())
    frame = pd.read_csv(out / "readings.csv").set_index("case")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True, trust_remote_code=False
    )
    errors, address_rows = [], []
    for row in bank:
        key = row["case"]
        state = np.load(out / "states" / f"{key}.npz", allow_pickle=False)
        b, bt, wi = decompose_query(state["g"], state["h"], model)
        rebuilt = {
            "query": float(b + bt.sum() + wi.sum()),
            "grain_f03": float(state["slice_scores"][:, 2].sum() / 4),
        }
        for name, value in rebuilt.items():
            errors.append(abs(value - frame.loc[key, name]))
        packet = json.loads((out / "cards" / f"{key}.json").read_text())
        trusted = {
            m["id"]: {**m, "value": float(frame.loc[key, m["id"]])}
            for m in packet["measurements"]
        }
        # JSON and decimal CSV round trips can differ by one ulp; authoritative values are stored states.
        for m in packet["measurements"]:
            assert abs(m["value"] - rebuilt[m["id"]]) < 1e-8
            trusted[m["id"]]["value"] = m["value"]
        validate_packet(packet, {key: row["text"]}, trusted)
        rendered, spans, encoded = token_view(tokenizer, [row["text"]], "TF_cont")
        np.testing.assert_array_equal(state["input_ids"], encoded["input_ids"])
        a, z = spans[0]
        assert rendered[a:z] == row["text"]
        for pos, between, within in zip(state["positions"], bt, wi):
            left, right = encoded["offset_mapping"][pos]
            start, end = max(a, left) - a, min(z, right) - a
            assert 0 <= start < end <= len(row["text"])
            address_rows.append(
                {
                    "case": key,
                    "model_position": int(pos),
                    "raw_start": start,
                    "raw_end": end,
                    "quote": row["text"][start:end],
                    "between_allocation": between,
                    "within_allocation": within,
                }
            )
    assert max(errors) < 1e-8
    pd.DataFrame(address_rows).to_csv(dst / "token_address_accounting.csv", index=False)
    condition = pd.read_csv(out / "condition_profiles.csv")
    cards = []
    for row in condition.to_dict("records"):
        lo, hi = frame.loc[row["case_low"]], frame.loc[row["case_high"]]
        name = row["measure"]
        assert abs(row["level"] - (lo[name] + hi[name]) / 2) < 1e-8
        assert abs(row["half_change"] - (hi[name] - lo[name]) / 2) < 1e-8
        card = {
            **row,
            "low": float(lo[name]),
            "high": float(hi[name]),
            "candidate": "Finite response of a frozen measurement function to the stated time-budget edit.",
            "alternative": "Numeral/tokenization, position, shared prompt and contextual semantics all change together.",
            "person_parameter_estimated": False,
        }
        if name == "query":
            components = {
                k: float((hi[k] - lo[k]) / 2) for k in ["anchor", "between", "within"]
            }
            assert abs(sum(components.values()) - row["half_change"]) < 1e-8
            card["half_change_accounting"] = components
        cards.append(card)
    write(dst / "condition_cards.json", cards)
    paragraphs = [
        "# Fixed-function condition-change explanations",
        "",
        "These are paired authored stimuli, not twelve real people. The numbers below are fixed-function changes, not psychological response coefficients.",
        "",
    ]
    for card in cards:
        paragraphs.extend(
            [
                f"## {card['case_low']} / {card['case_high']} | {card['measure']}",
                f"5 min: {card['low']:.8f}; 45 min: {card['high']:.8f}; level: {card['level']:.8f}; half-change: {card['half_change']:.8f}.",
                card["candidate"] + " " + card["alternative"],
                "",
            ]
        )
    (dst / "CONDITION_CARDS.md").write_text("\n".join(paragraphs))
    source = pd.read_csv(CRITERION)
    history = pd.read_csv(out / "historical_f03_criterion_links.csv")
    pd.testing.assert_frame_equal(
        history.reset_index(drop=True),
        source.loc[source.factor.eq("whole:F03")].reset_index(drop=True),
    )
    selected = history.loc[
        history["query"].eq("openness") & history.arm.isin(["U", "S"])
    ]
    write(
        dst / "evidence_levels.json",
        {
            "model_function": {
                "current": "24 actual prefix observations; exact numerical changes; no suffix generated judgments"
            },
            "stimulus_identity": "Authored manifest only; validator checks addresses, not semantic correctness of arbitrary prose",
            "anonymous_g03_historical_self_report": selected.to_dict("records"),
            "historical_object_warning": "S is covariance of F03 with a distinct E5 Openness query; it is not F03 mean U and not the new confirmation query",
            "current_self_report": "No new participant label association",
            "actual_behavior": "Not observed; reported action is not independently observed action",
            "independent_human_coding": "D01 remains pending",
        },
    )
    summary = json.loads((out / "summary.json").read_text())
    assert all(not t.get("forbidden_calls", []) for t in summary["traces"])
    write(
        dst / "audit.json",
        {
            "status": "pass",
            "post_result_companion": True,
            "independent_executor": False,
            "frozen_files": len(freeze["files"]),
            "sealed_outputs": len(manifest),
            "raw_packets": len(bank),
            "token_addresses": len(address_rows),
            "condition_target_cards": len(cards),
            "historical_aggregate_rows": len(history),
            "max_score_replay_error": max(errors),
            "new_neural_calls": 0,
            "script_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE / "explanation_chain_r1")
    audit(parser.parse_args().out)
