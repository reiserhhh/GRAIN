"""Synthetic attacks on order ledgers; output checksum gate is mocked separately."""

import json

import pandas as pd
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from grain_research.vector_mechanisms.response_order import (
    MODES,
    SEEDS,
    permute_prefix,
    summarize_order,
)
from scripts import audit_grain_response_order as checker


@pytest.mark.parametrize("count", [8, 9, 16, 17, 33, 640])
def test_permutation_hash_independent_replay(count):
    for mode in MODES:
        for seed in SEEDS:
            _, meta = permute_prefix(
                list(range(count)), list(range(count)), "X", mode, seed
            )
            hash_, groups = checker.replay_permutation("X", mode, seed, count)
            assert (
                hash_ == meta["permutation_sha256"]
                and groups == meta["permutation_groups"]
            )


@pytest.fixture
def order_ledger(tmp_path, monkeypatch):
    """Complete four-person scalar records with globally shared synthetic contexts."""
    run, parent = tmp_path / "run", tmp_path / "parent"
    run.mkdir()
    parent.mkdir()
    codes = [f"P{i}" for i in range(4)]
    cases = [
        {
            "participant_code": c,
            "response_tokens": 20,
            "matched_donor_code": codes[(i + 1) % 4],
        }
        for i, c in enumerate(codes)
    ]
    material = {"cases": cases, "identity": material_hash(cases)}
    for root in (run, parent):
        (root / "private_material_manifest.json").write_text(json.dumps(material))
    rows, meta, bases = [], [], []
    for i, code in enumerate(codes):
        bases.append(
            {
                "participant_code": code,
                "actual_mean_logp": -3.0,
                "actual_tail_logp": -3.0,
                "matched_mean_logp": -3.4,
                "matched_tail_logp": -3.4,
            }
        )
        for role, owner, base in (("actual", i, -3.0), ("matched", (i + 1) % 4, -3.4)):
            prefix = [0] + list(range(20 + owner * 40, 52 + owner * 40)) + [1]
            rows.append(
                {
                    "participant_code": code,
                    "condition": role + "_native",
                    "mean_logp": base,
                    "tail_logp": base,
                    "response_tokens": 20,
                    "input_embedding_max_error": 0,
                }
            )
            for mode in MODES:
                for seed in SEEDS:
                    _, record = permute_prefix(
                        prefix, list(range(1, 33)), codes[owner], mode, seed
                    )
                    meta.append(
                        {
                            "participant_code": code,
                            "context_code": codes[owner],
                            "role": role,
                            "mode": mode,
                            "seed": seed,
                            "prefix_tokens": len(prefix),
                            "response_tokens": 20,
                            "fixed_boundary_crossing_tokens": 0,
                            **record,
                        }
                    )
                    penalty = 0.3 if role == "actual" else 0.2
                    rows.append(
                        {
                            "participant_code": code,
                            "condition": f"{role}_{mode}_{seed}",
                            "mean_logp": base - penalty,
                            "tail_logp": base - penalty,
                            "response_tokens": 20,
                            "input_embedding_max_error": 0,
                        }
                    )
    pd.DataFrame(rows).to_csv(run / "private_order_metrics.csv", index=False)
    pd.DataFrame(meta).to_csv(run / "private_order_manifest.csv", index=False)
    pd.DataFrame(bases).to_csv(parent / "private_response_metrics.csv", index=False)
    manifest = {"rc1_identity": "parent-id", "modes": MODES, "seeds": SEEDS}
    manifest["identity"] = material_hash(manifest)
    (run / "manifest.json").write_text(json.dumps(manifest))
    numerical = [
        {
            "passed": True,
            "self_patch_max_logp_error": 0,
            "compact_fixed_mean_logp_difference": 0,
            "minimum_site_cosine": 1,
        }
    ] * 2
    technical = []
    for lang in ("en", "ja"):
        for mode in MODES:
            for seed in SEEDS:
                _, record = permute_prefix(
                    list(range(32)), list(range(32)), lang, mode, seed
                )
                technical.append(
                    {"language": lang, "mode": mode, "seed": seed, **record}
                )
    (run / "technical_numerical_gate.json").write_text(json.dumps(numerical))
    (run / "technical_order_gate.json").write_text(json.dumps(technical))
    result = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "rc1_identity": "parent-id",
        "participants": 4,
        "forwards": 46,
        "effects": summarize_order(pd.DataFrame(rows)),
    }
    monkeypatch.setattr(
        checker,
        "verify_outputs",
        lambda path: (
            result
            if path == run
            else {"status": "COMPLETE", "parent_identity": "parent-id"}
        ),
    )
    return run, parent, result


def test_order_ledger_replays_without_pseudoreplicating_seeds(order_ledger):
    run, parent, _ = order_ledger
    result, rows = checker.audit(run, parent)
    assert (
        result["model_rows"] == 40 and result["metadata_rows"] == 32 and len(rows) == 36
    )
    assert (
        result["effects"]["mean_logp/token/seed_mean"]["order_interaction"][
            "participants"
        ]
        == 4
    )


@pytest.mark.parametrize(
    "attack", ["drop_row", "owner", "permutation", "bag", "summary"]
)
def test_order_audit_rejects_contract_failures(order_ledger, attack):
    run, parent, result = order_ledger
    path = run / "private_order_manifest.csv"
    frame = pd.read_csv(path)
    if attack == "drop_row":
        frame = frame.iloc[1:]
    elif attack == "owner":
        frame.loc[0, "context_code"] = "wrong"
    elif attack == "permutation":
        frame.loc[0, "permutation_sha256"] = "wrong"
    elif attack == "bag":
        mask = (frame.context_code == "P0") & (frame["mode"] == "token")
        frame.loc[mask, "bag_sha256"] = "wrong"
    else:
        result["effects"]["mean_logp/block8/seed_mean"]["order_interaction"][
            "mean"
        ] += 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        checker.audit(run, parent)
