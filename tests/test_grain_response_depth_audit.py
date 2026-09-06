"""Synthetic scalar-contract attacks; checksum verification is mocked separately."""

import json

import pandas as pd
import pytest

from grain_research.vector_mechanisms.controls import material_hash
from scripts import audit_grain_response_depth as checker
from scripts.run_grain_response_depth import SITES, summarize


@pytest.fixture
def synthetic_audit(tmp_path, monkeypatch):
    """Build a small scalar ledger, never raw MEPS text or neural activations."""
    run, parent = tmp_path / "run", tmp_path / "parent"
    run.mkdir()
    parent.mkdir()
    cases = [{"participant_code": f"P{i}", "response_tokens": 20} for i in range(4)]
    material = {"cases": cases, "identity": material_hash(cases)}
    for root in (run, parent):
        (root / "private_material_manifest.json").write_text(json.dumps(material))
    rows, bases = [], []
    for i, gap in enumerate([0.2, 0.4, -0.1, 0.0]):
        base = {
            "participant_code": f"P{i}",
            "response_tokens": 20,
            "actual_tail_logp": -4 + gap,
            "matched_tail_logp": -4,
            "parent_replay_max_error": 0,
        }
        bases.append(base)
        for site, fraction in zip(SITES, [0, 0.1, 0.3, 0.8, 1, 1]):
            rows.append(
                {
                    "participant_code": f"P{i}",
                    "site": site,
                    "actual_tail_logp": -4 + gap,
                    "matched_tail_logp": -4,
                    "patch_tail_logp": -4 + fraction * gap,
                    "self_patch_max_error": 0,
                    "patch_first_token_change": 0,
                    "patch_actual_tail_max_error": abs((1 - fraction) * gap),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(run / "private_depth_metrics.csv", index=False)
    pd.DataFrame(bases).to_csv(run / "private_baseline_metrics.csv", index=False)
    pd.DataFrame(bases).to_csv(parent / "private_response_metrics.csv", index=False)
    manifest = {"parent_identity": "parent-id", "sites": list(SITES)}
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
    readout = [{"passed": True, "tail_max_error": 0, "first_token_change": 0}] * 2
    (run / "technical_numerical_gate.json").write_text(json.dumps(numerical))
    (run / "technical_readout_gate.json").write_text(json.dumps(readout))
    result = {
        "status": "COMPLETE",
        "parent_identity": manifest["identity"],
        "rc1_identity": "parent-id",
        "participants": 4,
        "forwards": 68,
        **summarize(frame),
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


def test_independent_depth_scalar_replay(synthetic_audit):
    run, parent, _ = synthetic_audit
    result, rows = checker.audit(run, parent)
    assert result["layer_rows"] == 24 and len(rows) == 6
    assert result["effects"]["final_norm"]["ratio_of_group_means"] == pytest.approx(1)


@pytest.mark.parametrize(
    "attack", ["drop_row", "first_token", "terminal", "case", "summary"]
)
def test_depth_audit_rejects_material_and_claim_errors(synthetic_audit, attack):
    run, parent, result = synthetic_audit
    path = run / "private_depth_metrics.csv"
    frame = pd.read_csv(path, dtype={"patch_first_token_change": float})
    if attack == "drop_row":
        frame = frame.iloc[1:]
    elif attack == "first_token":
        frame.loc[0, "patch_first_token_change"] = 0.01
    elif attack == "terminal":
        frame.loc[frame.site == "final_norm", "patch_actual_tail_max_error"] = 0.01
    elif attack == "case":
        frame.loc[0, "participant_code"] = "wrong"
    else:
        result["effects"]["block_10"]["patch_minus_matched"]["mean"] += 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        checker.audit(run, parent)
