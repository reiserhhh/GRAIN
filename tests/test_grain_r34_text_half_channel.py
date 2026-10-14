"""Focused tests for packet20 B (R34 text half-channel sensitivity pilot)."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/grain_gptpro_foundation_v3/F16/text_half_channel_sensitivity_r1"
PROTOCOL = ROOT / "docs/GRAIN_R34_TEXT_HALF_CHANNEL_PROTOCOL_V1.md"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_protocol_bindings_hold():
    assert sha(PROTOCOL) == "6469a1885d7e16801a7615b337cb09b74c46a8f4cb1dc50b02b2fdc48ba46829"
    for rel, h in re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", PROTOCOL.read_text()):
        assert sha(ROOT / rel) == h


def test_halves_are_disjoint_whole_cores_and_balanced():
    ch = json.loads((OUT / "channels.json").read_text())
    A, B = set(ch["split"]["half_A_materials"]), set(ch["split"]["half_B_materials"])
    assert len(A) == len(B) == 16 and not A & B
    for arm in ("nli_joint", "hybrid_joint"):
        cA, cB = np.array(ch["arms"][arm]["counts_A"]), np.array(ch["arms"][arm]["counts_B"])
        assert (cA.sum(2) == 4).all() and (cB.sum(2) == 4).all()
        assert np.array_equal(cA + cB, np.array(ch["arms"][arm]["counts_FULL"]))


def test_smoothed_mean_rule():
    ch = json.loads((OUT / "channels.json").read_text())
    c = np.array(ch["arms"]["nli_joint"]["counts_A"], float) + .5
    assert np.allclose(np.array(ch["arms"]["nli_joint"]["mean_A"]), c / c.sum(2, keepdims=True), atol=1e-15)


def test_one_component_label_and_wrapper_replay():
    res = json.loads((OUT / "results.json").read_text())
    cfg = json.loads((OUT / "config.json").read_text())
    assert cfg["actual_component_count"] == 1 and "DIAGNOSTIC_ONE_COMPONENT_VIEW" in res["label"]
    assert res["cold_replay_max_err"] <= 1e-12
    assert res["neural_calls"] == 0 and res["persons_used"] == 20 and not res["stopped_early"]


def test_same_persons_all_arms_and_no_calibration_material():
    sup = json.loads((OUT / "support.json").read_text())
    for route in ("text_T1", "text_dynamic"):
        ids = {a: [s["author_id"] for s in sup if s["route"] == route and s["arm"] == a] for a in ("nli_joint", "hybrid_joint")}
        assert ids["nli_joint"] == ids["hybrid_joint"] == sorted(ids["nli_joint"])
    assert all(s["event_roles"] == ["application"] for s in sup)
    cal = set(json.loads((OUT / "channels.json").read_text())["split"]["half_A_materials"]) | set(json.loads((OUT / "channels.json").read_text())["split"]["half_B_materials"])
    assert not any(e[1] in cal for s in sup for e in s["event_addresses"])


def test_tv_zero_implies_identical_scores():
    d = pd.read_csv(OUT / "per_person.csv")
    ch = json.loads((OUT / "channels.json").read_text())
    for arm in ("nli_joint", "hybrid_joint"):
        g = d[d.arm == arm]
        if ch["arms"][arm]["tv_mean"] == 0:
            assert (g.d == 0).all()
        else:
            assert (g.d != 0).any()
    assert np.allclose(d.z, d.d / np.sqrt((d.var_A + d.var_B) / 2))
