"""R15 local NLI hedge-commitment reader: budget, frozen materials, independent recomputation of flags/contrasts."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grain_gptpro_foundation_v3/F17/hedge_commitment_reader_r1"
ROWS = [json.loads(l) for l in (OUT / "readings.jsonl").read_text().splitlines()]
C = json.loads((OUT / "contrasts.json").read_text())
LOG = json.loads((OUT / "run_log.json").read_text())


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def test_budget_and_offline_pinned_model():
    assert LOG["pairs_scored"] == 576 and LOG["invocations"] <= 576 and LOG["failures"] == 0
    assert LOG["revision"] == "6c749ce3425cd33b46d187e45b92bbf96ee12ec7" and LOG["seconds"] < 600
    assert LOG["boundary"]["pairs"] == 384
    assert len(ROWS) == 576 and len({r["pair_id"] for r in ROWS}) == 576


def test_materials_frozen_hash_and_split():
    h = hashlib.sha256((OUT / "materials_frozen.json").read_bytes()).hexdigest()
    assert h == "1b07fb5336e4909e267d97e844b47d6cc5330abc62a00fefeca20adbc53d12ac"
    mats = json.loads((OUT / "materials_frozen.json").read_text())
    pairs = {p["pair_id"] for p in mats["pairs"]} if "pairs" in mats else None
    if pairs is not None:
        assert pairs == {r["pair_id"] for r in ROWS}
    assert {r["core"] for r in ROWS if r["split"] == "source"} == {1, 2}
    assert {r["core"] for r in ROWS if r["split"] == "test"} == {3}


def test_probabilities_normalized():
    for r in ROWS:
        assert abs(r["p_entailment"] + r["p_neutral"] + r["p_contradiction"] - 1) < 1e-5


def test_source_boundary_flags_recomputed_without_pandas():
    src = [r for r in ROWS if r["split"] == "source"]
    sel = lambda **k: [r["p_entailment"] for r in src if all(r[a] == b for a, b in k.items())]
    own_self = mean(sel(owner="self", form="bare", role="own_commitment"))
    own_other = mean(sel(owner="other", form="bare", role="own_commitment"))
    rep_self, rep_other = mean(sel(owner="self", role="reported_other")), mean(sel(owner="other", role="reported_other"))
    f = C["source_boundary_flags"]
    assert abs(own_self - f["own_commitment_self_bare"]) < 1e-12 and abs(rep_other - f["reported_other_other"]) < 1e-12
    assert f["ownership_failure"] is (own_other >= own_self or rep_other <= rep_self) is False
    expect = {"own_commitment": "bare", "uncertainty": "epistemic", "deference": "deferential", "no_commitment": "noncommittal"}
    for role, form in expect.items():
        by = defaultdict(list)
        for r in src:
            if r["owner"] == "self" and r["role"] == role:
                by[r["form"]].append(r["p_entailment"])
        best = max(by, key=lambda k: mean(by[k]))
        assert f["form_failure"][role] is (best != form)
    assert f["form_failure"]["own_commitment"] is True  # the frozen flag that disables own-commitment attribution


def test_test_core_ownership_contrast_recomputed():
    t = [r for r in ROWS if r["core"] == 3]
    cell = defaultdict(dict)
    for r in t:
        cell[(r["form"], r["frame"], r["wording"], r["role"])][r["owner"]] = r["p_entailment"]
    d = mean(v["other"] - v["self"] for k, v in cell.items() if k[0] == "bare" and k[3] == "own_commitment")
    assert abs(d - C["per_core"]["3"]["reported_minus_self_by_form"]["own_commitment"]["bare"]) < 5e-5
    assert d < 0
