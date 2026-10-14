"""Tests for the GRAIN-C4-01 anonymous cold bundle and the C2/C3 addenda."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grain_research.construction import anonymous_bundle as AB
from grain_research.construction import declared_target as DT

ROOT = Path(__file__).resolve().parents[1]


def _bundle():
    rng = np.random.default_rng(0)
    dim, k = 8, 2
    d = rng.normal(size=(dim, k))
    d /= np.linalg.norm(d, axis=0)
    vecs = [rng.normal(size=(25, dim)) for _ in range(60)]
    venues = [rng.choice(["v1", "v2", "v3"], 25) for _ in range(60)]
    raw = [AB.normalize(v) @ d for v in vecs]
    table, pooled = AB.venue_offsets(np.concatenate(raw), np.concatenate(venues))
    rule = DT.context_rule(["v1"])
    fits = {}
    for m in AB.MODES:
        ys = [AB.apply_mode(r, list(vn), table, pooled, m)[0] for r, vn in zip(raw, venues)]
        fits[m], _ = DT.fit_moments(ys, [DT.contexts(vn, rule) for vn in venues])
    return AB.build_bundle(d, ["a_0", "a_1"], {"id": "enc@x"}, rule, table, pooled, fits, {}, {}), d


def _rec(vectors, venues, enc="enc@x"):
    return {"author_key": "a", "events": [{"event_id": f"e{i}", "venue": v, "encoder_id": enc, "vector": list(x)}
                                         for i, (x, v) in enumerate(zip(vectors, venues))]}


def test_tamper_detection():
    b, _ = _bundle()
    AB.verify_bundle(b)
    b2 = json.loads(json.dumps(b))
    b2["directions"][0][0] += 1e-6
    with pytest.raises(ValueError):
        AB.verify_bundle(b2)
    b3 = json.loads(json.dumps(b))
    b3["venue_offsets"]["v1"][0] += 1e-6
    with pytest.raises(ValueError):
        AB.verify_bundle(b3)


def test_input_contract():
    b, _ = _bundle()
    x = np.ones((2, 8))
    with pytest.raises(ValueError):
        AB.cold_score(b, [_rec(x, ["v1", "v2"], enc="other")])
    bad = _rec(x, ["v1", "v2"])
    bad["events"][0]["text"] = "no"
    with pytest.raises(ValueError):
        AB.cold_score(b, [bad])
    dup = _rec(x, ["v1", "v2"])
    dup["events"][1]["event_id"] = "e0"
    with pytest.raises(ValueError):
        AB.cold_score(b, [dup])


def test_unseen_venue_and_normalization_and_null():
    b, d = _bundle()
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 8))
    r1 = AB.cold_score(b, [_rec(x, ["v1", "v2", "zz", "zz"])])[0]
    r2 = AB.cold_score(b, [_rec(5 * x, ["v1", "v2", "zz", "zz"])])[0]
    assert r1["within_venue"]["unseen_venue_share"] == 0.5
    np.testing.assert_allclose(r1["within_venue"]["level"]["mean"], r2["within_venue"]["level"]["mean"], atol=1e-12)
    raw = AB.normalize(x) @ d
    y, unseen = AB.apply_mode(raw, ["v1", "v2", "zz", "zz"], b["venue_offsets"], b["pooled_offset"], "within_venue")
    np.testing.assert_allclose(y[2], raw[2] - np.array(b["pooled_offset"]))
    assert unseen.tolist() == [False, False, True, True]
    none = AB.cold_score(b, [{"author_key": "n", "events": []}])[0]
    assert none["composition_inclusive"]["support"] == "no_events_prior_only"


def test_bundle_hash_roundtrip(tmp_path):
    b, _ = _bundle()
    h = AB.save_bundle(b, tmp_path / "b.json")
    assert AB.load_bundle(tmp_path / "b.json", h)["directions_sha256"] == b["directions_sha256"]
    with pytest.raises(ValueError):
        AB.load_bundle(tmp_path / "b.json", "0" * 64)


def test_c2_addendum_matches_saved_intervals_and_old_roots_intact():
    add = ROOT / "results/grain_construction_v1/C2/evidence_unit_r1_addendum_20261003/c2_interpretation_addendum.json"
    if not add.exists():
        pytest.skip("addendum not generated in this checkout")
    a = json.loads(add.read_text())
    assert a["old_root_hashes_intact"]
    p = pd.read_csv(ROOT / "results/grain_construction_v1/C2/evidence_unit_r1/paired_differences.csv")
    p = p[(p.reference == "naive_uncalibrated") & (p.family_role == "application")]
    for row in a["rows"]:
        r = p[(p.reader == row["reader"]) & (p.kappa == row["kappa"])].iloc[0]
        assert row["ci_hi"] == r.ci_hi and row["zero_margin_upper_bound_lt_0 (primary better than naive)"] == (r.ci_hi < 0)
    nli = [r for r in a["rows"] if r["reader"].startswith("nli")]
    assert all(r["zero_margin_upper_bound_lt_0 (primary better than naive)"] for r in nli)
