"""Focused tests for GRAIN-C2-01 evidence unit / observation law."""


import numpy as np
import pandas as pd
import pytest

from grain_research.construction import evidence_unit as EU

BANK = "results/grain_gptpro_foundation_v3/F08/nli_reader_r1/reader_scores.csv"


def _rows(role="source", reader_perfect=True):
    rows = []
    for b in (0, 1):
        for s in EU.STATUSES:
            for v in range(4):
                c = 2 * b + int(s in EU.QUALIFIED) if reader_perfect else 2 * b + 1
                rows.append({"material_id": f"m{b}{s}{v}", "role": role, "pole": b, "subject_status": s,
                             "variant": v, "category": c})
    return rows


def test_bundle_rejects_non_source_rows():
    with pytest.raises(ValueError):
        EU.build_bundle(_rows("application"), EU.READERS[0], 1, {})


def test_family_role_split_in_bank():
    df = pd.read_csv(BANK)
    assert (df.family == df.role.map(EU.FAMILY_OF_ROLE)).all()
    assert df.groupby(["role", "pole", "subject_status"]).size().eq(4).all()


def test_no_writer_ids_or_labels_accepted():
    b = EU.build_bundle(_rows(), EU.READERS[0], 1, {})
    assert not any("writer" in k for k in b)
    bad = [{"author_id": "a", "reader_id": EU.READERS[0], "writer_id": "w1", "events": []}]
    with pytest.raises(ValueError):
        EU.records_to_counts(bad, EU.READERS[0])
    bad_ev = [{"author_id": "a", "reader_id": EU.READERS[0],
               "events": [{"event_id": "e", "pole_score": .9, "qualification_score": .9, "pole": 1}]}]
    with pytest.raises(ValueError):
        EU.records_to_counts(bad_ev, EU.READERS[0])


def test_null_personal_evidence_is_prior_not_neutral_trait():
    b = EU.build_bundle(_rows(), EU.READERS[0], 1, {})
    out = EU.cold_score(b, [{"author_id": "a", "reader_id": EU.READERS[0], "events": []}], mode="point")[0]
    assert out["status"] == "no_events" and abs(out["pi_mean"] - 0.5) < 1e-9
    assert "NOT an observed neutral trait" in out["interpretation"]


def test_serialization_and_cold_replay(tmp_path):
    b = EU.build_bundle(_rows(), EU.READERS[1], 3, {})
    h = EU.save_bundle(b, tmp_path / "b.json")
    loaded = EU.load_bundle(tmp_path / "b.json", h)
    recs = [{"author_id": "a", "reader_id": EU.READERS[1],
             "events": [{"event_id": f"e{i}", "pole_score": .9, "qualification_score": .8} for i in range(5)]}]
    assert EU.cold_score(loaded, recs) == EU.cold_score(b, recs)
    with pytest.raises(ValueError):
        EU.load_bundle(tmp_path / "b.json", "0" * 64)
    with pytest.raises(ValueError):
        EU.cold_score(b, [{**recs[0], "reader_id": EU.READERS[0]}])


def test_calibration_uncertainty_widens_intervals():
    n = EU.channel_counts(_rows())
    counts = np.array([[2, 1, 3, 10]], float)
    point = EU.summarize_grid(EU.marginal_posteriors(counts, EU.point_channel(n)[None])[0])
    mix = EU.summarize_grid(EU.marginal_posteriors(counts, EU.dirichlet_channels(n, 5))[0])
    assert (mix[3] - mix[2])[0] > (point[3] - point[2])[0]


def test_wrong_reader_law_counterexample():
    """A reader that marks every report as qualified: its own law vs an identity law."""
    lax = EU.channel_counts(_rows(reader_perfect=False))
    ident = EU.channel_counts(_rows())
    rng = np.random.default_rng(0)
    pi, kappa, n = 0.8, 0.5, 4000
    qual = rng.random(n) < kappa
    pole = np.where(qual, rng.random(n) < pi, rng.random(n) < 0.5).astype(int)
    cats = 2 * pole + 1
    counts = np.bincount(cats, minlength=4)[None].astype(float)
    right = EU.summarize_grid(EU.marginal_posteriors(counts, EU.point_channel(lax)[None])[0])[0][0]
    wrong = EU.summarize_grid(EU.marginal_posteriors(counts, EU.point_channel(ident)[None])[0])[0][0]
    # correct law recovers pi; the identity law treats unqualified reports as qualified
    # and is pulled toward the unqualified pole rate (exact value depends on smoothing).
    assert abs(right - pi) < 0.05
    assert abs(wrong - pi) > 0.08 and wrong < right


def test_identity_channel_matches_beta_reference():
    e = np.zeros((2, 4, 4))
    for b in (0, 1):
        for s, st in enumerate(EU.STATUSES):
            e[b, s, 2 * b + int(st in EU.QUALIFIED)] = 1
    e = 0.999 * e + 0.001 / 4
    counts = np.array([[3, 4, 2, 9]], float)
    m = EU.summarize_grid(EU.marginal_posteriors(counts, e[None])[0])[0][0]
    ref = EU.beta_summary([9], [4])[0][0]
    assert abs(m - ref) < 0.01
    assert np.allclose(EU.category_probabilities(e).sum(-1), 1)
