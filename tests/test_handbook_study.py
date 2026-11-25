#!/usr/bin/env python3
"""Reader-facing contracts for the all-system handbook; no data assets or neural jobs."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples import handbook_study as study


@pytest.fixture(scope="module")
def worked(tmp_path_factory):
    dest = tmp_path_factory.mktemp("example") / "study"
    with threadpool_limits(limits=1):
        study.run(dest)
    return dest


def test_walkthrough_distinguishes_order_from_composition(worked):
    load = study.reader.load
    rows = load(worked / "target.json")["episodes"]
    a, b = rows[:4], rows[4:]
    assert [Counter(x["events"]) for x in a] == [Counter(x["events"]) for x in b]
    table = load(worked / "target-units.json")
    (level0, step0), (level1, step1) = [r["values"] for r in table["units"]]
    assert level0 == pytest.approx(level1)
    assert 0 < step0 < step1
    chart = load(worked / "chart.json")
    expected = study.population_io("locate", table, chart=chart)
    assert expected == load(worked / "positions.json")
    assert expected["reference_n"] == 10
    assert (
        expected["units"][1]["percentiles_0_1"][1]
        > expected["units"][0]["percentiles_0_1"][1]
    )
    families = load(worked / "target-families.json")
    assert len(families) == 4
    assert all({"static", "dynamic", "hybrid"} <= r.keys() for r in families)


def test_study_accepts_prepared_inputs_and_protects_existing_outputs(worked, tmp_path):
    out = tmp_path / "rebuild"
    with threadpool_limits(limits=1):
        result = study.run(out, inputs=worked)
    np.testing.assert_allclose(
        [r["percentiles_0_1"] for r in result["units"]],
        [
            r["percentiles_0_1"]
            for r in study.reader.load(worked / "positions.json")["units"]
        ],
    )
    with pytest.raises(FileExistsError):
        study.run(out, inputs=worked)


def test_cold_score_uses_just_two_files_and_no_fit(worked, tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for name in ("native-reference.json", "chart.json"):
        study.reader.save(frozen / name, study.reader.load(worked / name))

    def unexpected(*args, **kwargs):
        pytest.fail("cold scoring called a fit function")

    monkeypatch.setattr(study.reader, "fit_atlas", unexpected)
    monkeypatch.setattr(study.reader, "fit_axes", unexpected)
    assert study.score(
        tmp_path, frozen, study.reader.load(worked / "target.json")
    ) == study.reader.load(worked / "positions.json")
    with pytest.raises(ValueError, match="discovery author"):
        study.score(tmp_path, frozen, study.reader.load(worked / "discovery.json"))
    with pytest.raises(ValueError, match="calibration unit"):
        study.score(tmp_path, frozen, study.reader.load(worked / "calibration.json"))


def test_cross_cohort_author_leak_is_rejected(tmp_path):
    for role in ("discovery", "calibration", "target"):
        doc = study.corpus("duplicate", 3)
        study.reader.save(tmp_path / f"{role}.json", doc)
    with pytest.raises(ValueError, match="another cohort"):
        study.run(tmp_path / "bad-output", inputs=tmp_path)


@pytest.mark.parametrize("lang", ["", "_ZH", "_JA"])
def test_readme_routes_to_complete_handbook_and_links_resolve(lang):
    root = Path(__file__).resolve().parents[1]
    manual = root / f"docs/GRAIN_USER_MANUAL{lang}.md"
    readme = (
        root / {"": "README.md", "_ZH": "README.zh.md", "_JA": "README.ja.md"}[lang]
    )
    assert str(manual.relative_to(root)) in readme.read_text()
    body = manual.read_text()
    for name in (
        "handbook_study",
        "liquid_reference_error",
        "liquid_instrument",
        "event_response",
        "dual_view_increment",
        "CONSTRUCT_REGISTRY",
        "CLAIMS_LEDGER",
    ):
        assert name in body
    for href in re.findall(r"\]\(([^)]+)\)", body):
        if not href.startswith(("http:", "https:", "#")):
            assert (manual.parent / href.split("#")[0]).is_file(), href
