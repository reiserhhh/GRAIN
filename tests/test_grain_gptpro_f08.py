#!/usr/bin/env python3
"""F08 synthetic reader checks."""

import numpy as np

from grain_research.gptpro_foundation_v3.f08 import FAMILIES, STATUSES, fit_reader, render_event


def test_renderer_has_controlled_status_without_id_leakage():
    rng = np.random.default_rng(1)
    text = render_event(1, "actual", rng, family="plain", variant=0)
    assert "material_" not in text
    assert "actual" in text


def test_reader_fits_on_small_material_bank():
    rng = np.random.default_rng(2)
    texts, poles, qualifications = [], [], []
    for family in FAMILIES:
        for pole in (0, 1):
            for status in STATUSES:
                texts.append(render_event(pole, status, rng, family=family, variant=0))
                poles.append(pole)
                qualifications.append(int(status in ("actual", "negated")))
    reader = fit_reader(texts, poles, qualifications)
    assert reader["n_features"] > 0
