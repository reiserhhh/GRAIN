import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dif", ROOT / "scripts/run_grain_r23_fixed_core_dif.py")
dif = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dif)


def test_generator_formula_matches_parsed_text():
    d = dif.load()
    assert (d.frame_f == d.frame_t).all() and (d.day_f == d.day_t).all() and (d.topic_f == d.topic_t).all()
    assert d.groupby("role").frame_f.nunique().max() == 3
    frames = {r: set(g.frame_f) for r, g in d.groupby("role")}
    assert not (frames["source"] & frames["application"])  # role-disjoint frames: coefficient transport unsupported


def test_aliased_design_reports_null_space():
    g = pd.DataFrame({"core": ["a", "b"] * 6, "frame_f": ["x", "y"] * 6, "topic_f": ["t"] * 12, "day_f": ["d"] * 12})
    _coef, rank, p, null = dif.fit(g, np.arange(12, dtype=float))
    assert rank < p and null  # core and frame perfectly aliased


def test_core_contrast_undefined_for_constant_reader():
    d = dif.load()
    g = d[d.role == "application"]
    assert abs(dif.core_contrast(g, "grainridge_p", "pole")) < 0.05
