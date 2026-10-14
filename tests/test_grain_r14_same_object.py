import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r14", ROOT / "scripts/run_grain_r14_same_object_channels.py")
r14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r14)


def toy_strata():
    strata = {}
    for rk in ("application", "pilot_validation", "source"):
        for pole in (0, 1):
            for st in r14.STATUSES:
                strata[(rk, pole, st)] = [f"{rk}_{pole}_{st}_{i}" for i in range(3)]
    return strata


def test_render_deterministic_and_role_separated():
    ys = [i % 2 for i in range(48)]
    s = toy_strata()
    a = r14.render_transport([1, 8, 0, 5], ys, "mixed", "application", s)
    b = r14.render_transport([1, 8, 0, 5], ys, "mixed", "application", s)
    assert a == b
    assert all(e["material_id"].startswith("application_") for e in a)
    # qualified events carry the latent observation as pole
    assert all(e["material_pole"] == ys[e["t"]] for e in a if e["q_meta"] == 1)


def test_complete_panel_statuses_and_q1():
    ys = [1] * 48
    ev = r14.render_transport([2, 8, 1, 0], ys, "complete", "application", toy_strata())
    assert {e["status"] for e in ev} <= {"actual", "negated"}
    rd = {e["material_id"]: {"oracle": (1.0, 0.0)} for e in ev}
    seq = r14.readings_for(ev, rd, "complete", ["oracle"])["oracle"]
    assert all(q == 1.0 for _p, q in seq)


def test_unqualified_reading_adds_no_evidence_mass():
    fitted = {"b1": {"source_p": [0.5] * 4}, "b2": {"coef": [0.0] * 4}, "m": {"mean": [0.5] * 4, "right": np.zeros((2, 4)).tolist()}}
    h = r14.QualifiedHistory(fitted, None, 0.5, mode="filtered")
    h.observe(1.0, 0.0)
    h.observe(1.0, 0.0)
    h.observe(1.0, 0.0)
    assert h.mass == 0.0 and h.predict()["status"] == "prior_only"


def test_same_material_shared_across_channels():
    ev = r14.render_transport([3, 8, 0, 1], [0, 1] * 24, "mixed", "application", toy_strata())
    rd = {e["material_id"]: {"x": (0.2, 1.0), "y": (0.9, 0.0)} for e in ev}
    r = r14.readings_for(ev, rd, "mixed", ["x", "y"])
    assert len(r["x"]) == len(r["y"]) == 48
