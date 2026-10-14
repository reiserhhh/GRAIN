#!/usr/bin/env python3
"""F16-03: an old C2 (binary history) request must fail against a T1 bundle."""

import copy

from grain_research.gptpro_foundation_v3 import f16_serialize as pub
from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.measurement_discovery.estimators import ordered_rows
from tests.test_grain_gptpro_f16_serialize import _t1_bundle, _t1_requests


def c2_history():
    records, _ = generate_world("c2_xor", 20260915, events=48, counts={"source": 0, "validation": 0, "application": 1})
    return next(iter(ordered_rows(records).values()))


def test_c2_requests_are_rejected_by_t1_bundle():
    bundle, z = _t1_bundle()
    good = _t1_requests(bundle, z, n=1)[0]
    rows = c2_history()
    native = copy.deepcopy(good)
    native["data"] = {"author_id": "c2", "records": [dict(r) for r in rows]}
    unconditioned = copy.deepcopy(good)
    unconditioned["data"] = {"author_id": "c2", "records": [{"t": int(r["t"]), "z": None, "y": int(r["observation"])} for r in rows[:32]]}
    full_clock = copy.deepcopy(good)
    full_clock["data"] = {"author_id": "c2", "records": [{"t": int(r["t"]), "z": -1.0, "y": int(r["observation"])} for r in rows]}
    wrong_target = copy.deepcopy(good)
    wrong_target["contract"]["target_id"] = "C2_next_probability"
    out = pub.score(bundle, [good, native, unconditioned, full_clock, wrong_target])
    assert out[0]["status"] == "SCORED"
    assert [r["code"] for r in out[1:]] == ["HIDDEN_OR_UNDECLARED_FIELD", "UNSUPPORTED_CONDITION", "WRONG_CLOCK", "WRONG_TARGET"]
