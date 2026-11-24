#!/usr/bin/env python3
"""Wall-clock schedules for a discrete observation arriving after an unobserved interval."""

import numpy as np


def run_cases(ref, cases, transport="ssh"):
    """Numeric worker only, same-length batches and one matched schedule per case."""
    from . import liquid_instrument as li

    if not cases or ref["worker"] != li.digest(li.WORKER.read_text()):
        raise ValueError("missing episodes or incompatible frozen reader")
    code = (
        li.WORKER.read_text().split('if __name__ == "__main__":')[0]
        + li.WORKER.with_name("liquid_time_worker.py").read_text()
    )
    return li.call(
        {"weights": ref["weights"], "cases": cases}, transport, worker_code=code
    )


def score(doc, ref, calibration, transport="ssh"):
    from . import liquid_instrument as li
    from .liquid_task_paths import encode

    if (
        ref.get("version") != "task-path-ltc-1"
        or doc["axes"] != ref["axes"]
        or doc["basis"] != ref["basis"]
        or calibration["reader"] != li.digest(ref)
        or doc["time_unit"] != "seconds"
    ):
        raise ValueError("incompatible readout, seconds calibration or axes")
    result = {}
    for ep in doc["episodes"]:
        key = ep["id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("unique nonempty episode ID required")
        x = np.asarray(ep["values"], float)
        g = schedules(ep["times"], calibration["seconds_per_unit"])
        if x.shape != (len(g["wall"]), len(ref["axes"])) or not np.isfinite(x).all():
            raise ValueError("one observed coordinate vector per timestamp required")
        g["wall_two_steps"] = g["wall"]
        case = {
            "x": encode(x, ref).tolist(),
            "gaps": {a: b.tolist() for a, b in g.items()},
        }
        arms = run_cases(ref, [case], transport)
        result[key] = {
            a: {name: array[0] for name, array in v.items()} for a, v in arms.items()
        }
    return {
        "reader": calibration["reader"],
        "seconds_per_unit": calibration["seconds_per_unit"],
        "readout": "node-membership forecast of the frozen reader, not observed personality coordinates",
        "episodes": result,
    }


def clock_unit(source_gaps):
    """Median of person medians. Seconds per model time unit are a reference convention."""
    medians = []
    for values in source_gaps:
        a = np.asarray(values, float)
        if a.ndim != 1 or not np.isfinite(a).all() or np.any(a <= 0):
            raise ValueError("source gaps must be positive finite observed seconds")
        if len(a):
            medians.append(np.median(a))
    if not medians:
        raise ValueError("no timed calibration people")
    return float(np.median(medians))


def schedules(stamps, unit):
    """Negative/reset or unknown times are caller-visible breaks. A simultaneous event has no idle evolution."""
    t = np.asarray(stamps, float)
    if (
        t.ndim != 1
        or not len(t)
        or not np.isfinite(t).all()
        or np.any(np.diff(t) < 0)
        or not np.isfinite(unit)
        or unit <= 0
    ):
        raise ValueError(
            "one finite ordered episode and an explicit positive clock unit required"
        )
    elapsed = np.diff(t) / unit
    positive = elapsed > 0
    shuffled = elapsed.copy()
    shuffled[positive] = elapsed[positive][::-1]
    return {
        k: np.r_[0.0, v]
        for k, v in {
            "event": np.zeros_like(elapsed),
            "equal": positive.astype(float),
            "wall": elapsed,
            "reversed_gaps": shuffled,
            "double_unit": elapsed / 2,
        }.items()
    }
