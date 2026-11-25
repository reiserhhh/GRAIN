#!/usr/bin/env python3
"""Fit/score an LTC opportunity clock on a frozen, signed numerical factor series."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax

from . import liquid_instrument as li
from .liquid_frozen_bank import mix

VERSION = "signed-axis-clock-v1"


def inputs(doc):
    names = doc["axes"]
    if (
        doc.get("clock") != "observed_opportunity"
        or not names
        or len(set(names)) != len(names)
    ):
        raise ValueError("named frozen axes and an explicit observation clock required")
    paths = {}
    for seq in doc["series"]:
        x = np.asarray(seq["values"], float)
        ts = seq["time"]
        if (
            x.ndim != 2
            or len(x) != len(ts)
            or len(ts) == 0
            or x.shape[1] != len(names)
            or not np.isfinite(x).all()
        ):
            raise ValueError(
                "missing or invalid numerical observation; split incomplete series explicitly"
            )
        if list(ts) != sorted(set(ts)) or seq["id"] in paths:
            raise ValueError("non-monotonic or duplicate opportunity / stream")
        paths[seq["id"]] = (ts, x)
    if not paths:
        raise ValueError("empty source")
    return paths


def encode_bipolar_origin(z, ref=None):
    if ref is not None:
        z = (np.asarray(z, float) - ref["mean"]) / np.asarray(ref["scale"])
    z = np.asarray(z, float)
    if z.ndim != 2 or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError("finite (events, axes) required")
    return softmax(np.column_stack((z, -z, np.zeros(len(z)))), axis=1)


def encode(x, ref):
    z = (x - ref["mean"]) / np.array(ref["scale"])
    # The fixed zero logit preserves the original signed origin (unlike ordinary softmax).
    p = (
        encode_bipolar_origin(z)
        if ref.get("encoding") == "bipolar_origin"
        else softmax(np.column_stack([z, np.zeros(len(z))]), axis=1)
    )
    if p.min() < np.finfo(np.float32).tiny:
        raise ValueError("outside float32 clock-input support; no silent clipping")
    return p


def decode_bipolar_origin(p, ref):
    """Invert symmetric (2K+1)-simplex probabilities back to original signed scale."""
    p_arr = np.asarray(p, dtype=float)
    k = (p_arr.shape[1] - 1) // 2
    if p_arr.ndim != 2 or 2 * k + 1 != p_arr.shape[1]:
        raise ValueError("expected (N, 2K+1) bipolar simplex probabilities")
    z = 0.5 * np.log(
        np.maximum(p_arr[:, :k], 1e-300) / np.maximum(p_arr[:, k : 2 * k], 1e-300)
    )
    return z * np.asarray(ref["scale"], dtype=float) + np.asarray(
        ref["mean"], dtype=float
    )


def bipolar_polarization(p):
    """Signed opposite-channel difference p_+ - p_- (strictly odd under z -> -z)."""
    p_arr = np.asarray(p, dtype=float)
    k = (p_arr.shape[1] - 1) // 2
    if p_arr.ndim != 2 or 2 * k + 1 != p_arr.shape[1]:
        raise ValueError("expected (N, 2K+1) bipolar simplex probabilities")
    return p_arr[:, :k] - p_arr[:, k : 2 * k]


def fit(doc, transport, *, encoding="signed_origin"):
    if encoding not in ("signed_origin", "bipolar_origin"):
        raise ValueError("unknown encoding")
    paths = inputs(doc)
    all_x = np.concatenate([x for _, x in paths.values()])
    scale = all_x.std(0)
    ref = {
        "version": VERSION,
        **({"encoding": encoding} if encoding != "signed_origin" else {}),
        "axes": doc["axes"],
        "coordinate_system": doc["coordinate_system"],
        "worker_digest": li.digest(li.WORKER.read_text()),
        "mean": all_x.mean(0).tolist(),
        "scale": np.where(scale > 1e-8, scale, 1.0).tolist(),
        "constant_axes": [doc["axes"][i] for i in np.flatnonzero(scale <= 1e-8)],
        "source_digest": li.digest(doc),
        "source_latest_time": max(ts[-1] for ts, _ in paths.values()),
    }
    windows = []
    for _, x in paths.values():
        p = encode(x, ref)
        for i in range(0, len(p) - 1, 12):
            # Adjacent training windows share only their boundary observation.
            windows.append(p[i : i + 13].tolist())
    if not windows:
        raise ValueError("source has no transitions")
    ref.update(li.call({"mode": "fit", "paths": windows}, transport))
    return ref


def score(doc, ref, transport):
    paths = inputs(doc)
    if (
        ref["axes"] != doc["axes"]
        or ref["coordinate_system"] != doc["coordinate_system"]
        or ref["version"] != VERSION
        or ref["worker_digest"] != li.digest(li.WORKER.read_text())
    ):
        raise ValueError("incompatible frozen coordinates/worker")
    if any(ts[0] <= ref["source_latest_time"] for ts, _ in paths.values()):
        raise ValueError(
            "cold temporal application must start after source calibration ends"
        )
    extension = Path(__file__).with_name("liquid_clock_worker.py").read_text()
    worker = li.WORKER.read_text().split('if __name__ == "__main__":')[0] + extension
    clocks = li.call(
        {
            "paths": [encode(x, ref).tolist() for _, x in paths.values()],
            "weights": ref["weights"],
        },
        transport,
        worker_code=worker,
    )
    out = {}
    for (key, (ts, x)), r in zip(paths.items(), clocks["paths"], strict=True):
        out[key] = {
            "time": ts,
            "original": x.tolist(),
            "fixed_half": mix(x, np.full((len(x), 1), 0.5)).tolist(),
            "liquid_in_original_units": mix(x, r).tolist(),
            "carry_per_cell": r,
        }
    return {
        "version": VERSION,
        "axes": ref["axes"],
        "series": out,
        "reference_digest": li.digest(ref),
        "input_digest": li.digest(doc),
        "official_state_max_error": clocks["official_state_max_error"],
        "clock": "one retained observation; each series starts independently at its first observation",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "score"))
    for k in ("input", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--reference")
    p.add_argument("--transport", choices=("local", "ssh"), default="local")
    a = p.parse_args()
    if Path(a.out).exists() or (a.mode == "score" and not a.reference):
        p.error("use a new output and a frozen reference for score")
    doc = li.load(a.input)
    li.save(
        a.out,
        fit(doc, a.transport)
        if a.mode == "fit"
        else score(doc, li.load(a.reference), a.transport),
    )


if __name__ == "__main__":
    main()
