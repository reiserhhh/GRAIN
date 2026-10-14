#!/usr/bin/env python3
"""Cold source replay and enumerated state/weighted-least-squares score audit."""

import argparse
import gzip
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grain_research.gptpro_foundation_v3.f02 import generate_world
from grain_research.measurement_discovery.estimators import ordered_rows

STATES = list(itertools.product((0, 1), repeat=2))
FEATURES = np.array([[1, b, a, a * b] for a, b in STATES], float)


def source_replay(source, fitted):
    """Reconstruct source regression by augmented least squares and table SVD."""
    tables = []
    x = []
    y = []
    total = np.zeros((4, 2))
    initial = np.ones((2, 2))
    for rows in ordered_rows(source).values():
        values = [r["observation"] for r in rows]
        initial[values[0], values[1]] += 1
        counts = np.zeros((4, 2))
        for i in range(2, len(values)):
            a, b = values[i - 2 : i]
            outcome = values[i]
            counts[2 * a + b, outcome] += 1
            x.append([1, b, a, a * b])
            y.append(outcome)
        total += counts
        tables.append((counts[:, 1] + 1) / (counts.sum(axis=1) + 2))
    np.testing.assert_allclose(
        (total[:, 1] + 1) / (total.sum(axis=1) + 2),
        fitted["b1"]["source_p"],
        atol=1e-12,
    )
    coef = np.linalg.lstsq(
        np.vstack([x, np.eye(4)]), np.r_[y, np.zeros(4)], rcond=None
    )[0]
    np.testing.assert_allclose(coef, fitted["b2"]["coef"], atol=1e-10)
    tables = np.array(tables)
    center = tables.mean(axis=0)
    _, _, right = np.linalg.svd(tables - center, full_matrices=False)
    np.testing.assert_allclose(center, fitted["m"]["mean"], atol=1e-12)
    np.testing.assert_allclose(
        right[:2].T @ right[:2],
        fitted["m"]["right"].T @ fitted["m"]["right"],
        atol=1e-10,
    )
    return initial / initial.sum(), float(np.mean([r["observation"] for r in source]))


class Reference:
    """Explicit eight-state enumeration; no production filter/scorer imports."""

    def __init__(self, fitted, initial, mean, panel):
        self.fitted = fitted
        self.pair = initial.copy()
        self.mean = mean
        self.panel = panel
        self.history = []
        self.counts = np.zeros((4, 2))
        self.mass = self.total = 0.0

    def update(self, p, q):
        """Retain time and enumerate prior states, emitted bit and qualification."""
        t = len(self.history)
        if self.panel == "complete":
            if t >= 2:
                a_prob, b_prob = self.history[-2:]
                for a, b, y in itertools.product((0, 1), repeat=3):
                    self.counts[2 * a + b, y] += (
                        (a_prob if a else 1 - a_prob)
                        * (b_prob if b else 1 - b_prob)
                        * (p if y else 1 - p)
                    )
            if t >= 1:
                previous = self.history[-1]
                self.pair = np.array(
                    [
                        [
                            (previous if a else 1 - previous) * (p if b else 1 - p)
                            for b in (0, 1)
                        ]
                        for a in (0, 1)
                    ]
                )
        elif t < 2:
            for a, b in STATES:
                bit = a if t == 0 else b
                self.pair[a, b] *= q * (p if bit else 1 - p) + (1 - q) / 2
            self.pair /= self.pair.sum()
        else:
            joint = np.zeros((2, 2, 2))
            for a, b, y in itertools.product((0, 1), repeat=3):
                probability = self.fitted["b1"]["source_p"][2 * a + b]
                joint[a, b, y] = (
                    self.pair[a, b]
                    * (probability if y else 1 - probability)
                    * (q * (p if y else 1 - p) + (1 - q) / 2)
                )
            joint /= joint.sum()
            self.pair = np.zeros((2, 2))
            for a, b, y in itertools.product((0, 1), repeat=3):
                self.pair[b, y] += joint[a, b, y]
                self.counts[2 * a + b, y] += q * joint[a, b, y]
        self.history.append(p)
        self.mass += q
        self.total += q * p

    def scores(self):
        """Solve the augmented weighted observations, not accumulated normal equations."""
        x = []
        y = []
        for state in range(4):
            for bit in (0, 1):
                weight = np.sqrt(self.counts[state, bit])
                x.append(weight * FEATURES[state])
                y.append(weight * bit)
        coef = np.linalg.lstsq(
            np.vstack([x, np.eye(4)]), np.r_[y, self.fitted["b2"]["coef"]], rcond=None
        )[0]
        model = self.fitted["m"]
        raw = (self.counts[:, 1] + 1) / (self.counts.sum(axis=1) + 2)
        projected = model["mean"].copy()
        for axis in model["right"]:
            projected += np.dot(raw - model["mean"], axis) * axis
        answer = {
            "b1": 0.0,
            "b2": 0.0,
            "m": 0.0,
            "source_mean": self.mean,
            "personal_mean": self.total / self.mass if self.mass else self.mean,
        }
        for i, (a, b) in enumerate(STATES):
            answer["b1"] += self.pair[a, b] * self.fitted["b1"]["source_p"][i]
            answer["b2"] += self.pair[a, b] * np.dot(FEATURES[i], coef)
            answer["m"] += self.pair[a, b] * projected[i]
        return {k: float(np.clip(v, 1e-6, 1 - 1e-6)) for k, v in answer.items()}


def main():
    """Stream all frozen observations and predictions in causal order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.input / "protocol.json").read_text())
    if (
        json.loads((args.input / "summary.json").read_text())["status"]
        != "COMPUTED_AUDIT_PENDING"
    ):
        raise ValueError("complete run required")
    bundles = {}
    for world in ("c2_xor", "c2_iid"):
        for seed in protocol["seeds"]:
            with (args.input / f"source_{world}_{seed}.pkl").open("rb") as stream:
                fitted = pickle.load(stream)
            records, _ = generate_world(world, seed, counts=protocol["counts"])
            source = [r for r in records if r["role"] == "source"]
            initial, mean = source_replay(source, fitted)
            bundles[(world, seed)] = (fitted, initial, mean)
    maximum = 0.0
    checks = 0
    previous = None
    with (
        gzip.open(args.input / "observations.jsonl.gz", "rt") as observations,
        gzip.open(args.input / "predictions.jsonl.gz", "rt") as stream,
    ):
        predictions = iter(stream)
        for line in observations:
            row = json.loads(line)
            key = (row["author_id"], row["panel"])
            if key != previous:
                # Event IDs include the public simulator world and seed, never a scoring feature.
                parts = row["event_id"].split("_")
                world = "_".join(parts[:2])
                seed = int(parts[2])
                fitted, initial, mean = bundles[(world, seed)]
                refs = {
                    c: Reference(fitted, initial, mean, row["panel"])
                    for c in ("oracle", "soft", "hard")
                }
                previous = key
            for channel, ref in refs.items():
                if row["t"] >= 12:
                    for candidate, value in ref.scores().items():
                        saved = json.loads(next(predictions))
                        if (
                            saved["event_id"],
                            saved["panel"],
                            saved["channel"],
                            saved["candidate"],
                        ) != (row["event_id"], row["panel"], channel, candidate):
                            raise ValueError("prediction identity/order")
                        maximum = max(maximum, abs(value - saved["p"]))
                        if abs(value - saved["p"]) > 1e-10:
                            raise ValueError("score discrepancy")
                        checks += 1
                p = (
                    row["material_pole"]
                    if channel == "oracle"
                    else row["pole_score"]
                    if channel == "soft"
                    else int(row["pole_score"] >= 0.5)
                )
                q = (
                    1
                    if row["panel"] == "complete"
                    else row["qualification"]
                    if channel == "oracle"
                    else row["qualification_score"]
                    if channel == "soft"
                    else int(row["qualification_score"] >= 0.5)
                )
                ref.update(p, q)
        if next(predictions, None) is not None:
            raise ValueError("unconsumed predictions")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "SOURCE_AND_ENUMERATED_STATE_SCORE_REPLAY_PASS",
        "source_bundles": len(bundles),
        "predictions": checks,
        "max_difference": maximum,
        "scope": "same-agent alternate enumeration/augmented least squares;no production filter/scorer imported",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
