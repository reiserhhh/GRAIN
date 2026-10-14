#!/usr/bin/env python3
"""Independent four-state transition-matrix replay of every B1 application row."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


class Reference:
    """Row-vector Markov recursion, separate from production tensor filtering."""

    def __init__(self, fits, complete):
        self.p = np.array([f["b1"]["source_p"] for f in fits])
        self.w = np.array([f["initial_pair"] for f in fits]).reshape(-1, 4)
        self.matrix = np.zeros((len(fits), 4, 4))
        for state in range(4):
            self.matrix[:, state, 2 * (state % 2)] = 1 - self.p[:, state]
            self.matrix[:, state, 2 * (state % 2) + 1] = self.p[:, state]
        self.complete, self.history = complete, []

    def observe(self, p, q):
        """Update the same observation law by explicit state addresses."""
        t = len(self.history)
        if self.complete:
            if t:
                a = self.history[-1]
                self.w[:] = [(1 - a) * (1 - p), (1 - a) * p, a * (1 - p), a * p]
        else:
            likelihood = q * np.array([1 - p, p]) + (1 - q) / 2
            if t >= 2:
                self.w = np.einsum("bi,bij->bj", self.w, self.matrix)
            axis = np.array([0, 0, 1, 1]) if t == 0 else np.array([0, 1, 0, 1])
            self.w *= likelihood[axis]
            self.w /= self.w.sum(axis=1, keepdims=True)
        self.history.append(p)

    def components(self):
        """Raw second moments independently reconstruct variance components."""
        mu = np.einsum("bs,bs->b", self.w, self.p)
        second = np.einsum("bs,bs->b", self.w, self.p**2)
        return {
            "mean": mu.mean(),
            "state_variance": (second - mu**2).mean(),
            "source_variance": (mu**2).mean() - mu.mean() ** 2,
            "event_variance": (mu - second).mean(),
            "probability_variance": second.mean() - mu.mean() ** 2,
            "outcome_variance": mu.mean() - mu.mean() ** 2,
        }


def main():
    """Read frozen outputs and reject misalignment or altered inputs."""
    parser = argparse.ArgumentParser()
    for name in ("input", "observations", "source", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.input / "protocol_before_run.json").read_text())
    for item in protocol["files"]:
        if (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            != item["sha256"]
        ):
            raise ValueError("frozen input mismatch")
    models = {}
    for world in ("c2_xor", "c2_iid"):
        for seed in (20260915, 20260916, 20260917):
            models[world, seed] = [
                json.loads((args.source / f"{world}_{seed}_b{b:03d}.json").read_text())[
                    "fit"
                ]
                for b in range(200)
            ]
    maximum, count = 0.0, 0
    with (
        gzip.open(args.input / "predictions.csv.gz", "rt") as saved,
        gzip.open(args.observations, "rt") as stream,
    ):
        expected = iter(csv.DictReader(saved))
        for _, group in itertools.groupby(
            (json.loads(x) for x in stream), key=lambda r: (r["author_id"], r["panel"])
        ):
            records = list(group)
            if records[0]["role"] != "application":
                continue
            parts = records[0]["event_id"].split("_")
            world, seed = "_".join(parts[:2]), int(parts[2])
            panel = records[0]["panel"]
            histories = {
                c: Reference(models[world, seed], panel == "complete")
                for c in ("oracle", "soft", "hard")
            }
            for row in records:
                for c, h in histories.items():
                    if row["t"] >= 12:
                        old = next(expected)
                        assert (
                            old["world"],
                            int(old["seed"]),
                            old["author_id"],
                            old["panel"],
                            old["channel"],
                            int(old["t"]),
                        ) == (world, seed, row["author_id"], panel, c, row["t"])
                        values = h.components()
                        maximum = max(
                            maximum,
                            *(abs(v - float(old[k])) for k, v in values.items()),
                        )
                        count += 1
                    p, q = (
                        (row["material_pole"], row["qualification"])
                        if c == "oracle"
                        else (row["pole_score"], row["qualification_score"])
                    )
                    if c == "hard":
                        p, q = float(p >= 0.5), float(q >= 0.5)
                    h.observe(p, 1 if panel == "complete" else q)
        assert next(expected, None) is None
    assert count == 259200 and maximum < 1e-12
    result = {
        "status": "FULL_APPLICATION_FOUR_STATE_REPLAY_PASS",
        "rows": count,
        "source_conditioned_predictions": count * 200,
        "max_error": maximum,
        "scope": "all analytic means/variance components and input hashes; not new calibration evidence or MC RNG replay",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
